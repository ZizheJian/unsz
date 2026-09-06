import argparse,csv,math,os,re,subprocess,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data")
compressor=project_root/"build/s1_compress"
decompressor=project_root/"build/s1_decompress"
rel_eb_vec=(1e-3,9e-4)
dataset_config={
    "LES":{"dim":3,"field_vec":("velocity_x","velocity_y","velocity_z")},
    "Tsunami":{"dim":2,"field_vec":("momentum_x","momentum_y","velocity_x","velocity_y")},
}

def run_command(command):
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(str(value) for value in command)}")
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def read_results(result_path):
    if not result_path.is_file():
        return []
    with result_path.open(newline="") as result_file:
        return list(csv.DictReader(result_file))

def write_csv(path,row_vec):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary_path=path.with_suffix(path.suffix+".tmp")
    with temporary_path.open("w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=row_vec[0].keys())
        writer.writeheader()
        writer.writerows(row_vec)
    temporary_path.replace(path)

def row_exists(row_vec,dataset,field,config,rel_eb):
    return any(row["dataset"]==dataset and row["field"]==field and row["config"]==config and math.isclose(float(row["rel_eb"]),rel_eb) for row in row_vec)

def cleanup(data_path):
    for suffix in (".unsz",".unsz.bin"):
        path=Path(str(data_path)+suffix)
        if path.is_file() or path.is_symlink():
            path.unlink()

def benchmark(dataset,field,dim,matrix_config,test_dir,rel_eb,prediction_structure_path):
    matrix_path=project_root/"adjust_coord_matrices"/f"{dataset}_{field}_{matrix_config['suffix']}.txt"
    if not matrix_path.is_file():
        raise FileNotFoundError(f"缺少坐标变换矩阵：{matrix_path}")
    arguments=["-data-dir",test_dir,"-variable",field,"-time-step","{0}","-M","REL",rel_eb,"-dim",dim,"-adjust-coord-matrix-file",matrix_path,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",10,"-actual-ref-num",10,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",8,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental","-global-order-before-round-split-experimental"]
    structure_existed=prediction_structure_path.is_file()
    compression_output=run_command([compressor,*arguments])
    decompression_output=run_command([decompressor,*arguments])
    psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
    cleanup(Path(test_dir)/f"{field}.dat.0")
    return {"dataset":dataset,"field":field,"config":matrix_config["config"],"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"BR"),"psnr":psnr,"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间"),"prediction_structure_mode":"load" if structure_existed else "generate","matrix_path":str(matrix_path.relative_to(project_root))}

def interpolate_log_br(row_vec,target_psnr):
    sorted_row_vec=sorted(row_vec,key=lambda row:float(row["psnr"]))
    selected_pair=None
    for lower,upper in zip(sorted_row_vec[:-1],sorted_row_vec[1:]):
        if float(lower["psnr"])<=target_psnr<=float(upper["psnr"]):
            selected_pair=(lower,upper)
            break
    if selected_pair is None:
        selected_pair=tuple(sorted_row_vec[:2] if target_psnr<float(sorted_row_vec[0]["psnr"]) else sorted_row_vec[-2:])
    lower,upper=selected_pair
    ratio=(target_psnr-float(lower["psnr"]))/(float(upper["psnr"])-float(lower["psnr"]))
    return math.exp(math.log(float(lower["br"]))*(1-ratio)+math.log(float(upper["br"]))*ratio)

def summarize(result_vec,matrix_config_vec,summary_path):
    candidate_config=matrix_config_vec[1]["config"]
    summary_vec=[]
    for dataset,config in dataset_config.items():
        for field in config["field_vec"]:
            ref5_vec=[row for row in result_vec if row["dataset"]==dataset and row["field"]==field and row["config"]=="ref5"]
            candidate_vec=[row for row in result_vec if row["dataset"]==dataset and row["field"]==field and row["config"]==candidate_config]
            if len(ref5_vec)<2 or len(candidate_vec)<2:
                continue
            target_psnr=next(float(row["psnr"]) for row in ref5_vec if math.isclose(float(row["rel_eb"]),rel_eb_vec[0]))
            ref5_br=interpolate_log_br(ref5_vec,target_psnr)
            candidate_br=interpolate_log_br(candidate_vec,target_psnr)
            summary_vec.append({"dataset":dataset,"field":field,"target_psnr":target_psnr,"ref5_matched_br":ref5_br,f"{candidate_config}_matched_br":candidate_br,f"{candidate_config}_br_change_percent":(candidate_br/ref5_br-1)*100})
    if summary_vec:
        write_csv(summary_path,summary_vec)
        for row in summary_vec:
            print(f"{row['dataset']}-{row['field']}:同PSNR下{candidate_config}相对ref5的BR变化={row[f'{candidate_config}_br_change_percent']:.6f}%",flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-dataset",choices=dataset_config,required=True)
    parser.add_argument("-field",action="append",default=[])
    parser.add_argument("-candidate-ref-num",type=int,default=10)
    args=parser.parse_args()
    if args.candidate_ref_num<=0 or args.candidate_ref_num==5:
        raise ValueError("candidate-ref-num必须是正数且不能等于5")
    candidate_config=f"ref{args.candidate_ref_num}"
    matrix_config_vec=({"config":"ref5","suffix":"spatial"},{"config":candidate_config,"suffix":f"spatial_{candidate_config}_candidate"})
    result_path=project_root/f"py_codes/trash/results/ref5_{candidate_config}_spatial_matrix_benchmark.csv"
    summary_path=project_root/f"py_codes/trash/results/ref5_{candidate_config}_spatial_matrix_summary.csv"
    config=dataset_config[args.dataset]
    field_vec=[field for field in config["field_vec"] if not args.field or field in args.field]
    if not field_vec:
        raise ValueError("没有符合筛选条件的field")
    result_vec=read_results(result_path)
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    for field in field_vec:
        source_dir=data_root/args.dataset
        with tempfile.TemporaryDirectory(prefix=f"unsz_ref_matrix_{args.dataset}_{field}_",dir=temporary_root) as temporary_dir:
            test_dir=Path(temporary_dir)
            (test_dir/"coords.dat").symlink_to(source_dir/"coords.dat")
            (test_dir/f"{field}.dat.0").symlink_to(source_dir/f"{field}.dat.0")
            for matrix_config in matrix_config_vec:
                prediction_structure_path=test_dir/f"prediction_{matrix_config['config']}.bin"
                for rel_eb in rel_eb_vec:
                    if row_exists(result_vec,args.dataset,field,matrix_config["config"],rel_eb):
                        continue
                    row=benchmark(args.dataset,field,config["dim"],matrix_config,test_dir,rel_eb,prediction_structure_path)
                    result_vec.append(row)
                    write_csv(result_path,result_vec)
                    print(f"{args.dataset}-{field} {matrix_config['config']} REL={rel_eb:g}:BR={row['br']:.9f},PSNR={row['psnr']:.6f},pre={row['preprocessing_time']:.6f}s",flush=True)
            ref5_target_psnr=next(float(row["psnr"]) for row in result_vec if row["dataset"]==args.dataset and row["field"]==field and row["config"]=="ref5" and math.isclose(float(row["rel_eb"]),rel_eb_vec[0]))
            candidate_vec=[row for row in result_vec if row["dataset"]==args.dataset and row["field"]==field and row["config"]==candidate_config]
            if not min(float(row["psnr"]) for row in candidate_vec)<=ref5_target_psnr<=max(float(row["psnr"]) for row in candidate_vec):
                closest_row=min(candidate_vec,key=lambda row:abs(float(row["psnr"])-ref5_target_psnr))
                matching_rel_eb=float(closest_row["rel_eb"])*10**((float(closest_row["psnr"])-ref5_target_psnr)/20)
                prediction_structure_path=test_dir/f"prediction_{candidate_config}.bin"
                if not row_exists(result_vec,args.dataset,field,candidate_config,matching_rel_eb):
                    row=benchmark(args.dataset,field,config["dim"],matrix_config_vec[1],test_dir,matching_rel_eb,prediction_structure_path)
                    result_vec.append(row)
                    write_csv(result_path,result_vec)
                    print(f"{args.dataset}-{field} {candidate_config}匹配点 REL={matching_rel_eb:.9g}:BR={row['br']:.9f},PSNR={row['psnr']:.6f},pre={row['preprocessing_time']:.6f}s",flush=True)
    summarize(result_vec,matrix_config_vec,summary_path)

if __name__=="__main__":
    main()
