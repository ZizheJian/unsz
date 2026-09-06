import csv,math,os,re,subprocess,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean")
compressor=project_root/"build/s1_compress"
decompressor=project_root/"build/s1_decompress"
field_vec=("layerThickness","salinity","temperature","velocityMeridional","velocityX","velocityY","velocityZ","velocityZonal","zTop")
rel_eb_vec=(1e-3,9e-4)
result_path=project_root/"py_codes/trash/results/ocean_spatial_adjust_ref4_screen.csv"
summary_path=project_root/"py_codes/trash/results/ocean_spatial_adjust_ref4_summary.csv"
prediction_ggd_summary_path=project_root/"py_codes/trash/results/ocean_spatial_adjust_prediction_ggd_summary.csv"

def run_command(command):
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(str(value) for value in command)}")
    return result.stdout

def read_float(pattern,output,name):
    match=re.search(pattern,output,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def cleanup(data_path):
    for suffix in (".unsz",".unsz.bin"):
        path=Path(str(data_path)+suffix)
        if path.is_file() or path.is_symlink():
            path.unlink()

def read_results():
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

def row_exists(row_vec,field,config,rel_eb):
    return any(row["field"]==field and row["config"]==config and math.isclose(float(row["rel_eb"]),rel_eb,rel_tol=1e-12) for row in row_vec)

def benchmark(field,config,test_dir,rel_eb,prediction_structure_path):
    arguments=["-data-dir",test_dir,"-variable",field,"-time-step","{0}","-M","REL",rel_eb,"-dim",3,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",10,"-actual-ref-num",10,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",8,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental","-global-order-before-round-split-experimental"]
    if config=="ref4_adjust":
        arguments.extend(("-adjust-coord-matrix-file",project_root/"adjust_coord_matrices"/f"Ocean_{field}_spatial_ref4_candidate.txt"))
    elif config=="existing_adjust":
        arguments.extend(("-adjust-coord-matrix-file",project_root/"adjust_coord_matrices"/f"Ocean_{field}_spatial.txt"))
    elif config=="prediction_ggd":
        arguments.extend(("-adjust-coord-matrix-file",project_root/"adjust_coord_matrices"/f"Ocean_{field}_spatial_prediction_ggd_ref10_candidate.txt"))
    compression_output=run_command([compressor,*arguments])
    decompression_output=run_command([decompressor,*arguments])
    psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
    cleanup(Path(test_dir)/f"{field}.dat.0")
    return {"field":field,"config":config,"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"BR"),"psnr":psnr}

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

def main():
    result_vec=read_results()
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    for field in field_vec:
        with tempfile.TemporaryDirectory(prefix=f"unsz_ocean_adjust_{field}_",dir=temporary_root) as temporary_dir:
            test_dir=Path(temporary_dir)
            (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
            (test_dir/f"{field}.dat.0").symlink_to(data_dir/f"{field}.dat.0")
            config_vec=list(("original_coord","ref4_adjust","existing_adjust") if field in ("temperature","velocityZonal") else ("original_coord","ref4_adjust"))
            if (project_root/"adjust_coord_matrices"/f"Ocean_{field}_spatial_prediction_ggd_ref10_candidate.txt").is_file():
                config_vec.append("prediction_ggd")
            for config in config_vec:
                prediction_structure_path=test_dir/f"prediction_{config}.bin"
                for rel_eb in rel_eb_vec:
                    if row_exists(result_vec,field,config,rel_eb):
                        continue
                    row=benchmark(field,config,test_dir,rel_eb,prediction_structure_path)
                    result_vec.append(row)
                    write_csv(result_path,result_vec)
                    print(f"{field} {config} REL={rel_eb:g}:BR={row['br']:.9f},PSNR={row['psnr']:.6f}",flush=True)
    summary_vec=[]
    for field in field_vec:
        original_vec=[row for row in result_vec if row["field"]==field and row["config"]=="original_coord"]
        adjust_vec=[row for row in result_vec if row["field"]==field and row["config"]=="ref4_adjust"]
        target_psnr=next(float(row["psnr"]) for row in original_vec if math.isclose(float(row["rel_eb"]),rel_eb_vec[0]))
        original_br=interpolate_log_br(original_vec,target_psnr)
        adjust_br=interpolate_log_br(adjust_vec,target_psnr)
        summary_vec.append({"field":field,"target_psnr":target_psnr,"original_coord_br":original_br,"ref4_adjust_br":adjust_br,"br_change_percent":(adjust_br/original_br-1)*100})
    write_csv(summary_path,summary_vec)
    for row in summary_vec:
        print(f"Ocean-{row['field']}:同PSNR下adjust相对原坐标的BR变化={row['br_change_percent']:.6f}%",flush=True)
    for field in ("temperature","velocityZonal"):
        original_vec=[row for row in result_vec if row["field"]==field and row["config"]=="original_coord"]
        existing_vec=[row for row in result_vec if row["field"]==field and row["config"]=="existing_adjust"]
        ref4_vec=[row for row in result_vec if row["field"]==field and row["config"]=="ref4_adjust"]
        target_psnr=next(float(row["psnr"]) for row in original_vec if math.isclose(float(row["rel_eb"]),rel_eb_vec[0]))
        original_br=interpolate_log_br(original_vec,target_psnr)
        existing_br=interpolate_log_br(existing_vec,target_psnr)
        ref4_br=interpolate_log_br(ref4_vec,target_psnr)
        print(f"Ocean-{field}:旧正式矩阵相对原坐标={(existing_br/original_br-1)*100:.6f}%,ref4相对旧正式矩阵={(ref4_br/existing_br-1)*100:.6f}%",flush=True)
    prediction_ggd_summary=[]
    for field in field_vec:
        prediction_ggd_vec=[row for row in result_vec if row["field"]==field and row["config"]=="prediction_ggd"]
        if len(prediction_ggd_vec)<2:
            continue
        original_vec=[row for row in result_vec if row["field"]==field and row["config"]=="original_coord"]
        ref4_vec=[row for row in result_vec if row["field"]==field and row["config"]=="ref4_adjust"]
        target_psnr=next(float(row["psnr"]) for row in original_vec if math.isclose(float(row["rel_eb"]),rel_eb_vec[0]))
        original_br=interpolate_log_br(original_vec,target_psnr)
        ref4_br=interpolate_log_br(ref4_vec,target_psnr)
        prediction_ggd_br=interpolate_log_br(prediction_ggd_vec,target_psnr)
        prediction_ggd_summary.append({"field":field,"target_psnr":target_psnr,"original_coord_br":original_br,"ref4_adjust_br":ref4_br,"prediction_ggd_br":prediction_ggd_br,"prediction_ggd_vs_original_percent":(prediction_ggd_br/original_br-1)*100,"prediction_ggd_vs_ref4_percent":(prediction_ggd_br/ref4_br-1)*100})
    if prediction_ggd_summary:
        write_csv(prediction_ggd_summary_path,prediction_ggd_summary)
        for row in prediction_ggd_summary:
            print(f"Ocean-{row['field']}:码长loss相对原坐标={row['prediction_ggd_vs_original_percent']:.6f}%,相对二次方ref4={row['prediction_ggd_vs_ref4_percent']:.6f}%",flush=True)

if __name__=="__main__":
    main()
