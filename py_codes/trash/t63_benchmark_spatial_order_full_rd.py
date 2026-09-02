import argparse,csv,math,os,re,subprocess,sys,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))

data_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data")
compressor=project_root/"build/s1_compress"
decompressor=project_root/"build/s1_decompress"
umc_compressor=Path("/home/x-zjian1/umc/build/test/test_compress")
umc_decompressor=Path("/home/x-zjian1/umc/build/test/test_decompress")
matrix_dir=project_root/"py_codes/trash/results/adjust_coord_history/spatial_parameter_screen"
result_path=project_root/"py_codes/trash/results/spatial_order_full_rd.csv"
geometry_path=project_root/"py_codes/trash/results/spatial_order_geometry.csv"
summary_path=project_root/"py_codes/trash/results/spatial_order_full_rd_summary.csv"
rel_eb_vec=(1e-2,3e-3,1e-3,3e-4,1e-4)
dataset_vec=(
    {"dataset":"LES","field":"velocity_x","dim":3,"umc_from_edge":0},
    {"dataset":"LES","field":"velocity_y","dim":3,"umc_from_edge":0},
    {"dataset":"LES","field":"velocity_z","dim":3,"umc_from_edge":0},
    {"dataset":"Tsunami","field":"momentum_x","dim":2,"umc_from_edge":1},
    {"dataset":"Tsunami","field":"momentum_y","dim":2,"umc_from_edge":1},
    {"dataset":"Tsunami","field":"velocity_x","dim":2,"umc_from_edge":1},
    {"dataset":"Tsunami","field":"velocity_y","dim":2,"umc_from_edge":1},
)
config_vec=(
    {"config":"random_10_8","global_order":False},
    {"config":"global_10_8","global_order":True},
)
result_field_vec=("dataset","field","method","config","rel_eb","br","psnr","preprocessing_time","compression_time","decompression_time","preprocessing_mb_s","compression_mb_s","decompression_mb_s","prediction_structure_mode","matrix")
geometry_field_vec=("dataset","field","config","round","target_num","span_ratio","ref_num","mean_ref_distance","mean_ref_distance_ratio","mean_nearest_ref_distance","mean_nearest_ref_distance_ratio")

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

def read_csv(path):
    if not path.is_file():
        return []
    with path.open(newline="") as file:
        return list(csv.DictReader(file))

def write_csv(path,field_vec,row_vec):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary_path=path.with_suffix(path.suffix+".tmp")
    with temporary_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=field_vec)
        writer.writeheader()
        writer.writerows(row_vec)
    temporary_path.replace(path)

def load_matrix(dataset):
    path=matrix_dir/f"{dataset['dataset']}_{dataset['field']}_gauss_newton.npz"
    if not path.is_file():
        raise FileNotFoundError(f"缺少训练好的坐标变换矩阵：{path}")
    with np.load(path) as history:
        return history["matrix"][-1]

def calculate_bd_rate(base_result_vec,test_result_vec):
    base_psnr=np.asarray([float(result["psnr"]) for result in base_result_vec])
    test_psnr=np.asarray([float(result["psnr"]) for result in test_result_vec])
    base_log_br=np.log(np.asarray([float(result["br"]) for result in base_result_vec]))
    test_log_br=np.log(np.asarray([float(result["br"]) for result in test_result_vec]))
    min_psnr=max(base_psnr.min(),test_psnr.min())
    max_psnr=min(base_psnr.max(),test_psnr.max())
    if not min_psnr<max_psnr:
        raise RuntimeError("两条RD曲线没有重叠的PSNR区间")
    base_polynomial=np.polyfit(base_psnr,base_log_br,3)
    test_polynomial=np.polyfit(test_psnr,test_log_br,3)
    base_integral=np.polyval(np.polyint(base_polynomial),max_psnr)-np.polyval(np.polyint(base_polynomial),min_psnr)
    test_integral=np.polyval(np.polyint(test_polynomial),max_psnr)-np.polyval(np.polyint(test_polynomial),min_psnr)
    return (math.exp((test_integral-base_integral)/(max_psnr-min_psnr))-1)*100,min_psnr,max_psnr

def get_unsz_arguments(dataset,config,matrix,test_dir,rel_eb,prediction_structure_path):
    matrix_parameter=matrix[np.tril_indices(dataset["dim"])]
    arguments=["-data-dir",test_dir,"-variable",dataset["field"],"-time-step","{0}","-M","REL",str(rel_eb),"-dim",str(dataset["dim"]),"-adjust-coord-matrix",*[f"{value:.17g}" for value in matrix_parameter],"-reverse-mean-distance-only-order-experimental","-prediction-structure",prediction_structure_path,"-max-ref-num","10","-actual-ref-num","10","-inverse-distance-power","3.61","-d-only-cache-refill-threshold-experimental","8"]
    if config["global_order"]:
        arguments.append("-global-order-before-round-split-experimental")
    return arguments

def cleanup_output(data_path,suffix_vec):
    for suffix in suffix_vec:
        path=Path(str(data_path)+suffix)
        if path.is_file() or path.is_symlink():
            path.unlink()

def benchmark_unsz(dataset,config,matrix,test_dir,rel_eb,prediction_structure_path,data_path):
    structure_existed=prediction_structure_path.is_file()
    arguments=get_unsz_arguments(dataset,config,matrix,test_dir,rel_eb,prediction_structure_path)
    compression_output=run_command([compressor,*arguments])
    br=read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"UNSZ总BR")
    psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"UNSZ总PSNR")
    preprocessing_time=read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"UNSZ预处理时间")
    compression_time=read_float(r"^total compression time= ([^ ]+) s$",compression_output,"UNSZ压缩时间")
    decompression_output=run_command([decompressor,*arguments])
    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"UNSZ解压PSNR")
    decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"UNSZ解压时间")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"UNSZ压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
    data_bytes=data_path.stat().st_size
    cleanup_output(Path(test_dir)/f"{dataset['field']}.dat.0",(".unsz",".unsz.bin"))
    return {"dataset":dataset["dataset"],"field":dataset["field"],"method":"unsz","config":config["config"],"rel_eb":rel_eb,"br":br,"psnr":psnr,"preprocessing_time":preprocessing_time,"compression_time":compression_time,"decompression_time":decompression_time,"preprocessing_mb_s":data_bytes/preprocessing_time/1e6,"compression_mb_s":data_bytes/compression_time/1e6,"decompression_mb_s":data_bytes/decompression_time/1e6,"prediction_structure_mode":"load" if structure_existed else "generate","matrix":";".join(f"{value:.17g}" for value in matrix[np.tril_indices(dataset['dim'])])}

def benchmark_umc(dataset,test_dir,rel_eb,data_path):
    common=[str(test_dir)+"/",dataset["field"],"coords.dat","conn.dat",str(dataset["umc_from_edge"]),str(dataset["dim"]),"2","1","1"]
    compression_output=run_command([umc_compressor,*common,"0",str(rel_eb),"10"])
    compressed_size=int(read_float(r"compressed_size\s*=\s*(\d+)",compression_output,"UMC压缩大小"))
    mse=read_float(r"^MSE\s*=\s*([^\s]+)",compression_output,"UMC MSE")
    range_match=re.search(r"Max value\s*=\s*([^,]+),\s*min value\s*=\s*([^,]+),",compression_output)
    if range_match is None:
        raise RuntimeError("UMC输出中缺少值域")
    data_range=float(range_match.group(1))-float(range_match.group(2))
    psnr=10*math.log10(data_range*data_range/mse)
    preprocessing_time=read_float(r"^preprocessing time\s*=\s*([^\s]+)",compression_output,"UMC预处理时间")
    reorder_time=read_float(r"^reorder time\s*=\s*([^\s]+)",compression_output,"UMC重排时间")
    full_compression_time=read_float(r"^compression time\s*=\s*([^\s]+)",compression_output,"UMC压缩时间")
    compression_time=full_compression_time-reorder_time
    decompression_output=run_command([umc_decompressor,*common])
    decompression_time=read_float(r"^core decompression time\s*=\s*([^\s]+)",decompression_output,"UMC解压时间")
    data_bytes=data_path.stat().st_size
    point_num=data_bytes//4
    cleanup_output(Path(test_dir)/f"{dataset['field']}.dat.0",(".umc",".umc.out"))
    return {"dataset":dataset["dataset"],"field":dataset["field"],"method":"umc","config":"umc","rel_eb":rel_eb,"br":compressed_size*8/point_num,"psnr":psnr,"preprocessing_time":preprocessing_time,"compression_time":compression_time,"decompression_time":decompression_time,"preprocessing_mb_s":data_bytes/preprocessing_time/1e6,"compression_mb_s":data_bytes/compression_time/1e6,"decompression_mb_s":data_bytes/decompression_time/1e6,"prediction_structure_mode":"generate","matrix":""}

def benchmark_les_umc_combined(result_vec):
    dataset_field_vec=[dataset for dataset in dataset_vec if dataset["dataset"]=="LES"]
    source_dir=data_root/"LES"
    temporary_root=Path(os.environ.get("SLURM_TMPDIR","/tmp"))
    with tempfile.TemporaryDirectory(prefix="unsz_order_LES_umc_",dir=temporary_root) as temporary_dir:
        test_dir=Path(temporary_dir)
        (test_dir/"coords.dat").symlink_to(source_dir/"coords.dat")
        (test_dir/"conn.dat").symlink_to(source_dir/"conn.dat")
        data_path_vec=[]
        for time_step,dataset in enumerate(dataset_field_vec):
            data_path=source_dir/f"{dataset['field']}.dat.0"
            data_path_vec.append(data_path)
            (test_dir/f"les_velocity.dat.{time_step}").symlink_to(data_path)
        total_data_bytes=sum(path.stat().st_size for path in data_path_vec)
        for rel_eb in rel_eb_vec:
            if all(row_exists(result_vec,dataset,"umc","umc",rel_eb) for dataset in dataset_field_vec):
                continue
            common=[str(test_dir)+"/","les_velocity","coords.dat","conn.dat","0","3","2","1",str(len(dataset_field_vec))]
            compression_output=run_command([umc_compressor,*common,"0",str(rel_eb),"10"])
            compressed_size_vec=[int(value) for value in re.findall(r"compressed_size\s*=\s*(\d+)",compression_output)]
            mse_vec=[float(value) for value in re.findall(r"^MSE\s*=\s*([^\s]+)",compression_output,re.MULTILINE)]
            range_vec=[(float(maximum),float(minimum)) for maximum,minimum in re.findall(r"Max value\s*=\s*([^,]+),\s*min value\s*=\s*([^,]+),",compression_output)]
            reorder_time_vec=[float(value) for value in re.findall(r"^reorder time\s*=\s*([^\s]+)",compression_output,re.MULTILINE)]
            full_compression_time_vec=[float(value) for value in re.findall(r"^compression time\s*=\s*([^\s]+)",compression_output,re.MULTILINE)]
            preprocessing_time_vec=[float(value) for value in re.findall(r"^preprocessing time\s*=\s*([^\s]+)",compression_output,re.MULTILINE)]
            expected_num=len(dataset_field_vec)
            if len(compressed_size_vec)!=expected_num or len(mse_vec)!=expected_num or len(range_vec)!=expected_num or len(reorder_time_vec)!=expected_num or len(full_compression_time_vec)!=expected_num or len(preprocessing_time_vec)!=1:
                raise RuntimeError(f"LES UMC多文件输出数量错误：size={len(compressed_size_vec)},MSE={len(mse_vec)},range={len(range_vec)},reorder={len(reorder_time_vec)},compression={len(full_compression_time_vec)},preprocessing={len(preprocessing_time_vec)}")
            decompression_output=run_command([umc_decompressor,*common])
            decompression_time_vec=[float(value) for value in re.findall(r"^core decompression time\s*=\s*([^\s]+)",decompression_output,re.MULTILINE)]
            if len(decompression_time_vec)!=expected_num:
                raise RuntimeError(f"LES UMC多文件解压时间数量错误：{len(decompression_time_vec)}")
            preprocessing_time=preprocessing_time_vec[0]
            for dataset,data_path,compressed_size,mse,data_range,reorder_time,full_compression_time,decompression_time in zip(dataset_field_vec,data_path_vec,compressed_size_vec,mse_vec,range_vec,reorder_time_vec,full_compression_time_vec,decompression_time_vec):
                if row_exists(result_vec,dataset,"umc","umc",rel_eb):
                    continue
                data_bytes=data_path.stat().st_size
                compression_time=full_compression_time-reorder_time
                value_range=data_range[0]-data_range[1]
                row={"dataset":"LES","field":dataset["field"],"method":"umc","config":"umc","rel_eb":rel_eb,"br":compressed_size*8/(data_bytes//4),"psnr":10*math.log10(value_range*value_range/mse),"preprocessing_time":preprocessing_time,"compression_time":compression_time,"decompression_time":decompression_time,"preprocessing_mb_s":total_data_bytes/preprocessing_time/1e6,"compression_mb_s":data_bytes/compression_time/1e6,"decompression_mb_s":data_bytes/decompression_time/1e6,"prediction_structure_mode":"generate_shared_3_fields","matrix":""}
                result_vec.append(row)
                print(f"  UMC {dataset['field']}:REL={rel_eb:g},BR={row['br']:.9f},PSNR={row['psnr']:.6f},shared_preprocessing={preprocessing_time:.6f}s",flush=True)
            write_csv(result_path,result_field_vec,result_vec)
            for time_step in range(expected_num):
                cleanup_output(test_dir/f"les_velocity.dat.{time_step}",(".umc",".umc.out"))

def collect_geometry(dataset,config,matrix,test_dir,prediction_structure_path,data_path):
    arguments=get_unsz_arguments(dataset,config,matrix,test_dir,rel_eb_vec[0],prediction_structure_path)
    output=run_command([compressor,*arguments,"-prediction-geometry-stats"])
    pattern=r"^Prediction geometry: round=(\d+),target_num=(\d+),span_ratio=\[([^\]]*)\],ref_num=(\d+),mean_ref_distance=([^,]+),mean_ref_distance_ratio=([^,]+),mean_nearest_ref_distance=([^,]+),mean_nearest_ref_distance_ratio=([^\s]+)$"
    match_vec=re.findall(pattern,output,re.MULTILINE)
    if len(match_vec)!=10:
        raise RuntimeError(f"预测几何统计轮数应为10，实际为{len(match_vec)}")
    cleanup_output(Path(test_dir)/f"{dataset['field']}.dat.0",(".unsz",".unsz.bin"))
    return [{"dataset":dataset["dataset"],"field":dataset["field"],"config":config["config"],"round":round_id,"target_num":target_num,"span_ratio":span_ratio,"ref_num":ref_num,"mean_ref_distance":mean_ref_distance,"mean_ref_distance_ratio":mean_ref_distance_ratio,"mean_nearest_ref_distance":mean_nearest_ref_distance,"mean_nearest_ref_distance_ratio":mean_nearest_ref_distance_ratio} for round_id,target_num,span_ratio,ref_num,mean_ref_distance,mean_ref_distance_ratio,mean_nearest_ref_distance,mean_nearest_ref_distance_ratio in match_vec]

def row_exists(row_vec,dataset,method,config,rel_eb):
    return any(row["dataset"]==dataset["dataset"] and row["field"]==dataset["field"] and row["method"]==method and row["config"]==config and math.isclose(float(row["rel_eb"]),rel_eb) for row in row_vec)

def summarize(result_vec):
    summary_vec=[]
    for dataset in dataset_vec:
        field_result_vec=[row for row in result_vec if row["dataset"]==dataset["dataset"] and row["field"]==dataset["field"]]
        curve_map={name:[row for row in field_result_vec if row["config"]==name] for name in ("random_10_8","global_10_8","umc")}
        if any(len(curve)!=len(rel_eb_vec) for curve in curve_map.values()):
            continue
        for test_name,base_name in (("global_10_8","random_10_8"),("random_10_8","umc"),("global_10_8","umc")):
            bd_rate,min_psnr,max_psnr=calculate_bd_rate(curve_map[base_name],curve_map[test_name])
            test_curve=curve_map[test_name]
            summary_vec.append({"dataset":dataset["dataset"],"field":dataset["field"],"test":test_name,"base":base_name,"bd_rate_percent":bd_rate,"min_psnr":min_psnr,"max_psnr":max_psnr,"mean_compression_mb_s":sum(float(row["compression_mb_s"]) for row in test_curve)/len(test_curve),"mean_decompression_mb_s":sum(float(row["decompression_mb_s"]) for row in test_curve)/len(test_curve),"generation_preprocessing_time":next((row["preprocessing_time"] for row in test_curve if row["prediction_structure_mode"].startswith("generate")),""),"mean_load_preprocessing_time":sum(float(row["preprocessing_time"]) for row in test_curve if row["prediction_structure_mode"]=="load")/max(1,sum(row["prediction_structure_mode"]=="load" for row in test_curve))})
            print(f"{dataset['dataset']} {dataset['field']}: BD-rate({test_name} vs {base_name})={bd_rate:.6f}%,overlap=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
    if summary_vec:
        write_csv(summary_path,tuple(summary_vec[0]),summary_vec)

def benchmark_dataset(dataset,result_vec,geometry_vec,method):
    source_dir=data_root/dataset["dataset"]
    data_path=source_dir/f"{dataset['field']}.dat.0"
    temporary_root=Path(os.environ.get("SLURM_TMPDIR","/tmp"))
    with tempfile.TemporaryDirectory(prefix=f"unsz_order_{dataset['dataset']}_{dataset['field']}_",dir=temporary_root) as temporary_dir:
        test_dir=Path(temporary_dir)
        (test_dir/"coords.dat").symlink_to(source_dir/"coords.dat")
        (test_dir/"conn.dat").symlink_to(source_dir/"conn.dat")
        (test_dir/f"{dataset['field']}.dat.0").symlink_to(data_path)
        matrix=load_matrix(dataset)
        print(f"dataset={dataset['dataset']},field={dataset['field']},matrix=\n{matrix}",flush=True)
        if method in ("all","unsz"):
            for config in config_vec:
                prediction_structure_path=test_dir/f"prediction_{config['config']}.bin"
                for rel_eb in rel_eb_vec:
                    if row_exists(result_vec,dataset,"unsz",config["config"],rel_eb):
                        continue
                    row=benchmark_unsz(dataset,config,matrix,test_dir,rel_eb,prediction_structure_path,data_path)
                    result_vec.append(row)
                    write_csv(result_path,result_field_vec,result_vec)
                    print(f"  {config['config']}:REL={rel_eb:g},BR={row['br']:.9f},PSNR={row['psnr']:.6f},preprocessing={row['preprocessing_time']:.6f}s",flush=True)
                if not any(row["dataset"]==dataset["dataset"] and row["field"]==dataset["field"] and row["config"]==config["config"] for row in geometry_vec):
                    geometry_vec.extend(collect_geometry(dataset,config,matrix,test_dir,prediction_structure_path,data_path))
                    write_csv(geometry_path,geometry_field_vec,geometry_vec)
        if method in ("all","umc"):
            for rel_eb in rel_eb_vec:
                if row_exists(result_vec,dataset,"umc","umc",rel_eb):
                    continue
                row=benchmark_umc(dataset,test_dir,rel_eb,data_path)
                result_vec.append(row)
                write_csv(result_path,result_field_vec,result_vec)
                print(f"  umc:REL={rel_eb:g},BR={row['br']:.9f},PSNR={row['psnr']:.6f},preprocessing={row['preprocessing_time']:.6f}s",flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-dataset",choices=("LES","Tsunami","all"),default="all")
    parser.add_argument("-field",default="all")
    parser.add_argument("-method",choices=("unsz","umc","all"),default="all")
    parser.add_argument("-summary-only",action="store_true")
    arguments=parser.parse_args()
    result_vec=read_csv(result_path)
    geometry_vec=read_csv(geometry_path)
    if not arguments.summary_only:
        for dataset in dataset_vec:
            if arguments.dataset!="all" and dataset["dataset"]!=arguments.dataset:
                continue
            if arguments.field!="all" and dataset["field"]!=arguments.field:
                continue
            if dataset["dataset"]=="LES" and arguments.field=="all" and arguments.method=="umc":
                continue
            current_method="unsz" if dataset["dataset"]=="LES" and arguments.field=="all" and arguments.method=="all" else arguments.method
            benchmark_dataset(dataset,result_vec,geometry_vec,current_method)
        if arguments.dataset in ("LES","all") and arguments.field=="all" and arguments.method in ("umc","all"):
            benchmark_les_umc_combined(result_vec)
    summarize(result_vec)

if __name__=="__main__":
    main()
