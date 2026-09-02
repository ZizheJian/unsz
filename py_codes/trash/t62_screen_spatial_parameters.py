import argparse,csv,math,os,re,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np
project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.py_unsz_core.adjust_coord import adjust_coord
from py_codes.py_unsz_core.init import init
from py_codes.py_unsz_core.param import Param

data_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data")
compressor=project_root/"build/s1_compress"
decompressor=project_root/"build/s1_decompress"
result_path=project_root/"py_codes/trash/results/spatial_parameter_screen.csv"
summary_path=project_root/"py_codes/trash/results/spatial_parameter_screen_summary.csv"
matrix_dir=project_root/"py_codes/trash/results/adjust_coord_history/spatial_parameter_screen"
rel_eb=1e-3
screen_rel_eb_vec=(1e-3,9e-4)
first_sample_budget=117580
katrina_matrix=np.asarray([[1.320727945153478,0,0],[-0.050022502016393564,1.119176596028389,0],[-0.0022639755300411725,0.00041386801901295816,0.02477688868808782]],dtype=np.float64)
dataset_vec=[
    {"dataset":"Katrina","field":"attr0","dim":3,"matrix":katrina_matrix},
    {"dataset":"LES","field":"velocity_x","dim":3},
    {"dataset":"LES","field":"velocity_y","dim":3},
    {"dataset":"LES","field":"velocity_z","dim":3},
    {"dataset":"Tsunami","field":"momentum_x","dim":2},
    {"dataset":"Tsunami","field":"momentum_y","dim":2},
    {"dataset":"Tsunami","field":"velocity_x","dim":2},
    {"dataset":"Tsunami","field":"velocity_y","dim":2},
]
config_vec=[
    {"config":"random_10_8","power":3.61,"ref_num":10,"threshold":8,"global_order":False},
    {"config":"random_8_6","power":3.5,"ref_num":8,"threshold":6,"global_order":False},
    {"config":"global_10_8","power":3.61,"ref_num":10,"threshold":8,"global_order":True},
    {"config":"global_8_6","power":3.5,"ref_num":8,"threshold":6,"global_order":True},
]

def run_command(command):
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(str(value) for value in command)}")
    return result.stdout

def read_value(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def load_or_train_matrix(dataset):
    if "matrix" in dataset:
        return dataset["matrix"],0,0
    matrix_dir.mkdir(parents=True,exist_ok=True)
    history_path=matrix_dir/f"{dataset['dataset']}_{dataset['field']}_gauss_newton.npz"
    if history_path.is_file():
        history=np.load(history_path)
        return history["matrix"][-1],float(history["elapsed_time"][-1]),int(history["first_sample_num"][-1])
    data_dir=data_root/dataset["dataset"]
    param=Param()
    param.coord_path=str(data_dir/"coords.dat")
    param.original_data_path_vec=[str(data_dir/f"{dataset['field']}.dat.0")]
    param.dim=dataset["dim"]
    param.rel_eb=rel_eb
    param.actual_ref_num=5
    init(param)
    first_sample_ratio=min(0.25,first_sample_budget/param.point_num)
    adjust_coord(param,loss_function_name="second_power",iteration_num=4,matrix_history_path=str(history_path),first_sample_ratio=first_sample_ratio,second_sample_ratio=0.10,new_matrix_weight=0.5,adjust_max_nfev=100,sample_file_num=1,matrix_save_interval=1,optimizer_name="gauss_newton")
    history=np.load(history_path)
    return history["matrix"][-1],float(history["elapsed_time"][-1]),int(history["first_sample_num"][-1])

def benchmark(dataset,config,matrix,test_dir,current_rel_eb):
    data_dir=data_root/dataset["dataset"]
    variable=dataset["field"]
    data_path=data_dir/f"{variable}.dat.0"
    temporary_data_path=Path(test_dir)/f"{variable}.dat.0"
    if not temporary_data_path.exists():
        temporary_data_path.symlink_to(data_path)
    prediction_structure_path=Path(test_dir)/f"prediction_{config['config']}.bin"
    matrix_parameter=matrix[np.tril_indices(dataset["dim"])]
    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step","{0}","-M","REL",str(current_rel_eb),"-dim",str(dataset["dim"]),"-adjust-coord-matrix",*[f"{value:.17g}" for value in matrix_parameter],"-reverse-mean-distance-only-order-experimental","-prediction-structure",prediction_structure_path,"-max-ref-num",str(config["ref_num"]),"-actual-ref-num",str(config["ref_num"]),"-inverse-distance-power",str(config["power"]),"-d-only-cache-refill-threshold-experimental",str(config["threshold"])]
    if config["global_order"]:
        arguments.append("-global-order-before-round-split-experimental")
    compression_output=run_command([compressor,*arguments])
    decompression_output=run_command([decompressor,*arguments])
    result={
        "dataset":dataset["dataset"],"field":variable,"dim":dataset["dim"],"config":config["config"],"global_order":int(config["global_order"]),"power":config["power"],"ref_num":config["ref_num"],"cache_refill_threshold":config["threshold"],"rel_eb":current_rel_eb,
        "br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR"),
        "preprocessing_time":read_value(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":read_value(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_value(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间"),
        "matrix":";".join(f"{value:.17g}" for value in matrix_parameter),
    }
    decompressed_psnr=read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    if not math.isclose(result["psnr"],decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{result['psnr']} != {decompressed_psnr}")
    return result

def save_results(result_vec):
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)

def interpolate_log_br(result_vec,target_psnr):
    sorted_result_vec=sorted(result_vec,key=lambda result:float(result["psnr"]))
    selected_pair=None
    for lower,upper in zip(sorted_result_vec[:-1],sorted_result_vec[1:]):
        lower_psnr=float(lower["psnr"])
        upper_psnr=float(upper["psnr"])
        if lower_psnr<=target_psnr<=upper_psnr:
            selected_pair=(lower,upper)
            break
    if selected_pair is None:
        selected_pair=tuple(sorted_result_vec[:2] if target_psnr<float(sorted_result_vec[0]["psnr"]) else sorted_result_vec[-2:])
    lower,upper=selected_pair
    lower_psnr=float(lower["psnr"])
    upper_psnr=float(upper["psnr"])
    ratio=(target_psnr-lower_psnr)/(upper_psnr-lower_psnr)
    return math.exp(math.log(float(lower["br"]))*(1-ratio)+math.log(float(upper["br"]))*ratio)

def save_summary(result_vec):
    summary_vec=[]
    field_key_vec=sorted({(row["dataset"],row["field"]) for row in result_vec})
    for dataset_name,field in field_key_vec:
        field_result_vec=[row for row in result_vec if row["dataset"]==dataset_name and row["field"]==field]
        baseline=next(row for row in field_result_vec if row["config"]=="random_10_8" and math.isclose(float(row["rel_eb"]),rel_eb))
        target_psnr=float(baseline["psnr"])
        baseline_matched_br=interpolate_log_br([row for row in field_result_vec if row["config"]=="random_10_8"],target_psnr)
        for config in config_vec:
            config_result_vec=[row for row in field_result_vec if row["config"]==config["config"]]
            matched_br=interpolate_log_br(config_result_vec,target_psnr)
            summary_vec.append({"dataset":dataset_name,"field":field,"config":config["config"],"target_psnr":target_psnr,"matched_br":matched_br,"br_change_vs_random_10_8_percent":math.nan if not math.isfinite(matched_br) else (matched_br/baseline_matched_br-1)*100})
    with open(summary_path,"w",newline="") as summary_file:
        writer=csv.DictWriter(summary_file,fieldnames=summary_vec[0].keys())
        writer.writeheader()
        writer.writerows(summary_vec)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-dataset",action="append",default=[])
    parser.add_argument("-field",action="append",default=[])
    parser.add_argument("-config",action="append",default=[])
    parser.add_argument("-skip-existing",action="store_true")
    parser.add_argument("-summary-only",action="store_true")
    args=parser.parse_args()
    selected_dataset_vec=[dataset for dataset in dataset_vec if (not args.dataset or dataset["dataset"] in args.dataset) and (not args.field or dataset["field"] in args.field)]
    selected_config_vec=[config for config in config_vec if not args.config or config["config"] in args.config]
    if not selected_dataset_vec:
        raise ValueError("没有符合筛选条件的数据")
    if not selected_config_vec:
        raise ValueError("没有符合筛选条件的配置")
    existing_result_vec=[]
    if result_path.is_file():
        with open(result_path,newline="") as result_file:
            existing_result_vec=list(csv.DictReader(result_file))
    if args.summary_only:
        if not existing_result_vec:
            raise RuntimeError("没有可汇总的筛选结果")
        save_summary(existing_result_vec)
        return
    result_vec=existing_result_vec.copy()
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    for dataset in selected_dataset_vec:
        matrix,training_time,first_sample_num=load_or_train_matrix(dataset)
        print(f"{dataset['dataset']}-{dataset['field']}矩阵训练完成：time={training_time:.6f}s,first_sample_num={first_sample_num},matrix={matrix.tolist()}",flush=True)
        with tempfile.TemporaryDirectory(prefix=f"unsz_spatial_screen_{dataset['dataset']}_{dataset['field']}_",dir=temporary_root) as test_dir:
            (Path(test_dir)/"coords.dat").symlink_to(data_root/dataset["dataset"]/"coords.dat")
            baseline_result=next((row for row in result_vec if row["dataset"]==dataset["dataset"] and row["field"]==dataset["field"] and row["config"]=="random_10_8" and math.isclose(float(row["rel_eb"]),rel_eb)),None)
            target_psnr=None if baseline_result is None else float(baseline_result["psnr"])
            for config in selected_config_vec:
                existing_config_result_vec=[row for row in result_vec if row["dataset"]==dataset["dataset"] and row["field"]==dataset["field"] and row["config"]==config["config"]]
                if args.skip_existing and all(any(math.isclose(float(row["rel_eb"]),current_rel_eb) for row in existing_config_result_vec) for current_rel_eb in screen_rel_eb_vec):
                    print(f"跳过已有结果：{dataset['dataset']}-{dataset['field']} {config['config']}",flush=True)
                    continue
                result_vec=[row for row in result_vec if not (row["dataset"]==dataset["dataset"] and row["field"]==dataset["field"] and row["config"]==config["config"])]
                config_result_vec=[]
                for current_rel_eb in screen_rel_eb_vec:
                    start_time=time.perf_counter()
                    result=benchmark(dataset,config,matrix,test_dir,current_rel_eb)
                    result["matrix_training_time"]=training_time
                    result["matrix_first_sample_num"]=first_sample_num
                    result_vec.append(result)
                    config_result_vec.append(result)
                    save_results(result_vec)
                    print(f"{dataset['dataset']}-{dataset['field']} {config['config']} REL={current_rel_eb}: BR={result['br']:.6f},PSNR={result['psnr']:.6f},pre={result['preprocessing_time']:.3f}s,comp={result['compression_time']:.3f}s,decomp={result['decompression_time']:.3f}s,wall={time.perf_counter()-start_time:.3f}s",flush=True)
                if target_psnr is None:
                    target_psnr=config_result_vec[0]["psnr"]
                min_psnr=min(result["psnr"] for result in config_result_vec)
                max_psnr=max(result["psnr"] for result in config_result_vec)
                if not min_psnr<=target_psnr<=max_psnr:
                    closest_result=min(config_result_vec,key=lambda result:abs(result["psnr"]-target_psnr))
                    matched_rel_eb=closest_result["rel_eb"]*10**((closest_result["psnr"]-target_psnr)/20)
                    start_time=time.perf_counter()
                    result=benchmark(dataset,config,matrix,test_dir,matched_rel_eb)
                    result["matrix_training_time"]=training_time
                    result["matrix_first_sample_num"]=first_sample_num
                    result_vec.append(result)
                    config_result_vec.append(result)
                    save_results(result_vec)
                    print(f"{dataset['dataset']}-{dataset['field']} {config['config']} matching REL={matched_rel_eb}: BR={result['br']:.6f},PSNR={result['psnr']:.6f},pre={result['preprocessing_time']:.3f}s,comp={result['compression_time']:.3f}s,decomp={result['decompression_time']:.3f}s,wall={time.perf_counter()-start_time:.3f}s",flush=True)
                (Path(test_dir)/f"prediction_{config['config']}.bin").unlink()
        save_summary(result_vec)

if __name__=="__main__":
    main()
