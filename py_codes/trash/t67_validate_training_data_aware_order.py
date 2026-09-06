import argparse,csv,math,os,re,statistics,subprocess,sys,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
data_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data")
compressor=project_root/"build/s1_compress"
decompressor=project_root/"build/s1_decompress"
result_path=project_root/"py_codes/trash/results/training_data_aware_order_validation.csv"
summary_path=project_root/"py_codes/trash/results/training_data_aware_order_validation_summary.csv"
rel_eb_vec=(0.02,0.01,0.005,0.0025,0.00125)
dataset_vec=(
    {"name":"Katrina-attr0","directory":"Katrina","field":"attr0","dim":3,"time_step_num":162,"matrix":"Katrina_attr0_spatial.txt","training_split_vec":((40,80,120),(20,80,140),(10,70,130))},
    {"name":"Ocean-temperature","directory":"Ocean","field":"temperature","dim":3,"time_step_num":31,"matrix":"Ocean_temperature_spatial.txt","training_split_vec":((8,16,24),(4,15,26),(2,15,28))},
)
config_vec=(
    {"name":"reverse_d_only","arguments":["-reverse-mean-distance-only-order-experimental"]},
    {"name":"global_before_round_split","arguments":["-reverse-mean-distance-only-order-experimental","-global-order-before-round-split-experimental"]},
    {"name":"training_data_aware","arguments":None},
)
result_field_vec=("experiment","dataset","split","config","scope","time_step","rel_eb","br","psnr","preprocessing_time_s","compression_time_s","decompression_time_s","prediction_structure_mode")
summary_field_vec=("experiment","dataset","split","config","total_bd_rate","mean_time_step_bd_rate","median_time_step_bd_rate","worst_time_step_bd_rate","best_time_step_bd_rate","time_step_num")

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
    return (math.exp((test_integral-base_integral)/(max_psnr-min_psnr))-1)*100

def result_exists(result_vec,experiment,dataset_name,split_name,config_name,rel_eb):
    return any(row["experiment"]==experiment and row["dataset"]==dataset_name and row["split"]==split_name and row["config"]==config_name and row["scope"]=="total" and math.isclose(float(row["rel_eb"]),rel_eb,rel_tol=0,abs_tol=1e-15) for row in result_vec)

def make_common_arguments(dataset,test_dir,test_time_step_vec,rel_eb,prediction_structure_path):
    time_step_expression="{"+",".join(str(value) for value in test_time_step_vec)+"}"
    return ["-data-dir",test_dir,"-variable",dataset["field"],"-time-step",time_step_expression,"-dim",dataset["dim"],"-M","REL",rel_eb,"-adjust-coord-matrix-file",project_root/"adjust_coord_matrices"/dataset["matrix"],"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",10,"-actual-ref-num",10,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",8,"-eb-schedule-experimental","round_point_ratio",0.8]

def benchmark(result_vec,experiment,dataset,split_name,training_time_step_vec,test_time_step_vec,config,test_dir,rel_eb,prediction_structure_path):
    if result_exists(result_vec,experiment,dataset["name"],split_name,config["name"],rel_eb):
        return
    common_arguments=make_common_arguments(dataset,test_dir,test_time_step_vec,rel_eb,prediction_structure_path)
    order_arguments=config["arguments"]
    if order_arguments is None:
        config_training_time_step_vec=config.get("training_time_step_vec",training_time_step_vec)
        training_expression="{"+",".join(str(value) for value in config_training_time_step_vec)+"}"
        order_arguments=["-reference-data-order-experimental",training_expression]
    structure_existed=prediction_structure_path.is_file()
    compression_output=run_command([compressor,*common_arguments,*order_arguments])
    per_file_metric_vec=[tuple(float(value) for value in values) for values in re.findall(r"^cr= ([^,]+), br= ([^,]+), psnr= ([^ ]+)$",compression_output,re.MULTILINE)]
    per_file_compression_time_vec=[float(value) for value in re.findall(r"^file compression time= ([^ ]+) s$",compression_output,re.MULTILINE)]
    if len(per_file_metric_vec)!=len(test_time_step_vec) or len(per_file_compression_time_vec)!=len(test_time_step_vec):
        raise RuntimeError(f"逐文件压缩结果数量错误：metric={len(per_file_metric_vec)},time={len(per_file_compression_time_vec)},expected={len(test_time_step_vec)}")
    total_br=read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR")
    total_psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR")
    preprocessing_time=read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间")
    total_compression_time=read_float(r"^total compression time= ([^ ]+) s$",compression_output,"总压缩时间")
    decompression_output=run_command([decompressor,*common_arguments])
    per_file_decompression_psnr_vec=[float(value) for value in re.findall(r"^Decompressed .*, psnr= ([^ ]+)$",decompression_output,re.MULTILINE)]
    per_file_decompression_time_vec=[float(value) for value in re.findall(r"^file decompression time= ([^ ]+) s$",decompression_output,re.MULTILINE)]
    total_decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"总解压时间")
    decompressed_total_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"总解压PSNR")
    if len(per_file_decompression_psnr_vec)!=len(test_time_step_vec) or len(per_file_decompression_time_vec)!=len(test_time_step_vec):
        raise RuntimeError(f"逐文件解压结果数量错误：psnr={len(per_file_decompression_psnr_vec)},time={len(per_file_decompression_time_vec)},expected={len(test_time_step_vec)}")
    if not math.isclose(total_psnr,decompressed_total_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"总压缩与解压PSNR不一致：{total_psnr}!={decompressed_total_psnr}")
    prediction_structure_mode="load" if structure_existed else "generate"
    for time_step,metric,compression_time,decompression_psnr,decompression_time in zip(test_time_step_vec,per_file_metric_vec,per_file_compression_time_vec,per_file_decompression_psnr_vec,per_file_decompression_time_vec):
        _,br,psnr=metric
        if not math.isclose(psnr,decompression_psnr,rel_tol=1e-6,abs_tol=1e-6):
            raise RuntimeError(f"time step {time_step}压缩与解压PSNR不一致：{psnr}!={decompression_psnr}")
        result_vec.append({"experiment":experiment,"dataset":dataset["name"],"split":split_name,"config":config["name"],"scope":"time_step","time_step":time_step,"rel_eb":rel_eb,"br":br,"psnr":psnr,"preprocessing_time_s":"","compression_time_s":compression_time,"decompression_time_s":decompression_time,"prediction_structure_mode":prediction_structure_mode})
    result_vec.append({"experiment":experiment,"dataset":dataset["name"],"split":split_name,"config":config["name"],"scope":"total","time_step":"","rel_eb":rel_eb,"br":total_br,"psnr":total_psnr,"preprocessing_time_s":preprocessing_time,"compression_time_s":total_compression_time,"decompression_time_s":total_decompression_time,"prediction_structure_mode":prediction_structure_mode})
    write_csv(result_path,result_field_vec,result_vec)
    print(f"{dataset['name']} {split_name} {config['name']} REL={rel_eb:g}:BR={total_br:.9f},PSNR={total_psnr:.6f},preprocessing={preprocessing_time:.6f}s,compression={total_compression_time:.6f}s",flush=True)

def generate_summary(result_vec,experiment):
    summary_vec=[row for row in read_csv(summary_path) if row["experiment"]!=experiment]
    for dataset in dataset_vec:
        dataset_rows=[row for row in result_vec if row["experiment"]==experiment and row["dataset"]==dataset["name"]]
        for split_name in sorted({row["split"] for row in dataset_rows}):
            split_rows=[row for row in dataset_rows if row["split"]==split_name]
            base_total_vec=sorted((row for row in split_rows if row["config"]=="reverse_d_only" and row["scope"]=="total"),key=lambda row:float(row["rel_eb"]),reverse=True)
            if len(base_total_vec)!=len(rel_eb_vec):
                continue
            comparison_config_name_vec=sorted({row["config"] for row in split_rows if row["config"]!="reverse_d_only"})
            for config_name in comparison_config_name_vec:
                test_total_vec=sorted((row for row in split_rows if row["config"]==config_name and row["scope"]=="total"),key=lambda row:float(row["rel_eb"]),reverse=True)
                if len(test_total_vec)!=len(rel_eb_vec):
                    continue
                time_step_bd_rate_vec=[]
                for time_step in sorted({int(row["time_step"]) for row in split_rows if row["scope"]=="time_step"}):
                    base_vec=sorted((row for row in split_rows if row["config"]=="reverse_d_only" and row["scope"]=="time_step" and int(row["time_step"])==time_step),key=lambda row:float(row["rel_eb"]),reverse=True)
                    test_vec=sorted((row for row in split_rows if row["config"]==config_name and row["scope"]=="time_step" and int(row["time_step"])==time_step),key=lambda row:float(row["rel_eb"]),reverse=True)
                    if len(base_vec)==len(rel_eb_vec) and len(test_vec)==len(rel_eb_vec):
                        time_step_bd_rate_vec.append(calculate_bd_rate(base_vec,test_vec))
                if not time_step_bd_rate_vec:
                    continue
                summary={"experiment":experiment,"dataset":dataset["name"],"split":split_name,"config":config_name,"total_bd_rate":calculate_bd_rate(base_total_vec,test_total_vec),"mean_time_step_bd_rate":statistics.fmean(time_step_bd_rate_vec),"median_time_step_bd_rate":statistics.median(time_step_bd_rate_vec),"worst_time_step_bd_rate":max(time_step_bd_rate_vec),"best_time_step_bd_rate":min(time_step_bd_rate_vec),"time_step_num":len(time_step_bd_rate_vec)}
                summary_vec.append(summary)
                print(f"SUMMARY {dataset['name']} {split_name} {config_name}:total={summary['total_bd_rate']:.6f}%,mean={summary['mean_time_step_bd_rate']:.6f}%,median={summary['median_time_step_bd_rate']:.6f}%,worst={summary['worst_time_step_bd_rate']:.6f}%",flush=True)
    write_csv(summary_path,summary_field_vec,summary_vec)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=("all","katrina","ocean"),default="all")
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--training-file-num",action="store_true")
    args=parser.parse_args()
    if args.quick and args.training_file_num:
        raise ValueError("--quick和--training-file-num不能同时使用")
    experiment="training_file_num" if args.training_file_num else ("quick" if args.quick else "full")
    selected_dataset_vec=[dataset for dataset in dataset_vec if args.dataset=="all" or dataset["directory"].lower()==args.dataset]
    result_vec=read_csv(result_path)
    temporary_root=Path(os.environ.get("SLURM_TMPDIR","/tmp"))
    with tempfile.TemporaryDirectory(prefix="unsz_training_order_validation_",dir=temporary_root) as temporary_directory:
        for dataset in selected_dataset_vec:
            source_dir=data_root/dataset["directory"]
            test_dir=Path(temporary_directory)/dataset["name"]
            test_dir.mkdir()
            (test_dir/"coords.dat").symlink_to(source_dir/"coords.dat")
            for time_step in range(dataset["time_step_num"]):
                (test_dir/f"{dataset['field']}.dat.{time_step}").symlink_to(source_dir/f"{dataset['field']}.dat.{time_step}")
            if args.training_file_num:
                full_training_time_step_vec=dataset["training_split_vec"][0]
                training_split_vec=(("training_file_num_"+"-".join(str(value) for value in full_training_time_step_vec),full_training_time_step_vec,(
                    {"name":"reverse_d_only","arguments":["-reverse-mean-distance-only-order-experimental"]},
                    {"name":"training_data_aware_1","arguments":None,"training_time_step_vec":(full_training_time_step_vec[1],)},
                    {"name":"training_data_aware_2","arguments":None,"training_time_step_vec":(full_training_time_step_vec[0],full_training_time_step_vec[2])},
                    {"name":"training_data_aware_3","arguments":None,"training_time_step_vec":full_training_time_step_vec},
                )),)
            else:
                selected_training_split_vec=dataset["training_split_vec"][:1] if args.quick else dataset["training_split_vec"]
                training_split_vec=tuple((f"split{split_id}_"+"-".join(str(value) for value in training_time_step_vec),training_time_step_vec,config_vec) for split_id,training_time_step_vec in enumerate(selected_training_split_vec))
            for split_name,training_time_step_vec,current_config_vec in training_split_vec:
                test_time_step_num=min(dataset["time_step_num"],5) if args.quick else dataset["time_step_num"]
                test_time_step_vec=[time_step for time_step in range(test_time_step_num) if time_step not in training_time_step_vec]
                for config in current_config_vec:
                    structure_name=config["name"] if config["name"]!="training_data_aware" else f"{config['name']}_{split_name}"
                    prediction_structure_path=test_dir/f"{structure_name}.pred"
                    for rel_eb in rel_eb_vec:
                        benchmark(result_vec,experiment,dataset,split_name,training_time_step_vec,test_time_step_vec,config,test_dir,rel_eb,prediction_structure_path)
    generate_summary(result_vec,experiment)

if __name__=="__main__":
    main()
