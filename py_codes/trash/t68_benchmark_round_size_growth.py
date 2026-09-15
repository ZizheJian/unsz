import csv,math,os,re,subprocess,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
data_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data")
compressor=project_root/"build/s1_compress"
decompressor=project_root/"build/s1_decompress"
result_path=project_root/"py_codes/trash/results/round_size_growth_benchmark.csv"
summary_path=project_root/"py_codes/trash/results/round_size_growth_summary.csv"
rel_eb_vec=(0.02,0.01,0.005,0.0025,0.00125)
growth_ratio_vec=(1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.5,4.0)
dataset_vec=(
    {"dataset":"Katrina","field":"attr0","dim":3,"matrix_file":"Katrina_attr0_spatial.txt"},
    {"dataset":"Ocean","field":"temperature","dim":3,"matrix_file":"Ocean_temperature_spatial.txt"},
    {"dataset":"Ocean","field":"velocityZonal","dim":3,"matrix_file":"Ocean_velocityZonal_spatial.txt"},
    {"dataset":"Tsunami","field":"velocity_y","dim":2,"matrix_history":"Tsunami_velocity_y_gauss_newton.npz"},
)
result_field_vec=("dataset","field","growth_ratio","rel_eb","br","psnr","preprocessing_time_s","compression_time_s","decompression_time_s","prediction_structure_mode","round_target_num")
summary_field_vec=("dataset","field","growth_ratio","bd_rate_vs_2_percent","generation_preprocessing_time_s","mean_load_preprocessing_time_s","mean_compression_time_s","mean_decompression_time_s","round_target_num")

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

def row_exists(row_vec,dataset,growth_ratio,rel_eb):
    return any(row["dataset"]==dataset["dataset"] and row["field"]==dataset["field"] and math.isclose(float(row["growth_ratio"]),growth_ratio,rel_tol=0,abs_tol=1e-12) and math.isclose(float(row["rel_eb"]),rel_eb,rel_tol=0,abs_tol=1e-15) for row in row_vec)

def get_matrix_arguments(dataset):
    if "matrix_file" in dataset:
        return ["-adjust-coord-matrix-file",project_root/"adjust_coord_matrices"/dataset["matrix_file"]]
    matrix_path=project_root/"py_codes/trash/results/adjust_coord_history/spatial_parameter_screen"/dataset["matrix_history"]
    with np.load(matrix_path) as history:
        matrix=history["matrix"][-1]
    parameter=matrix[np.tril_indices(dataset["dim"])]
    return ["-adjust-coord-matrix",*[f"{value:.17g}" for value in parameter]]

def get_arguments(dataset,test_dir,growth_ratio,rel_eb,prediction_structure_path):
    return ["-data-dir",test_dir,"-variable",dataset["field"],"-time-step","{0}","-dim",dataset["dim"],"-M","REL",rel_eb,*get_matrix_arguments(dataset),"-reverse-mean-distance-only-order-experimental","-dag-cache",prediction_structure_path,"-predict-round-num",8,"-round-size-growth-ratio-experimental",growth_ratio,"-max-ref-num",10,"-actual-ref-num",10,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",8,"-eb-schedule-experimental","round_point_ratio",0.8]

def calculate_bd_rate(base_result_vec,test_result_vec):
    base_psnr=np.asarray([float(row["psnr"]) for row in base_result_vec])
    test_psnr=np.asarray([float(row["psnr"]) for row in test_result_vec])
    base_log_br=np.log(np.asarray([float(row["br"]) for row in base_result_vec]))
    test_log_br=np.log(np.asarray([float(row["br"]) for row in test_result_vec]))
    min_psnr=max(base_psnr.min(),test_psnr.min())
    max_psnr=min(base_psnr.max(),test_psnr.max())
    if not min_psnr<max_psnr:
        raise RuntimeError("两条RD曲线没有重叠的PSNR区间")
    base_polynomial=np.polyfit(base_psnr,base_log_br,3)
    test_polynomial=np.polyfit(test_psnr,test_log_br,3)
    base_integral=np.polyval(np.polyint(base_polynomial),max_psnr)-np.polyval(np.polyint(base_polynomial),min_psnr)
    test_integral=np.polyval(np.polyint(test_polynomial),max_psnr)-np.polyval(np.polyint(test_polynomial),min_psnr)
    return (math.exp((test_integral-base_integral)/(max_psnr-min_psnr))-1)*100

def benchmark_dataset(dataset,result_vec):
    source_dir=data_root/dataset["dataset"]
    temporary_root=Path(os.environ.get("SLURM_TMPDIR","/tmp"))
    with tempfile.TemporaryDirectory(prefix=f"unsz_round_growth_{dataset['dataset']}_{dataset['field']}_",dir=temporary_root) as temporary_directory:
        test_dir=Path(temporary_directory)
        (test_dir/"coords.dat").symlink_to(source_dir/"coords.dat")
        (test_dir/f"{dataset['field']}.dat.0").symlink_to(source_dir/f"{dataset['field']}.dat.0")
        for growth_ratio in growth_ratio_vec:
            prediction_structure_path=test_dir/f"growth_{growth_ratio:g}.dag"
            for rel_eb in rel_eb_vec:
                if row_exists(result_vec,dataset,growth_ratio,rel_eb):
                    continue
                structure_existed=prediction_structure_path.is_file()
                arguments=get_arguments(dataset,test_dir,growth_ratio,rel_eb,prediction_structure_path)
                compression_output=run_command([compressor,*arguments])
                decompression_output=run_command([decompressor,*arguments])
                psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩与解压PSNR不一致：{psnr}!={decompressed_psnr}")
                round_target_num=",".join(re.findall(r"^Round \d+ target num: (\d+)$",compression_output,re.MULTILINE))
                if not round_target_num:
                    round_target_num="loaded"
                row={"dataset":dataset["dataset"],"field":dataset["field"],"growth_ratio":growth_ratio,"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time_s":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time_s":read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time_s":read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间"),"prediction_structure_mode":"load" if structure_existed else "generate","round_target_num":round_target_num}
                result_vec.append(row)
                write_csv(result_path,result_field_vec,result_vec)
                print(f"{dataset['dataset']} {dataset['field']} ratio={growth_ratio:g},REL={rel_eb:g}:BR={row['br']:.9f},PSNR={psnr:.6f},preprocessing={row['preprocessing_time_s']:.6f}s",flush=True)

def summarize(result_vec):
    summary_vec=[]
    for dataset in dataset_vec:
        dataset_rows=[row for row in result_vec if row["dataset"]==dataset["dataset"] and row["field"]==dataset["field"]]
        base_result_vec=[row for row in dataset_rows if math.isclose(float(row["growth_ratio"]),2,rel_tol=0,abs_tol=1e-12)]
        if len(base_result_vec)!=len(rel_eb_vec):
            continue
        for growth_ratio in growth_ratio_vec:
            test_result_vec=[row for row in dataset_rows if math.isclose(float(row["growth_ratio"]),growth_ratio,rel_tol=0,abs_tol=1e-12)]
            if len(test_result_vec)!=len(rel_eb_vec):
                continue
            generation_row=next(row for row in test_result_vec if row["prediction_structure_mode"]=="generate")
            load_rows=[row for row in test_result_vec if row["prediction_structure_mode"]=="load"]
            summary={"dataset":dataset["dataset"],"field":dataset["field"],"growth_ratio":growth_ratio,"bd_rate_vs_2_percent":0 if growth_ratio==2 else calculate_bd_rate(base_result_vec,test_result_vec),"generation_preprocessing_time_s":generation_row["preprocessing_time_s"],"mean_load_preprocessing_time_s":sum(float(row["preprocessing_time_s"]) for row in load_rows)/len(load_rows),"mean_compression_time_s":sum(float(row["compression_time_s"]) for row in test_result_vec)/len(test_result_vec),"mean_decompression_time_s":sum(float(row["decompression_time_s"]) for row in test_result_vec)/len(test_result_vec),"round_target_num":generation_row["round_target_num"]}
            summary_vec.append(summary)
            print(f"SUMMARY {dataset['dataset']} {dataset['field']} ratio={growth_ratio:g}:BD-rate vs 2={float(summary['bd_rate_vs_2_percent']):+.6f}%",flush=True)
    write_csv(summary_path,summary_field_vec,summary_vec)

def main():
    result_vec=read_csv(result_path)
    for dataset in dataset_vec:
        benchmark_dataset(dataset,result_vec)
    summarize(result_vec)

if __name__=="__main__":
    main()
