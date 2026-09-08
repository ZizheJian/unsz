import csv,json,math,os,re,subprocess,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
result_path=project_root/"py_codes/trash/results/katrina_reference_order_predictors_rd.csv"
training_time_step_vec=(40,80,120)
test_time_step_vec=tuple(time_step for time_step in range(162) if time_step not in training_time_step_vec)
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
predictor_vec=(
    {"name":"idw_ref4_power2","method":"inverse_proportion","ref_num":4,"power":2,"cache_threshold":3,"compress_extra":["-fused-two-point-set-prediction-experimental","-normalized-weight-experimental"],"decompress_extra":["-fused-two-point-set-prediction-experimental","-linear-previous-label-experimental"]},
    {"name":"idw_ref5_power2.5","method":"inverse_proportion","ref_num":5,"power":2.5,"cache_threshold":4,"compress_extra":["-fused-two-point-set-prediction-experimental","-normalized-weight-experimental"],"decompress_extra":["-fused-two-point-set-prediction-experimental","-linear-previous-label-experimental"]},
    {"name":"rbf_ref8_matern32","method":"rbf_fitting","ref_num":8,"power":3.6,"cache_threshold":8,"compress_extra":["-rbf-filtered-fallback-idw-experimental","-fused-two-point-set-prediction-experimental","-linear-previous-label-experimental"],"decompress_extra":["-rbf-filtered-fallback-idw-experimental","-fused-two-point-set-prediction-experimental","-linear-previous-label-experimental"]},
)

def run_command(command):
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError("命令执行失败："+" ".join(str(value) for value in command))
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def calculate_bd_rate(reference,test):
    reference_psnr=np.asarray([row["psnr"] for row in reference])
    reference_br=np.asarray([row["br"] for row in reference])
    test_psnr=np.asarray([row["psnr"] for row in test])
    test_br=np.asarray([row["br"] for row in test])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    reference_integral=np.polyint(np.polyfit(reference_psnr,np.log(reference_br),3))
    test_integral=np.polyint(np.polyfit(test_psnr,np.log(test_br),3))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (np.exp((test_area-reference_area)/(upper-lower))-1)*100

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def write_point_set_config(path,method,rel_eb):
    with path.open("w",encoding="utf-8") as file:
        json.dump({"version":1,"rule_type":"ordered","labels":[{"method":method,"eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]},file,ensure_ascii=False)

def cleanup(test_dir):
    for time_step in test_time_step_vec:
        for suffix in (".unsz",".unsz.bin"):
            path=test_dir/f"attr0.dat.{time_step}{suffix}"
            if path.exists():
                path.unlink()

def write_results(result_vec):
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with result_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=("predictor","order","rel_eb","br","psnr","preprocessing_time","compression_time","decompression_time"),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_vec)

def main():
    with result_path.open(newline="") if result_path.is_file() else open(os.devnull,newline="") as file:
        result_vec=list(csv.DictReader(file)) if result_path.is_file() else []
    for row in result_vec:
        for key in ("rel_eb","br","psnr","preprocessing_time","compression_time","decompression_time"):
            row[key]=float(row[key])
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_reference_order_predictors_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(162):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        time_step_expression="{"+",".join(str(time_step) for time_step in test_time_step_vec)+"}"
        training_expression="{"+",".join(str(time_step) for time_step in training_time_step_vec)+"}"
        for predictor in predictor_vec:
            order_vec=[("d_only",["-reverse-mean-distance-only-order-experimental"]),("reference_data",["-reference-data-order-experimental",training_expression])]
            if predictor["method"]=="rbf_fitting":
                order_vec.append(("reference_data_rbf_score",["-reference-data-order-experimental",training_expression,"-reference-data-rbf-score-experimental"]))
            for order_name,order_extra in order_vec:
                if sum(row["predictor"]==predictor["name"] and row["order"]==order_name for row in result_vec)==len(rel_eb_vec):
                    continue
                prediction_structure_path=temporary_dir/f"{predictor['name']}_{order_name}.pred"
                for rel_eb in rel_eb_vec:
                    config_path=temporary_dir/f"point_sets_{predictor['name']}_{rel_eb:g}.json"
                    write_point_set_config(config_path,predictor["method"],rel_eb)
                    common_arguments=["-data-dir",test_dir,"-variable","attr0","-time-step",time_step_expression,"-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",matrix_path,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",predictor["ref_num"],"-actual-ref-num",predictor["ref_num"],"-inverse-distance-power",predictor["power"],"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",predictor["cache_threshold"],"-eb-schedule-experimental","round_point_ratio",0.8,"-cross-time-label-predictor-experimental","previous","-rbf-kernel","matern32","-rbf-shape",0.01,"-rbf-nugget",1e-6]
                    compression_output=run_command([compressor,*common_arguments,*order_extra,*predictor["compress_extra"]])
                    decompression_output=run_command([decompressor,*common_arguments,*predictor["decompress_extra"]])
                    psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                    result={"predictor":predictor["name"],"order":order_name,"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
                    result_vec.append(result)
                    write_results(result_vec)
                    print(f"{predictor['name']} {order_name} REL={rel_eb:g}:BR={result['br']:.6f},PSNR={result['psnr']:.6f},compression={result['compression_time']:.6f}s,decompression={result['decompression_time']:.6f}s",flush=True)
                    cleanup(test_dir)
    for predictor in predictor_vec:
        d_only_rows=[row for row in result_vec if row["predictor"]==predictor["name"] and row["order"]=="d_only"]
        for order_name in sorted({row["order"] for row in result_vec if row["predictor"]==predictor["name"] and row["order"]!="d_only"}):
            reference_rows=[row for row in result_vec if row["predictor"]==predictor["name"] and row["order"]==order_name]
            print(f"{predictor['name']} {order_name}相对D-only BD-rate={calculate_bd_rate(d_only_rows,reference_rows):+.6f}%，压缩时间倍率={geometric_mean([row['compression_time'] for row in reference_rows])/geometric_mean([row['compression_time'] for row in d_only_rows]):.6f}，解压时间倍率={geometric_mean([row['decompression_time'] for row in reference_rows])/geometric_mean([row['decompression_time'] for row in d_only_rows]):.6f}",flush=True)
    idw_reference=[row for row in result_vec if row["predictor"]=="idw_ref4_power2" and row["order"]=="d_only"]
    for predictor in predictor_vec:
        for order_name in sorted({row["order"] for row in result_vec if row["predictor"]==predictor["name"]}):
            rows=[row for row in result_vec if row["predictor"]==predictor["name"] and row["order"]==order_name]
            print(f"{predictor['name']} {order_name}相对当前4点IDW D-only BD-rate={calculate_bd_rate(idw_reference,rows):+.6f}%",flush=True)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
