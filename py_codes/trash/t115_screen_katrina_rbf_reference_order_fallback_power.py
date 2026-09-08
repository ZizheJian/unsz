import csv,json,math,os,re,subprocess,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
result_path=project_root/"py_codes/trash/results/katrina_rbf_reference_order_fallback_power_screen.csv"
training_time_step_vec=(40,80,120)
test_time_step_vec=tuple(range(20))
rel_eb_vec=(1e-2,1e-3,1e-4)
power_vec=(1,2,2.5,3,3.6,4.5,5,6,8,12)

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
    reference_integral=np.polyint(np.polyfit(reference_psnr,np.log(reference_br),2))
    test_integral=np.polyint(np.polyfit(test_psnr,np.log(test_br),2))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (np.exp((test_area-reference_area)/(upper-lower))-1)*100

def write_point_set_config(path,rel_eb):
    with path.open("w",encoding="utf-8") as file:
        json.dump({"version":1,"rule_type":"ordered","labels":[{"method":"rbf_fitting","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]},file,ensure_ascii=False)

def cleanup(test_dir):
    for time_step in test_time_step_vec:
        for suffix in (".unsz",".unsz.bin"):
            path=test_dir/f"attr0.dat.{time_step}{suffix}"
            if path.exists():
                path.unlink()

def write_results(result_vec):
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with result_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=("power","rel_eb","br","psnr","preprocessing_time","compression_time","decompression_time"),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_vec)

def main():
    if result_path.is_file():
        with result_path.open(newline="") as file:
            result_vec=list(csv.DictReader(file))
        for row in result_vec:
            for key in ("power","rel_eb","br","psnr","preprocessing_time","compression_time","decompression_time"):
                row[key]=float(row[key])
    else:
        result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_rbf_fallback_power_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in set(test_time_step_vec)|set(training_time_step_vec):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        time_step_expression="{"+",".join(str(time_step) for time_step in test_time_step_vec)+"}"
        training_expression="{"+",".join(str(time_step) for time_step in training_time_step_vec)+"}"
        for power in power_vec:
            if sum(math.isclose(row["power"],power) for row in result_vec)==len(rel_eb_vec):
                continue
            prediction_structure_path=temporary_dir/f"power_{power:g}.pred"
            for rel_eb in rel_eb_vec:
                config_path=temporary_dir/f"point_sets_{power:g}_{rel_eb:g}.json"
                write_point_set_config(config_path,rel_eb)
                common_arguments=["-data-dir",test_dir,"-variable","attr0","-time-step",time_step_expression,"-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",matrix_path,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",8,"-actual-ref-num",8,"-inverse-distance-power",power,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",8,"-eb-schedule-experimental","round_point_ratio",0.8,"-cross-time-label-predictor-experimental","previous","-rbf-kernel","matern32","-rbf-shape",0.01,"-rbf-nugget",1e-6]
                compression_output=run_command([compressor,*common_arguments,"-reference-data-order-experimental",training_expression,"-reference-data-rbf-score-experimental","-rbf-filtered-fallback-idw-experimental","-fused-two-point-set-prediction-experimental","-linear-previous-label-experimental"])
                decompression_output=run_command([decompressor,*common_arguments,"-rbf-filtered-fallback-idw-experimental","-fused-two-point-set-prediction-experimental","-linear-previous-label-experimental"])
                psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                result={"power":power,"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
                result_vec.append(result)
                write_results(result_vec)
                print(f"power={power:g},REL={rel_eb:g}:BR={result['br']:.6f},PSNR={result['psnr']:.6f},preprocessing={result['preprocessing_time']:.6f}s",flush=True)
                cleanup(test_dir)
    reference=[row for row in result_vec if row["power"]==3.6]
    for power in power_vec:
        current=[row for row in result_vec if row["power"]==power]
        print(f"power={power:g}相对3.6的BD-rate={calculate_bd_rate(reference,current):+.6f}%",flush=True)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
