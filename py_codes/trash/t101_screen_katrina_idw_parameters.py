import csv,json,math,os,re,subprocess,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
result_path=project_root/"py_codes/trash/results/katrina_current_idw_parameter_screen.csv"
summary_path=project_root/"py_codes/trash/results/katrina_current_idw_parameter_screen_summary.csv"
time_step_num=20
rel_eb_vec=(1e-2,1e-3,1e-4)
power_vec=(1,2,2.5,3,3.2,3.4,3.5,3.6,3.7,3.8,4,4.5,5)
config_vec=[{"name":f"ref{ref_num}_power{power:g}","ref_num":ref_num,"cache_threshold":ref_num,"power":power} for ref_num in (4,5) for power in power_vec]

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

def write_point_set_config(path,rel_eb):
    config={"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]}
    with open(path,"w",encoding="utf-8") as file:
        json.dump(config,file,ensure_ascii=False)

def cleanup(test_dir):
    for time_step in range(time_step_num):
        for suffix in (".unsz",".unsz.bin"):
            path=test_dir/f"attr0.dat.{time_step}{suffix}"
            if path.exists():
                path.unlink()

def calculate_bd_rate(reference,test):
    reference_psnr=np.asarray([result["psnr"] for result in reference])
    reference_br=np.asarray([result["br"] for result in reference])
    test_psnr=np.asarray([result["psnr"] for result in test])
    test_br=np.asarray([result["br"] for result in test])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    reference_integral=np.polyint(np.polyfit(reference_psnr,np.log(reference_br),2))
    test_integral=np.polyint(np.polyfit(test_psnr,np.log(test_br),2))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (np.exp((test_area-reference_area)/(upper-lower))-1)*100

def main():
    result_vec=[]
    result_map={}
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_idw_screen_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        for config in config_vec:
            current=[]
            prediction_structure_path=temporary_dir/f"{config['name']}.prediction_structure.bin"
            for rel_eb in rel_eb_vec:
                point_set_config_path=temporary_dir/f"point_sets_{rel_eb}.json"
                write_point_set_config(point_set_config_path,rel_eb)
                arguments=["-data-dir",test_dir,"-variable","attr0","-time-step",f"[0,{time_step_num-1}]","-dim",3,"-point-set-config",point_set_config_path,"-adjust-coord-matrix-file",matrix_path,"-dag-cache",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",config["ref_num"],"-actual-ref-num",config["ref_num"],"-inverse-distance-power",config["power"],"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",config["cache_threshold"],"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental","-cross-time-label-predictor-experimental","previous","-fused-two-point-set-prediction-experimental","-linear-previous-label-experimental"]
                output=run_command([compressor,*arguments])
                result={**config,"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",output,"总BR"),"psnr":read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",output,"总PSNR"),"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",output,"预处理时间"),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",output,"压缩时间")}
                result_vec.append(result)
                current.append(result)
                cleanup(test_dir)
            result_map[config["name"]]=current
            print(f"完成{config['name']}",flush=True)
    baseline=result_map["ref4_power3.6"]
    summary_vec=[]
    for config in config_vec:
        current=result_map[config["name"]]
        summary={**config,"bd_rate_vs_ref4_power3_6":calculate_bd_rate(baseline,current),"first_preprocessing_time":current[0]["preprocessing_time"],"compression_time_geomean":math.exp(sum(math.log(result["compression_time"]) for result in current)/len(current))}
        summary_vec.append(summary)
    summary_vec.sort(key=lambda result:result["bd_rate_vs_ref4_power3_6"])
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_vec)
    with open(summary_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=summary_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_vec)
    for result in summary_vec[:10]:
        print(f"{result['name']}：BD-rate={result['bd_rate_vs_ref4_power3_6']:+.6f}%，preprocessing={result['first_preprocessing_time']:.6f} s，compression={result['compression_time_geomean']:.6f} s",flush=True)

if __name__=="__main__":
    main()
