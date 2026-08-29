import os
import re
import subprocess
import tempfile

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
compressor=os.path.join(project_root,"install/bin/s1_compress")
time_step_num=5
baseline_rel_eb_vec=(0.01,0.008,0.006,0.004,0.002)
round_ratio_vec=(0.5,0.6,0.7,0.8,0.9)
schedule_mode_vec=("round_ratio","round_point_ratio")
datasets=[
    {"name":"Katrina","variable":"attr0","adjust_coord":True},
    {"name":"Ocean","variable":"temperature","adjust_coord":False},
]

def run_command(command):
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(command)}")
    return result.stdout

def read_result(output):
    match=re.search(r"^total cr= [^,]+, total br= ([^,]+), total psnr= ([^ ]+)$",output,re.MULTILINE)
    if not match:
        raise RuntimeError("输出中缺少总BR或总PSNR")
    return float(match.group(1)),float(match.group(2))

def interpolate_br(result_vec,target_psnr):
    sorted_result=sorted(result_vec,key=lambda result:result[1])
    for low,high in zip(sorted_result[:-1],sorted_result[1:]):
        if low[1]<=target_psnr<=high[1]:
            ratio=(target_psnr-low[1])/(high[1]-low[1])
            return low[0]+ratio*(high[0]-low[0]),low,high
    raise RuntimeError(f"默认曲线没有覆盖PSNR={target_psnr}")

def run_config(test_dir,variable,rel_eb,adjust_coord,schedule_mode="none",round_ratio=1):
    command=[compressor,"-data-dir",test_dir,"-variable",variable,"-time-step",f"[0,{time_step_num-1}]","-M","REL",str(rel_eb)]
    if adjust_coord:
        command.append("-adjust-coord")
    command.extend(["-eb-schedule-experimental",schedule_mode,str(round_ratio)])
    return read_result(run_command(command))

def benchmark_dataset(dataset,test_dir):
    source_dir=os.path.join(data_root,dataset["name"])
    os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
    config_name_vec=[f"base_{str(rel_eb).replace('.','_')}" for rel_eb in baseline_rel_eb_vec]+[f"{mode}_{str(ratio).replace('.','_')}" for mode in schedule_mode_vec for ratio in round_ratio_vec]
    for config_name in config_name_vec:
        for time_step in range(time_step_num):
            os.symlink(os.path.join(source_dir,f"{dataset['variable']}.dat.{time_step}"),os.path.join(test_dir,f"{config_name}.dat.{time_step}"))
    print(dataset["name"],flush=True)
    baseline_result_vec=[]
    for rel_eb in baseline_rel_eb_vec:
        config_name=f"base_{str(rel_eb).replace('.','_')}"
        br,psnr=run_config(test_dir,config_name,rel_eb,dataset["adjust_coord"])
        baseline_result_vec.append((br,psnr,rel_eb))
        print(f"  base eb={rel_eb}: br={br:.9f}, psnr={psnr:.6f}",flush=True)
    for mode in schedule_mode_vec:
        print(f"  {mode}",flush=True)
        for round_ratio in round_ratio_vec:
            config_name=f"{mode}_{str(round_ratio).replace('.','_')}"
            br,psnr=run_config(test_dir,config_name,0.01,dataset["adjust_coord"],mode,round_ratio)
            baseline_equal_br,low,high=interpolate_br(baseline_result_vec,psnr)
            br_saving=(baseline_equal_br-br)/baseline_equal_br*100
            print(f"    ratio={round_ratio}: br={br:.9f}, psnr={psnr:.6f}, equal_psnr_base_br={baseline_equal_br:.9f}, br_saving={br_saving:.4f}%, interpolation_eb=[{low[2]},{high[2]}]",flush=True)

def main():
    for dataset in datasets:
        with tempfile.TemporaryDirectory(prefix=f"unsz_round_ratio_{dataset['name']}_") as test_dir:
            benchmark_dataset(dataset,test_dir)

if __name__=="__main__":
    main()
