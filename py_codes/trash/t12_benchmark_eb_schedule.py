import os
import re
import subprocess
import tempfile

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
compressor=os.path.join(project_root,"install/bin/s1_compress")
time_step_num=5
ratios=(0.25,0.5,0.75,0.9,0.99)
configs=[("base",None,1)]+[(f"{mode}_{str(ratio).replace('.','_')}",mode,ratio) for mode in ("round","round_point","point") for ratio in ratios]
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

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if not match:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def benchmark_dataset(dataset,test_dir):
    source_dir=os.path.join(data_root,dataset["name"])
    os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
    for config_name,_,_ in configs:
        for time_step in range(time_step_num):
            os.symlink(os.path.join(source_dir,f"{dataset['variable']}.dat.{time_step}"),os.path.join(test_dir,f"{config_name}.dat.{time_step}"))
    common_arguments=["-data-dir",test_dir,"-time-step",f"[0,{time_step_num-1}]","-M","REL","0.01","-round-stats"]
    if dataset["adjust_coord"]:
        common_arguments.append("-adjust-coord")
    print(dataset["name"],flush=True)
    for config_name,mode,ratio in configs:
        flags=["-eb-schedule-experimental","none","1"] if mode is None else ["-eb-schedule-experimental",mode,str(ratio)]
        output=run_command([compressor,"-variable",config_name,*common_arguments,*flags])
        br=read_float(r"^total cr= [^,]+, total br= ([^,]+),",output,"总比特率")
        psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",output,"总PSNR")
        value_zstd_bytes=sum(int(value) for value in re.findall(r"^value_huffman_bytes=\d+, value_zstd_bytes=(\d+),",output,re.MULTILINE))
        round_matches=re.findall(r"^Round (\d+): point_num=\d+, qb_num=\d+, qb_br=([^,]+), psnr=([^ ]+)$",output,re.MULTILINE)
        if len(round_matches)!=time_step_num*10:
            raise RuntimeError(f"逐轮统计数量应为{time_step_num*10}，实际为{len(round_matches)}")
        round_br=[]
        for round_id in range(10):
            values=[float(qb_br) for current_round,qb_br,_ in round_matches if int(current_round)==round_id]
            round_br.append(sum(values)/len(values))
        print(f"  {config_name}: br={br:.9f}, psnr={psnr:.6f}, value_zstd_bytes={value_zstd_bytes}, round_qb_br=[{','.join(f'{value:.4f}' for value in round_br)}]",flush=True)

def main():
    for dataset in datasets:
        with tempfile.TemporaryDirectory(prefix=f"unsz_eb_schedule_{dataset['name']}_") as test_dir:
            benchmark_dataset(dataset,test_dir)

if __name__=="__main__":
    main()
