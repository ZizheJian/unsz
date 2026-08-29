import math
import os
import re
import subprocess
import tempfile
import numpy as np

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
compressor=os.path.join(project_root,"install/bin/s1_compress")
decompressor=os.path.join(project_root,"install/bin/s1_decompress")
rel_eb_vec=(0.02,0.01,0.005,0.0025,0.00125)
config_vec=(
    {"name":"none","flags":["-eb-schedule-experimental","none","1"]},
    {"name":"round_point_ratio_0_8","flags":["-eb-schedule-experimental","round_point_ratio","0.8"]},
)
dataset_vec=(
    {"name":"Katrina-attr0","folder":"Katrina","variable":"attr0","time_step_num":162,"adjust_coord":True},
    {"name":"Ocean-temperature","folder":"Ocean","variable":"temperature","time_step_num":31,"adjust_coord":False},
)

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

def calculate_bd_rate(base_result_vec,test_result_vec):
    base_psnr=np.asarray([result["psnr"] for result in base_result_vec])
    test_psnr=np.asarray([result["psnr"] for result in test_result_vec])
    base_log_br=np.log(np.asarray([result["br"] for result in base_result_vec]))
    test_log_br=np.log(np.asarray([result["br"] for result in test_result_vec]))
    min_psnr=max(base_psnr.min(),test_psnr.min())
    max_psnr=min(base_psnr.max(),test_psnr.max())
    if not min_psnr<max_psnr:
        raise RuntimeError("两条RD曲线没有重叠的PSNR区间")
    base_polynomial=np.polyfit(base_psnr,base_log_br,3)
    test_polynomial=np.polyfit(test_psnr,test_log_br,3)
    base_integral=np.polyval(np.polyint(base_polynomial),max_psnr)-np.polyval(np.polyint(base_polynomial),min_psnr)
    test_integral=np.polyval(np.polyint(test_polynomial),max_psnr)-np.polyval(np.polyint(test_polynomial),min_psnr)
    bd_rate=(math.exp((test_integral-base_integral)/(max_psnr-min_psnr))-1)*100
    return bd_rate,min_psnr,max_psnr

def cleanup(test_dir,variable,time_step_num):
    for time_step in range(time_step_num):
        for suffix in (".unsz",".unsz.bin"):
            path=os.path.join(test_dir,f"{variable}.dat.{time_step}{suffix}")
            if os.path.isfile(path):
                os.remove(path)

def benchmark_config(dataset,test_dir,config,rel_eb):
    eb_name=str(rel_eb).replace(".","_")
    variable=f"{config['name']}_{eb_name}"
    common_arguments=["-data-dir",test_dir,"-variable",variable,"-time-step",f"[0,{dataset['time_step_num']-1}]","-M","REL",str(rel_eb)]
    if dataset["adjust_coord"]:
        common_arguments.append("-adjust-coord")
    compression_output=run_command([compressor,*common_arguments,*config["flags"]])
    br=read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR")
    psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR")
    compression_time=read_float(r"^total compression time= ([^ ]+) s$",compression_output,"总压缩时间")
    decompression_output=run_command([decompressor,*common_arguments,*config["flags"]])
    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"总解压时间")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
    cleanup(test_dir,variable,dataset["time_step_num"])
    return {"rel_eb":rel_eb,"br":br,"psnr":psnr,"compression_time":compression_time,"decompression_time":decompression_time}

def benchmark_dataset(dataset):
    source_dir=os.path.join(data_root,dataset["folder"])
    total_mb=sum(os.path.getsize(os.path.join(source_dir,f"{dataset['variable']}.dat.{time_step}")) for time_step in range(dataset["time_step_num"]))/1e6
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix=f"unsz_eb_rd_{dataset['folder']}_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for config in config_vec:
            for rel_eb in rel_eb_vec:
                eb_name=str(rel_eb).replace(".","_")
                variable=f"{config['name']}_{eb_name}"
                for time_step in range(dataset["time_step_num"]):
                    os.symlink(os.path.join(source_dir,f"{dataset['variable']}.dat.{time_step}"),os.path.join(test_dir,f"{variable}.dat.{time_step}"))
        result_map={}
        print(f"dataset={dataset['name']}",flush=True)
        for config in config_vec:
            result_vec=[]
            for rel_eb in rel_eb_vec:
                result=benchmark_config(dataset,test_dir,config,rel_eb)
                result_vec.append(result)
                compression_mb_s=total_mb/result["compression_time"]
                decompression_mb_s=total_mb/result["decompression_time"]
                print(f"  {config['name']}: rel_eb={rel_eb}, br={result['br']:.9f}, psnr={result['psnr']:.6f}, compression={compression_mb_s:.3f} MB/s, decompression={decompression_mb_s:.3f} MB/s",flush=True)
            result_map[config["name"]]=result_vec
        base_name=config_vec[0]["name"]
        test_name=config_vec[1]["name"]
        bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map[base_name],result_map[test_name])
        print(f"  BD-rate({test_name} vs {base_name})={bd_rate:.6f}%, overlap_psnr=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)

def main():
    for dataset in dataset_vec:
        benchmark_dataset(dataset)

if __name__=="__main__":
    main()
