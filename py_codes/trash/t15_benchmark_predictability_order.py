import os
import re
import subprocess
import tempfile

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
compressor=os.path.join(project_root,"install/bin/s1_compress")
time_step_num=5
base_rel_eb_vec=(0.015,0.012,0.01,0.008,0.006)
dataset_vec=(
    {"name":"Katrina","variable":"attr0","adjust_coord":True},
    {"name":"Ocean","variable":"temperature","adjust_coord":False},
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

def run_config(test_dir,variable,rel_eb,adjust_coord,order_flags=()):
    command=[compressor,"-data-dir",test_dir,"-variable",variable,"-time-step",f"[0,{time_step_num-1}]","-M","REL",str(rel_eb)]
    if adjust_coord:
        command.append("-adjust-coord")
    command.extend(order_flags)
    output=run_command(command)
    return {
        "br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",output,"总BR"),
        "psnr":read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",output,"总PSNR"),
        "preprocessing":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",output,"预处理时间"),
        "compression":read_float(r"^total compression time= ([^ ]+) s$",output,"压缩时间"),
        "rel_eb":rel_eb,
    }

def interpolate_br(base_result_vec,target_psnr):
    sorted_result=sorted(base_result_vec,key=lambda result:result["psnr"])
    for low,high in zip(sorted_result[:-1],sorted_result[1:]):
        if low["psnr"]<=target_psnr<=high["psnr"]:
            ratio=(target_psnr-low["psnr"])/(high["psnr"]-low["psnr"])
            br=low["br"]+ratio*(high["br"]-low["br"])
            return br,low["rel_eb"],high["rel_eb"]
    raise RuntimeError(f"基线曲线没有覆盖PSNR={target_psnr}")

def benchmark_dataset(dataset):
    source_dir=os.path.join(data_root,dataset["name"])
    total_mb=sum(os.path.getsize(os.path.join(source_dir,f"{dataset['variable']}.dat.{time_step}")) for time_step in range(time_step_num))/1e6
    with tempfile.TemporaryDirectory(prefix=f"unsz_predictability_order_{dataset['name']}_") as test_dir:
        os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        order_config_vec=(
            ("predictability",("-predictability-order-experimental",)),
            ("forward_mean_distance_only",("-forward-mean-distance-only-order-experimental",)),
            ("forward_s2_only",("-forward-s2-only-order-experimental",)),
            ("forward_direction_only",("-forward-direction-only-order-experimental",)),
            ("forward_mean_distance_s2",("-forward-mean-distance-s2-order-experimental",)),
            ("forward_mean_distance_direction",("-forward-mean-distance-direction-order-experimental",)),
            ("reverse_predictability",("-reverse-predictability-order-experimental",)),
            ("reverse_mean_distance_only",("-reverse-mean-distance-only-order-experimental",)),
            ("reverse_s2_only",("-reverse-s2-only-order-experimental",)),
            ("reverse_distribution",("-reverse-distribution-order-experimental",)),
            ("reverse_direction_only",("-reverse-direction-only-order-experimental",)),
            ("reverse_direction",("-reverse-direction-order-experimental",)),
            ("reverse_direction_distribution",("-reverse-direction-distribution-order-experimental",)),
        )
        variable_vec=[f"base_{str(rel_eb).replace('.','_')}" for rel_eb in base_rel_eb_vec]+[name for name,_ in order_config_vec]
        for variable in variable_vec:
            for time_step in range(time_step_num):
                os.symlink(os.path.join(source_dir,f"{dataset['variable']}.dat.{time_step}"),os.path.join(test_dir,f"{variable}.dat.{time_step}"))
        print(dataset["name"],flush=True)
        base_result_vec=[]
        for rel_eb in base_rel_eb_vec:
            result=run_config(test_dir,f"base_{str(rel_eb).replace('.','_')}",rel_eb,dataset["adjust_coord"])
            base_result_vec.append(result)
            print(f"  base eb={rel_eb}: br={result['br']:.9f}, psnr={result['psnr']:.6f}, preprocessing={result['preprocessing']:.6f} s, compression={total_mb/result['compression']:.3f} MB/s",flush=True)
        for name,order_flags in order_config_vec:
            result=run_config(test_dir,name,0.01,dataset["adjust_coord"],order_flags)
            equal_psnr_base_br,low_eb,high_eb=interpolate_br(base_result_vec,result["psnr"])
            br_saving=(equal_psnr_base_br-result["br"])/equal_psnr_base_br*100
            print(f"  {name}: br={result['br']:.9f}, psnr={result['psnr']:.6f}, equal_psnr_base_br={equal_psnr_base_br:.9f}, br_saving={br_saving:.4f}%, interpolation_eb=[{low_eb},{high_eb}], preprocessing={result['preprocessing']:.6f} s, compression={total_mb/result['compression']:.3f} MB/s",flush=True)

def main():
    for dataset in dataset_vec:
        benchmark_dataset(dataset)

if __name__=="__main__":
    main()
