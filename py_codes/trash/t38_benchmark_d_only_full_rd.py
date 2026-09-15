import csv,math,os,re,subprocess,tempfile
import numpy as np

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
compressor=os.path.join(project_root,"install/bin/s1_compress")
decompressor=os.path.join(project_root,"install/bin/s1_decompress")
result_path=os.path.join(project_root,"d_only_full_rd_benchmark.csv")
rel_eb_vec=(0.02,0.01,0.005,0.0025,0.00125)
dataset_vec=(
    {"name":"Ocean-temperature","folder":"Ocean","variable":"temperature","time_step_num":31,"adjust_coord":False},
    {"name":"Katrina-attr0","folder":"Katrina","variable":"attr0","time_step_num":162,"adjust_coord":True},
)
config_vec=(
    {"name":"exact_10","flags":["-d-only-cache-refill-threshold-experimental","0"]},
    {"name":"cache_8","flags":["-d-only-cache-refill-threshold-experimental","8"]},
    {"name":"candidate_5","flags":["-order-candidate-num-experimental","5"]},
    {"name":"candidate_5_cache_8","flags":["-order-candidate-num-experimental","5","-d-only-cache-refill-threshold-experimental","8"]},
)

def run_command(command):
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(command)}")
    return result.stdout

def read_value(pattern,text,name,value_type=float):
    match=re.search(pattern,text,re.MULTILINE)
    if not match:
        raise RuntimeError(f"输出中缺少{name}")
    return value_type(match.group(1))

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
    return (math.exp((test_integral-base_integral)/(max_psnr-min_psnr))-1)*100,min_psnr,max_psnr

def cleanup(test_dir,variable,time_step_num):
    for time_step in range(time_step_num):
        for suffix in (".unsz",".unsz.bin"):
            path=os.path.join(test_dir,f"{variable}.dat.{time_step}{suffix}")
            if os.path.isfile(path):
                os.remove(path)

def benchmark(dataset,config,rel_eb,test_dir,total_mb,prediction_structure_path):
    eb_name=str(rel_eb).replace(".","_")
    variable=f"{config['name']}_{eb_name}"
    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step",f"[0,{dataset['time_step_num']-1}]","-M","REL",str(rel_eb),"-kd-leaf-size","32","-reverse-mean-distance-only-order-experimental","-dag-cache",prediction_structure_path,*config["flags"]]
    if dataset["adjust_coord"]:
        arguments.append("-adjust-coord")
    compression_output=run_command([compressor,*arguments])
    preprocessing=read_value(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间")
    br=read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR")
    psnr=read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR")
    compression_time=read_value(r"^total compression time= ([^ ]+) s$",compression_output,"总压缩时间")
    structure_generated="Saved prediction DAG:" in compression_output
    sorting=read_value(r"^Target sorting time: ([^ ]+) s$",compression_output,"排序时间") if structure_generated else 0
    reference_generation=read_value(r"^Prediction reference generation time: ([^ ]+) s$",compression_output,"参考生成时间") if structure_generated else 0
    decompression_output=run_command([decompressor,*arguments])
    decompressed_psnr=read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    decompression_time=read_value(r"^total decompression time= ([^ ]+) s$",decompression_output,"总解压时间")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
    cleanup(test_dir,variable,dataset["time_step_num"])
    return {"dataset":dataset["name"],"config":config["name"],"rel_eb":rel_eb,"br":br,"psnr":psnr,"preprocessing_s":preprocessing,"sorting_s":sorting,"reference_generation_s":reference_generation,"compression_mb_s":total_mb/compression_time,"decompression_mb_s":total_mb/decompression_time,"structure_generated":int(structure_generated)}

def benchmark_dataset(dataset,result_vec):
    source_dir=os.path.join(data_root,dataset["folder"])
    total_mb=sum(os.path.getsize(os.path.join(source_dir,f"{dataset['variable']}.dat.{time_step}")) for time_step in range(dataset["time_step_num"]))/1e6
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix=f"unsz_d_only_full_{dataset['folder']}_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for config in config_vec:
            for rel_eb in rel_eb_vec:
                variable=f"{config['name']}_{str(rel_eb).replace('.','_')}"
                for time_step in range(dataset["time_step_num"]):
                    os.symlink(os.path.join(source_dir,f"{dataset['variable']}.dat.{time_step}"),os.path.join(test_dir,f"{variable}.dat.{time_step}"))
        result_map={}
        for config in config_vec:
            prediction_structure_path=os.path.join(test_dir,f"{config['name']}.prediction_structure")
            config_result_vec=[]
            for rel_eb in rel_eb_vec:
                result=benchmark(dataset,config,rel_eb,test_dir,total_mb,prediction_structure_path)
                result_vec.append(result)
                config_result_vec.append(result)
                print(f"{dataset['name']} {config['name']} rel_eb={rel_eb}: br={result['br']:.9f},psnr={result['psnr']:.6f},preprocessing={result['preprocessing_s']:.6f} s,compression={result['compression_mb_s']:.3f} MB/s,decompression={result['decompression_mb_s']:.3f} MB/s",flush=True)
            result_map[config["name"]]=config_result_vec
            print(f"{dataset['name']} {config['name']} prediction_structure_bytes={os.path.getsize(prediction_structure_path)}",flush=True)
        for config in config_vec[1:]:
            bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map["exact_10"],result_map[config["name"]])
            print(f"{dataset['name']} BD-rate({config['name']} vs exact_10)={bd_rate:.6f}%,overlap_psnr=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)

def main():
    result_vec=[]
    for dataset in dataset_vec:
        benchmark_dataset(dataset,result_vec)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
