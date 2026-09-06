import csv,math,os,re,subprocess,tempfile
from collections import Counter
import t44_benchmark_ocean_idw_power as common

project_root=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean"
compressor=os.path.join(project_root,"build/s1_compress")
decompressor=os.path.join(project_root,"build/s1_decompress")
result_path=os.path.join(project_root,"py_codes/trash/results/partition_adaptive_confidence_benchmark.csv")
summary_path=os.path.join(project_root,"py_codes/trash/results/partition_adaptive_confidence_summary.csv")
field_vec=common.field_vec
rel_eb_vec=(0.01,0.003,0.001,0.0003,0.0001)
threshold_vec=(0,0.005)

def run_command(command):
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(command)}")
    return result.stdout

def read_value(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if not match:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def benchmark(field,rel_eb,threshold,test_dir,prediction_structure_path):
    variable=f"{field}_threshold_{threshold:g}_eb_{rel_eb:g}".replace(".","_")
    os.symlink(os.path.join(data_dir,f"{field}.dat.0"),os.path.join(test_dir,f"{variable}.dat.0"))
    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step","{0}","-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental","-prediction-structure",prediction_structure_path,"-predictor","inverse_proportion","-max-ref-num","10","-actual-ref-num","10","-inverse-distance-power","3.5","-d-only-cache-refill-threshold-experimental","8","-partition-num","4","-partition-adaptive-predictor-experimental","-partition-adaptive-sample-num-experimental","256","-partition-fitting-candidates-experimental","original"]
    if threshold>0:
        arguments.extend(["-partition-adaptive-min-improvement-experimental",str(threshold)])
    compression_output=run_command([compressor,*arguments])
    decompression_arguments=[]
    skip_next=False
    for argument in arguments:
        if skip_next:
            skip_next=False
            continue
        if argument=="-predictor":
            skip_next=True
            continue
        decompression_arguments.append(argument)
    decompression_output=run_command([decompressor,*decompression_arguments])
    psnr=read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR")
    decompressed_psnr=read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
    selected_config_counter=Counter(int(value) for value in re.findall(r"^partition_adaptive_config: partition=\d+,sample_num=\d+,config=(\d+),",compression_output,re.MULTILINE))
    local_config_counter=Counter(int(value) for value in re.findall(r"^partition_adaptive_config: partition=\d+,sample_num=\d+,config=\d+,local_config=(\d+),",compression_output,re.MULTILINE))
    return {"field":field,"rel_eb":rel_eb,"threshold":threshold,"br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"compression_time":read_value(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_value(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间"),"selected_configs":";".join(f"{key}:{value}" for key,value in sorted(selected_config_counter.items())),"local_configs":";".join(f"{key}:{value}" for key,value in sorted(local_config_counter.items()))}

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_partition_confidence_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        prediction_structure_path=os.path.join(test_dir,"prediction_structure_b4.bin")
        for field in field_vec:
            for threshold in threshold_vec:
                for rel_eb in rel_eb_vec:
                    result=benchmark(field,rel_eb,threshold,test_dir,prediction_structure_path)
                    result_vec.append(result)
                    print(f"{field},threshold={threshold:g},rel_eb={rel_eb:g}:BR={result['br']:.6f},PSNR={result['psnr']:.6f},configs={result['selected_configs']}",flush=True)
    summary_vec=[]
    gain_vec=[]
    for field in field_vec:
        base_vec=[result for result in result_vec if result["field"]==field and result["threshold"]==0]
        test_vec=[result for result in result_vec if result["field"]==field and result["threshold"]==0.005]
        bd_rate,min_psnr,max_psnr=common.calculate_bd_rate(base_vec,test_vec)
        gain=-bd_rate
        gain_vec.append(gain)
        summary_vec.append({"scope":field,"br_reduction_percent":gain,"min_psnr":min_psnr,"max_psnr":max_psnr})
        print(f"{field}:0.5%置信门槛同PSNR BR降低={gain:.6f}%",flush=True)
    summary_vec.append({"scope":"all_fields_mean","br_reduction_percent":sum(gain_vec)/len(gain_vec),"min_psnr":math.nan,"max_psnr":math.nan})
    summary_vec.append({"scope":"all_fields_median","br_reduction_percent":sorted(gain_vec)[len(gain_vec)//2],"min_psnr":math.nan,"max_psnr":math.nan})
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    with open(summary_path,"w",newline="") as summary_file:
        writer=csv.DictWriter(summary_file,fieldnames=summary_vec[0].keys())
        writer.writeheader()
        writer.writerows(summary_vec)
    print(f"结果已写入{result_path}和{summary_path}",flush=True)

if __name__=="__main__":
    main()
