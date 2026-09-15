import csv,math,os,re,subprocess,tempfile
from collections import Counter
import t44_benchmark_ocean_idw_power as common

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean"
compressor=os.path.join(project_root,"build/s1_compress")
decompressor=os.path.join(project_root,"build/s1_decompress")
result_path=os.path.join(project_root,"ocean_weighted_fitting_rd_benchmark.csv")
summary_path=os.path.join(project_root,"ocean_weighted_fitting_rd_summary.csv")
field_vec=common.field_vec
block_num_vec=(1,4)
mode_vec=("original","weighted")
rel_eb_vec=(0.01,0.003,0.001,0.0003,0.0001)

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

def benchmark(field,block_num,mode,rel_eb,test_dir,prediction_structure_path):
    arguments=["-data-dir",test_dir,"-variable",field,"-time-step","{0}","-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental","-dag-cache",prediction_structure_path,"-predictor","inverse_proportion","-max-ref-num","10","-actual-ref-num","10","-inverse-distance-power","3.5","-d-only-cache-refill-threshold-experimental","8","-partition-num",str(block_num),"-partition-adaptive-predictor-experimental","-partition-adaptive-sample-num-experimental","256","-partition-fitting-candidates-experimental",mode]
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
    predictor_counter=Counter(re.findall(r"^block_adaptive_config: block=\d+,sample_num=\d+,config=\d+,predictor=([^,]+),",compression_output,re.MULTILINE))
    return {"field":field,"block_num":block_num,"mode":mode,"rel_eb":rel_eb,"br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"compression_time":read_value(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_value(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间"),"selected_predictors":";".join(f"{name}:{predictor_counter[name]}" for name in sorted(predictor_counter))}

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_weighted_fitting_rd_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for field in field_vec:
            os.symlink(os.path.join(data_dir,f"{field}.dat.0"),os.path.join(test_dir,f"{field}.dat.0"))
        for block_num in block_num_vec:
            prediction_structure_path=os.path.join(test_dir,f"prediction_structure_b{block_num}.bin")
            for field in field_vec:
                for mode in mode_vec:
                    for rel_eb in rel_eb_vec:
                        result=benchmark(field,block_num,mode,rel_eb,test_dir,prediction_structure_path)
                        result_vec.append(result)
                        print(f"{field} block_num={block_num},mode={mode},rel_eb={rel_eb}: br={result['br']:.9f},psnr={result['psnr']:.6f},predictors={result['selected_predictors']}",flush=True)
    summary_vec=[]
    for block_num in block_num_vec:
        gain_vec=[]
        for field in field_vec:
            original_vec=[result for result in result_vec if result["field"]==field and result["block_num"]==block_num and result["mode"]=="original"]
            weighted_vec=[result for result in result_vec if result["field"]==field and result["block_num"]==block_num and result["mode"]=="weighted"]
            bd_rate,min_psnr,max_psnr=common.calculate_bd_rate(original_vec,weighted_vec)
            gain=-bd_rate
            gain_vec.append(gain)
            summary_vec.append({"scope":field,"block_num":block_num,"weighted_fitting_br_reduction":gain,"min_psnr":min_psnr,"max_psnr":max_psnr})
            print(f"{field} block_num={block_num}: weighted fitting同PSNR BR降低={gain:.6f}%,overlap=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
        summary_vec.append({"scope":"all_fields_mean","block_num":block_num,"weighted_fitting_br_reduction":sum(gain_vec)/len(gain_vec),"min_psnr":math.nan,"max_psnr":math.nan})
        summary_vec.append({"scope":"all_fields_median","block_num":block_num,"weighted_fitting_br_reduction":sorted(gain_vec)[len(gain_vec)//2],"min_psnr":math.nan,"max_psnr":math.nan})
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    with open(summary_path,"w",newline="") as summary_file:
        writer=csv.DictWriter(summary_file,fieldnames=summary_vec[0].keys())
        writer.writeheader()
        writer.writerows(summary_vec)
    print(f"逐点结果已写入{result_path}",flush=True)
    print(f"汇总结果已写入{summary_path}",flush=True)

if __name__=="__main__":
    main()
