import argparse,csv,math,os,re,statistics,subprocess,tempfile
from collections import Counter
import t44_benchmark_ocean_idw_power as common

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean"
compressor=os.path.join(project_root,"build/s1_compress")
decompressor=os.path.join(project_root,"build/s1_decompress")
result_path=os.path.join(project_root,"ocean_block_adaptive_benchmark.csv")
summary_path=os.path.join(project_root,"ocean_block_adaptive_summary.csv")
field_vec=common.field_vec
block_num_vec=(1,2,4,8)
rel_eb_vec=(0.01,0.003,0.001,0.0003,0.0001)
time_step_num=31

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

def benchmark(field,block_num,rel_eb,test_dir,prediction_structure_path,test_time_step_num=time_step_num):
    variable=f"{field}_block_adaptive_b{block_num}_{str(rel_eb).replace('.','_')}"
    for time_step in range(test_time_step_num):
        os.symlink(os.path.join(data_dir,f"{field}.dat.{time_step}"),os.path.join(test_dir,f"{variable}.dat.{time_step}"))
    time_step_expression=f"[0,{test_time_step_num-1}]"
    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step",time_step_expression,"-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental","-dag-cache",prediction_structure_path,"-predictor","inverse_proportion","-max-ref-num","10","-actual-ref-num","10","-inverse-distance-power","3.5","-d-only-cache-refill-threshold-experimental","8","-partition-num",str(block_num),"-partition-adaptive-predictor-experimental","-partition-adaptive-sample-num-experimental","256"]
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
    config_counter=Counter(int(value) for value in re.findall(r"block_adaptive_config: block=\d+,sample_num=\d+,config=(\d+),",compression_output))
    expected_config_num=block_num*test_time_step_num
    if sum(config_counter.values())!=expected_config_num:
        raise RuntimeError(f"block配置输出数量不一致：期望{expected_config_num}，实际{sum(config_counter.values())}")
    return {"field":field,"block_num":block_num,"rel_eb":rel_eb,"br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":read_value(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":read_value(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_value(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间"),"config_count":";".join(f"{config_id}:{config_counter[config_id]}" for config_id in sorted(config_counter))}

def read_umc_results(field):
    result_vec=[]
    with open(os.path.join(data_dir,f"{field}.csv"),newline="") as result_file:
        for row in csv.DictReader(result_file):
            if int(row["TIME_STEP_NUM"])==time_step_num and row.get("UMC_BR") and row.get("UMC_PSNR"):
                result_vec.append({"br":float(row["UMC_BR"]),"psnr":float(row["UMC_PSNR"])})
    result_vec.sort(key=lambda result:result["psnr"])
    if len(result_vec)<3:
        raise RuntimeError(f"{field}的UMC曲线不足3点")
    return result_vec

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    test_field_vec=field_vec[:1] if args.smoke else field_vec
    test_block_num_vec=block_num_vec[:1] if args.smoke else block_num_vec
    test_rel_eb_vec=rel_eb_vec[:1] if args.smoke else rel_eb_vec
    test_time_step_num=1 if args.smoke else time_step_num
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_ocean_block_adaptive_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for block_num in test_block_num_vec:
            prediction_structure_path=os.path.join(test_dir,f"prediction_structure_b{block_num}.bin")
            for field in test_field_vec:
                for rel_eb in test_rel_eb_vec:
                    result=benchmark(field,block_num,rel_eb,test_dir,prediction_structure_path,test_time_step_num)
                    result_vec.append(result)
                    print(f"{field} block_num={block_num},rel_eb={rel_eb}: br={result['br']:.6f},psnr={result['psnr']:.6f},configs={result['config_count']}",flush=True)
    if args.smoke:
        print("smoke test passed",flush=True)
        return
    summary_vec=[]
    for block_num in block_num_vec:
        field_gain_vec=[]
        for field in field_vec:
            unsz_result_vec=[result for result in result_vec if result["field"]==field and result["block_num"]==block_num]
            bd_rate,min_psnr,max_psnr=common.calculate_bd_rate(read_umc_results(field),unsz_result_vec)
            gain=-bd_rate
            field_gain_vec.append(gain)
            summary_vec.append({"scope":field,"block_num":block_num,"br_reduction_vs_umc":gain,"min_psnr":min_psnr,"max_psnr":max_psnr})
            print(f"{field} block_num={block_num}: UNSZ同PSNR BR降低={gain:.6f}%,overlap=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
        summary_vec.append({"scope":"all_fields_mean","block_num":block_num,"br_reduction_vs_umc":statistics.fmean(field_gain_vec),"min_psnr":math.nan,"max_psnr":math.nan})
        summary_vec.append({"scope":"all_fields_median","block_num":block_num,"br_reduction_vs_umc":statistics.median(field_gain_vec),"min_psnr":math.nan,"max_psnr":math.nan})
        print(f"all fields block_num={block_num}: mean BR reduction={statistics.fmean(field_gain_vec):.6f}%,median={statistics.median(field_gain_vec):.6f}%",flush=True)
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
