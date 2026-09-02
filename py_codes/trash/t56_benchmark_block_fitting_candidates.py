import argparse,csv,math,os,re,subprocess,tempfile
from collections import Counter
import t44_benchmark_ocean_idw_power as common

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean"
compressor=os.path.join(project_root,"build/s1_compress")
decompressor=os.path.join(project_root,"build/s1_decompress")
result_path=os.path.join(project_root,"ocean_block_fitting_candidates_benchmark.csv")
summary_path=os.path.join(project_root,"ocean_block_fitting_candidates_summary.csv")
field_vec=common.field_vec
block_num_vec=(1,4)
mode_vec=("original","weighted","ridge","all")
rel_eb=0.001

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

def benchmark(field,block_num,mode,test_dir,prediction_structure_path):
    arguments=["-data-dir",test_dir,"-variable",field,"-time-step","{0}","-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental","-prediction-structure",prediction_structure_path,"-predictor","inverse_proportion","-max-ref-num","10","-actual-ref-num","10","-inverse-distance-power","3.5","-d-only-cache-refill-threshold-experimental","8","-partition-num",str(block_num),"-partition-adaptive-predictor-experimental","-partition-adaptive-sample-num-experimental","256","-partition-fitting-candidates-experimental",mode]
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
    predictor_counter=Counter(f"{predictor}_weight_{weight_power}_ridge_{ridge}_normalized_{normalized}" for predictor,weight_power,ridge,normalized in re.findall(r"^block_adaptive_config: block=\d+,sample_num=\d+,config=\d+,predictor=([^,]+),ref_num=\d+,power=[^,]+,fitting_weight_power=([^,]+),ridge=([^,]+),normalize_coord=(\d+),",compression_output,re.MULTILINE))
    if sum(predictor_counter.values())!=block_num:
        raise RuntimeError("block预测配置输出数量不正确")
    return {"field":field,"block_num":block_num,"mode":mode,"rel_eb":rel_eb,"br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"file_bytes":int(read_value(r"^value_huffman_bytes=\d+, value_zstd_bytes=\d+, file_bytes=(\d+)$",compression_output,"文件字节数")),"compression_time":read_value(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_value(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间"),"selected_predictors":";".join(f"{name}:{predictor_counter[name]}" for name in sorted(predictor_counter))}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    test_field_vec=field_vec[:1] if args.smoke else field_vec
    test_block_num_vec=block_num_vec[:1] if args.smoke else block_num_vec
    test_mode_vec=mode_vec if not args.smoke else ("original","all")
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_block_fitting_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for field in test_field_vec:
            os.symlink(os.path.join(data_dir,f"{field}.dat.0"),os.path.join(test_dir,f"{field}.dat.0"))
        for block_num in test_block_num_vec:
            prediction_structure_path=os.path.join(test_dir,f"prediction_structure_b{block_num}.bin")
            for field in test_field_vec:
                for mode in test_mode_vec:
                    result=benchmark(field,block_num,mode,test_dir,prediction_structure_path)
                    result_vec.append(result)
                    print(f"{field} block_num={block_num},mode={mode}: br={result['br']:.9f},psnr={result['psnr']:.6f},predictors={result['selected_predictors']},compression={result['compression_time']:.6f}s,decompression={result['decompression_time']:.6f}s",flush=True)
    if args.smoke:
        print("block fitting candidate smoke test passed",flush=True)
        return
    summary_vec=[]
    for block_num in block_num_vec:
        for field in field_vec:
            field_result_vec=[result for result in result_vec if result["block_num"]==block_num and result["field"]==field]
            baseline=next(result for result in field_result_vec if result["mode"]=="original")
            for result in field_result_vec:
                summary_vec.append({"field":field,"block_num":block_num,"mode":result["mode"],"br_change_vs_original_percent":(result["br"]/baseline["br"]-1)*100,"psnr_change_vs_original":result["psnr"]-baseline["psnr"],"compression_time_ratio":result["compression_time"]/baseline["compression_time"],"decompression_time_ratio":result["decompression_time"]/baseline["decompression_time"]})
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    with open(summary_path,"w",newline="") as summary_file:
        writer=csv.DictWriter(summary_file,fieldnames=summary_vec[0].keys())
        writer.writeheader()
        writer.writerows(summary_vec)
    print(f"逐项结果已写入{result_path}",flush=True)
    print(f"汇总结果已写入{summary_path}",flush=True)

if __name__=="__main__":
    main()
