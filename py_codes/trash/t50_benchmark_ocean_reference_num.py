import csv,math,os,re,statistics,subprocess,tempfile
import t44_benchmark_ocean_idw_power as common

result_path=os.path.join(common.project_root,"ocean_reference_num_benchmark.csv")
summary_path=os.path.join(common.project_root,"ocean_reference_num_summary.csv")
time_step=0
power=3.5
ref_num_vec=tuple(range(4,11))
missing_num_vec=(0,1,2)

def benchmark(field,ref_num,missing_num,rel_eb,test_dir,prediction_structure_path,current_time_step=time_step):
    threshold=ref_num-missing_num
    variable=f"{field}_t{current_time_step}_r{ref_num}_m{missing_num}_{str(rel_eb).replace('.','_')}"
    os.symlink(os.path.join(common.data_dir,f"{field}.dat.{current_time_step}"),os.path.join(test_dir,f"{variable}.dat.{current_time_step}"))
    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step",f"{{{current_time_step}}}","-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental","-dag-cache",prediction_structure_path,"-predictor","inverse_proportion","-max-ref-num",str(ref_num),"-actual-ref-num",str(ref_num),"-d-only-cache-refill-threshold-experimental",str(threshold),"-inverse-distance-power",str(power)]
    compression_output=common.run_command([common.compressor,*arguments])
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
    decompression_output=common.run_command([common.decompressor,*decompression_arguments])
    result={"field":field,"time_step":current_time_step,"power":power,"ref_num":ref_num,"cache_missing_num":missing_num,"cache_refill_threshold":threshold,"rel_eb":rel_eb,"br":common.read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":common.read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR"),"preprocessing_time":common.read_value(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":common.read_value(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":common.read_value(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
    decompressed_psnr=common.read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    if not math.isclose(result["psnr"],decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{result['psnr']} != {decompressed_psnr}")
    cache_match=re.search(r"^D-only score cache: (\d+) hits, (\d+) misses$",compression_output,re.MULTILINE)
    reduced_match=re.search(r"^D-only reduced-reference cache hits: (\d+)$",compression_output,re.MULTILINE)
    result["cache_hit_num"]=int(cache_match.group(1)) if cache_match else math.nan
    result["cache_miss_num"]=int(cache_match.group(2)) if cache_match else math.nan
    result["reduced_ref_cache_hit_num"]=int(reduced_match.group(1)) if reduced_match else math.nan
    return result

def main():
    result_vec=[]
    summary_vec=[]
    result_map={}
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_ocean_reference_num_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(common.data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for ref_num in ref_num_vec:
            for missing_num in missing_num_vec:
                config=(ref_num,missing_num)
                result_map[config]={}
                prediction_structure_path=os.path.join(test_dir,f"prediction_structure_r{ref_num}_m{missing_num}.bin")
                generation_preprocessing_time=math.nan
                cache_hit_num=math.nan
                cache_miss_num=math.nan
                reduced_ref_cache_hit_num=math.nan
                for field in common.field_vec:
                    field_result_vec=[]
                    for rel_eb in common.rel_eb_vec:
                        result=benchmark(field,ref_num,missing_num,rel_eb,test_dir,prediction_structure_path)
                        result_vec.append(result)
                        field_result_vec.append(result)
                        if not math.isnan(result["cache_miss_num"]):
                            generation_preprocessing_time=result["preprocessing_time"]
                            cache_hit_num=result["cache_hit_num"]
                            cache_miss_num=result["cache_miss_num"]
                            reduced_ref_cache_hit_num=result["reduced_ref_cache_hit_num"]
                    result_map[config][field]=field_result_vec
                summary_vec.append({"ref_num":ref_num,"cache_missing_num":missing_num,"cache_refill_threshold":ref_num-missing_num,"mean_bd_rate_vs_10_8":math.nan,"median_bd_rate_vs_10_8":math.nan,"worst_field_bd_rate_vs_10_8":math.nan,"generation_preprocessing_time":generation_preprocessing_time,"cache_hit_num":cache_hit_num,"cache_miss_num":cache_miss_num,"reduced_ref_cache_hit_num":reduced_ref_cache_hit_num})
                print(f"完成ref_num={ref_num},允许缺少={missing_num},预处理={generation_preprocessing_time:.6f}s,cache miss={cache_miss_num}",flush=True)
    baseline_config=(10,2)
    for summary in summary_vec:
        config=(summary["ref_num"],summary["cache_missing_num"])
        bd_rate_vec=[]
        for field in common.field_vec:
            bd_rate,_,_=common.calculate_bd_rate(result_map[baseline_config][field],result_map[config][field])
            bd_rate_vec.append(bd_rate)
        summary["mean_bd_rate_vs_10_8"]=statistics.fmean(bd_rate_vec)
        summary["median_bd_rate_vs_10_8"]=statistics.median(bd_rate_vec)
        summary["worst_field_bd_rate_vs_10_8"]=max(bd_rate_vec)
        print(f"ref_num={summary['ref_num']},允许缺少={summary['cache_missing_num']}: mean BD-rate={summary['mean_bd_rate_vs_10_8']:.6f}%,median={summary['median_bd_rate_vs_10_8']:.6f}%,worst={summary['worst_field_bd_rate_vs_10_8']:.6f}%,预处理={summary['generation_preprocessing_time']:.6f}s",flush=True)
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
