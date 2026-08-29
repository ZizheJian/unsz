import csv,math,os,statistics,tempfile
import t44_benchmark_ocean_idw_power as common
import t53_benchmark_ocean_block_adaptive as adaptive

result_path=os.path.join(adaptive.project_root,"ocean_block_fixed_benchmark.csv")
summary_path=os.path.join(adaptive.project_root,"ocean_block_adaptive_ablation_summary.csv")

def benchmark(field,block_num,rel_eb,test_dir,prediction_structure_path):
    variable=f"{field}_block_fixed_b{block_num}_{str(rel_eb).replace('.','_')}"
    for time_step in range(adaptive.time_step_num):
        os.symlink(os.path.join(adaptive.data_dir,f"{field}.dat.{time_step}"),os.path.join(test_dir,f"{variable}.dat.{time_step}"))
    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step",f"[0,{adaptive.time_step_num-1}]","-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental","-prediction-structure",prediction_structure_path,"-predictor","inverse_proportion","-max-ref-num","10","-actual-ref-num","10","-inverse-distance-power","3.5","-d-only-cache-refill-threshold-experimental","8","-independent-block-num-experimental",str(block_num)]
    compression_output=adaptive.run_command([adaptive.compressor,*arguments])
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
    decompression_output=adaptive.run_command([adaptive.decompressor,*decompression_arguments])
    psnr=adaptive.read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR")
    decompressed_psnr=adaptive.read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
    return {"field":field,"block_num":block_num,"rel_eb":rel_eb,"br":adaptive.read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":adaptive.read_value(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":adaptive.read_value(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":adaptive.read_value(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}

def read_adaptive_results():
    result_map={}
    with open(adaptive.result_path,newline="") as result_file:
        for row in csv.DictReader(result_file):
            result_map.setdefault((row["field"],int(row["block_num"])),[]).append({"br":float(row["br"]),"psnr":float(row["psnr"])})
    return result_map

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_ocean_block_fixed_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(adaptive.data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for block_num in adaptive.block_num_vec:
            prediction_structure_path=os.path.join(test_dir,f"prediction_structure_b{block_num}.bin")
            for field in adaptive.field_vec:
                for rel_eb in adaptive.rel_eb_vec:
                    result=benchmark(field,block_num,rel_eb,test_dir,prediction_structure_path)
                    result_vec.append(result)
                    print(f"fixed {field} block_num={block_num},rel_eb={rel_eb}: br={result['br']:.6f},psnr={result['psnr']:.6f}",flush=True)
    fixed_map={(field,block_num):[result for result in result_vec if result["field"]==field and result["block_num"]==block_num] for field in adaptive.field_vec for block_num in adaptive.block_num_vec}
    adaptive_map=read_adaptive_results()
    summary_vec=[]
    for block_num in adaptive.block_num_vec:
        fixed_umc_gain_vec=[]
        boundary_cost_vec=[]
        adaptive_gain_vec=[]
        for field in adaptive.field_vec:
            fixed_umc_bd_rate,_,_=common.calculate_bd_rate(adaptive.read_umc_results(field),fixed_map[(field,block_num)])
            boundary_bd_rate,_,_=common.calculate_bd_rate(fixed_map[(field,1)],fixed_map[(field,block_num)])
            adaptive_bd_rate,_,_=common.calculate_bd_rate(fixed_map[(field,block_num)],adaptive_map[(field,block_num)])
            summary={"scope":field,"block_num":block_num,"fixed_br_reduction_vs_umc":-fixed_umc_bd_rate,"fixed_block_boundary_cost_vs_1":boundary_bd_rate,"adaptive_br_reduction_vs_same_block_fixed":-adaptive_bd_rate}
            summary_vec.append(summary)
            fixed_umc_gain_vec.append(summary["fixed_br_reduction_vs_umc"])
            boundary_cost_vec.append(summary["fixed_block_boundary_cost_vs_1"])
            adaptive_gain_vec.append(summary["adaptive_br_reduction_vs_same_block_fixed"])
            print(f"{field} block_num={block_num}: fixed vs UMC={summary['fixed_br_reduction_vs_umc']:.6f}%,boundary cost={summary['fixed_block_boundary_cost_vs_1']:.6f}%,adaptive gain={summary['adaptive_br_reduction_vs_same_block_fixed']:.6f}%",flush=True)
        for scope,function in (("all_fields_mean",statistics.fmean),("all_fields_median",statistics.median)):
            summary_vec.append({"scope":scope,"block_num":block_num,"fixed_br_reduction_vs_umc":function(fixed_umc_gain_vec),"fixed_block_boundary_cost_vs_1":function(boundary_cost_vec),"adaptive_br_reduction_vs_same_block_fixed":function(adaptive_gain_vec)})
        print(f"all fields block_num={block_num}: mean fixed vs UMC={statistics.fmean(fixed_umc_gain_vec):.6f}%,boundary cost={statistics.fmean(boundary_cost_vec):.6f}%,adaptive gain={statistics.fmean(adaptive_gain_vec):.6f}%",flush=True)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    with open(summary_path,"w",newline="") as summary_file:
        writer=csv.DictWriter(summary_file,fieldnames=summary_vec[0].keys())
        writer.writeheader()
        writer.writerows(summary_vec)
    print(f"固定配置结果已写入{result_path}",flush=True)
    print(f"消融汇总已写入{summary_path}",flush=True)

if __name__=="__main__":
    main()
