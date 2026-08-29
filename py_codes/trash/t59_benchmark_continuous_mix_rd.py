import csv,math,os,tempfile
import t44_benchmark_ocean_idw_power as common
import t57_benchmark_weighted_fitting_rd as benchmark

result_path=os.path.join(benchmark.project_root,"ocean_continuous_mix_rd_benchmark.csv")
summary_path=os.path.join(benchmark.project_root,"ocean_continuous_mix_rd_summary.csv")

def main():
    with open(benchmark.result_path,newline="") as result_file:
        existing_result_vec=list(csv.DictReader(result_file))
    mixed_result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_continuous_mix_rd_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(benchmark.data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for field in benchmark.field_vec:
            os.symlink(os.path.join(benchmark.data_dir,f"{field}.dat.0"),os.path.join(test_dir,f"{field}.dat.0"))
        for block_num in benchmark.block_num_vec:
            prediction_structure_path=os.path.join(test_dir,f"prediction_structure_b{block_num}.bin")
            for field in benchmark.field_vec:
                for rel_eb in benchmark.rel_eb_vec:
                    result=benchmark.benchmark(field,block_num,"mixed",rel_eb,test_dir,prediction_structure_path)
                    mixed_result_vec.append(result)
                    print(f"{field} block_num={block_num},mode=mixed,rel_eb={rel_eb}: br={result['br']:.9f},psnr={result['psnr']:.6f},predictors={result['selected_predictors']}",flush=True)
    summary_vec=[]
    for block_num in benchmark.block_num_vec:
        for field in benchmark.field_vec:
            original_vec=[{"br":float(result["br"]),"psnr":float(result["psnr"])} for result in existing_result_vec if result["field"]==field and int(result["block_num"])==block_num and result["mode"]=="original"]
            weighted_vec=[{"br":float(result["br"]),"psnr":float(result["psnr"])} for result in existing_result_vec if result["field"]==field and int(result["block_num"])==block_num and result["mode"]=="weighted"]
            mixed_vec=[result for result in mixed_result_vec if result["field"]==field and result["block_num"]==block_num]
            mixed_vs_original,original_min_psnr,original_max_psnr=common.calculate_bd_rate(original_vec,mixed_vec)
            mixed_vs_weighted,weighted_min_psnr,weighted_max_psnr=common.calculate_bd_rate(weighted_vec,mixed_vec)
            summary_vec.append({"field":field,"block_num":block_num,"mixed_br_reduction_vs_original":-mixed_vs_original,"mixed_br_reduction_vs_weighted":-mixed_vs_weighted,"original_overlap_min_psnr":original_min_psnr,"original_overlap_max_psnr":original_max_psnr,"weighted_overlap_min_psnr":weighted_min_psnr,"weighted_overlap_max_psnr":weighted_max_psnr})
            print(f"{field} block_num={block_num}: mixed相对original同PSNR BR降低={-mixed_vs_original:.6f}%,相对weighted降低={-mixed_vs_weighted:.6f}%",flush=True)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=mixed_result_vec[0].keys())
        writer.writeheader()
        writer.writerows(mixed_result_vec)
    with open(summary_path,"w",newline="") as summary_file:
        writer=csv.DictWriter(summary_file,fieldnames=summary_vec[0].keys())
        writer.writeheader()
        writer.writerows(summary_vec)
    print(f"连续混合结果已写入{result_path}",flush=True)
    print(f"连续混合汇总已写入{summary_path}",flush=True)

if __name__=="__main__":
    main()
