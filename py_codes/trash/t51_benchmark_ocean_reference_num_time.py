import csv,math,os,statistics,tempfile
from collections import defaultdict
import t44_benchmark_ocean_idw_power as common
import t50_benchmark_ocean_reference_num as ref_common

result_path=os.path.join(common.project_root,"ocean_reference_num_time_benchmark.csv")
summary_path=os.path.join(common.project_root,"ocean_reference_num_time_summary.csv")
time_step_vec=(15,30)
config_vec=((4,1),(5,1),(6,0),(7,1),(8,2),(9,2),(10,2))

def read_time_zero_results():
    result_map=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    with open(ref_common.result_path,newline="") as result_file:
        for row in csv.DictReader(result_file):
            config=(int(row["ref_num"]),int(row["cache_missing_num"]))
            if config in config_vec:
                result_map[config][row["field"]][0].append({"br":float(row["br"]),"psnr":float(row["psnr"])})
    return result_map

def main():
    result_map=read_time_zero_results()
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_ocean_reference_num_time_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(common.data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for ref_num,missing_num in config_vec:
            prediction_structure_path=os.path.join(test_dir,f"prediction_structure_r{ref_num}_m{missing_num}.bin")
            for field in common.field_vec:
                for current_time_step in time_step_vec:
                    for rel_eb in common.rel_eb_vec:
                        result=ref_common.benchmark(field,ref_num,missing_num,rel_eb,test_dir,prediction_structure_path,current_time_step)
                        result_vec.append(result)
                        result_map[(ref_num,missing_num)][field][current_time_step].append(result)
            print(f"完成ref_num={ref_num},允许缺少={missing_num}",flush=True)
    summary_vec=[]
    baseline_config=(10,2)
    for config in config_vec:
        field_mean_bd_rate_vec=[]
        for field in common.field_vec:
            time_bd_rate_vec=[]
            for current_time_step in (0,*time_step_vec):
                bd_rate,_,_=common.calculate_bd_rate(result_map[baseline_config][field][current_time_step],result_map[config][field][current_time_step])
                time_bd_rate_vec.append(bd_rate)
            field_mean_bd_rate_vec.append(statistics.fmean(time_bd_rate_vec))
        summary={"ref_num":config[0],"cache_missing_num":config[1],"cache_refill_threshold":config[0]-config[1],"mean_bd_rate_vs_10_8":statistics.fmean(field_mean_bd_rate_vec),"median_bd_rate_vs_10_8":statistics.median(field_mean_bd_rate_vec),"worst_field_bd_rate_vs_10_8":max(field_mean_bd_rate_vec)}
        summary_vec.append(summary)
        print(f"ref_num={config[0]},允许缺少={config[1]}: 三time step mean={summary['mean_bd_rate_vs_10_8']:.6f}%,median={summary['median_bd_rate_vs_10_8']:.6f}%,worst={summary['worst_field_bd_rate_vs_10_8']:.6f}%",flush=True)
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
