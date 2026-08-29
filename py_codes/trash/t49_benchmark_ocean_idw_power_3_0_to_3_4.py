import csv,os,statistics,tempfile
import t44_benchmark_ocean_idw_power as common
import t48_benchmark_ocean_idw_power_near_3_61 as near

result_path=os.path.join(common.project_root,"ocean_idw_power_3_0_to_3_4_benchmark.csv")
summary_path=os.path.join(common.project_root,"ocean_idw_power_3_0_to_3_4_summary.csv")
time_step_vec=(0,15,30)
power_vec=(3.0,3.1,3.2,3.3,3.4)

def main():
    baseline_map=near.read_existing_baseline()
    result_vec=[]
    summary_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_ocean_power_3_0_to_3_4_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(common.data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        prediction_structure_path=os.path.join(test_dir,"prediction_structure.bin")
        for field in common.field_vec:
            field_bd_rate_map={power:[] for power in power_vec}
            for time_step in time_step_vec:
                for power in power_vec:
                    power_result_vec=[]
                    for rel_eb in common.rel_eb_vec:
                        result=common.benchmark(field,power,rel_eb,test_dir,prediction_structure_path,time_step)
                        result_vec.append(result)
                        power_result_vec.append(result)
                    bd_rate,min_psnr,max_psnr=common.calculate_bd_rate(baseline_map[(field,time_step)],power_result_vec)
                    field_bd_rate_map[power].append(bd_rate)
                    print(f"{field} time_step={time_step},power={power:g}: BD-rate vs 3.61={bd_rate:.6f}%",flush=True)
            for power,bd_rate_vec in field_bd_rate_map.items():
                summary_vec.append({"field":field,"power":power,"mean_bd_rate_vs_3_61":statistics.fmean(bd_rate_vec),"median_bd_rate_vs_3_61":statistics.median(bd_rate_vec),"worst_bd_rate_vs_3_61":max(bd_rate_vec)})
            best_summary=min((summary for summary in summary_vec if summary["field"]==field),key=lambda summary:summary["mean_bd_rate_vs_3_61"])
            print(f"{field} best power={best_summary['power']:g},mean BD-rate={best_summary['mean_bd_rate_vs_3_61']:.6f}%",flush=True)
    for power in power_vec:
        value_vec=[summary["mean_bd_rate_vs_3_61"] for summary in summary_vec if summary["power"]==power]
        print(f"all fields power={power:g}: mean={statistics.fmean(value_vec):.6f}%,median={statistics.median(value_vec):.6f}%,worst field={max(value_vec):.6f}%",flush=True)
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
