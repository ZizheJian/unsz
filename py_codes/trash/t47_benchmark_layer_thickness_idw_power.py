import csv,os,statistics,tempfile
import t44_benchmark_ocean_idw_power as common

result_path=os.path.join(common.project_root,"layer_thickness_idw_power_benchmark.csv")
summary_path=os.path.join(common.project_root,"layer_thickness_idw_power_summary.csv")
time_step_vec=(0,15,30)
power_vec=tuple(round(5.6+0.1*i,1) for i in range(20))

def main():
    result_vec=[]
    summary_vec=[]
    bd_rate_map={power:[] for power in power_vec}
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_layer_power_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(common.data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        prediction_structure_path=os.path.join(test_dir,"prediction_structure.bin")
        for time_step in time_step_vec:
            result_map={}
            for power in (common.base_power,*power_vec):
                power_result_vec=[]
                for rel_eb in common.rel_eb_vec:
                    result=common.benchmark("layerThickness",power,rel_eb,test_dir,prediction_structure_path,time_step)
                    result_vec.append(result)
                    power_result_vec.append(result)
                result_map[power]=power_result_vec
            for power in power_vec:
                bd_rate,min_psnr,max_psnr=common.calculate_bd_rate(result_map[common.base_power],result_map[power])
                bd_rate_map[power].append(bd_rate)
                print(f"layerThickness time_step={time_step},power={power:g}: BD-rate vs 3.61={bd_rate:.6f}%",flush=True)
    for power,bd_rate_vec in bd_rate_map.items():
        summary_vec.append({"power":power,"mean_bd_rate_vs_3_61":statistics.fmean(bd_rate_vec),"median_bd_rate_vs_3_61":statistics.median(bd_rate_vec),"worst_bd_rate_vs_3_61":max(bd_rate_vec)})
    best_summary=min(summary_vec,key=lambda summary:summary["mean_bd_rate_vs_3_61"])
    print(f"layerThickness best power={best_summary['power']:g},mean BD-rate={best_summary['mean_bd_rate_vs_3_61']:.6f}%,worst={best_summary['worst_bd_rate_vs_3_61']:.6f}%",flush=True)
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
