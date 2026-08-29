import csv,math,os,tempfile
import t44_benchmark_ocean_idw_power as common

result_path=os.path.join(common.project_root,"ocean_idw_power_time_benchmark.csv")
summary_path=os.path.join(common.project_root,"ocean_idw_power_time_summary.csv")
time_step_vec=(15,30)
candidate_power_map={"layerThickness":5.0,"salinity":3.0,"temperature":2.0,"velocityMeridional":4.0,"velocityX":4.0,"velocityY":4.0,"velocityZ":3.0,"velocityZonal":4.0,"zTop":3.0}

def main():
    result_vec=[]
    summary_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_ocean_idw_power_time_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(common.data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        prediction_structure_path=os.path.join(test_dir,"prediction_structure.bin")
        for field in common.field_vec:
            candidate_power=candidate_power_map[field]
            for time_step in time_step_vec:
                result_map={}
                for power in (common.base_power,candidate_power):
                    power_result_vec=[]
                    for rel_eb in common.rel_eb_vec:
                        result=common.benchmark(field,power,rel_eb,test_dir,prediction_structure_path,time_step)
                        result_vec.append(result)
                        power_result_vec.append(result)
                        print(f"{field} time_step={time_step},power={power:g},rel_eb={rel_eb}: br={result['br']:.6f},psnr={result['psnr']:.6f}",flush=True)
                    result_map[power]=power_result_vec
                bd_rate,min_psnr,max_psnr=common.calculate_bd_rate(result_map[common.base_power],result_map[candidate_power])
                summary={"field":field,"time_step":time_step,"candidate_power":candidate_power,"bd_rate_vs_3_61":bd_rate,"min_psnr":min_psnr,"max_psnr":max_psnr}
                summary_vec.append(summary)
                print(f"{field} time_step={time_step}: BD-rate(power={candidate_power:g} vs 3.61)={bd_rate:.6f}%",flush=True)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    with open(summary_path,"w",newline="") as summary_file:
        writer=csv.DictWriter(summary_file,fieldnames=summary_vec[0].keys())
        writer.writeheader()
        writer.writerows(summary_vec)
    stable_field_num=0
    for field in common.field_vec:
        field_summary_vec=[summary for summary in summary_vec if summary["field"]==field]
        stable=all(summary["bd_rate_vs_3_61"]<0 for summary in field_summary_vec)
        stable_field_num+=stable
        print(f"{field}: candidate_power={candidate_power_map[field]:g},time15/30={[round(summary['bd_rate_vs_3_61'],6) for summary in field_summary_vec]},stable_improvement={stable}",flush=True)
    print(f"两个新增time step均改善的field数量：{stable_field_num}/{len(common.field_vec)}",flush=True)
    print(f"逐点结果已写入{result_path}",flush=True)
    print(f"汇总结果已写入{summary_path}",flush=True)

if __name__=="__main__":
    main()
