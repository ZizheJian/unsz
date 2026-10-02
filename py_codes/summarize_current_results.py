import csv,math
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[1]
result_root=project_root/"results/current"

def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))

def calculate_bd_rate(reference,test):
    reference_points=sorted((float(row["psnr"]),float(row["br"])) for row in reference)
    test_points=sorted((float(row["psnr"]),float(row["br"])) for row in test)
    reference_psnr=np.asarray([point[0] for point in reference_points])
    reference_br=np.asarray([point[1] for point in reference_points])
    test_psnr=np.asarray([point[0] for point in test_points])
    test_br=np.asarray([point[1] for point in test_points])
    validate_rd_curve(reference_br,reference_psnr)
    validate_rd_curve(test_br,test_psnr)
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    if not upper>lower:
        raise RuntimeError("RD曲线没有共同PSNR范围")
    knots=np.unique(np.concatenate(([lower,upper],reference_psnr[(reference_psnr>lower)&(reference_psnr<upper)],test_psnr[(test_psnr>lower)&(test_psnr<upper)])))
    reference_log_br=np.interp(knots,reference_psnr,np.log(reference_br))
    test_log_br=np.interp(knots,test_psnr,np.log(test_br))
    mean_log_br_ratio=np.trapezoid(test_log_br-reference_log_br,knots)/(upper-lower)
    return math.expm1(mean_log_br_ratio)*100

def validate_rd_curve(br,psnr):
    if br.size<2 or psnr.size!=br.size:
        raise RuntimeError("RD曲线至少需要两个BR/PSNR点")
    if not np.all(np.isfinite(br)) or not np.all(np.isfinite(psnr)) or np.any(br<=0):
        raise RuntimeError("RD曲线的BR必须是有限正数，PSNR必须是有限数")
    if np.any(np.diff(psnr)<=0) or np.any(np.diff(br)<=0):
        raise RuntimeError("RD曲线必须随PSNR严格增加BR；请删除重复或被支配的实验点")

def interpolate_positive_at_psnr(rows,psnr,name):
    points=sorted((float(row["psnr"]),float(row[name])) for row in rows)
    for (left_psnr,left_value),(right_psnr,right_value) in zip(points,points[1:]):
        if left_psnr<=psnr<=right_psnr:
            if not math.isfinite(left_value) or not math.isfinite(right_value) or left_value<=0 or right_value<=0:
                raise RuntimeError(f"{name}必须是有限正数，才能在相同PSNR下插值")
            ratio=(psnr-left_psnr)/(right_psnr-left_psnr)
            return math.exp(math.log(left_value)*(1-ratio)+math.log(right_value)*ratio)
    raise RuntimeError(f"PSNR {psnr}不在曲线范围内")

def common_psnr_knots(reference,test):
    reference_psnr=np.asarray([float(row["psnr"]) for row in reference])
    test_psnr=np.asarray([float(row["psnr"]) for row in test])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    if not upper>lower:
        raise RuntimeError("RD曲线没有共同PSNR范围")
    return np.unique(np.concatenate(([lower,upper],reference_psnr[(reference_psnr>lower)&(reference_psnr<upper)],test_psnr[(test_psnr>lower)&(test_psnr<upper)])))

def geometric_metric_means_at_same_psnr(reference,test,value_function,name):
    reference_points=sorted((float(row["psnr"]),value_function(row)) for row in reference)
    test_points=sorted((float(row["psnr"]),value_function(row)) for row in test)
    reference_psnr=np.asarray([point[0] for point in reference_points])
    reference_metric=np.asarray([point[1] for point in reference_points])
    test_psnr=np.asarray([point[0] for point in test_points])
    test_metric=np.asarray([point[1] for point in test_points])
    if not np.all(np.isfinite(reference_metric)) or not np.all(np.isfinite(test_metric)) or np.any(reference_metric<=0) or np.any(test_metric<=0):
        raise RuntimeError(f"{name}必须全部是有限正数，才能计算几何平均倍率")
    if np.any(np.diff(reference_psnr)<=0) or np.any(np.diff(test_psnr)<=0):
        raise RuntimeError(f"{name}曲线的PSNR必须严格增加")
    knots=common_psnr_knots(reference,test)
    lower=knots[0]
    upper=knots[-1]
    reference_log_metric=np.interp(knots,reference_psnr,np.log(reference_metric))
    test_log_metric=np.interp(knots,test_psnr,np.log(test_metric))
    reference_mean=math.exp(np.trapezoid(reference_log_metric,knots)/(upper-lower))
    test_mean=math.exp(np.trapezoid(test_log_metric,knots)/(upper-lower))
    return reference_mean,test_mean

def geometric_metric_ratio_at_same_psnr(reference,test,name):
    reference_mean,test_mean=geometric_metric_means_at_same_psnr(reference,test,lambda row:float(row[name]),name)
    return test_mean/reference_mean

def experiment_identity(row):
    return row["task"],row["field"],row["time_step_num"],row["label_flow"],row["value_flow"],row.get("label_value_execution","not_applicable")

def main():
    summary=[]
    for path in sorted((result_root/"fields").glob("*.csv")):
        rows=read_rows(path)
        if rows and "task" not in rows[0]:
            raise RuntimeError(f"{path}是没有task身份的旧格式CSV，不能用于正式汇总")
        unsz_group_dict={}
        for row in rows:
            if row["compressor"]=="unsz":
                unsz_group_dict.setdefault(experiment_identity(row),[]).append(row)
        for identity,unsz in unsz_group_dict.items():
            task,field,time_step_num,label_flow,value_flow,label_value_execution=identity
            unsz=sorted(unsz,key=lambda row:float(row["psnr"]))
            umc=sorted((row for row in rows if row["compressor"]=="umc" and row["task"]==task and row["field"]==field and row["time_step_num"]==time_step_num),key=lambda row:float(row["psnr"]))
            if len(unsz)<2 or umc and len(umc)<2:
                raise RuntimeError(f"{path}的实验{identity}没有可插值的完整曲线：UNSZ={len(unsz)},UMC={len(umc)}")
            data_dir=unsz[0].get("data_dir",unsz[0].get("dataset",""))
            staged=label_value_execution=="staged"
            unsz_total_throughput=""
            item={"task":task,"field":field,"label_flow":label_flow,"value_flow":value_flow,"label_value_execution":label_value_execution,"data_dir":data_dir,"time_step_num":time_step_num,"representative_psnr":"","representative_cr_ratio_vs_umc":"","representative_total_time_ratio_vs_umc":"","bd_rate_vs_umc":"","compression_ratio_vs_umc":"","total_time_ratio_vs_umc":"","compression_time_ratio_vs_umc":"","decompression_time_ratio_vs_umc":"","unsz_total_throughput_MB_s":unsz_total_throughput,"umc_total_throughput_MB_s":""}
            if umc:
                bd_rate=calculate_bd_rate(umc,unsz)
                representative_row=min(umc,key=lambda row:abs(float(row["rel_eb"])-1e-3))
                representative_psnr=float(representative_row["psnr"])
                unsz_br=interpolate_positive_at_psnr(unsz,representative_psnr,"br")
                umc_br=interpolate_positive_at_psnr(umc,representative_psnr,"br")
                item.update({"representative_psnr":representative_psnr,"representative_cr_ratio_vs_umc":umc_br/unsz_br,"bd_rate_vs_umc":bd_rate,"compression_ratio_vs_umc":1/(1+bd_rate/100)})
                if not staged:
                    unsz_total_time=interpolate_positive_at_psnr(unsz,representative_psnr,"compression_decompression_time")
                    umc_total_time=interpolate_positive_at_psnr(umc,representative_psnr,"compression_decompression_time")
                    throughput_function=lambda row:float(row["input_bytes"])/float(row["compression_decompression_time"])/1e6
                    umc_total_throughput,unsz_total_throughput=geometric_metric_means_at_same_psnr(umc,unsz,throughput_function,"total throughput")
                    item.update({"representative_total_time_ratio_vs_umc":unsz_total_time/umc_total_time,"total_time_ratio_vs_umc":geometric_metric_ratio_at_same_psnr(umc,unsz,"compression_decompression_time"),"compression_time_ratio_vs_umc":geometric_metric_ratio_at_same_psnr(umc,unsz,"compression_time"),"decompression_time_ratio_vs_umc":geometric_metric_ratio_at_same_psnr(umc,unsz,"decompression_time"),"unsz_total_throughput_MB_s":unsz_total_throughput,"umc_total_throughput_MB_s":umc_total_throughput})
            summary.append(item)
    if not summary:
        raise RuntimeError(f"{result_root/'fields'}中没有可汇总的field CSV")
    output=result_root/"summary.csv"
    with output.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=summary[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(f"已汇总{len(summary)}个field：{output}")

if __name__=="__main__":
    main()
