import csv,math
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[1]
result_root=project_root/"results/current"

def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))

def calculate_average_br_change(reference,test):
    reference_points=sorted((float(row["psnr"]),float(row["br"])) for row in reference)
    test_points=sorted((float(row["psnr"]),float(row["br"])) for row in test)
    reference_psnr=np.asarray([point[0] for point in reference_points])
    reference_br=np.asarray([point[1] for point in reference_points])
    test_psnr=np.asarray([point[0] for point in test_points])
    test_br=np.asarray([point[1] for point in test_points])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    if not upper>lower:
        raise RuntimeError("RD曲线没有共同PSNR范围")
    knots=np.unique(np.concatenate(([lower,upper],reference_psnr[(reference_psnr>lower)&(reference_psnr<upper)],test_psnr[(test_psnr>lower)&(test_psnr<upper)])))
    reference_area=np.trapezoid(np.interp(knots,reference_psnr,reference_br),knots)
    test_area=np.trapezoid(np.interp(knots,test_psnr,test_br),knots)
    return (test_area/reference_area-1)*100

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def interpolate_at_psnr(rows,psnr,name):
    points=sorted((float(row["psnr"]),float(row[name])) for row in rows)
    for (left_psnr,left_value),(right_psnr,right_value) in zip(points,points[1:]):
        if left_psnr<=psnr<=right_psnr:
            return left_value+(right_value-left_value)*(psnr-left_psnr)/(right_psnr-left_psnr)
    raise RuntimeError(f"PSNR {psnr}不在曲线范围内")

def common_psnr_knots(reference,test):
    reference_psnr=np.asarray([float(row["psnr"]) for row in reference])
    test_psnr=np.asarray([float(row["psnr"]) for row in test])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    if not upper>lower:
        raise RuntimeError("RD曲线没有共同PSNR范围")
    return np.unique(np.concatenate(([lower,upper],reference_psnr[(reference_psnr>lower)&(reference_psnr<upper)],test_psnr[(test_psnr>lower)&(test_psnr<upper)])))

def geometric_metric_ratio_at_same_psnr(reference,test,name):
    ratios=[interpolate_at_psnr(test,psnr,name)/interpolate_at_psnr(reference,psnr,name) for psnr in common_psnr_knots(reference,test)]
    return geometric_mean(ratios)

def main():
    summary=[]
    for path in sorted((result_root/"fields").glob("*.csv")):
        rows=read_rows(path)
        unsz=sorted((row for row in rows if row["compressor"]=="unsz"),key=lambda row:float(row["psnr"]))
        umc=sorted((row for row in rows if row["compressor"]=="umc"),key=lambda row:float(row["psnr"]))
        if len(unsz)<2 or umc and len(umc)<2:
            raise RuntimeError(f"{path}没有可插值的完整曲线：UNSZ={len(unsz)},UMC={len(umc)}")
        data_dir=unsz[0].get("data_dir",unsz[0].get("dataset",""))
        item={"field":unsz[0]["field"],"data_dir":data_dir,"time_step_num":unsz[0]["time_step_num"],"representative_psnr":"","representative_cr_ratio_vs_umc":"","representative_total_time_ratio_vs_umc":"","average_br_change_vs_umc":"","compression_ratio_vs_umc":"","total_time_ratio_vs_umc":"","compression_time_ratio_vs_umc":"","decompression_time_ratio_vs_umc":"","unsz_total_throughput_MB_s":geometric_mean([float(row["input_bytes"])/(float(row["compression_time"])+float(row["decompression_time"]))/1e6 for row in unsz]),"umc_total_throughput_MB_s":""}
        if umc:
            average_br_change=calculate_average_br_change(umc,unsz)
            representative_row=min(umc,key=lambda row:abs(float(row["rel_eb"])-1e-3))
            representative_psnr=float(representative_row["psnr"])
            unsz_br=interpolate_at_psnr(unsz,representative_psnr,"br")
            umc_br=interpolate_at_psnr(umc,representative_psnr,"br")
            unsz_total_time=interpolate_at_psnr(unsz,representative_psnr,"compression_decompression_time")
            umc_total_time=interpolate_at_psnr(umc,representative_psnr,"compression_decompression_time")
            item.update({"representative_psnr":representative_psnr,"representative_cr_ratio_vs_umc":umc_br/unsz_br,"representative_total_time_ratio_vs_umc":unsz_total_time/umc_total_time,"average_br_change_vs_umc":average_br_change,"compression_ratio_vs_umc":1/(1+average_br_change/100),"total_time_ratio_vs_umc":geometric_metric_ratio_at_same_psnr(umc,unsz,"compression_decompression_time"),"compression_time_ratio_vs_umc":geometric_metric_ratio_at_same_psnr(umc,unsz,"compression_time"),"decompression_time_ratio_vs_umc":geometric_metric_ratio_at_same_psnr(umc,unsz,"decompression_time"),"umc_total_throughput_MB_s":geometric_mean([float(row["input_bytes"])/(float(row["compression_time"])+float(row["decompression_time"]))/1e6 for row in umc])})
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
