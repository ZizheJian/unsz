import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv_rows(csv_path):
    with open(csv_path,newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def read_rows(csv_path,time_step_num,reference_csv_path=None):
    source=read_csv_rows(csv_path)
    reference_source=read_csv_rows(reference_csv_path) if reference_csv_path is not None else source
    if source and "compressor" in source[0]:
        unsz_rows=[row for row in source if row["compressor"]=="unsz" and int(row["time_step_num"])==time_step_num]
    elif source and "UNSZ_BR" in source[0]:
        unsz_rows=[{"br":row["UNSZ_BR"],"psnr":row["UNSZ_PSNR"],"compression_MB_s":row["UNSZ_COMPRESSION_MB_S"],"decompression_MB_s":row["UNSZ_DECOMPRESSION_MB_S"]} for row in source if int(row["TIME_STEP_NUM"])==time_step_num and row.get("UNSZ_BR","")]
    else:
        unsz_rows=[row for row in source if int(row["time_step_num"])==time_step_num]
    if reference_source and "compressor" in reference_source[0]:
        umc_rows=[row for row in reference_source if row["compressor"]=="umc" and int(row["time_step_num"])==time_step_num]
    elif reference_source and "UMC_BR" in reference_source[0]:
        umc_rows=[{"br":row["UMC_BR"],"psnr":row["UMC_PSNR"],"compression_MB_s":row["UMC_COMPRESSION_MB_S"],"decompression_MB_s":row["UMC_DECOMPRESSION_MB_S"]} for row in reference_source if int(row["TIME_STEP_NUM"])==time_step_num and row.get("UMC_BR","")]
    else:
        umc_rows=[]
    if len(unsz_rows)<2 or len(umc_rows)<2:
        raise ValueError(f"{csv_path}没有可插值的完整UNSZ/UMC曲线")
    return unsz_rows,umc_rows


def read_curve(rows):
    rows=sorted(rows,key=lambda row:float(row["psnr"]))
    return np.asarray([float(row["br"]) for row in rows]),np.asarray([float(row["psnr"]) for row in rows])


def calculate_average_br_change(reference_br,reference_psnr,test_br,test_psnr):
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    knots=np.unique(np.concatenate(([lower,upper],reference_psnr[(reference_psnr>lower)&(reference_psnr<upper)],test_psnr[(test_psnr>lower)&(test_psnr<upper)])))
    reference_area=np.trapezoid(np.interp(knots,reference_psnr,reference_br),knots)
    test_area=np.trapezoid(np.interp(knots,test_psnr,test_br),knots)
    return (test_area/reference_area-1)*100


def save_figure(figure,output_path):
    output_path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output_path.with_suffix(".png"),dpi=220,bbox_inches="tight",facecolor="white")
    figure.savefig(output_path.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(figure)


def combined_throughput(compression_throughput,decompression_throughput):
    return 1/(1/compression_throughput+1/decompression_throughput)


def read_throughput_curve(rows):
    rows=sorted(rows,key=lambda row:float(row["psnr"]))
    psnr=np.asarray([float(row["psnr"]) for row in rows])
    throughput=np.asarray([combined_throughput(float(row["compression_MB_s"]),float(row["decompression_MB_s"])) for row in rows])
    return psnr,throughput


def calculate_total_time_ratio(unsz_rows,umc_rows):
    unsz_psnr,unsz_throughput=read_throughput_curve(unsz_rows)
    umc_psnr,umc_throughput=read_throughput_curve(umc_rows)
    lower=max(unsz_psnr.min(),umc_psnr.min())
    upper=min(unsz_psnr.max(),umc_psnr.max())
    knots=np.unique(np.concatenate(([lower,upper],unsz_psnr[(unsz_psnr>lower)&(unsz_psnr<upper)],umc_psnr[(umc_psnr>lower)&(umc_psnr<upper)])))
    ratios=np.interp(knots,umc_psnr,umc_throughput)/np.interp(knots,unsz_psnr,unsz_throughput)
    return math.exp(np.log(ratios).mean())


def resolve_field_csv(data_dir,dataset,field):
    direct_path=data_dir/f"{field}.csv"
    if direct_path.exists():
        return direct_path
    prefixed_path=data_dir/f"{dataset.lower()}_{field}.csv"
    if prefixed_path.exists():
        return prefixed_path
    raise FileNotFoundError(f"{data_dir}中找不到{field}的CSV")


def plot(data_dir,dataset,fields,time_step_num,output_prefix,column_num,reference_data_dir):
    column_num=min(column_num,len(fields))
    row_num=math.ceil(len(fields)/column_num)
    plt.rcParams.update({"font.size":10,"axes.labelsize":10,"axes.titlesize":11,"legend.fontsize":9,"pdf.fonttype":42,"ps.fonttype":42})
    figure,axes=plt.subplots(row_num,column_num,figsize=(4.25*column_num,3.45*row_num),squeeze=False,layout="constrained")
    summary=[]
    for field,axis in zip(fields,axes.ravel()):
        csv_path=resolve_field_csv(data_dir,dataset,field)
        reference_csv_path=resolve_field_csv(reference_data_dir,dataset,field) if reference_data_dir is not None else None
        unsz_rows,umc_rows=read_rows(csv_path,time_step_num,reference_csv_path)
        unsz_br,unsz_psnr=read_curve(unsz_rows)
        umc_br,umc_psnr=read_curve(umc_rows)
        average_br_change=calculate_average_br_change(umc_br,umc_psnr,unsz_br,unsz_psnr)
        summary.append((field,average_br_change))
        axis.plot(umc_br,umc_psnr,color="#D55E00",marker="o",markersize=5,linewidth=2,label="UMC")
        axis.plot(unsz_br,unsz_psnr,color="#0072B2",marker="s",markersize=5,linewidth=2,label="UNSZ")
        axis.set_xlim(left=0)
        axis.set_xlabel("Bit rate (bit/value)")
        axis.set_ylabel("PSNR (dB)")
        axis.set_title(field.removeprefix(dataset.lower()+"_"))
        axis.grid(True,which="both",linestyle=":",alpha=0.55)
        axis.text(0.97,0.04,f"Average BR change: {average_br_change:+.2f}%",transform=axis.transAxes,ha="right",va="bottom",fontsize=9,bbox={"boxstyle":"round,pad=0.25","facecolor":"white","edgecolor":"0.8","alpha":0.9})
    for axis in axes.ravel()[len(fields):]:
        axis.remove()
    axes.ravel()[0].legend(frameon=False)
    figure.suptitle(f"{dataset}: UNSZ vs. UMC",fontsize=14)
    save_figure(figure,output_prefix.parent/f"{output_prefix.name}_rd")

    figure,axes=plt.subplots(row_num,column_num,figsize=(4.25*column_num,3.45*row_num),squeeze=False,layout="constrained")
    for field,axis in zip(fields,axes.ravel()):
        csv_path=resolve_field_csv(data_dir,dataset,field)
        reference_csv_path=resolve_field_csv(reference_data_dir,dataset,field) if reference_data_dir is not None else None
        unsz_rows,umc_rows=read_rows(csv_path,time_step_num,reference_csv_path)
        unsz_psnr,unsz_total=read_throughput_curve(unsz_rows)
        umc_psnr,umc_total=read_throughput_curve(umc_rows)
        total_time_ratio=calculate_total_time_ratio(unsz_rows,umc_rows)
        axis.plot(unsz_psnr,unsz_total,color="#0072B2",marker="s",linewidth=2,label="UNSZ")
        axis.plot(umc_psnr,umc_total,color="#D55E00",marker="o",linewidth=2,label="UMC")
        axis.set_xlabel("PSNR (dB)")
        axis.set_ylabel("Combined throughput (MB/s)")
        axis.set_ylim(bottom=0)
        axis.set_title(field.removeprefix(dataset.lower()+"_"))
        axis.grid(True,which="both",linestyle=":",alpha=0.55)
        axis.text(0.97,0.04,f"Total time / UMC: {total_time_ratio:.3f}×",transform=axis.transAxes,ha="right",va="bottom",fontsize=9,bbox={"boxstyle":"round,pad=0.25","facecolor":"white","edgecolor":"0.8","alpha":0.9})
    for axis in axes.ravel()[len(fields):]:
        axis.remove()
    axes.ravel()[0].legend(frameon=False)
    figure.suptitle(f"{dataset}: combined serial core throughput",fontsize=14)
    save_figure(figure,output_prefix.parent/f"{output_prefix.name}_throughput")
    for field,average_br_change in summary:
        print(f"{dataset},{field},{average_br_change:.9f}")


def main():
    parser=argparse.ArgumentParser(description="绘制一个数据集中各field独立的UNSZ/UMC RD曲线")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-dataset",required=True)
    parser.add_argument("-fields",nargs="+",required=True)
    parser.add_argument("-time-step-num",type=int,required=True)
    parser.add_argument("-output-prefix",type=Path,required=True)
    parser.add_argument("-column-num",type=int,default=6)
    parser.add_argument("-reference-data-dir",type=Path)
    args=parser.parse_args()
    plot(args.data_dir,args.dataset,args.fields,args.time_step_num,args.output_prefix,args.column_num,args.reference_data_dir)


if __name__=="__main__":
    main()
