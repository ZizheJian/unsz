import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_rows(csv_path,time_step_num):
    with open(csv_path,newline="") as csv_file:
        rows=[row for row in csv.DictReader(csv_file) if int(row["TIME_STEP_NUM"])==time_step_num and row.get("UNSZ_BR","") and row.get("UMC_BR","")]
    rows.sort(key=lambda row:float(row["REL_EB"]),reverse=True)
    if len(rows)!=5:
        raise ValueError(f"{csv_path}没有完整的五点UNSZ/UMC曲线")
    return rows


def read_curve(rows,method):
    rows=sorted(rows,key=lambda row:float(row[f"{method}_PSNR"]))
    return np.asarray([float(row[f"{method}_BR"]) for row in rows]),np.asarray([float(row[f"{method}_PSNR"]) for row in rows])


def calculate_bd_rate(reference_br,reference_psnr,test_br,test_psnr):
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    reference_integral=np.polyint(np.polyfit(reference_psnr,np.log(reference_br),3))
    test_integral=np.polyint(np.polyfit(test_psnr,np.log(test_br),3))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (np.exp((test_area-reference_area)/(upper-lower))-1)*100


def save_figure(figure,output_path):
    output_path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output_path.with_suffix(".png"),dpi=220,bbox_inches="tight",facecolor="white")
    figure.savefig(output_path.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(figure)


def plot(data_dir,dataset,fields,time_step_num,output_prefix):
    column_num=2 if len(fields)==4 else min(3,len(fields))
    row_num=math.ceil(len(fields)/column_num)
    plt.rcParams.update({"font.size":10,"axes.labelsize":10,"axes.titlesize":11,"legend.fontsize":9,"pdf.fonttype":42,"ps.fonttype":42})
    figure,axes=plt.subplots(row_num,column_num,figsize=(4.25*column_num,3.45*row_num),squeeze=False,layout="constrained")
    summary=[]
    for field,axis in zip(fields,axes.ravel()):
        rows=read_rows(data_dir/f"{field}.csv",time_step_num)
        unsz_br,unsz_psnr=read_curve(rows,"UNSZ")
        umc_br,umc_psnr=read_curve(rows,"UMC")
        bd_rate=calculate_bd_rate(umc_br,umc_psnr,unsz_br,unsz_psnr)
        summary.append((field,bd_rate))
        axis.plot(umc_br,umc_psnr,color="#D55E00",marker="o",markersize=5,linewidth=2,label="UMC")
        axis.plot(unsz_br,unsz_psnr,color="#0072B2",marker="s",markersize=5,linewidth=2,label="UNSZ")
        axis.set_xlim(left=0)
        axis.set_xlabel("Bit rate (bit/value)")
        axis.set_ylabel("PSNR (dB)")
        axis.set_title(field)
        axis.grid(True,which="both",linestyle=":",alpha=0.55)
        axis.text(0.97,0.04,f"BD-rate: {bd_rate:+.2f}%",transform=axis.transAxes,ha="right",va="bottom",fontsize=9,bbox={"boxstyle":"round,pad=0.25","facecolor":"white","edgecolor":"0.8","alpha":0.9})
    for axis in axes.ravel()[len(fields):]:
        axis.remove()
    axes.ravel()[0].legend(frameon=False)
    figure.suptitle(f"{dataset}: UNSZ vs. UMC",fontsize=14)
    save_figure(figure,output_prefix.parent/f"{output_prefix.name}_rd")

    figure,axes=plt.subplots(row_num,column_num,figsize=(4.25*column_num,3.45*row_num),squeeze=False,layout="constrained")
    for field,axis in zip(fields,axes.ravel()):
        rows=read_rows(data_dir/f"{field}.csv",time_step_num)
        rel_eb=np.asarray([float(row["REL_EB"]) for row in rows])
        axis.plot(rel_eb,[float(row["UNSZ_COMPRESSION_MB_S"]) for row in rows],color="#0072B2",marker="s",linewidth=2,label="UNSZ compression")
        axis.plot(rel_eb,[float(row["UNSZ_DECOMPRESSION_MB_S"]) for row in rows],color="#0072B2",marker="s",linestyle="--",linewidth=2,label="UNSZ decompression")
        axis.plot(rel_eb,[float(row["UMC_COMPRESSION_MB_S"]) for row in rows],color="#D55E00",marker="o",linewidth=2,label="UMC compression")
        axis.plot(rel_eb,[float(row["UMC_DECOMPRESSION_MB_S"]) for row in rows],color="#D55E00",marker="o",linestyle="--",linewidth=2,label="UMC decompression")
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel("Relative error bound")
        axis.set_ylabel("Throughput (MB/s)")
        axis.set_title(field)
        axis.grid(True,which="both",linestyle=":",alpha=0.55)
    for axis in axes.ravel()[len(fields):]:
        axis.remove()
    axes.ravel()[0].legend(frameon=False)
    figure.suptitle(f"{dataset}: serial core throughput",fontsize=14)
    save_figure(figure,output_prefix.parent/f"{output_prefix.name}_throughput")
    for field,bd_rate in summary:
        print(f"{dataset},{field},{bd_rate:.9f}")


def main():
    parser=argparse.ArgumentParser(description="绘制一个数据集中各field独立的UNSZ/UMC RD曲线")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-dataset",required=True)
    parser.add_argument("-fields",nargs="+",required=True)
    parser.add_argument("-time-step-num",type=int,required=True)
    parser.add_argument("-output-prefix",type=Path,required=True)
    args=parser.parse_args()
    plot(args.data_dir,args.dataset,args.fields,args.time_step_num,args.output_prefix)


if __name__=="__main__":
    main()
