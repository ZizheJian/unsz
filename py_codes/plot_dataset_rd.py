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
    if reference_csv_path is not None:
        reference_source=read_csv_rows(reference_csv_path)
        unsz_rows={float(row["rel_eb"]):row for row in source}
        umc_rows={float(row["rel_eb"]):row for row in reference_source if row["compressor"]=="umc" and int(row["time_step_num"])==time_step_num}
        rows=[]
        for rel_eb in sorted(set(unsz_rows)&set(umc_rows),reverse=True):
            unsz_row=unsz_rows[rel_eb]
            umc_row=umc_rows[rel_eb]
            input_mb=float(umc_row["input_bytes"])/1e6
            rows.append({
                "REL_EB":rel_eb,
                "TIME_STEP_NUM":time_step_num,
                "UNSZ_BR":unsz_row["br"],
                "UNSZ_PSNR":unsz_row["psnr"],
                "UNSZ_COMPRESSION_MB_S":input_mb/float(unsz_row["compression_time"]),
                "UNSZ_DECOMPRESSION_MB_S":input_mb/float(unsz_row["decompression_time"]),
                "UMC_BR":umc_row["br"],
                "UMC_PSNR":umc_row["psnr"],
                "UMC_COMPRESSION_MB_S":umc_row["compression_MB_s"],
                "UMC_DECOMPRESSION_MB_S":umc_row["decompression_MB_s"],
            })
        if len(rows)!=5:
            raise ValueError(f"{csv_path}和{reference_csv_path}没有完整的五点UNSZ/UMC曲线")
        return rows
    if source and "compressor" in source[0]:
        grouped={}
        for row in source:
            if int(row["time_step_num"])==time_step_num:
                grouped.setdefault(float(row["rel_eb"]),{})[row["compressor"]]=row
        rows=[]
        for rel_eb,group in grouped.items():
            if "unsz" not in group or "umc" not in group:
                continue
            row={"REL_EB":rel_eb,"TIME_STEP_NUM":time_step_num}
            for method in ("UNSZ","UMC"):
                source_row=group[method.lower()]
                row[f"{method}_BR"]=source_row["br"]
                row[f"{method}_PSNR"]=source_row["psnr"]
                row[f"{method}_COMPRESSION_MB_S"]=source_row["compression_MB_s"]
                row[f"{method}_DECOMPRESSION_MB_S"]=source_row["decompression_MB_s"]
            rows.append(row)
    else:
        rows=[row for row in source if int(row["TIME_STEP_NUM"])==time_step_num and row.get("UNSZ_BR","") and row.get("UMC_BR","")]
    rows.sort(key=lambda row:float(row["REL_EB"]),reverse=True)
    if len(rows)!=5:
        raise ValueError(f"{csv_path}没有完整的五点UNSZ/UMC曲线")
    return rows


def read_curve(rows,method):
    rows=sorted(rows,key=lambda row:float(row[f"{method}_PSNR"]))
    return np.asarray([float(row[f"{method}_BR"]) for row in rows]),np.asarray([float(row[f"{method}_PSNR"]) for row in rows])


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
        rows=read_rows(csv_path,time_step_num,reference_csv_path)
        unsz_br,unsz_psnr=read_curve(rows,"UNSZ")
        umc_br,umc_psnr=read_curve(rows,"UMC")
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
        rows=read_rows(csv_path,time_step_num,reference_csv_path)
        rel_eb=np.asarray([float(row["REL_EB"]) for row in rows])
        unsz_total=np.asarray([combined_throughput(float(row["UNSZ_COMPRESSION_MB_S"]),float(row["UNSZ_DECOMPRESSION_MB_S"])) for row in rows])
        umc_total=np.asarray([combined_throughput(float(row["UMC_COMPRESSION_MB_S"]),float(row["UMC_DECOMPRESSION_MB_S"])) for row in rows])
        total_time_ratio=math.prod(umc_total/unsz_total)**(1/len(rows))
        axis.plot(rel_eb,unsz_total,color="#0072B2",marker="s",linewidth=2,label="UNSZ")
        axis.plot(rel_eb,umc_total,color="#D55E00",marker="o",linewidth=2,label="UMC")
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set_xlabel("Relative error bound")
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
