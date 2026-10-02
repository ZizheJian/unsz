import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv_rows(csv_path):
    with open(csv_path,newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def read_rows(csv_path,field,time_step_num,task,label_flow,value_flow,label_value_execution=None,reference_csv_path=None,reference_task=None):
    source=read_csv_rows(csv_path)
    reference_source=read_csv_rows(reference_csv_path) if reference_csv_path is not None else source
    if not source or "compressor" not in source[0] or "task" not in source[0]:
        raise ValueError(f"{csv_path}是没有task身份的旧格式CSV，不能用于正式绘图")
    if not reference_source or "compressor" not in reference_source[0] or "task" not in reference_source[0]:
        raise ValueError(f"{reference_csv_path or csv_path}是没有task身份的旧格式CSV，不能用于正式绘图")
    unsz_rows=[row for row in source if row["compressor"]=="unsz" and row.get("field")==field and int(row["time_step_num"])==time_step_num and row["task"]==task and row["label_flow"]==label_flow and row["value_flow"]==value_flow]
    execution_vec={row.get("label_value_execution","not_applicable") for row in unsz_rows}
    if label_value_execution is None:
        if len(execution_vec)>1:
            raise ValueError(f"{csv_path}包含多种label/value执行方式，请用-label-value-execution明确选择")
        selected_execution=next(iter(execution_vec),"not_applicable")
    else:
        selected_execution=label_value_execution
        unsz_rows=[row for row in unsz_rows if row.get("label_value_execution","not_applicable")==selected_execution]
    selected_reference_task=task if reference_task is None else reference_task
    umc_rows=[row for row in reference_source if row["compressor"]=="umc" and row.get("field")==field and int(row["time_step_num"])==time_step_num and row["task"]==selected_reference_task]
    if len(unsz_rows)<2 or len(umc_rows)<2:
        raise ValueError(f"{csv_path}没有可插值的完整UNSZ/UMC曲线")
    return unsz_rows,umc_rows,selected_execution


def read_curve(rows):
    rows=sorted(rows,key=lambda row:float(row["psnr"]))
    br=np.asarray([float(row["br"]) for row in rows])
    psnr=np.asarray([float(row["psnr"]) for row in rows])
    validate_rd_curve(br,psnr)
    return br,psnr


def validate_rd_curve(br,psnr):
    if br.size<2 or psnr.size!=br.size:
        raise ValueError("RD曲线至少需要两个BR/PSNR点")
    if not np.all(np.isfinite(br)) or not np.all(np.isfinite(psnr)) or np.any(br<=0):
        raise ValueError("RD曲线的BR必须是有限正数，PSNR必须是有限数")
    if np.any(np.diff(psnr)<=0) or np.any(np.diff(br)<=0):
        raise ValueError("RD曲线必须随PSNR严格增加BR；请删除重复或被支配的实验点")


def calculate_bd_rate(reference_br,reference_psnr,test_br,test_psnr):
    validate_rd_curve(reference_br,reference_psnr)
    validate_rd_curve(test_br,test_psnr)
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    if not upper>lower:
        raise ValueError("UNSZ和UMC的RD曲线没有共同PSNR范围")
    knots=np.unique(np.concatenate(([lower,upper],reference_psnr[(reference_psnr>lower)&(reference_psnr<upper)],test_psnr[(test_psnr>lower)&(test_psnr<upper)])))
    reference_log_br=np.interp(knots,reference_psnr,np.log(reference_br))
    test_log_br=np.interp(knots,test_psnr,np.log(test_br))
    mean_log_br_ratio=np.trapezoid(test_log_br-reference_log_br,knots)/(upper-lower)
    return math.expm1(mean_log_br_ratio)*100


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
    if np.any(np.diff(psnr)<=0):
        raise ValueError("吞吐率曲线的PSNR必须严格增加")
    return psnr,throughput


def calculate_total_time_ratio(unsz_rows,umc_rows):
    unsz_psnr,unsz_throughput=read_throughput_curve(unsz_rows)
    umc_psnr,umc_throughput=read_throughput_curve(umc_rows)
    if not np.all(np.isfinite(unsz_throughput)) or not np.all(np.isfinite(umc_throughput)) or np.any(unsz_throughput<=0) or np.any(umc_throughput<=0):
        raise ValueError("UNSZ和UMC吞吐率必须全部是有限正数")
    lower=max(unsz_psnr.min(),umc_psnr.min())
    upper=min(unsz_psnr.max(),umc_psnr.max())
    if not upper>lower:
        raise ValueError("UNSZ和UMC吞吐率曲线没有共同PSNR范围")
    knots=np.unique(np.concatenate(([lower,upper],unsz_psnr[(unsz_psnr>lower)&(unsz_psnr<upper)],umc_psnr[(umc_psnr>lower)&(umc_psnr<upper)])))
    unsz_log_throughput=np.interp(knots,unsz_psnr,np.log(unsz_throughput))
    umc_log_throughput=np.interp(knots,umc_psnr,np.log(umc_throughput))
    mean_log_time_ratio=np.trapezoid(umc_log_throughput-unsz_log_throughput,knots)/(upper-lower)
    return math.exp(mean_log_time_ratio)


def resolve_field_csv(data_dir,dataset,field):
    direct_path=data_dir/f"{field}.csv"
    if direct_path.exists():
        return direct_path
    prefixed_path=data_dir/f"{dataset.lower()}_{field}.csv"
    if prefixed_path.exists():
        return prefixed_path
    raise FileNotFoundError(f"{data_dir}中找不到{field}的CSV")


def plot(data_dir,dataset,fields,time_step_num,task,label_flow,value_flow,label_value_execution,output_prefix,column_num,reference_data_dir,reference_task):
    column_num=min(column_num,len(fields))
    row_num=math.ceil(len(fields)/column_num)
    plt.rcParams.update({"font.size":10,"axes.labelsize":10,"axes.titlesize":11,"legend.fontsize":9,"pdf.fonttype":42,"ps.fonttype":42})
    figure,axes=plt.subplots(row_num,column_num,figsize=(4.25*column_num,3.45*row_num),squeeze=False,layout="constrained")
    summary=[]
    selected_execution=None
    for field,axis in zip(fields,axes.ravel()):
        csv_path=resolve_field_csv(data_dir,dataset,field)
        reference_csv_path=resolve_field_csv(reference_data_dir,dataset,field) if reference_data_dir is not None else None
        unsz_rows,umc_rows,field_execution=read_rows(csv_path,field,time_step_num,task,label_flow,value_flow,label_value_execution,reference_csv_path,reference_task)
        if selected_execution is None:
            selected_execution=field_execution
        elif field_execution!=selected_execution:
            raise ValueError("同一张图中的field使用了不同label/value执行方式")
        unsz_br,unsz_psnr=read_curve(unsz_rows)
        umc_br,umc_psnr=read_curve(umc_rows)
        bd_rate=calculate_bd_rate(umc_br,umc_psnr,unsz_br,unsz_psnr)
        summary.append((field,bd_rate))
        axis.plot(umc_br,umc_psnr,color="#D55E00",marker="o",markersize=5,linewidth=2,label="UMC")
        axis.plot(unsz_br,unsz_psnr,color="#0072B2",marker="s",markersize=5,linewidth=2,label="UNSZ")
        axis.set_xlim(left=0)
        axis.set_xlabel("Bit rate (bit/value)")
        axis.set_ylabel("PSNR (dB)")
        axis.set_title(field.removeprefix(dataset.lower()+"_"))
        axis.grid(True,which="both",linestyle=":",alpha=0.55)
        axis.text(0.97,0.04,f"BD-rate: {bd_rate:+.2f}%",transform=axis.transAxes,ha="right",va="bottom",fontsize=9,bbox={"boxstyle":"round,pad=0.25","facecolor":"white","edgecolor":"0.8","alpha":0.9})
    for axis in axes.ravel()[len(fields):]:
        axis.remove()
    axes.ravel()[0].legend(frameon=False)
    figure.suptitle(f"{dataset}: UNSZ vs. UMC",fontsize=14)
    save_figure(figure,output_prefix.parent/f"{output_prefix.name}_rd")

    if selected_execution=="staged":
        for field,bd_rate in summary:
            print(f"{dataset},{field},{bd_rate:.9f}")
        return

    figure,axes=plt.subplots(row_num,column_num,figsize=(4.25*column_num,3.45*row_num),squeeze=False,layout="constrained")
    for field,axis in zip(fields,axes.ravel()):
        csv_path=resolve_field_csv(data_dir,dataset,field)
        reference_csv_path=resolve_field_csv(reference_data_dir,dataset,field) if reference_data_dir is not None else None
        unsz_rows,umc_rows,_=read_rows(csv_path,field,time_step_num,task,label_flow,value_flow,label_value_execution,reference_csv_path,reference_task)
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
    for field,bd_rate in summary:
        print(f"{dataset},{field},{bd_rate:.9f}")


def main():
    parser=argparse.ArgumentParser(description="绘制一个数据集中各field独立的UNSZ/UMC RD曲线")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-dataset",required=True)
    parser.add_argument("-fields",nargs="+",required=True)
    parser.add_argument("-time-step-num",type=int,required=True)
    parser.add_argument("-task",required=True)
    parser.add_argument("-label-flow",required=True)
    parser.add_argument("-value-flow",required=True)
    parser.add_argument("-label-value-execution",choices=("staged","fused"))
    parser.add_argument("-output-prefix",type=Path,required=True)
    parser.add_argument("-column-num",type=int,default=6)
    parser.add_argument("-reference-data-dir",type=Path)
    parser.add_argument("-reference-task")
    args=parser.parse_args()
    plot(args.data_dir,args.dataset,args.fields,args.time_step_num,args.task,args.label_flow,args.value_flow,args.label_value_execution,args.output_prefix,args.column_num,args.reference_data_dir,args.reference_task)


if __name__=="__main__":
    main()
