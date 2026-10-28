import argparse,csv,json,math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from summarize_current_results import calculate_bd_rate


project_root=Path(__file__).resolve().parents[1]
# 每种误差模式下参与比较的压缩器及其128核结果目录
mode_codecs={"slice":(("UNSZ","unsz/16_fields_8_partitions"),("UMC","umc/128_field_workers"),("SZ3-1D","sz3_1d/32_slice_workers_4_threads_per_slice"),("PFPL","pfpl/64_slice_workers_2_threads_per_slice")),"field":(("UNSZ","unsz/16_fields_8_partitions"),("SZ3-ND","sz3/4_fields_32_partitions"),("QoZ-ND","qoz/4_fields_32_partitions"))}
mode_labels={"slice":"per-slice","field":"per-field"}
styles={"UNSZ":("#0072B2","s"),"UMC":("#D55E00","o"),"SZ3-1D":("#009E73","^"),"SZ3-ND":("#009E73","^"),"PFPL":("#CC79A7","D"),"QoZ-ND":("#E69F00","v")}
shape_groups=(("12x1","58 fields, shape [12]"),("12x80","33 fields, shape [12,80]"),("12x81","4 fields, shape [12,81]"),(None,"All 95 fields"))


def load_runs(result_root,mode):
    # 只使用95个field全部完成且通过校验的EB；每个EB按形状分组汇总码率和PSNR
    runs={}
    for codec,directory in mode_codecs[mode]:
        points=[]
        for summary_path in sorted((result_root/f"ocean_large_per_{mode}_rd_128"/directory).glob("rel_*/summary.json")):
            summary=json.loads(summary_path.read_text())
            if summary.get("field_num")!=95 or not summary.get("valid"):
                continue
            with (summary_path.parent/"fields.csv").open(newline="") as file:
                rows=list(csv.DictReader(file))
            groups={}
            for shape,_ in shape_groups:
                selected=[row for row in rows if shape is None or row["shape"]==shape]
                error=sum(float(row["normalized_squared_error"]) for row in selected)
                groups[shape]={"br":8*sum(int(row["compressed_bytes"]) for row in selected)/sum(int(row["value_num"]) for row in selected),"psnr":-10*math.log10(error/sum(int(row["valid_value_num"]) for row in selected))}
            points.append({"rel_eb":summary["rel_eb"],"groups":groups,"compression_time":summary["reported_compression_time"],"decompression_time":summary["reported_decompression_time"],"total_codec_time":summary["total_codec_time"]})
        runs[codec]=sorted(points,key=lambda point:point["rel_eb"],reverse=True)
    return runs


def legend_handles(codecs):
    return [Line2D([],[],color=styles[codec][0],marker=styles[codec][1],linewidth=1.5,markersize=5,label=codec) for codec in codecs]


def plot_rd(runs,mode,output_path):
    figure,axes=plt.subplots(1,len(shape_groups),figsize=(9,2.6),layout="constrained")
    for (shape,title),axis in zip(shape_groups,axes):
        for codec,points in runs.items():
            color,marker=styles[codec]
            axis.plot([point["groups"][shape]["br"] for point in points],[point["groups"][shape]["psnr"] for point in points],color=color,marker=marker,linewidth=1.5,markersize=5)
        axis.set_xscale("log")
        axis.set_title(title)
        axis.grid(True,which="major",linestyle=":",alpha=.55)
    figure.supxlabel("Bit rate (bits/value, log scale)")
    figure.supylabel(f"{mode_labels[mode].capitalize()} PSNR (dB)")
    figure.legend(handles=legend_handles(runs),loc="lower center",bbox_to_anchor=(.5,1.0),ncol=len(runs),frameon=False)
    save(figure,output_path)


def plot_time(slice_runs,field_runs,output_path):
    figure,axes=plt.subplots(1,2,figsize=(9,2.6),layout="constrained",sharey=True)
    for (mode,runs),axis in zip((("slice",slice_runs),("field",field_runs)),axes):
        rel_ebs=sorted({point["rel_eb"] for points in runs.values() for point in points},reverse=True)
        width=.8/len(runs)
        for codec_index,(codec,points) in enumerate(runs.items()):
            times={point["rel_eb"]:point["total_codec_time"] for point in points}
            x=[eb_index-.4+width*(codec_index+.5) for eb_index,rel_eb in enumerate(rel_ebs) if rel_eb in times]
            axis.bar(x,[times[rel_eb] for rel_eb in rel_ebs if rel_eb in times],width=width*.9,color=styles[codec][0],label=codec)
        axis.set_yscale("log")
        axis.set_xticks(range(len(rel_ebs)),[f"$10^{{{round(math.log10(rel_eb))}}}$" for rel_eb in rel_ebs])
        axis.set_xlabel("REL bound")
        axis.set_title(f"{mode_labels[mode].capitalize()} mode")
        axis.grid(True,axis="y",which="major",linestyle=":",alpha=.55)
        axis.legend(loc="upper left",bbox_to_anchor=(1.0,1.0),frameon=False)
    axes[0].set_ylabel("Total codec time (s, log scale)")
    save(figure,output_path)


def save(figure,output_path):
    output_path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(output_path.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    figure.savefig(output_path.with_suffix(".png"),dpi=220,bbox_inches="tight",facecolor="white")
    plt.close(figure)


def report_ratios(runs,mode):
    # 打印同PSNR码率比：基线码率除以UNSZ码率，在两条曲线的公共PSNR区间内按对数码率平均
    for codec,points in runs.items():
        if codec=="UNSZ":
            continue
        ratios=[]
        for shape,title in shape_groups:
            unsz=[point["groups"][shape] for point in runs["UNSZ"]]
            baseline=[point["groups"][shape] for point in points]
            ratios.append(f"{title}: {1+calculate_bd_rate(unsz,baseline)/100:.2f}x")
        print(f"{mode_labels[mode]} {codec}/UNSZ bit rate at equal PSNR: "+", ".join(ratios))
    for codec,points in runs.items():
        print(f"{mode_labels[mode]} {codec} total codec time: "+", ".join(f"REL {point['rel_eb']:g} {point['total_codec_time']:.1f} s" for point in points))


def main():
    parser=argparse.ArgumentParser(description="绘制Ocean-large 128核两种误差模式下的RD和total codec time")
    parser.add_argument("--result-root",type=Path,default=project_root/"results")
    parser.add_argument("--output-dir",type=Path,default=project_root/"results/ocean_large_figures")
    arguments=parser.parse_args()
    plt.rcParams.update({"font.size":9,"axes.titlesize":9,"legend.fontsize":8,"pdf.fonttype":42,"ps.fonttype":42})
    slice_runs=load_runs(arguments.result_root,"slice")
    field_runs=load_runs(arguments.result_root,"field")
    plot_rd(slice_runs,"slice",arguments.output_dir/"ocean_large_slice_rd")
    plot_rd(field_runs,"field",arguments.output_dir/"ocean_large_field_rd")
    plot_time(slice_runs,field_runs,arguments.output_dir/"ocean_large_128_time")
    report_ratios(slice_runs,"slice")
    report_ratios(field_runs,"field")


if __name__=="__main__":
    main()
