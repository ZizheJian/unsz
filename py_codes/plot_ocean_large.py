import argparse,csv,json,math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator,LogLocator,MaxNLocator,NullLocator
from paperfig import PANEL_GAP_WITH_YLABEL_PT,PaperFigureSpec,add_panel_captions,align_xlabels,apply_paper_style,layout_panel_row,legend_spacing_kwargs,panel_aspect_ratios,save_paper_png_vertical_crop,style_plot_axis

from plot_current_benchmarks import add_top_legend
from summarize_current_results import calculate_bd_rate


project_root=Path(__file__).resolve().parents[1]
# 每种误差模式下参与比较的压缩器及其128核结果目录
mode_codecs={"slice":(("UNSZ","unsz/16_fields_8_partitions"),("UMC","umc/128_field_workers"),("SZ3-1D","sz3_1d/32_slice_workers_4_threads_per_slice"),("PFPL","pfpl/64_slice_workers_2_threads_per_slice")),"field":(("UNSZ","unsz/16_fields_8_partitions"),("SZ3-ND","sz3/4_fields_32_partitions"),("QoZ-ND","qoz/4_fields_32_partitions"))}
# REL 1e-3 scaling所用的配置：核数减半时只减半worker数或每个worker的并行度之一；128核以下的点都申请整节点
scaling_configs={"slice":{"UNSZ":{128:"rd_128/unsz/16_fields_8_partitions",64:"rd_64_whole_node/unsz/8_fields_8_partitions",32:"rd_32_whole_node/unsz/4_fields_8_partitions",16:"rd_16_whole_node/unsz/2_fields_8_partitions"},"UMC":{128:"rd_128/umc/128_field_workers",64:"rd_64/umc/64_field_workers",32:"rd_32/umc/32_field_workers",16:"rd_16/umc/16_field_workers"},"SZ3-1D":{128:"rd_128/sz3_1d/32_slice_workers_4_threads_per_slice",64:"rd_64_whole_node/sz3_1d/16_slice_workers_4_threads_per_slice",32:"rd_32_whole_node/sz3_1d/16_slice_workers_2_threads_per_slice",16:"rd_16_whole_node/sz3_1d/16_slice_workers_1_threads_per_slice"},"PFPL":{128:"rd_128/pfpl/64_slice_workers_2_threads_per_slice",64:"rd_64_whole_node/pfpl/64_slice_workers_1_threads_per_slice",32:"rd_32_whole_node/pfpl/32_slice_workers_1_threads_per_slice",16:"rd_16_whole_node/pfpl/16_slice_workers_1_threads_per_slice"}},"field":{"UNSZ":{128:"rd_128/unsz/16_fields_8_partitions",64:"rd_64_whole_node/unsz/8_fields_8_partitions",32:"rd_32_whole_node/unsz/4_fields_8_partitions",16:"rd_16_whole_node/unsz/2_fields_8_partitions"},"SZ3-ND":{128:"rd_128/sz3/4_fields_32_partitions",64:"rd_64/sz3/2_fields_32_partitions",32:"rd_32/sz3/1_fields_32_partitions",16:"rd_16/sz3/1_fields_16_partitions"},"QoZ-ND":{128:"rd_128/qoz/4_fields_32_partitions",64:"rd_64/qoz/4_fields_16_partitions",32:"rd_32/qoz/4_fields_8_partitions",16:"rd_16/qoz/2_fields_8_partitions"}}}
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


def load_scaling(result_root,mode):
    # 每个压缩器在各核数下REL 1e-3的total codec time；尚未完成的点跳过
    scaling={}
    for codec,configs in scaling_configs[mode].items():
        points=[]
        for cpu_num,directory in sorted(configs.items()):
            for summary_path in (result_root/f"ocean_large_per_{mode}_{directory}").glob("rel_*/summary.json"):
                summary=json.loads(summary_path.read_text())
                if summary["rel_eb"]==1e-3 and summary.get("field_num")==95 and summary.get("valid"):points.append((cpu_num,summary["total_codec_time"]))
        scaling[codec]=points
    return scaling


def plot_time(runs,scaling,mode,output_path):
    # 左：128核下五个EB的total codec time；右：REL 1e-3下total codec time随核数的变化
    figure,(bar_axis,scaling_axis)=plt.subplots(1,2,figsize=(9,2.6),layout="constrained")
    rel_ebs=sorted({point["rel_eb"] for points in runs.values() for point in points},reverse=True)
    width=.8/len(runs)
    for codec_index,(codec,points) in enumerate(runs.items()):
        times={point["rel_eb"]:point["total_codec_time"] for point in points}
        x=[eb_index-.4+width*(codec_index+.5) for eb_index,rel_eb in enumerate(rel_ebs) if rel_eb in times]
        bar_axis.bar(x,[times[rel_eb] for rel_eb in rel_ebs if rel_eb in times],width=width*.9,color=styles[codec][0])
    bar_axis.set_xticks(range(len(rel_ebs)),[f"$10^{{{round(math.log10(rel_eb))}}}$" for rel_eb in rel_ebs])
    bar_axis.set_xlabel("Relative error bound")
    bar_axis.set_title("(a) 128 cores")
    for codec,points in scaling.items():
        color,marker=styles[codec]
        scaling_axis.plot([cpu_num for cpu_num,_ in points],[time for _,time in points],color=color,marker=marker,linewidth=1.5,markersize=5)
    cpu_nums=sorted({cpu_num for configs in scaling_configs[mode].values() for cpu_num in configs})
    scaling_axis.set_xscale("log",base=2)
    scaling_axis.set_xticks(cpu_nums,[str(cpu_num) for cpu_num in cpu_nums])
    scaling_axis.minorticks_off()
    scaling_axis.set_xlabel("Cores")
    scaling_axis.set_title(r"(b) $\epsilon_{\mathrm{rel}}=10^{-3}$")
    # 耗时轴一律用从0开始的线性坐标
    for axis in(bar_axis,scaling_axis):
        axis.set_ylim(bottom=0)
        axis.grid(True,axis="y",which="major",linestyle=":",alpha=.55)
    bar_axis.set_ylabel("Total codec time (s)")
    figure.legend(handles=legend_handles(runs),loc="lower center",bbox_to_anchor=(.5,1.0),ncol=len(runs),frameon=False)
    save(figure,output_path)


def plot_summary(runs,mode,output_path):
    # 2/3栏宽一行两个子图：全部95个field的RD、128核下各EB的total codec time
    with plt.rc_context():
        apply_paper_style()
        spec=PaperFigureSpec.column(width_fraction=2/3)
        figure,axes=plt.subplots(1,2,figsize=spec.size_in,dpi=spec.dpi)
        rd_axis,time_axis=axes
        rel_ebs=sorted({point["rel_eb"] for points in runs.values() for point in points},reverse=True)
        for codec,points in runs.items():
            color=styles[codec][0]
            rd_axis.plot([point["groups"][None]["br"] for point in points],[point["groups"][None]["psnr"] for point in points],color=color,linewidth=.9)
            times={point["rel_eb"]:point["total_codec_time"] for point in points}
            time_axis.plot([eb_index for eb_index,rel_eb in enumerate(rel_ebs) if rel_eb in times],[times[rel_eb] for rel_eb in rel_ebs if rel_eb in times],color=color,linewidth=.9)
        rd_axis.set_xscale("log")
        x_left,x_right=rd_axis.get_xlim()
        rd_axis.xaxis.set_major_locator(FixedLocator([tick for tick in LogLocator(base=10,numticks=4).tick_values(x_left,x_right) if x_left<=tick<=x_right]))
        rd_axis.set_xlabel("Bit rate (bits/value)")
        rd_axis.set_ylabel("PSNR (dB)")
        # 2/3栏宽下五个刻度标签放不下，隔一个标一个
        time_axis.set_xticks(range(len(rel_ebs)),[f"$10^{{{round(math.log10(rel_eb))}}}$" if eb_index%2==0 else "" for eb_index,rel_eb in enumerate(rel_ebs)])
        time_axis.set_xlim(-.35,len(rel_ebs)-.65)
        time_axis.set_xlabel("Relative error bound")
        # 耗时轴一律用从0开始的线性坐标
        time_axis.set_ylim(bottom=0)
        time_axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
        time_axis.set_ylabel("Time (s)")
        for axis in axes:
            axis.xaxis.set_minor_locator(NullLocator())
            axis.yaxis.set_minor_locator(NullLocator())
            axis.set_axisbelow(True)
            axis.grid(True,which="major",linestyle="--",linewidth=.35,alpha=.35)
            style_plot_axis(axis)
        layout_panel_row(figure,axes,panel_aspect_ratios(1,1),bottom_pt=40,panel_gap_pt=PANEL_GAP_WITH_YLABEL_PT,left_edge_gap_pt=3,right_edge_gap_pt=3)
        add_top_legend(figure,[axes],[Line2D([],[],color=styles[codec][0],linewidth=.9,label=codec) for codec in runs],ncol=2 if len(runs)>3 else len(runs),**legend_spacing_kwargs(12))
        align_xlabels(figure,axes)
        add_panel_captions(figure,axes,["(a) Rate\u2013distortion","(b) Time on 128 cores"])
        output_path.parent.mkdir(parents=True,exist_ok=True)
        save_paper_png_vertical_crop(figure,output_path.with_suffix(".png"),spec,edge_gap_pt=3)
        plt.close(figure)


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


def report_scaling(scaling,mode):
    for codec,points in scaling.items():
        times=dict(points)
        speedup=f", {min(times)} to 128 cores: {times[min(times)]/times[128]:.2f}x" if 128 in times else ""
        print(f"{mode_labels[mode]} {codec} REL 1e-3 total codec time: "+", ".join(f"{cpu_num} cores {time:.1f} s" for cpu_num,time in points)+speedup)


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
    for mode,runs in(("slice",slice_runs),("field",field_runs)):
        scaling=load_scaling(arguments.result_root,mode)
        plot_time(runs,scaling,mode,arguments.output_dir/f"ocean_large_{mode}_time")
        plot_summary(runs,mode,arguments.output_dir/f"ocean_large_{mode}_summary")
        report_ratios(runs,mode)
        report_scaling(scaling,mode)


if __name__=="__main__":
    main()
