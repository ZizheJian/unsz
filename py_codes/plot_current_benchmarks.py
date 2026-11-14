import argparse,csv,json,math,string
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import FixedLocator,LogLocator,MaxNLocator,NullLocator
from paperfig import DEFAULT_LEGEND_ROW_GAP_PT,PaperFigureSpec,add_shared_xlabel,align_xlabels,apply_paper_style,layout_panel_grid,legend_above,legend_spacing_kwargs,panel_aspect_ratios,save_paper_png,style_plot_axis
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


project_root=Path(__file__).resolve().parents[1]
# 与论文表I的数据集顺序一致，field按各配置文件中的顺序排列
# 按数据集的数值总大小从小到大排列，与论文中的数据集表格一致
default_configs=("tpost20","e3sm-h4","les","adcirc-haiyan-1.5km","ocean","katrina","taps")
# 论文中不展示的field，使25个field变为4x6的网格
excluded_fields={("taps","RMS_T")}
dataset_labels={"katrina":"Katrina","les":"LES","e3sm-h4":"E3SM-h4","adcirc-haiyan-1.5km":"Haiyan","ocean":"Ocean","taps":"TAPS","tpost20":"Tpost20"}
compressor_styles={"unsz":{"label":"UNSZ","color":"#0072B2","marker":"s"},"umc":{"label":"UMC","color":"#D55E00","marker":"o"},"sz3_1d":{"label":"SZ3-1D","color":"#009E73","marker":"^"},"pfpl":{"label":"PFPL","color":"#CC79A7","marker":"D"}}
# 一维基线由benchmarks/tensor_1d/run_moderate.py单独运行，没有预处理
tensor_1d_codecs=("sz3_1d","pfpl")
unsz_time_components=(("compression_preprocessing_time","Compression preprocessing","#56B4E9"),("compression_time","Compression","#0072B2"),("decompression_preprocessing_time","Decompression preprocessing","#E69F00"),("decompression_time","Decompression","#D55E00"))
umc_time_components=(("compression_preprocessing_time","Compression preprocessing","#56B4E9"),("compression_time","Compression","#0072B2"),("decompression_preprocessing_time","Decompression preprocessing","#E69F00"),("decompression_time","Decompression","#D55E00"))
# 4x6网格中每个子图绘图区的宽高比；比默认的sqrt(2)更扁，以节省论文篇幅
grid_panel_aspect_ratio=2.0
# 耗时堆叠柱：颜色区分压缩器，浅色为预处理、深色为压缩或解压，斜线为解压一侧
time_stage_labels=(("compression_preprocessing_time","Compression preprocessing"),("compression_time","Compression"),("decompression_preprocessing_time","Decompression preprocessing"),("decompression_time","Decompression"))
unavailable_bar_style={"color":".85","hatch":"xxxxxx"}
timing_summary_keys=("coordinate_rescaling_time","prediction_structure_time","automatic_pipeline_selection_time","compression_preprocessing_time","compression_time","decompression_preprocessing_time","decompression_time")


def load_rows(path):
    with path.open(newline="",encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_label_rel_eb(path):
    path=Path(path)
    if not path.is_absolute():
        path=project_root/path
    document=json.loads(path.read_text())
    values={float(label["eb_value"]) for label in document["labels"] if label["method"]=="predict" and label["eb_mode"]=="REL"}
    if len(values)!=1:
        raise ValueError(f"{path}必须恰好包含一个预测标签REL EB")
    return values.pop()


def task_rows(task,field):
    csv_path=project_root/task["csv_path"].format(field=field)
    rows=load_rows(csv_path)
    if task["slice_num"]==1:
        label_pipeline,value_pipeline="not_applicable","independent"
    elif "label_config_path_list" in task["unsz"]:
        label_pipeline,value_pipeline="auto","auto"
    else:
        label_pipeline,value_pipeline="not_applicable","auto"
    selected=[]
    for row in rows:
        if row["task"]!=task["name"] or row["field"]!=field:
            continue
        if row["compressor"]=="unsz" and (row["label_pipeline"]!=label_pipeline or row["value_pipeline"]!=value_pipeline):
            continue
        if row["compressor"] not in task["compressor_list"]:
            continue
        item=dict(row)
        item["rel_eb_value"]=float(row["rel_eb"]) if row["rel_eb"]!="not_applicable" else read_label_rel_eb(row["label_config"])
        selected.append(item)
    for compressor in task["compressor_list"]:
        compressor_rows=[row for row in selected if row["compressor"]==compressor]
        rel_values={row["rel_eb_value"] for row in compressor_rows}
        if len(compressor_rows)!=5 or rel_values!={1e-1,1e-2,1e-3,1e-4,1e-5}:
            raise ValueError(f"{csv_path}中的{compressor}正式曲线不完整：rows={len(compressor_rows)},REL={sorted(rel_values)}")
    return selected


def load_fields(config_names,selected_fields):
    fields=[]
    for config_name in config_names:
        tasks=json.loads((project_root/"configs"/(config_name+".json")).read_text())
        for task in tasks:
            for field in task["field_list"]:
                if selected_fields and field not in selected_fields:
                    continue
                if (config_name,field) in excluded_fields:
                    continue
                rows=task_rows(task,field)
                input_bytes={int(row["input_bytes"]) for row in rows}
                if len(input_bytes)!=1:
                    raise ValueError(f"{task['name']}:{field}的input_bytes不一致：{sorted(input_bytes)}")
                fields.append((config_name,task,field,rows,input_bytes.pop()))
    return fields


def load_tensor_1d_rows(fields):
    curves={}
    for codec in tensor_1d_codecs:
        for path in sorted((project_root/"results/moderate_tensor_1d"/codec).glob("rel_*/fields.csv")):
            for row in load_rows(path):
                curves.setdefault((row["task"],row["field"]),{}).setdefault(codec,[]).append(row)
    for _,task,field,_,_ in fields:
        for codec in tensor_1d_codecs:
            rows=curves.get((task["name"],field),{}).get(codec,[])
            if len(rows)!=5 or not all(row["valid"]=="True" for row in rows):
                raise ValueError(f"{task['name']}:{field}的{codec}曲线不完整或未通过误差检查")
    return curves


def time_segment_style(color,key):
    light=tuple(1-(1-channel)*.45 for channel in to_rgb(color))
    return (light if "preprocessing" in key else color),("//////" if key.startswith("decompression") else "")


def time_stage_handles():
    # 图例用灰色示意四个阶段，与压缩器的颜色无关
    return [Patch(facecolor=time_segment_style("#808080",key)[0],edgecolor="black",linewidth=.3,hatch=time_segment_style("#808080",key)[1],label=label) for key,label in time_stage_labels]


def subplot_grid(item_num,column_num=6):
    columns=min(column_num,item_num)
    rows=math.ceil(item_num/columns)
    figure,axes=plt.subplots(rows,columns,figsize=(4.5*columns,3.5*rows),squeeze=False,layout="constrained")
    return figure,axes.ravel()


def field_label(dataset,field,separator=" - "):
    return dataset_labels.get(dataset,dataset)+separator+field


def set_panel_caption(axis,dataset,field):
    axis.text(.5,-.18,field_label(dataset,field),transform=axis.transAxes,ha="center",va="top",fontsize=11)


def finish_subplots(figure,axes,item_num,title,output_path,legend_handles=(),legend_columns=None):
    for axis in axes[item_num:]:
        axis.remove()
    caption_y=-.015
    if legend_handles:
        columns=legend_columns or len(legend_handles)
        figure.legend(handles=legend_handles,loc="lower center",bbox_to_anchor=(.5,1.005),ncol=columns,frameon=False)
    figure.text(.5,caption_y,title,ha="center",va="top",fontsize=14)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(Path(str(output_path)+".png"),dpi=220,bbox_inches="tight",facecolor="white")
    figure.savefig(Path(str(output_path)+".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(figure)


def plot_rd(fields,tensor_1d_rows,output_dir,column_num=6):
    with plt.rc_context():
        apply_paper_style()
        spec=PaperFigureSpec.text()
        row_num=math.ceil(len(fields)/column_num)
        if row_num*column_num!=len(fields):
            raise ValueError(f"RD图需要填满{row_num}x{column_num}网格，当前有{len(fields)}个field")
        figure,axes=plt.subplots(row_num,column_num,figsize=spec.size_in,dpi=spec.dpi,squeeze=False)
        ylabel="PSNR (dB)"
        for index,((dataset,task,field,rows,_),axis) in enumerate(zip(fields,axes.ravel())):
            for compressor in task["compressor_list"]:
                curve=sorted((row for row in rows if row["compressor"]==compressor),key=lambda row:float(row["br"]))
                style=compressor_styles[compressor]
                axis.plot([float(row["br"]) for row in curve],[float(row["psnr"]) for row in curve],color=style["color"],linewidth=.9)
            for codec in tensor_1d_codecs:
                curve=sorted(tensor_1d_rows[(task["name"],field)][codec],key=lambda row:float(row["br"]))
                axis.plot([float(row["br"]) for row in curve],[float(row["psnr"]) for row in curve],color=compressor_styles[codec]["color"],linewidth=.9)
            axis.set_xscale("log")
            x_left,x_right=axis.get_xlim()
            axis.xaxis.set_major_locator(FixedLocator([tick for tick in LogLocator(base=10,numticks=4).tick_values(x_left,x_right) if x_left<=tick<=x_right]))
            axis.xaxis.set_minor_locator(NullLocator())
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3,integer=True))
            axis.set_xlabel(f"({string.ascii_lowercase[index]}) "+dataset_labels.get(dataset,dataset)+"\n"+field)
            if "umc" not in task["compressor_list"]:
                axis.text(.08,.94,"No UMC",transform=axis.transAxes,ha="left",va="top")
            axis.set_axisbelow(True)
            axis.grid(True,which="major",linestyle="--",linewidth=.35,alpha=.35)
            style_plot_axis(axis)
        layout_panel_grid(figure,axes,panel_aspect_ratios(*[1]*column_num,unit_aspect_ratio=grid_panel_aspect_ratio),bottom_pt=60,row_content_gap_pt=6,shared_ylabel=ylabel)
        for row in axes:
            align_xlabels(figure,row)
        add_shared_xlabel(figure,axes[-1],"Bit rate (bits/value)")
        legend_handles=[Line2D([],[],color=style["color"],linewidth=.9,label=style["label"]) for style in compressor_styles.values()]
        legend_above(axes[0],handles=legend_handles,ncol=len(legend_handles),**legend_spacing_kwargs(12))
        output_path=output_dir/"all_fields_rd.png"
        output_path.parent.mkdir(parents=True,exist_ok=True)
        save_paper_png(figure,output_path,spec)
        plt.close(figure)


def plot_time_by_eb(fields,tensor_1d_rows,output_dir,column_num=6):
    with plt.rc_context():
        apply_paper_style()
        plt.rcParams["hatch.linewidth"]=.4
        spec=PaperFigureSpec.text()
        row_num=math.ceil(len(fields)/column_num)
        if row_num*column_num!=len(fields):
            raise ValueError(f"耗时图需要填满{row_num}x{column_num}网格，当前有{len(fields)}个field")
        figure,axes=plt.subplots(row_num,column_num,figsize=spec.size_in,dpi=spec.dpi,squeeze=False)
        ylabel="Time (s)"
        rel_ebs=(1e-1,1e-2,1e-3,1e-4,1e-5)
        bar_width=.2
        compressor_offsets={"unsz":-1.5*bar_width,"umc":-.5*bar_width,"sz3_1d":.5*bar_width,"pfpl":1.5*bar_width}
        for index,((dataset,task,field,rows,_),axis) in enumerate(zip(fields,axes.ravel())):
            for eb_index,rel_eb in enumerate(rel_ebs):
                for compressor in (*task["compressor_list"],*tensor_1d_codecs):
                    if compressor in tensor_1d_codecs:
                        row=next(row for row in tensor_1d_rows[(task["name"],field)][compressor] if math.isclose(float(row["rel_eb"]),rel_eb,rel_tol=1e-9))
                    else:
                        row=next(row for row in rows if row["compressor"]==compressor and math.isclose(row["rel_eb_value"],rel_eb,rel_tol=1e-9))
                    bottom=0.0
                    for key,_ in time_stage_labels:
                        value=float(row.get(key,0))
                        color,hatch=time_segment_style(compressor_styles[compressor]["color"],key)
                        axis.bar(eb_index+compressor_offsets[compressor],value,width=bar_width,bottom=bottom,color=color,edgecolor="black",linewidth=.3,hatch=hatch)
                        bottom+=value
            axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
            axis.set_ylim(0,axis.get_ylim()[1])
            if "umc" not in task["compressor_list"]:
                top=axis.get_ylim()[1]
                for eb_index in range(len(rel_ebs)):
                    axis.bar(eb_index+compressor_offsets["umc"],top,width=bar_width,edgecolor="black",linewidth=.3,**unavailable_bar_style)
                axis.set_ylim(0,top)
            axis.set_xlim(-.6,len(rel_ebs)-.4)
            axis.set_xticks(range(len(rel_ebs)),[f"$10^{{{round(math.log10(rel_eb))}}}$" if eb_index%2==0 else "" for eb_index,rel_eb in enumerate(rel_ebs)])
            axis.set_xlabel(f"({string.ascii_lowercase[index]}) "+dataset_labels.get(dataset,dataset)+"\n"+field)
            axis.set_axisbelow(True)
            axis.grid(True,axis="y",which="major",linestyle="--",linewidth=.35,alpha=.35)
            style_plot_axis(axis)
        layout_panel_grid(figure,axes,panel_aspect_ratios(*[1]*column_num,unit_aspect_ratio=grid_panel_aspect_ratio),bottom_pt=60,row_content_gap_pt=6,shared_ylabel=ylabel)
        for row in axes:
            align_xlabels(figure,row)
        add_shared_xlabel(figure,axes[-1],"Relative error bound")
        component_handles=time_stage_handles()
        compressor_handles=[Patch(facecolor=compressor_styles[compressor]["color"],edgecolor="black",linewidth=.3,label=compressor_styles[compressor]["label"]) for compressor in compressor_offsets]
        compressor_handles.append(Patch(facecolor=unavailable_bar_style["color"],edgecolor="black",linewidth=.3,hatch=unavailable_bar_style["hatch"],label="UMC unavailable"))
        # 两行图例各自居中：下一行是压缩器，上一行是耗时的组成部分
        compressor_legend=legend_above(axes[0],handles=compressor_handles,gap_pt=3,ncol=len(compressor_handles),**legend_spacing_kwargs(8))
        legend_above(axes[0],handles=component_handles,above=compressor_legend,gap_pt=DEFAULT_LEGEND_ROW_GAP_PT,ncol=len(component_handles),**legend_spacing_kwargs(8))
        output_path=output_dir/"all_fields_releb_time.png"
        output_path.parent.mkdir(parents=True,exist_ok=True)
        save_paper_png(figure,output_path,spec)
        plt.close(figure)


def plot_time(fields,output_dir,summary_rows,subplot_num=6):
    group_num=min(subplot_num,len(fields))
    group_sizes=[len(fields)//group_num+(group_id<len(fields)%group_num) for group_id in range(group_num)]
    groups=[]
    begin=0
    for group_size in group_sizes:
        groups.append(fields[begin:begin+group_size])
        begin+=group_size
    figure,axes=subplot_grid(len(groups))
    bar_width=.36
    compressor_offsets={"unsz":-bar_width/2,"umc":bar_width/2}
    compressor_hatches={"unsz":"","umc":"//"}
    for group,axis in zip(groups,axes):
        unavailable=[]
        for field_index,(dataset,task,field,rows,_) in enumerate(group):
            selected=[row for row in rows if math.isclose(row["rel_eb_value"],1e-3,rel_tol=0,abs_tol=1e-12)]
            for compressor in ("unsz","umc"):
                x=field_index+compressor_offsets[compressor]
                if compressor not in task["compressor_list"]:
                    axis.bar(x,0,width=bar_width,color="white",edgecolor=".55",hatch="xx")
                    unavailable.append(x)
                    continue
                row=next(row for row in selected if row["compressor"]==compressor)
                bottom=0.0
                components=unsz_time_components if compressor=="unsz" else umc_time_components
                summary={key:float(row[key]) if row.get(key) else "" for key in timing_summary_keys}
                for key,_,color in components:
                    value=float(row[key])
                    axis.bar(x,value,width=bar_width,bottom=bottom,color=color,edgecolor="black",linewidth=.35,hatch=compressor_hatches[compressor])
                    bottom+=value
                summary_rows.append({"dataset":dataset,"task":task["name"],"field":field,"compressor":compressor,"rel_eb":1e-3,**summary,"total_time":bottom})
        for x in unavailable:
            axis.text(x,.02,"UMC unavailable",transform=axis.get_xaxis_transform(),ha="center",va="bottom",rotation=90,fontsize=7,color=".4")
        axis.set_xticks(range(len(group)),[field_label(dataset,field) for dataset,_,field,_,_ in group],rotation=32,ha="right",fontsize=8)
        axis.grid(True,axis="y",linestyle=":",alpha=.55)
    figure.supylabel("Time (s)")
    component_styles=[]
    for _,label,color in (*unsz_time_components,*umc_time_components):
        if (label,color) not in component_styles:
            component_styles.append((label,color))
    legend_handles=[Patch(facecolor=color,label=label) for label,color in component_styles]
    legend_handles.extend((Patch(facecolor="white",edgecolor="black",label="UNSZ"),Patch(facecolor="white",edgecolor="black",hatch="//",label="UMC")))
    finish_subplots(figure,axes,len(groups),"REL EB=1e-3 timing",output_dir/"all_fields_releb_1e-3_time",legend_handles,5)


def write_summary(rows,path):
    fieldnames=("dataset","task","field","compressor","rel_eb","coordinate_rescaling_time","prediction_structure_time","automatic_pipeline_selection_time","compression_preprocessing_time","compression_time","decompression_preprocessing_time","decompression_time","total_time")
    with path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=fieldnames,lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key:row.get(key,"") for key in fieldnames})


def main():
    parser=argparse.ArgumentParser(description="绘制当前正式配置的RD、各REL EB单核耗时和REL EB=1e-3耗时")
    parser.add_argument("--config",action="append",dest="configs",help="配置名，可重复；默认排除Ocean-large的全部正式配置")
    parser.add_argument("--field",action="append",dest="fields",help="只绘制指定field，可重复")
    parser.add_argument("--output-dir",type=Path,default=project_root/"results/current/figures")
    arguments=parser.parse_args()
    plt.rcParams.update({"font.size":10,"axes.labelsize":10,"axes.titlesize":11,"legend.fontsize":9,"pdf.fonttype":42,"ps.fonttype":42})
    fields=load_fields(arguments.configs or default_configs,arguments.fields)
    if not fields:
        raise RuntimeError("没有可绘制的field")
    summary_rows=[]
    tensor_1d_rows=load_tensor_1d_rows(fields)
    plot_rd(fields,tensor_1d_rows,arguments.output_dir)
    plot_time_by_eb(fields,tensor_1d_rows,arguments.output_dir)
    plot_time(fields,arguments.output_dir,summary_rows)
    write_summary(summary_rows,arguments.output_dir/"releb_1e-3_timing.csv")


if __name__=="__main__":
    main()
