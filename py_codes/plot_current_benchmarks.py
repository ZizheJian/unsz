import argparse,csv,json,math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


project_root=Path(__file__).resolve().parents[1]
default_configs=("katrina","les","e3sm-h4","adcirc-haiyan-1.5km","ocean","taps","tpost20")
dataset_labels={"katrina":"Katrina","les":"LES","e3sm-h4":"E3SM-h4","adcirc-haiyan-1.5km":"Haiyan","ocean":"Ocean","taps":"TAPS","tpost20":"Tpost20"}
compressor_styles={"unsz":{"label":"UNSZ","color":"#0072B2","marker":"s"},"umc":{"label":"UMC","color":"#D55E00","marker":"o"}}
unsz_time_components=(("compression_preprocessing_time","Compression preprocessing","#56B4E9"),("compression_time","Compression","#0072B2"),("decompression_preprocessing_time","Decompression preprocessing","#E69F00"),("decompression_time","Decompression","#D55E00"))
umc_time_components=(("compression_preprocessing_time","Compression preprocessing","#56B4E9"),("compression_time","Compression","#0072B2"),("decompression_preprocessing_time","Decompression preprocessing","#E69F00"),("decompression_time","Decompression","#D55E00"))
timing_summary_keys=("coordinate_transform_time","prediction_structure_time","automatic_flow_selection_time","compression_preprocessing_time","compression_time","decompression_preprocessing_time","decompression_time")


def load_rows(path):
    with path.open(newline="",encoding="utf-8") as file:
        return list(csv.DictReader(file))


def read_point_set_rel_eb(path):
    path=Path(path)
    if not path.is_absolute():
        path=project_root/path
    document=json.loads(path.read_text())
    values={float(label["eb_value"]) for label in document["labels"] if label["method"]=="predict" and label["eb_mode"]=="REL"}
    if len(values)!=1:
        raise ValueError(f"{path}必须恰好包含一个预测点集REL EB")
    return values.pop()


def task_rows(task,field):
    csv_path=project_root/task["csv_path"].format(field=field)
    rows=load_rows(csv_path)
    if task["time_step_num"]==1:
        label_flow,value_flow="not_applicable","spatial"
    elif "point_set_config_path_list" in task["unsz"]:
        label_flow,value_flow="auto","auto"
    else:
        label_flow,value_flow="not_applicable","auto"
    selected=[]
    for row in rows:
        if row["task"]!=task["name"] or row["field"]!=field:
            continue
        if row["compressor"]=="unsz" and (row["label_flow"]!=label_flow or row["value_flow"]!=value_flow):
            continue
        if row["compressor"] not in task["compressor_list"]:
            continue
        item=dict(row)
        item["rel_eb_value"]=float(row["rel_eb"]) if row["rel_eb"]!="not_applicable" else read_point_set_rel_eb(row["point_set_config"])
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
                rows=task_rows(task,field)
                input_bytes={int(row["input_bytes"]) for row in rows}
                if len(input_bytes)!=1:
                    raise ValueError(f"{task['name']}:{field}的input_bytes不一致：{sorted(input_bytes)}")
                fields.append((config_name,task,field,rows,input_bytes.pop()))
    return sorted(fields,key=lambda item:(item[4],item[2].casefold(),item[1]["name"]))


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


def plot_rd(fields,output_dir):
    figure,axes=subplot_grid(len(fields))
    for (dataset,task,field,rows,_),axis in zip(fields,axes):
        for compressor in task["compressor_list"]:
            curve=sorted((row for row in rows if row["compressor"]==compressor),key=lambda row:float(row["br"]))
            style=compressor_styles[compressor]
            axis.plot([float(row["br"]) for row in curve],[float(row["psnr"]) for row in curve],color=style["color"],marker=style["marker"],linewidth=2)
        axis.set_xscale("log")
        set_panel_caption(axis,dataset,field)
        axis.grid(True,linestyle=":",alpha=.55)
        if "umc" not in task["compressor_list"]:
            axis.text(.98,.04,"UMC unavailable",transform=axis.transAxes,ha="right",va="bottom",fontsize=9)
    figure.supxlabel("Bit rate (bit/value, log scale)")
    figure.supylabel("PSNR (dB)")
    legend_handles=[Line2D([],[],color=style["color"],marker=style["marker"],linewidth=2,label=style["label"]) for style in compressor_styles.values()]
    finish_subplots(figure,axes,len(fields),"Rate-distortion",output_dir/"all_fields_rd",legend_handles,2)


def plot_throughput(fields,output_dir):
    figure,axes=subplot_grid(len(fields))
    for (dataset,task,field,rows,_),axis in zip(fields,axes):
        for compressor in task["compressor_list"]:
            curve=sorted((row for row in rows if row["compressor"]==compressor),key=lambda row:row["rel_eb_value"])
            style=compressor_styles[compressor]
            x=[row["rel_eb_value"] for row in curve]
            axis.plot(x,[float(row["compression_MB_s"]) for row in curve],color=style["color"],marker=style["marker"],linewidth=2,linestyle="-")
            axis.plot(x,[float(row["decompression_MB_s"]) for row in curve],color=style["color"],marker=style["marker"],linewidth=2,linestyle="--")
        axis.set_xscale("log")
        axis.set_ylim(bottom=0)
        set_panel_caption(axis,dataset,field)
        axis.grid(True,which="both",linestyle=":",alpha=.55)
    figure.supxlabel("REL EB (log scale)")
    figure.supylabel("Core throughput (MB/s)")
    legend_handles=[]
    for compressor,style in compressor_styles.items():
        for operation,linestyle in (("compression","-"),("decompression","--")):
            legend_handles.append(Line2D([],[],color=style["color"],marker=style["marker"],linewidth=2,linestyle=linestyle,label=style["label"]+" "+operation))
    finish_subplots(figure,axes,len(fields),"Core compression/decompression throughput",output_dir/"all_fields_releb_throughput",legend_handles,4)


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
    fieldnames=("dataset","task","field","compressor","rel_eb","coordinate_transform_time","prediction_structure_time","automatic_flow_selection_time","compression_preprocessing_time","compression_time","decompression_preprocessing_time","decompression_time","total_time")
    with path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=fieldnames,lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key:row.get(key,"") for key in fieldnames})


def main():
    parser=argparse.ArgumentParser(description="绘制当前正式配置的RD、核心吞吐率和REL EB=1e-3耗时")
    parser.add_argument("--config",action="append",dest="configs",help="配置名，可重复；默认排除Ocean-large的全部正式配置")
    parser.add_argument("--field",action="append",dest="fields",help="只绘制指定field，可重复")
    parser.add_argument("--output-dir",type=Path,default=project_root/"results/current/figures")
    arguments=parser.parse_args()
    plt.rcParams.update({"font.size":10,"axes.labelsize":10,"axes.titlesize":11,"legend.fontsize":9,"pdf.fonttype":42,"ps.fonttype":42})
    fields=load_fields(arguments.configs or default_configs,arguments.fields)
    if not fields:
        raise RuntimeError("没有可绘制的field")
    summary_rows=[]
    plot_rd(fields,arguments.output_dir)
    plot_throughput(fields,arguments.output_dir)
    plot_time(fields,arguments.output_dir,summary_rows)
    write_summary(summary_rows,arguments.output_dir/"releb_1e-3_timing.csv")


if __name__=="__main__":
    main()
