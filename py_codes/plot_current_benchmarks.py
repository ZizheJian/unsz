import argparse,csv,json,math
from pathlib import Path

import matplotlib.pyplot as plt


project_root=Path(__file__).resolve().parents[1]
default_configs=("katrina","les","e3sm-h4","adcirc-haiyan-1.5km","ocean","taps","tpost20")
compressor_styles={"unsz":{"label":"UNSZ","color":"#0072B2","marker":"s"},"umc":{"label":"UMC","color":"#D55E00","marker":"o"}}
unsz_time_components=(("coordinate_transform_time","Adjust coordinates","#56B4E9"),("prediction_structure_time","DAG and weights","#009E73"),("automatic_flow_selection_time","Automatic flow","#F0E442"),("compression_time","Compression","#0072B2"),("decompression_preprocessing_time","Decompression preprocessing","#E69F00"),("decompression_time","Decompression","#D55E00"))
umc_time_components=(("compression_preprocessing_time","Compression preprocessing","#56B4E9"),("compression_time","Compression","#0072B2"),("decompression_preprocessing_time","Decompression preprocessing","#E69F00"),("decompression_time","Decompression","#D55E00"))


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


def load_datasets(config_names,selected_fields):
    datasets=[]
    for config_name in config_names:
        tasks=json.loads((project_root/"configs"/(config_name+".json")).read_text())
        fields=[]
        for task in tasks:
            for field in task["field_list"]:
                if selected_fields and field not in selected_fields:
                    continue
                fields.append((task,field,task_rows(task,field)))
        if fields:
            datasets.append((config_name,fields))
    return datasets


def subplot_grid(item_num,column_num=3):
    columns=min(column_num,item_num)
    rows=math.ceil(item_num/columns)
    figure,axes=plt.subplots(rows,columns,figsize=(4.5*columns,3.5*rows),squeeze=False,layout="constrained")
    return figure,axes.ravel()


def finish_subplots(figure,axes,item_num,title,output_path):
    for axis in axes[item_num:]:
        axis.remove()
    figure.suptitle(title,fontsize=14)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(Path(str(output_path)+".png"),dpi=220,bbox_inches="tight",facecolor="white")
    figure.savefig(Path(str(output_path)+".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(figure)


def plot_rd(dataset,fields,output_dir):
    figure,axes=subplot_grid(len(fields))
    for (task,field,rows),axis in zip(fields,axes):
        for compressor in task["compressor_list"]:
            curve=sorted((row for row in rows if row["compressor"]==compressor),key=lambda row:float(row["br"]))
            style=compressor_styles[compressor]
            axis.plot([float(row["br"]) for row in curve],[float(row["psnr"]) for row in curve],color=style["color"],marker=style["marker"],linewidth=2,label=style["label"])
        axis.set_xlabel("Bit rate (bit/value)")
        axis.set_ylabel("PSNR (dB)")
        axis.set_xscale("log")
        axis.set_title(field)
        axis.grid(True,linestyle=":",alpha=.55)
        axis.legend(frameon=False)
        if "umc" not in task["compressor_list"]:
            axis.text(.98,.04,"UMC unavailable",transform=axis.transAxes,ha="right",va="bottom",fontsize=9)
    finish_subplots(figure,axes,len(fields),f"{dataset}: rate-distortion",output_dir/(dataset+"_rd"))


def plot_throughput(dataset,fields,output_dir):
    figure,axes=subplot_grid(len(fields))
    for (task,field,rows),axis in zip(fields,axes):
        for compressor in task["compressor_list"]:
            curve=sorted((row for row in rows if row["compressor"]==compressor),key=lambda row:row["rel_eb_value"])
            style=compressor_styles[compressor]
            x=[row["rel_eb_value"] for row in curve]
            axis.plot(x,[float(row["compression_MB_s"]) for row in curve],color=style["color"],marker=style["marker"],linewidth=2,linestyle="-",label=style["label"]+" compression")
            axis.plot(x,[float(row["decompression_MB_s"]) for row in curve],color=style["color"],marker=style["marker"],linewidth=2,linestyle="--",label=style["label"]+" decompression")
        axis.set_xscale("log")
        axis.set_xlabel("REL EB")
        axis.set_ylabel("Core throughput (MB/s)")
        axis.set_title(field)
        axis.grid(True,which="both",linestyle=":",alpha=.55)
        axis.legend(frameon=False,fontsize=8)
    finish_subplots(figure,axes,len(fields),f"{dataset}: core compression/decompression throughput",output_dir/(dataset+"_releb_throughput"))


def plot_time(dataset,fields,output_dir,summary_rows):
    figure,axes=subplot_grid(len(fields))
    for (task,field,rows),axis in zip(fields,axes):
        selected=[row for row in rows if math.isclose(row["rel_eb_value"],1e-3,rel_tol=0,abs_tol=1e-12)]
        compressor_names=[compressor for compressor in ("unsz","umc") if compressor in task["compressor_list"]]
        compressor_labels=[compressor_styles[compressor]["label"] for compressor in compressor_names]
        bottoms=[]
        summary_by_compressor={}
        used_labels=set()
        for compressor,label in zip(compressor_names,compressor_labels):
            row=next(row for row in selected if row["compressor"]==compressor)
            bottom=0.0
            components=unsz_time_components if compressor=="unsz" else umc_time_components
            summary_by_compressor[compressor]={key:float(row[key]) for key,_,_ in components}
            for key,component_label,color in components:
                value=float(row[key])
                axis.bar(label,value,bottom=bottom,label=component_label if component_label not in used_labels else None,color=color)
                used_labels.add(component_label)
                bottom+=value
            bottoms.append(bottom)
        axis.set_ylabel("Time (s)")
        axis.set_title(field)
        axis.grid(True,axis="y",linestyle=":",alpha=.55)
        for compressor,total in zip(compressor_names,bottoms):
            summary_rows.append({"dataset":dataset,"task":task["name"],"field":field,"compressor":compressor,"rel_eb":1e-3,**summary_by_compressor[compressor],"total_time":total})
    axes[0].legend(frameon=False,fontsize=8)
    finish_subplots(figure,axes,len(fields),f"{dataset}: REL EB=1e-3 timing",output_dir/(dataset+"_releb_1e-3_time"))


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
    datasets=load_datasets(arguments.configs or default_configs,arguments.fields)
    summary_rows=[]
    for dataset,fields in datasets:
        plot_rd(dataset,fields,arguments.output_dir)
        plot_throughput(dataset,fields,arguments.output_dir)
        plot_time(dataset,fields,arguments.output_dir,summary_rows)
    write_summary(summary_rows,arguments.output_dir/"releb_1e-3_timing.csv")


if __name__=="__main__":
    main()
