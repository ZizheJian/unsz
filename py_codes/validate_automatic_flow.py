import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


project_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project_root))
import t1_start_c_code as runner


candidate_pattern=re.compile(r"automatic flow candidate: label=([^,]+),value=([^,]+),estimated_label_bytes=([0-9.eE+-]+),estimated_value_bytes=([0-9.eE+-]+),estimated_total_bytes=([0-9.eE+-]+),selection_score=([0-9.eE+-]+),estimated_label_br=([0-9.eE+-]+),estimated_value_br=([0-9.eE+-]+),evaluation_time=([0-9.eE+-]+) s")
selected_pattern=re.compile(r"automatic flow selected: label=([^,]+),value=([^,]+),selection_score=([0-9.eE+-]+)")
decision_eb_pattern=re.compile(r"automatic flow decision EB: configured_mode=([^,]+),configured_eb=([0-9.eE+-]+),decision_eb=([0-9.eE+-]+)")
selection_time_pattern=re.compile(r"automatic flow selection time excluding data input= ([0-9.eE+-]+) s")
flow_names=("none","constant_extrapolation","constant_extrapolation_spatial_residual","anchor_interpolation_stride_8","anchor_interpolation_stride_8_spatial_residual")


def normalize_flow(flow):
    return "none" if flow in (None,"","spatial","not_applicable") else flow


def point_set_config_rel_eb(path):
    values=runner.read_point_set_rel_eb_list(path)
    if len(values)!=1:
        raise ValueError(f"{path}必须只定义一个REL EB")
    return values[0]


def configurations(config_path):
    for task in json.loads(config_path.read_text()):
        base=task|task.get("unsz",{})
        point_set_paths=base.get("point_set_config_path_list")
        if point_set_paths:
            error_configs=[(point_set_config_rel_eb(path),{"point_set_config_path":path}) for path in point_set_paths]
            label_flow="auto"
        else:
            error_configs=[(float(rel_eb),{"rel_eb":float(rel_eb)}) for rel_eb in task["rel_eb_list"]]
            label_flow=None
        for field in task["field_list"]:
            for rel_eb,error_config in error_configs:
                yield base|error_config|{"field":field,"label_flow":label_flow,"value_flow":"auto","force_flow":False},rel_eb


def prepare_data_view(config,output_dir):
    source_dir=Path(config["data_dir"])
    view_dir=Path(tempfile.gettempdir())/"unsz-automatic-flow-validation"/(config["name"]+"_"+config["field"])
    view_dir.mkdir(parents=True,exist_ok=True)
    source_paths=[source_dir/"coords.dat",*runner.generate_input_file_paths(config)]
    if "time_step_coordinate_file" in config:
        source_paths.append(source_dir/config["time_step_coordinate_file"])
    for source_path in source_paths:
        destination=view_dir/source_path.name
        if not destination.exists():
            destination.symlink_to(source_path)
    return config|{"data_dir":str(view_dir)}


def run_selector(config):
    command,_=runner.build_unsz_commands(config,Path(config["data_dir"]))
    command=["stdbuf","-oL","-eL",*(str(value) for value in command)]
    process=subprocess.Popen(command,cwd=project_root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    lines=[]
    completed_selection=False
    try:
        for line in process.stdout:
            lines.append(line)
            if selection_time_pattern.search(line):
                completed_selection=True
                break
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
        try:
            remainder,_=process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            remainder,_=process.communicate()
        lines.append(remainder)
    output="".join(lines)
    if not completed_selection:
        raise RuntimeError("自动flow选择未完成：\n"+output[-4000:])
    candidates=[]
    for match in candidate_pattern.finditer(output):
        candidates.append({
            "label_flow":match[1],"value_flow":match[2],"estimated_label_bytes":float(match[3]),"estimated_value_bytes":float(match[4]),
            "estimated_total_bytes":float(match[5]),"selection_score":float(match[6]),"estimated_label_br":float(match[7]),
            "estimated_value_br":float(match[8]),"evaluation_time":float(match[9])})
    selected=selected_pattern.search(output)
    decision_eb=decision_eb_pattern.search(output)
    selection_time=selection_time_pattern.search(output)
    if len(candidates)!=5 or selected is None or decision_eb is None or selection_time is None:
        raise RuntimeError("无法解析自动flow输出：\n"+output[-4000:])
    return candidates,{
        "selected_label_flow":selected[1],"selected_value_flow":selected[2],"selected_score":float(selected[3]),
        "configured_eb":float(decision_eb[2]),"decision_eb":float(decision_eb[3]),"selection_time":float(selection_time[1]),
        "command":" ".join(command[3:])}


def read_existing(path,key_columns):
    if not path.is_file():
        return set()
    with path.open(newline="",encoding="utf-8") as file:
        return {tuple(row[column] for column in key_columns) for row in csv.DictReader(file)}


def append_rows(path,fieldnames,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    exists=path.is_file()
    with path.open("a",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=fieldnames,lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def collect(config_path,output_dir):
    decision_path=output_dir/"decisions.csv"
    candidate_path=output_dir/"candidates.csv"
    completed=read_existing(decision_path,("task","field","rel_eb"))
    decision_fields=("task","field","rel_eb","configured_eb","decision_eb","selected_label_flow","selected_value_flow","selected_score","selection_time","command")
    candidate_fields=("task","field","rel_eb","label_flow","value_flow","estimated_label_bytes","estimated_value_bytes","estimated_total_bytes","selection_score","estimated_label_br","estimated_value_br","evaluation_time")
    for config,rel_eb in configurations(config_path):
        key=(config["name"],config["field"],f"{rel_eb:.12g}")
        if key in completed:
            continue
        print(f"Selecting {key[0]} {key[1]} REL={key[2]}",flush=True)
        diagnostic_config=prepare_data_view(config,output_dir)
        candidates,decision=run_selector(diagnostic_config)
        common={"task":config["name"],"field":config["field"],"rel_eb":key[2]}
        append_rows(candidate_path,candidate_fields,[common|candidate for candidate in candidates])
        append_rows(decision_path,decision_fields,[common|decision])


def actual_rel_eb(row):
    if row["rel_eb"] not in ("","not_applicable"):
        return float(row["rel_eb"])
    return point_set_config_rel_eb(row["point_set_config"])


def read_actual_results(directory):
    actual={}
    curves={}
    for path in sorted(directory.glob("*.csv")):
        with path.open(newline="",encoding="utf-8") as file:
            for row in csv.DictReader(file):
                rel_eb=actual_rel_eb(row)
                flow=(normalize_flow(row["label_flow"]),normalize_flow(row["value_flow"]))
                key=(row["field"],rel_eb,flow)
                if key in actual:
                    raise ValueError(f"重复的实际flow结果：{key}")
                actual[key]={"br":float(row["br"]),"psnr":float(row["psnr"]),"label_br":float(row["label_stream_br"]),"value_br":float(row["value_stream_br"])}
                curves.setdefault((row["field"],flow),[]).append((float(row["psnr"]),float(row["br"])))
    return actual,curves


def interpolate_log_br(points,target_psnr):
    points=sorted(points)
    if target_psnr<points[0][0] or target_psnr>points[-1][0]:
        return None
    for left,right in zip(points,points[1:]):
        if target_psnr<=right[0]:
            if right[0]==left[0]:
                return min(left[1],right[1])
            ratio=(target_psnr-left[0])/(right[0]-left[0])
            return math.exp(math.log(left[1])+ratio*(math.log(right[1])-math.log(left[1])))
    return points[-1][1]


def kendall_tau_b(first,second):
    concordant=discordant=first_ties=second_ties=0
    for left in range(len(first)):
        for right in range(left+1,len(first)):
            first_difference=first[left]-first[right]
            second_difference=second[left]-second[right]
            if first_difference==0 and second_difference==0:
                continue
            if first_difference==0:
                first_ties+=1
            elif second_difference==0:
                second_ties+=1
            elif first_difference*second_difference>0:
                concordant+=1
            else:
                discordant+=1
    denominator=math.sqrt((concordant+discordant+first_ties)*(concordant+discordant+second_ties))
    return (concordant-discordant)/denominator if denominator else 1


def analyze(output_dir,actual_directory):
    actual,curves=read_actual_results(actual_directory)
    with (output_dir/"decisions.csv").open(newline="",encoding="utf-8") as file:
        decisions=list(csv.DictReader(file))
    with (output_dir/"candidates.csv").open(newline="",encoding="utf-8") as file:
        candidates=list(csv.DictReader(file))
    calibration=[]
    ranking_tau=[]
    for decision in decisions:
        field=decision["field"]
        rel_eb=float(decision["rel_eb"])
        current_candidates=[candidate for candidate in candidates if candidate["task"]==decision["task"] and candidate["field"]==field and math.isclose(float(candidate["rel_eb"]),rel_eb,rel_tol=1e-9)]
        for candidate in current_candidates:
            flow=(candidate["label_flow"],candidate["value_flow"])
            actual_value=actual[(field,rel_eb,flow)]
            calibration.append({
                "task":decision["task"],"field":field,"rel_eb":decision["rel_eb"],"label_flow":flow[0],"value_flow":flow[1],
                "estimated_label_br":candidate["estimated_label_br"],"actual_label_br":actual_value["label_br"],
                "estimated_value_br":candidate["estimated_value_br"],"actual_value_br":actual_value["value_br"],
                "estimated_selection_score":candidate["selection_score"],"actual_total_br":actual_value["br"]})
        ranking_tau.append(kendall_tau_b([float(candidate["selection_score"]) for candidate in current_candidates],[actual[(field,rel_eb,(candidate["label_flow"],candidate["value_flow"]))]["br"] for candidate in current_candidates]))
    calibration_path=output_dir/"calibration.csv"
    with calibration_path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=calibration[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(calibration)
    rows=[]
    for decision in decisions:
        field=decision["field"]
        rel_eb=float(decision["rel_eb"])
        selected_flow=(decision["selected_label_flow"],decision["selected_value_flow"])
        available={flow:value for (actual_field,actual_eb,flow),value in actual.items() if actual_field==field and math.isclose(actual_eb,rel_eb,rel_tol=1e-9)}
        if selected_flow not in available:
            raise ValueError(f"缺少选择flow的实际结果：{field},REL={rel_eb:g},{selected_flow}")
        best_flow,best_value=min(available.items(),key=lambda item:item[1]["br"])
        selected_value=available[selected_flow]
        equal_psnr_values={flow:interpolate_log_br(curves[(field,flow)],selected_value["psnr"]) for flow in available}
        equal_psnr_values={flow:br for flow,br in equal_psnr_values.items() if br is not None}
        equal_psnr_best_flow,equal_psnr_best_br=min(equal_psnr_values.items(),key=lambda item:item[1])
        selected_equal_psnr_br=equal_psnr_values[selected_flow]
        rows.append({
            "task":decision["task"],"field":field,"rel_eb":f"{rel_eb:.12g}","decision_eb":decision["decision_eb"],
            "selected_label_flow":selected_flow[0],"selected_value_flow":selected_flow[1],
            "same_eb_best_label_flow":best_flow[0],"same_eb_best_value_flow":best_flow[1],"same_eb_hit":int(selected_flow==best_flow),
            "selected_br":selected_value["br"],"same_eb_best_br":best_value["br"],"same_eb_regret":selected_value["br"]/best_value["br"]-1,
            "selected_psnr":selected_value["psnr"],"equal_psnr_best_label_flow":equal_psnr_best_flow[0],"equal_psnr_best_value_flow":equal_psnr_best_flow[1],
            "equal_psnr_hit":int(selected_flow==equal_psnr_best_flow),"selected_equal_psnr_br":selected_equal_psnr_br,
            "equal_psnr_best_br":equal_psnr_best_br,"equal_psnr_regret":selected_equal_psnr_br/equal_psnr_best_br-1,
            "selection_time":decision["selection_time"]})
    report_path=output_dir/"report.csv"
    with report_path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    manual_adjustments=[row for row in rows if not row["same_eb_hit"]]
    manual_path=output_dir/"manual_adjustments.csv"
    with manual_path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(manual_adjustments)
    field_summary=[]
    for field in sorted({row["field"] for row in rows}):
        field_rows=[row for row in rows if row["field"]==field]
        field_summary.append({
            "field":field,"sample_num":len(field_rows),"same_eb_hits":sum(row["same_eb_hit"] for row in field_rows),
            "equal_psnr_hits":sum(row["equal_psnr_hit"] for row in field_rows),
            "same_eb_mean_regret":sum(row["same_eb_regret"] for row in field_rows)/len(field_rows),"same_eb_max_regret":max(row["same_eb_regret"] for row in field_rows),
            "equal_psnr_mean_regret":sum(row["equal_psnr_regret"] for row in field_rows)/len(field_rows),"equal_psnr_max_regret":max(row["equal_psnr_regret"] for row in field_rows),
            "mean_selection_time":sum(float(row["selection_time"]) for row in field_rows)/len(field_rows)})
    field_summary_path=output_dir/"field_summary.csv"
    with field_summary_path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=field_summary[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(field_summary)
    summary={
        "sample_num":len(rows),"same_eb_hits":sum(row["same_eb_hit"] for row in rows),"equal_psnr_hits":sum(row["equal_psnr_hit"] for row in rows),
        "same_eb_mean_regret":sum(row["same_eb_regret"] for row in rows)/len(rows),"same_eb_max_regret":max(row["same_eb_regret"] for row in rows),
        "equal_psnr_mean_regret":sum(row["equal_psnr_regret"] for row in rows)/len(rows),"equal_psnr_max_regret":max(row["equal_psnr_regret"] for row in rows),
        "mean_candidate_ranking_tau_b":sum(ranking_tau)/len(ranking_tau),
        "mean_selection_time":sum(float(row["selection_time"]) for row in rows)/len(rows),"max_selection_time":max(float(row["selection_time"]) for row in rows)}
    (output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


parser=argparse.ArgumentParser()
parser.add_argument("--config",type=Path,default=project_root/"configs/flow-speed-screen.json")
parser.add_argument("--output-dir",type=Path,default=project_root/"results/automatic_flow_validation")
parser.add_argument("--actual-directory",type=Path,default=project_root/"results/flow_speed_screen")
parser.add_argument("--analyze-only",action="store_true")
arguments=parser.parse_args()
if not arguments.analyze_only:
    collect(arguments.config,arguments.output_dir)
analyze(arguments.output_dir,arguments.actual_directory)
