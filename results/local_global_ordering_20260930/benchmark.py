import argparse,csv,json,re,sys,time
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
import t1_start_c_code as t1

task_spec_vec={
    "e3sm_h4_mass_dst_srf":("Mass_dst_srf",None),
    "tsunami_rd":("height",None),
    "tpost20_scalar":("scalar",None),
    "ocean_temperature":("temperature",None),
    "katrina_attr0":("attr0",0.001),
    "adcirc_haiyan_depth":("depth",None),
    "ocean_large_temperature_surface_value":("temperatureSurfaceValue",None),
    "les_velocity_x":("velocity_x",None),
    "taps_temperature":("TEMPERATURE",None)
}

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def load_task_map():
    task_map={}
    for path in (project_root/"configs").glob("*.json"):
        for task in json.load(path.open(encoding="utf-8")):
            task_map[task["name"]]=task
    return task_map

def select_point_set_config(task,target_rel_eb):
    for path in task["unsz"]["point_set_config_path_list"]:
        if target_rel_eb in t1.read_point_set_rel_eb_list(path):
            return path
    raise ValueError(f"{task['name']}没有REL={target_rel_eb:g}的点集配置")

def build_config(task,field,point_set_rel_eb,ordering_scope,ordering_candidate_num,ordering_score_ref_num):
    config=t1.expand_unsz_flows(task|task["unsz"]|{"field":field})[0]|{"ordering_scope":ordering_scope}
    if ordering_candidate_num is not None:
        config["ordering_candidate_num"]=ordering_candidate_num
    if ordering_score_ref_num is not None:
        config["ordering_score_max_ref_num"]=ordering_score_ref_num
        config["ordering_score_min_ref_num"]=ordering_score_ref_num
    if point_set_rel_eb is None:
        config["rel_eb"]=0.001
    else:
        config["point_set_config_path"]=select_point_set_config(task,point_set_rel_eb)
    return config

def run_compression(config):
    data_dir=Path(config["data_dir"])
    compression_command,_=t1.build_unsz_commands(config,data_dir,None)
    output=t1.run_command(compression_command,silent=True)
    result={
        "br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",output,"BR"),
        "psnr":read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",output,"PSNR"),
        "preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",output,"预处理时间"),
        "sorting_time":read_float(r"^Target sorting time: ([^ ]+) s$",output,"排序时间"),
        "reference_generation_time":read_float(r"^Prediction reference generation time: ([^ ]+) s$",output,"参考生成时间"),
        "compression_time":read_float(r"^total compression time= ([^ ]+) s$",output,"压缩时间")
    }
    t1.cleanup_test_outputs(config,"unsz")
    return result

def append_row(path,row):
    fieldnames=("task","field","point_num","trial","ordering_scope","br","psnr","preprocessing_time","sorting_time","reference_generation_time","compression_time")
    new_file=not path.exists()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("a",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("task",nargs="+",choices=tuple(task_spec_vec))
    parser.add_argument("--trials",type=int,default=1)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--ordering-candidate-num",type=int)
    parser.add_argument("--ordering-score-ref-num",type=int)
    arguments=parser.parse_args()
    task_map=load_task_map()
    for task_name in arguments.task:
        task=task_map[task_name]
        field,point_set_rel_eb=task_spec_vec[task_name]
        point_num=(Path(task["data_dir"])/f"{field}.dat.0").stat().st_size//4
        for trial in range(arguments.trials):
            scope_vec=("local","global") if trial%2==0 else ("global","local")
            for ordering_scope in scope_vec:
                config=build_config(task,field,point_set_rel_eb,ordering_scope,arguments.ordering_candidate_num,arguments.ordering_score_ref_num)
                print(f"开始 task={task_name},scope={ordering_scope},trial={trial}",flush=True)
                start=time.monotonic()
                result=run_compression(config)
                row={"task":task_name,"field":field,"point_num":point_num,"trial":trial,"ordering_scope":ordering_scope,**result}
                append_row(arguments.output,row)
                print(f"完成 task={task_name},scope={ordering_scope},trial={trial},wall={time.monotonic()-start:.3f}s,br={result['br']},preprocessing={result['preprocessing_time']}",flush=True)

if __name__=="__main__":
    main()
