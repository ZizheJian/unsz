import copy,csv,math,os,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from t1_start_c_code import load_json_tasks,read_point_set_rel_eb_list,resolve_json_path,run_task

result_root=Path(__file__).resolve().parent
field_sources=(
    ("katrina","attr0"),
    ("e3sm-h4","AODDUST"),
    ("e3sm-h4","Mass_dst_srf"),
    ("ocean","layerThickness"),
    ("ocean","salinity"),
    ("ocean","temperature"),
    ("ocean","velocityMeridional"),
    ("ocean","velocityX"),
    ("ocean","velocityY"),
    ("ocean","velocityZ"),
    ("ocean","velocityZonal"),
    ("ocean","zTop"),
    ("adcirc-haiyan-1.5km","zeta"),
    ("tpost20","scalar"),
)

def select_rel_1e_3(task):
    point_set_paths=task.get("unsz",{}).get("point_set_config_path_list")
    if point_set_paths is None:
        task["rel_eb_list"]=[1e-3]
        return
    selected=[path for path in point_set_paths if any(math.isclose(value,1e-3,rel_tol=0,abs_tol=1e-15) for value in read_point_set_rel_eb_list(path))]
    if len(selected)!=1:
        raise RuntimeError(f"{task['name']}没有唯一的REL 1e-3点集配置")
    task["unsz"]["point_set_config_path_list"]=selected

def make_task(config_name,field,variant):
    matching=[task for task in load_json_tasks(resolve_json_path(config_name)) if field in task["field_list"]]
    if len(matching)!=1:
        raise RuntimeError(f"{config_name}中没有唯一包含{field}的task")
    task=copy.deepcopy(matching[0])
    task["name"]=f"coordinate_transform_group_loss_{config_name}_{field}_{variant}"
    task["field_list"]=[field]
    task["compressor_list"]=["unsz"]
    task["csv_path"]=str(result_root/f"{config_name}_{field}.csv")
    task["cleanup_after_each_test"]=True
    select_rel_1e_3(task)
    if variant=="off":
        task["unsz"]["coordinate_transform"]="off"
    else:
        power=float(variant.removeprefix("power"))
        task["unsz"].update({"coordinate_transform":"on","coordinate_transform_training_normalization":"time_range","coordinate_transform_sample_file_num":3,"coordinate_transform_second_sample_num":3000,"coordinate_transform_group_loss_power":power})
    return task

def completed(task):
    path=Path(task["csv_path"])
    if not path.is_file():
        return False
    with path.open(newline="",encoding="utf-8") as file:
        return any(row["task"]==task["name"] for row in csv.DictReader(file))

def main():
    selected_field_sources=field_sources
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        selected_field_sources=(field_sources[int(os.environ["SLURM_ARRAY_TASK_ID"])],)
    for config_name,field in selected_field_sources:
        for variant in ("off","power1","power2","power3","power4"):
            task=make_task(config_name,field,variant)
            if completed(task):
                print(f"跳过已完成任务：{task['name']}",flush=True)
                continue
            run_task(task)

if __name__=="__main__":
    main()
