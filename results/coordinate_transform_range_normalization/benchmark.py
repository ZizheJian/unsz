import copy,csv,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from t1_start_c_code import load_json_tasks,resolve_json_path,run_task

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

def make_task(config_name,field,mode):
    matching=[task for task in load_json_tasks(resolve_json_path(config_name)) if field in task["field_list"]]
    if len(matching)!=1:
        raise RuntimeError(f"{config_name}中没有唯一包含{field}的task")
    task=copy.deepcopy(matching[0])
    task["name"]=f"coordinate_transform_range_normalization_{config_name}_{field}_{mode}"
    task["field_list"]=[field]
    task["compressor_list"]=["unsz"]
    task["csv_path"]=str(result_root/f"{config_name}_{field}.csv")
    task["cleanup_after_each_test"]=True
    task["unsz"]["coordinate_transform"]=mode
    return task

def completed(task):
    path=Path(task["csv_path"])
    if not path.is_file():
        return False
    with path.open(newline="",encoding="utf-8") as file:
        rows=list(csv.DictReader(file))
    return sum(row["task"]==task["name"] for row in rows)>=5

def main():
    for config_name,field in field_sources:
        for mode in ("off","on"):
            task=make_task(config_name,field,mode)
            if completed(task):
                print(f"跳过已完成任务：{task['name']}",flush=True)
                continue
            run_task(task)

if __name__=="__main__":
    main()
