import copy,csv,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from t1_start_c_code import load_json_tasks,resolve_json_path,run_task

result_root=Path(__file__).resolve().parent
cases=(("katrina","attr0",3),("ocean","salinity",1),("ocean","salinity",3),("ocean","temperature",1),("ocean","temperature",3))

def completed(task):
    path=Path(task["csv_path"])
    if not path.is_file():
        return False
    with path.open(newline="",encoding="utf-8") as file:
        return any(row["task"]==task["name"] for row in csv.DictReader(file))

def main():
    for config_name,field,power in cases:
        matching=[task for task in load_json_tasks(resolve_json_path(config_name)) if field in task["field_list"]]
        if len(matching)!=1:
            raise RuntimeError(f"{config_name}中没有唯一包含{field}的task")
        task=copy.deepcopy(matching[0])
        task["name"]=f"coordinate_transform_group_loss_full_curve_{config_name}_{field}_power{power}"
        task["field_list"]=[field]
        task["compressor_list"]=["unsz"]
        task["csv_path"]=str(result_root/f"{config_name}_{field}.csv")
        task["cleanup_after_each_test"]=True
        task["unsz"].update({"coordinate_transform":"on","coordinate_transform_training_normalization":"time_range","coordinate_transform_sample_file_num":3,"coordinate_transform_second_sample_num":3000,"coordinate_transform_group_loss_power":power})
        if completed(task):
            print(f"跳过已完成任务：{task['name']}",flush=True)
            continue
        run_task(task)

if __name__=="__main__":
    main()
