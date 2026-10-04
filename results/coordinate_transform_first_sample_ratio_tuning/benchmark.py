import copy,csv,shutil,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from t1_start_c_code import load_json_tasks,resolve_json_path,run_task

result_root=project_root/"results/coordinate_transform_first_sample_ratio_tuning"
ratio_vec=(0.2,0.25,0.3,0.35,0.4,1.0)

def ratio_name(ratio):
    return f"{ratio:.2f}".replace(".","p")

def task_name(ratio):
    return "katrina_coordinate_transform_first_sample_ratio_"+ratio_name(ratio)

def completed(ratio):
    path=result_root/"results.csv"
    if not path.is_file():
        return False
    with path.open(newline="") as file:
        return sum(row["task"]==task_name(ratio) for row in csv.DictReader(file))>=5

def make_task(ratio):
    tasks=load_json_tasks(resolve_json_path("katrina"))
    if len(tasks)!=1:
        raise RuntimeError("Katrina正式配置必须恰好包含一个task")
    task=copy.deepcopy(tasks[0])
    task["name"]=task_name(ratio)
    task["compressor_list"]=["unsz"]
    task["csv_path"]=str(result_root/"results.csv")
    task["cleanup_after_each_test"]=True
    unsz=task["unsz"]
    unsz["coordinate_transform"]="on"
    unsz["coordinate_transform_first_sample_ratio"]=ratio
    unsz["dag_cache_path"]=str(result_root/"dag"/(ratio_name(ratio)+".dag"))
    return task

def main():
    result_root.mkdir(parents=True,exist_ok=True)
    for ratio in ratio_vec:
        if completed(ratio):
            print(f"跳过已完成比例：{ratio}",flush=True)
            continue
        task=make_task(ratio)
        run_task(task)
        source=Path(task["data_dir"])/"attr0.unsz.coord_transform.txt"
        shutil.copyfile(source,result_root/f"coordinate_transform_{ratio_name(ratio)}.txt")

if __name__=="__main__":
    main()
