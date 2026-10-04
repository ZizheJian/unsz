import copy,csv,shutil,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from t1_start_c_code import load_json_tasks,resolve_json_path,run_task

result_root=project_root/"results/coordinate_transform_sample_file_num_tuning"
configuration_vec=((2,3000),(3,3000),(4,3000),(4,4000),(5,3000),(5,5000))

def configuration_name(sample_file_num,second_sample_num):
    return str(sample_file_num) if second_sample_num==3000 else f"{sample_file_num}_second_sample_num_{second_sample_num}"

def task_name(sample_file_num,second_sample_num):
    return f"katrina_coordinate_transform_sample_file_num_{configuration_name(sample_file_num,second_sample_num)}"

def completed(sample_file_num,second_sample_num):
    path=result_root/"results.csv"
    if not path.is_file():
        return False
    with path.open(newline="") as file:
        return sum(row["task"]==task_name(sample_file_num,second_sample_num) for row in csv.DictReader(file))>=5

def make_task(sample_file_num,second_sample_num):
    tasks=load_json_tasks(resolve_json_path("katrina"))
    if len(tasks)!=1:
        raise RuntimeError("Katrina正式配置必须恰好包含一个task")
    task=copy.deepcopy(tasks[0])
    task["name"]=task_name(sample_file_num,second_sample_num)
    task["compressor_list"]=["unsz"]
    task["csv_path"]=str(result_root/"results.csv")
    task["cleanup_after_each_test"]=True
    unsz=task["unsz"]
    unsz["coordinate_transform"]="on"
    unsz["coordinate_transform_first_sample_ratio"]=0.3
    unsz["coordinate_transform_second_sample_num"]=second_sample_num
    unsz["coordinate_transform_sample_file_num"]=sample_file_num
    unsz["dag_cache_path"]=str(result_root/"dag"/f"{configuration_name(sample_file_num,second_sample_num)}.dag")
    return task

def main():
    result_root.mkdir(parents=True,exist_ok=True)
    for sample_file_num,second_sample_num in configuration_vec:
        if completed(sample_file_num,second_sample_num):
            print(f"跳过已完成配置：文件数={sample_file_num},第二轮目标数={second_sample_num}",flush=True)
            continue
        task=make_task(sample_file_num,second_sample_num)
        run_task(task)
        source=Path(task["data_dir"])/"attr0.unsz.coord_transform.txt"
        shutil.copyfile(source,result_root/f"coordinate_transform_{configuration_name(sample_file_num,second_sample_num)}.txt")

if __name__=="__main__":
    main()
