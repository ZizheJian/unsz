import argparse,copy,csv,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from t1_start_c_code import load_json_tasks,resolve_json_path,run_task

result_root=project_root/"results/coordinate_transform"
dataset_map={
    "tsunami":("tsunami","momentum_x"),
    "les":("les","velocity_x"),
    "e3sm":("e3sm-h4-aoddust-fused","AODDUST"),
    "haiyan":("adcirc-haiyan-zeta-fused","zeta"),
    "katrina":("katrina","attr0"),
}

def make_task(dataset,mode):
    config_name,field=dataset_map[dataset]
    base_tasks=load_json_tasks(resolve_json_path(config_name))
    matching_tasks=[task for task in base_tasks if field in task["field_list"]]
    if len(matching_tasks)!=1:
        raise RuntimeError(f"{dataset}没有唯一包含field={field}的task")
    task=copy.deepcopy(matching_tasks[0])
    task["name"]=f"coordinate_transform_{dataset}_{mode}"
    task["field_list"]=[field]
    task["compressor_list"]=["unsz"]
    task["csv_path"]=str(result_root/f"{dataset}.csv")
    task["cleanup_after_each_test"]=True
    unsz=task["unsz"]
    for key in tuple(unsz):
        if key.startswith("adjust_coord_"):
            del unsz[key]
    unsz["coordinate_transform"]=mode
    unsz["dag_cache_path"]=str(result_root/"dag"/f"{dataset}_{mode}.dag")
    return task

def completed(dataset,mode):
    path=result_root/f"{dataset}.csv"
    if not path.is_file():
        return False
    with path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    task_name=f"coordinate_transform_{dataset}_{mode}"
    return sum(row["task"]==task_name for row in rows)>=5

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("dataset",choices=(*dataset_map,"all"))
    parser.add_argument("--mode",choices=("off","on","both"),default="both")
    arguments=parser.parse_args()
    datasets=dataset_map if arguments.dataset=="all" else (arguments.dataset,)
    modes=("off","on") if arguments.mode=="both" else (arguments.mode,)
    for dataset in datasets:
        for mode in modes:
            if completed(dataset,mode):
                print(f"跳过已完成任务：dataset={dataset},mode={mode}",flush=True)
                continue
            run_task(make_task(dataset,mode))

if __name__=="__main__":
    main()
