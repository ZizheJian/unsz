import argparse,copy,csv,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from t1_start_c_code import load_json_tasks,resolve_json_path,run_task

result_root=project_root/"results/rbf_eb_base_tuning"
kernel_result_root=project_root/"results/rbf_kernel_tuning"
dataset_config_map={
    "tsunami":"tsunami",
    "les":"les",
    "e3sm":"e3sm-h4-aoddust-fused",
    "haiyan":"adcirc-haiyan-zeta-fused",
    "katrina":"katrina",
}
base_vec=(0.8,0.9)

def expected_row_num(task):
    unsz=task["unsz"]
    error_num=len(unsz.get("point_set_config_path_list",task.get("rel_eb_list",())))
    label_num=len(unsz["flow"].get("label",(None,)))
    value_num=len(unsz["flow"]["value"])
    return error_num*label_num*value_num

def result_complete(path,task_name,field,row_num):
    if not path.is_file():
        return False
    with path.open(newline="") as file:
        rows=[row for row in csv.DictReader(file) if row["task"]==task_name and row["field"]==field and row["compressor"]=="unsz"]
    return len(rows)==row_num

def base_name(base):
    return f"{base:g}".replace(".","p")

def make_task(base_task,dataset,base,field):
    task=copy.deepcopy(base_task)
    variant=f"exponential_shape_0p01_base_{base_name(base)}"
    task["name"]=f"rbf_eb_base_{dataset}_{variant}"
    task["field_list"]=[field]
    task["compressor_list"]=["unsz"]
    task["csv_path"]=str(result_root/"curves"/dataset/f"{variant}_{field}.csv")
    task["cleanup_after_each_test"]=True
    unsz=task["unsz"]
    unsz["predictor"]={"name":"rbf","rbf_kernel":"exponential","rbf_shape":0.01,"rbf_nugget":1e-6}
    unsz["prediction_min_ref_num"]=10
    unsz["prediction_max_ref_num"]=10
    unsz["dag_cache_path"]=str(kernel_result_root/"dag"/dataset/f"{field}_rbf10.dag")
    unsz["eb_schedule"]["base"]=base
    return task

def main():
    parser=argparse.ArgumentParser(description="固定最终RBF参数，测试两个额外inverse_rank_log2 EB base")
    parser.add_argument("dataset",choices=tuple(dataset_config_map))
    arguments=parser.parse_args()
    base_tasks=load_json_tasks(resolve_json_path(dataset_config_map[arguments.dataset]))
    for base in base_vec:
        for base_task in base_tasks:
            for field in base_task["field_list"]:
                task=make_task(base_task,arguments.dataset,base,field)
                result_path=Path(task["csv_path"])
                if result_complete(result_path,task["name"],field,expected_row_num(task)):
                    print(f"跳过完整结果：dataset={arguments.dataset},field={field},base={base:g}",flush=True)
                    continue
                if result_path.is_file():
                    result_path.unlink()
                print(f"开始EB base曲线：dataset={arguments.dataset},field={field},base={base:g}",flush=True)
                run_task(task)

if __name__=="__main__":
    main()
