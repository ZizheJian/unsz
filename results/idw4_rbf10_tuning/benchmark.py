import argparse,copy,csv,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from t1_start_c_code import load_json_tasks,resolve_json_path,run_task

result_root=project_root/"results/idw4_rbf10_tuning"
dataset_config_map={
    "tsunami":"tsunami",
    "les":"les",
    "e3sm":"e3sm-h4-aoddust-fused",
    "haiyan":"adcirc-haiyan-zeta-fused",
    "katrina":"katrina",
}
idw_power_vec=(1,2,3.6,5,8)
rbf_config_vec=(
    ("gaussian",0.1),("gaussian",0.25),("gaussian",0.5),
    ("exponential",0.05),("exponential",0.1),("exponential",0.25),
    ("matern32",0.01),("matern32",0.03),("matern32",0.1),
    ("matern52",0.03),("matern52",0.1),("inverse_multiquadric",0.25),
)

def variant_vec():
    variants=[]
    for power in idw_power_vec:
        variants.append((f"idw4_power_{power:g}",{"name":"idw","idw_power":power},1,4))
    for kernel,shape in rbf_config_vec:
        variants.append((f"rbf10_{kernel}_shape_{shape:g}",{"name":"rbf","rbf_kernel":kernel,"rbf_shape":shape,"rbf_nugget":1e-6},10,10))
    return variants

def select_representative_error(task):
    if "point_set_config_path_list" in task["unsz"]:
        selected=[path for path in task["unsz"]["point_set_config_path_list"] if "1e-3.json" in path]
        if len(selected)!=1:
            raise RuntimeError(f"{task['name']}没有唯一的REL 1e-3点集配置")
        task["unsz"]["point_set_config_path_list"]=selected
    else:
        task["rel_eb_list"]=[1e-3]

def make_task(base_task,dataset,variant_name,predictor,min_ref_num,max_ref_num):
    task=copy.deepcopy(base_task)
    task["name"]=f"{dataset}_{variant_name}"
    task["compressor_list"]=["unsz"]
    task["csv_path"]=str(result_root/f"{dataset}.csv")
    task["cleanup_after_each_test"]=True
    select_representative_error(task)
    unsz=task["unsz"]
    unsz["predictor"]=predictor
    unsz["prediction_min_ref_num"]=min_ref_num
    unsz["prediction_max_ref_num"]=max_ref_num
    predictor_family="idw4" if predictor["name"]=="idw" else "rbf10"
    if dataset=="les" and predictor_family=="rbf10":
        unsz["dag_cache_path"]=str(project_root/"results/les_spatial/rbf10_global.dag")
    else:
        unsz["dag_cache_path"]=str(result_root/"dag"/dataset/f"{predictor_family}.dag")
    if dataset=="katrina":
        unsz["coordinate_transform"]="off"
    return task

def remove_completed_fields(task):
    result_path=Path(task["csv_path"])
    if not result_path.is_file():
        return
    with result_path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    completed_fields={row["field"] for row in rows if row["task"]==task["name"] and row["compressor"]=="unsz"}
    task["field_list"]=[field for field in task["field_list"] if field not in completed_fields]

def main():
    parser=argparse.ArgumentParser(description="在五个正式数据上公平筛选IDW4与RBF10参数")
    parser.add_argument("dataset",choices=tuple(dataset_config_map))
    arguments=parser.parse_args()
    base_tasks=load_json_tasks(resolve_json_path(dataset_config_map[arguments.dataset]))
    for variant_name,predictor,min_ref_num,max_ref_num in variant_vec():
        for base_task in base_tasks:
            task=make_task(base_task,arguments.dataset,variant_name,predictor,min_ref_num,max_ref_num)
            remove_completed_fields(task)
            if not task["field_list"]:
                continue
            print(f"开始筛选：dataset={arguments.dataset},variant={variant_name}",flush=True)
            run_task(task)

if __name__=="__main__":
    main()
