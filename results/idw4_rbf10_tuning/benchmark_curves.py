import argparse,copy,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from t1_start_c_code import load_json_tasks,resolve_json_path,run_task
from benchmark import dataset_config_map

result_root=project_root/"results/idw4_rbf10_tuning"
idw_power_vec=(1,1.5,2,2.5,3.6)
rbf_candidate_map={
    "tsunami":(("matern32",0.03),("gaussian",0.1),("exponential",0.05)),
    "les":(("matern32",0.03),("matern32",0.01),("gaussian",0.1)),
    "e3sm":(("matern32",0.03),("matern32",0.01),("exponential",0.05)),
    "haiyan":(("exponential",0.05),("exponential",0.1),("exponential",0.25)),
    "katrina":(("matern32",0.01),("matern52",0.03),("exponential",0.05)),
}

def variant_vec(dataset):
    variants=[]
    for power in idw_power_vec:
        variants.append((f"idw4_power_{power:g}",{"name":"idw","idw_power":power},1,4))
    for kernel,shape in rbf_candidate_map[dataset]:
        variants.append((f"rbf10_{kernel}_shape_{shape:g}",{"name":"rbf","rbf_kernel":kernel,"rbf_shape":shape,"rbf_nugget":1e-6},10,10))
    return variants

def make_curve_task(base_task,dataset,variant_name,predictor,min_ref_num,max_ref_num):
    task=copy.deepcopy(base_task)
    task["name"]=f"curve_{dataset}_{variant_name}"
    task["compressor_list"]=["unsz"]
    task["csv_path"]=str(result_root/"curves"/f"{dataset}.csv")
    task["cleanup_after_each_test"]=True
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

def main():
    parser=argparse.ArgumentParser(description="对宽筛候选运行完整RD曲线")
    parser.add_argument("dataset",choices=tuple(dataset_config_map))
    arguments=parser.parse_args()
    base_tasks=load_json_tasks(resolve_json_path(dataset_config_map[arguments.dataset]))
    for variant_name,predictor,min_ref_num,max_ref_num in variant_vec(arguments.dataset):
        for base_task in base_tasks:
            task=make_curve_task(base_task,arguments.dataset,variant_name,predictor,min_ref_num,max_ref_num)
            print(f"开始完整曲线：dataset={arguments.dataset},variant={variant_name}",flush=True)
            run_task(task)

if __name__=="__main__":
    main()
