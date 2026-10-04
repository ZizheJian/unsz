import argparse,copy,csv,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from t1_start_c_code import load_json_tasks,resolve_json_path,run_task

result_root=project_root/"results/rbf_reference_num_tuning"
dataset_config_map={
    "tsunami":"tsunami",
    "les":"les",
    "e3sm":"e3sm-h4-aoddust-fused",
    "haiyan":"adcirc-haiyan-zeta-fused",
    "katrina":"katrina",
}
reference_num_vec=((10,10),(8,8),(9,9),(7,10),(8,10),(9,10),(10,12))

def variant_name(min_ref_num,max_ref_num):
    return f"min_{min_ref_num}_max_{max_ref_num}"

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

def make_task(base_task,dataset,min_ref_num,max_ref_num,field):
    task=copy.deepcopy(base_task)
    variant=variant_name(min_ref_num,max_ref_num)
    task["name"]=f"rbf_reference_num_{dataset}_{variant}"
    task["field_list"]=[field]
    task["compressor_list"]=["unsz"]
    task["csv_path"]=str(result_root/"curves"/dataset/f"{variant}_{field}.csv")
    task["cleanup_after_each_test"]=True
    unsz=task["unsz"]
    unsz["predictor"]={"name":"rbf","rbf_kernel":"exponential","rbf_shape":0.01,"rbf_nugget":1e-6}
    unsz["prediction_min_ref_num"]=min_ref_num
    unsz["prediction_max_ref_num"]=max_ref_num
    unsz["ordering_candidate_num"]=16
    unsz["ordering_score_max_ref_num"]=1
    unsz["ordering_score_min_ref_num"]=1
    unsz["ordering_scope"]="global"
    unsz.pop("prediction_round_num",None)
    unsz["eb_schedule"]={"name":"inverse_rank_log2","base":0.8,"log2_shift":0.5,"max_log2_exponent":7}
    unsz["dag_cache_path"]=str(result_root/"dag"/dataset/f"{field}_max_{max_ref_num}.dag")
    return task

def main():
    parser=argparse.ArgumentParser(description="比较RBF最小和最大预测参考点数的压缩效果")
    parser.add_argument("dataset",choices=tuple(dataset_config_map))
    parser.add_argument("--pairs",nargs="+",choices=tuple(variant_name(*pair) for pair in reference_num_vec))
    arguments=parser.parse_args()
    selected_pairs=reference_num_vec if arguments.pairs is None else tuple(reference_num_vec[tuple(variant_name(*pair) for pair in reference_num_vec).index(name)] for name in arguments.pairs)
    base_tasks=load_json_tasks(resolve_json_path(dataset_config_map[arguments.dataset]))
    for base_task in base_tasks:
        for field in base_task["field_list"]:
            for min_ref_num,max_ref_num in selected_pairs:
                task=make_task(base_task,arguments.dataset,min_ref_num,max_ref_num,field)
                result_path=Path(task["csv_path"])
                if result_complete(result_path,task["name"],field,expected_row_num(task)):
                    print(f"跳过完整结果：dataset={arguments.dataset},field={field},min={min_ref_num},max={max_ref_num}",flush=True)
                    continue
                if result_path.is_file():
                    result_path.unlink()
                print(f"开始参考点数测试：dataset={arguments.dataset},field={field},min={min_ref_num},max={max_ref_num}",flush=True)
                run_task(task)

if __name__=="__main__":
    main()
