import copy,csv,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from t1_start_c_code import load_json_tasks,resolve_json_path,run_task

result_root=project_root/"results/e3sm_label_flow_tuning"
label_flow_vec=("spatial","anchor_interpolation_stride_8")

def expected_row_num(task):
    unsz=task["unsz"]
    return len(unsz["point_set_config_path_list"])*len(unsz["flow"]["label"])*len(unsz["flow"]["value"])

def result_complete(path,task_name,row_num):
    if not path.is_file():
        return False
    with path.open(newline="") as file:
        rows=[row for row in csv.DictReader(file) if row["task"]==task_name and row["field"]=="AODDUST" and row["compressor"]=="unsz"]
    return len(rows)==row_num

def make_task(base_task,label_flow):
    task=copy.deepcopy(base_task)
    variant="label_spatial" if label_flow=="spatial" else "label_anchor8"
    task["name"]=f"e3sm_{variant}_value_anchor8"
    task["field_list"]=["AODDUST"]
    task["compressor_list"]=["unsz"]
    task["csv_path"]=str(result_root/"curves"/f"{variant}.csv")
    task["cleanup_after_each_test"]=True
    unsz=task["unsz"]
    unsz["flow"]={"label":[label_flow],"value":["anchor_interpolation_stride_8"]}
    unsz["label_value_execution"]="fused"
    unsz["predictor"]={"name":"rbf","rbf_kernel":"exponential","rbf_shape":0.01,"rbf_nugget":1e-6}
    unsz["prediction_min_ref_num"]=10
    unsz["prediction_max_ref_num"]=10
    unsz["ordering_scope"]="global"
    unsz["ordering_candidate_num"]=16
    unsz["ordering_score_max_ref_num"]=1
    unsz["ordering_score_min_ref_num"]=1
    unsz["eb_schedule"]={"name":"inverse_rank_log2","base":0.8,"log2_shift":0.5,"max_log2_exponent":7}
    unsz["dag_cache_path"]=str(result_root/"AODDUST_rbf10.dag")
    return task

def main():
    base_task=load_json_tasks(resolve_json_path("e3sm-h4-aoddust-fused"))[0]
    for label_flow in label_flow_vec:
        task=make_task(base_task,label_flow)
        result_path=Path(task["csv_path"])
        if result_complete(result_path,task["name"],expected_row_num(task)):
            print(f"跳过完整结果：label_flow={label_flow}",flush=True)
            continue
        if result_path.is_file():
            result_path.unlink()
        print(f"开始E3SM label流程：label_flow={label_flow}",flush=True)
        run_task(task)

if __name__=="__main__":
    main()
