import csv,copy,re,subprocess,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from t1_start_c_code import build_unsz_commands,expand_unsz_flows,load_json_tasks,resolve_json_path

result_root=project_root/"results/coordinate_transform_first_sample_ratio_tuning"
ratio_vec=(0.2,0.25,0.3,0.35,0.4,1.0)

def ratio_name(ratio):
    return f"{ratio:.2f}".replace(".","p")

def extract(pattern,text,name,convert=float):
    match=re.search(pattern,text)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return convert(match.group(1))

def main():
    tasks=load_json_tasks(resolve_json_path("katrina"))
    if len(tasks)!=1:
        raise RuntimeError("Katrina正式配置必须恰好包含一个task")
    task=tasks[0]
    field_config=task|{"field":"attr0"}|copy.deepcopy(task["unsz"])
    base=expand_unsz_flows(field_config)[0]
    point_set_config=task["unsz"]["point_set_config_path_list"][2]
    dag_directory=result_root/"timing_dag"
    dag_directory.mkdir(parents=True,exist_ok=True)
    result_path=result_root/"timing.csv"
    existing={}
    if result_path.is_file():
        with result_path.open(newline="") as file:
            existing={float(row["ratio"]):row for row in csv.DictReader(file)}
    rows=[]
    for ratio in ratio_vec:
        if ratio in existing:
            rows.append(existing[ratio])
            print(f"跳过已完成计时：{ratio}",flush=True)
            continue
        name=ratio_name(ratio)
        dag_path=dag_directory/f"{name}.dag"
        if dag_path.exists():
            dag_path.unlink()
        config=base|{"coordinate_transform":"on","coordinate_transform_first_sample_ratio":ratio,"point_set_config_path":point_set_config}
        compression_command,_=build_unsz_commands(config,task["data_dir"],dag_path)
        result=subprocess.run([str(value) for value in compression_command],cwd=project_root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=True)
        output=result.stdout
        (result_root/f"timing_{name}.log").write_text(output)
        transform_time=extract(r"coordinate transform training and application time= ([0-9.eE+-]+) s",output,"坐标变换时间")
        preprocessing_time=extract(r"preprocessing time excluding coordinate and training-data input= ([0-9.eE+-]+) s",output,"总预处理时间")
        rows.append({"ratio":ratio,"transform_time":transform_time,"preprocessing_time":preprocessing_time,"transform_fraction":transform_time/preprocessing_time,"accepted_steps":extract(r"coordinate transform optimization accepted steps=([0-9]+)",output,"接受步数",int),"function_evaluations":extract(r"function evaluations=([0-9]+)",output,"函数评估次数",int)})
        print(rows[-1],flush=True)
    with result_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

if __name__=="__main__":
    main()
