import copy,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
import t1_start_c_code as benchmark


task_template=benchmark.load_json_tasks(benchmark.resolve_json_path("remaining-single-step-auto-screen"))[0]
for field in task_template["field_list"]:
    transform_path=Path(task_template["data_dir"])/f"{field}.unsz.coord_transform.txt"
    tokens=transform_path.read_text().split()
    enabled=int(tokens[tokens.index("enabled")+1])
    mode="off" if enabled else "on"
    task=copy.deepcopy(task_template)
    task["name"]=f"taps_opposite_{mode}"
    task["csv_path"]="results/remaining_adjust_coord_screen/taps_opposite.csv"
    task["field_list"]=[field]
    task["unsz"]["coordinate_transform"]=mode
    benchmark.run_task(task)
