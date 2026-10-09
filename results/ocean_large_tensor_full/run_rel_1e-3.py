import json,subprocess,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/tensor_data")
metadata=json.loads((data_dir/"tensor_metadata.json").read_text())
field_list=list(metadata["fields"])
if len(field_list)!=95:
    raise RuntimeError(f"完整Ocean-large应包含95个field，实际为{len(field_list)}")
config=[{"data_dir":str(data_dir),"csv_path":"results/ocean_large_tensor_full/{field}.csv","compressor_list":["unsz"],"coordinate_dim":3,"name":"ocean_large_tensor_full","field_list":field_list,"tensor_metadata":"tensor_metadata.json","cleanup_after_each_test":True,"unsz":{"partition_num":1,"thread_num":1,"coordinate_transform":"auto","point_set_config_path_list":["point_set_configs/Ocean_large_monthly_rbf_rel_1e-3.json"]}}]
config_path=Path(__file__).with_name("config_rel_1e-3.json")
config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n")
subprocess.run([sys.executable,str(project_root/"t1_start_c_code.py"),str(config_path),"--skip-build"],cwd=project_root,check=True)
