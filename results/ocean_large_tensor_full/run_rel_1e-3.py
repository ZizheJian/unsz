import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--results-dir',default='results/ocean_large_tensor_full/sparse_window')
parser.add_argument('--require-memory-validation',action='store_true')
parser.add_argument('--prepare-only',action='store_true')
arguments=parser.parse_args()
output_dir=(project_root/arguments.results_dir).resolve()
output_dir.mkdir(parents=True,exist_ok=True)
binary_sha256={name:hashlib.sha256((project_root/'install/bin'/name).read_bytes()).hexdigest() for name in ('s1_compress','s1_decompress')}
if arguments.require_memory_validation:
    for directory,field in (('density_sparse_mem8','timeMonthly_avg_density'),('layerThickness_sparse_mem8','timeMonthly_avg_layerThickness'),('vertVelocityTop_sparse_mem8','timeMonthly_avg_vertVelocityTop')):
        validation_dir=Path(__file__).parent/directory
        summary=json.loads((validation_dir/'summary.json').read_text())
        if summary['field']!=field or summary['binary_sha256']!=binary_sha256:
            raise RuntimeError(f'内存验证版本或field不一致：{directory}')
        if max(summary['max_rss_kib'].values())>0.9*15168*1024:
            raise RuntimeError(f'8份内存的安全余量不足：{directory}')
        maximum_error=re.search(r'maximum error over eb= ([^\s,]+)',(validation_dir/'decompression.log').read_text())
        if maximum_error is None or not 0<=float(maximum_error[1])<=1.01:
            raise RuntimeError(f'逐点误差验证失败：{directory}')
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/tensor_data")
metadata=json.loads((data_dir/"tensor_metadata.json").read_text())
field_list=list(metadata["fields"])
if len(field_list)!=95:
    raise RuntimeError(f"完整Ocean-large应包含95个field，实际为{len(field_list)}")
config=[{"data_dir":str(data_dir),"csv_path":str(output_dir/'{field}.csv'),"compressor_list":["unsz"],"coordinate_dim":3,"name":"ocean_large_tensor_full","field_list":field_list,"tensor_metadata":"tensor_metadata.json","cleanup_after_each_test":True,"unsz":{"partition_num":1,"thread_num":1,"coordinate_transform":"auto","point_set_config_path_list":["point_set_configs/Ocean_large_monthly_rbf_rel_1e-3.json"]}}]
config_path=output_dir/'config_rel_1e-3.json'
config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n")
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip()
(output_dir/'manifest.json').write_text(json.dumps({'git_revision':revision,'binary_sha256':binary_sha256,'slurm_cpus':8,'partition_num':1,'thread_num':1,'rel_eb':0.001,'field_num':len(field_list)},indent=2)+'\n')
if arguments.prepare_only:
    print(config_path)
    sys.exit(0)
subprocess.run([sys.executable,str(project_root/"t1_start_c_code.py"),str(config_path),"--skip-build"],cwd=project_root,check=True)
