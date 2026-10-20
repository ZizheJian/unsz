import argparse,hashlib,json,re,subprocess,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--results-dir')
parser.add_argument('--rel-eb',type=float,choices=(1e-1,1e-2,1e-3,1e-4,1e-5),default=1e-3)
parser.add_argument('--require-memory-validation',action='store_true')
parser.add_argument('--prepare-only',action='store_true')
arguments=parser.parse_args()
eb_tag={1e-1:'1e-1',1e-2:'1e-2',1e-3:'1e-3',1e-4:'1e-4',1e-5:'1e-5'}[arguments.rel_eb]
default_results='results/ocean_large_tensor_full/'+('sparse_window' if arguments.rel_eb==1e-3 else 'rel_'+eb_tag)
output_dir=(project_root/(arguments.results_dir or default_results)).resolve()
output_dir.mkdir(parents=True,exist_ok=True)
binary_sha256={name:hashlib.sha256((project_root/'install/bin'/name).read_bytes()).hexdigest() for name in ('s1_compress','s1_decompress')}
reference=json.loads((Path(__file__).parent/'sparse_window/manifest.json').read_text())
if binary_sha256!=reference['binary_sha256']:
    raise RuntimeError('UNSZ二进制与已完成REL=1e-3不同，禁止混合版本')
manifest_path=output_dir/'manifest.json'
if manifest_path.exists():
    previous=json.loads(manifest_path.read_text())
    if previous['binary_sha256']!=binary_sha256 or previous['rel_eb']!=arguments.rel_eb:
        raise RuntimeError('已有结果的二进制或EB不同，禁止覆盖')
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
metadata_sha256=hashlib.sha256((data_dir/'tensor_metadata.json').read_bytes()).hexdigest()
if metadata_sha256!=json.loads((project_root/'results/ocean_large_tensor_1d/sz3/rel_0.001_t1/manifest.json').read_text())['metadata_sha256']:
    raise RuntimeError('tensor metadata与已完成实验不同')
field_list=list(metadata["fields"])
if len(field_list)!=95:
    raise RuntimeError(f"完整Ocean-large应包含95个field，实际为{len(field_list)}")
point_set_path=Path('point_set_configs')/f'Ocean_large_monthly_rbf_rel_{eb_tag}.json'
point_set=json.loads((project_root/point_set_path).read_text())
if point_set!={'version':3,'labels':[{'method':'predict','eb_mode':'REL','eb_value':arguments.rel_eb,'condition':{'operator':'!=','value':-99999}},{'method':'constant'}]}:
    raise RuntimeError('点集配置与正式实验规则不一致')
config=[{"data_dir":str(data_dir),"csv_path":str(output_dir/'{field}.csv'),"compressor_list":["unsz"],"coordinate_dim":3,"name":"ocean_large_tensor_full","field_list":field_list,"tensor_metadata":"tensor_metadata.json","cleanup_after_each_test":True,"unsz":{"partition_num":1,"thread_num":1,"coordinate_transform":"auto","point_set_config_path_list":[str(point_set_path)]}}]
config_path=output_dir/f'config_rel_{eb_tag}.json'
config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n")
revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip()
manifest_path.write_text(json.dumps({'git_revision':revision,'source_sha256':{str(path.relative_to(project_root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__).resolve(),project_root/'t1_start_c_code.py')},'binary_sha256':binary_sha256,'metadata_sha256':metadata_sha256,'point_set_config_sha256':hashlib.sha256((project_root/point_set_path).read_bytes()).hexdigest(),'slurm_cpus':8,'partition_num':1,'thread_num':1,'rel_eb':arguments.rel_eb,'field_num':len(field_list)},indent=2)+'\n')
if arguments.prepare_only:
    print(config_path)
    sys.exit(0)
subprocess.run([sys.executable,str(project_root/"t1_start_c_code.py"),str(config_path),"--skip-build"],cwd=project_root,check=True)
