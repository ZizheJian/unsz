import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--prepare-only',action='store_true')
arguments=parser.parse_args()
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/tensor_data")
metadata=json.loads((data_dir/"tensor_metadata.json").read_text())
result_dir=project_root/"results/ocean_large_tensor_full_umc/serial_rel_1e-3"
result_dir.mkdir(parents=True,exist_ok=True)
rel_eb=1e-3
umc_root=project_root/'third_party/umc'
for name in ('compress.hpp','decompress.hpp'):
    if (umc_root/'include'/name).read_bytes()!=(project_root/'third_party/umc_replacements/include'/name).read_bytes():
        raise RuntimeError(f'UMC源码不符合项目记录的替换文件：{name}')
binary_sha256={name:hashlib.sha256((umc_root/'build/test'/name).read_bytes()).hexdigest() for name in ('test_compress','test_decompress')}
manifest_path=result_dir/'manifest.json'
if manifest_path.exists() and json.loads(manifest_path.read_text())['binary_sha256']!=binary_sha256:
    raise RuntimeError('UMC二进制变化，请使用新的结果目录，不能混合版本')
manifest={'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip(),'umc_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=umc_root,text=True).strip(),'binary_sha256':binary_sha256,'slurm_cpus':8,'program_threads':1,'rel_eb':rel_eb,'field_num':len(metadata['fields']),'method':'NBP-IDW-INP','reorder':1,'neighbor_num':4,'batch':10}
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
if len(metadata['fields'])!=95:
    raise RuntimeError('完整Ocean-large应有95个field')
if arguments.prepare_only:
    print(manifest_path)
    sys.exit(0)

def completed(path):
    if not path.is_file():
        return False
    with path.open(newline="") as file:
        return any(row.get("compressor")=="umc" and float(row.get("rel_eb","nan"))==rel_eb and all(row.get(name) for name in ('br','psnr','compression_time','decompression_time','compression_preprocessing_time','decompression_preprocessing_time')) for row in csv.DictReader(file))

def cleanup(field,slice_num):
    for slice_id in range(slice_num):
        source=data_dir/f"{field}.dat.{slice_id}"
        Path(str(source)+".umc").unlink(missing_ok=True)
        Path(str(source)+".umc.out").unlink(missing_ok=True)

failures=[]
for field,field_metadata in metadata["fields"].items():
    slice_num=field_metadata["shape"][0]*field_metadata["shape"][1]
    result_path=result_dir/f"{field}.csv"
    if completed(result_path):
        print(f"跳过已完成field：{field}",flush=True)
        continue
    cleanup(field,slice_num)
    config=[{"name":"ocean_large_tensor_full_umc","data_dir":str(data_dir),"csv_path":str(result_dir/'{field}.csv'),"compressor_list":["umc"],"field_list":[field],"time_step_num":slice_num,"coordinate_dim":3,"rel_eb_list":[rel_eb],"cleanup_after_each_test":True,"umc":{"from_edge":1,"reorder":1,"method":"NBP-IDW-INP","neighbor_num":4}}]
    config_path=result_dir/"current_config.json"
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+"\n")
    print(f"开始UMC field：{field}，切片数={slice_num}",flush=True)
    try:
        subprocess.run([sys.executable,str(project_root/"t1_start_c_code.py"),str(config_path),"--skip-build"],cwd=project_root,check=True)
    except subprocess.CalledProcessError as error:
        failures.append((field,error.returncode))
        print(f"UMC field失败：{field}，returncode={error.returncode}",flush=True)
    finally:
        cleanup(field,slice_num)
        config_path.unlink(missing_ok=True)
        (result_dir/'failures.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2)+'\n')
if failures:
    raise RuntimeError("UMC失败field："+",".join(f"{field}({returncode})" for field,returncode in failures))
