import csv,json,math,os,re,subprocess,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
result_path=project_root/"py_codes/trash/results/katrina_reference_num_screen.csv"
time_step_num=162
point_num=417642
total_mb=point_num*time_step_num*4/1e6
rel_eb=1e-3
config_vec=(
    {"name":"ref10_cache8","ref_num":10,"cache_threshold":8},
    {"name":"ref4_cache4","ref_num":4,"cache_threshold":4},
    {"name":"ref4_cache3","ref_num":4,"cache_threshold":3},
)

def run_command(command):
    print("运行："+" ".join(str(value) for value in command),flush=True)
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(result.stdout,flush=True)
    if result.returncode!=0:
        raise RuntimeError("命令执行失败")
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def write_point_set_config(path):
    config={"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]}
    with open(path,"w",encoding="utf-8") as file:
        json.dump(config,file,ensure_ascii=False,indent=2)

def cleanup(test_dir):
    for time_step in range(time_step_num):
        for suffix in (".unsz",".unsz.bin"):
            path=test_dir/f"attr0.dat.{time_step}{suffix}"
            if path.exists():
                path.unlink()

def benchmark(test_dir,point_set_config_path,config,prediction_structure_path):
    ref_num=config["ref_num"]
    cache_threshold=config["cache_threshold"]
    common_arguments=["-data-dir",test_dir,"-variable","attr0","-time-step","[0,161]","-dim",3,"-point-set-config",point_set_config_path,"-adjust-coord-matrix-file",matrix_path,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",ref_num,"-actual-ref-num",ref_num,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",cache_threshold,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental"]
    compression_output=run_command([compressor,*common_arguments,"-compression-stage-time","-label-stats"])
    decompression_output=run_command([decompressor,*common_arguments])
    br=read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR")
    psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR")
    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
    preprocessing_time=read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间")
    compression_time=read_float(r"^total compression time= ([^ ]+) s$",compression_output,"总压缩时间")
    decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"总解压时间")
    cache_match=re.search(r"^D-only score cache: (\d+) hits, (\d+) misses$",compression_output,re.MULTILINE)
    reduced_match=re.search(r"^D-only reduced-reference cache hits: (\d+)$",compression_output,re.MULTILINE)
    label_zstd_bytes=sum(int(value) for value in re.findall(r"label_zstd_bytes=([0-9]+)",compression_output))
    result={"name":config["name"],"ref_num":ref_num,"cache_threshold":cache_threshold,"rel_eb":rel_eb,"br":br,"psnr":psnr,"preprocessing_time":preprocessing_time,"compression_time":compression_time,"decompression_time":decompression_time,"compression_mb_s":total_mb/compression_time,"decompression_mb_s":total_mb/decompression_time,"prediction_structure_bytes":prediction_structure_path.stat().st_size,"label_zstd_bytes":label_zstd_bytes,"label_br":label_zstd_bytes*8/(point_num*time_step_num),"cache_hit_num":int(cache_match.group(1)) if cache_match else -1,"cache_miss_num":int(cache_match.group(2)) if cache_match else -1,"reduced_ref_cache_hit_num":int(reduced_match.group(1)) if reduced_match else -1}
    cleanup(test_dir)
    return result

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_reference_num_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        point_set_config_path=temporary_dir/"point_sets.json"
        write_point_set_config(point_set_config_path)
        for config in config_vec:
            print(f"\n测试{config['name']}：ref_num={config['ref_num']},cache_threshold={config['cache_threshold']}",flush=True)
            prediction_structure_path=temporary_dir/f"{config['name']}.prediction_structure.bin"
            result=benchmark(test_dir,point_set_config_path,config,prediction_structure_path)
            result_vec.append(result)
            print(f"结果：BR={result['br']:.9f},PSNR={result['psnr']:.6f},preprocessing={result['preprocessing_time']:.6f} s,compression={result['compression_mb_s']:.3f} MB/s,decompression={result['decompression_mb_s']:.3f} MB/s",flush=True)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    baseline=result_vec[0]
    for result in result_vec:
        print(f"{result['name']}：BR变化={(result['br']/baseline['br']-1)*100:+.3f}%，压缩速度变化={(result['compression_mb_s']/baseline['compression_mb_s']-1)*100:+.3f}%，解压速度变化={(result['decompression_mb_s']/baseline['decompression_mb_s']-1)*100:+.3f}%",flush=True)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
