import csv,json,math,os,re,subprocess,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
result_path=project_root/"py_codes/trash/results/katrina_zstd_level.csv"
time_step_num=162
point_num=417642
total_mb=point_num*time_step_num*4/1e6
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
zstd_level_vec=(1,3)
repeat_num=2

def run_command(command):
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError("命令执行失败："+" ".join(str(value) for value in command))
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def sum_float(pattern,text):
    return sum(float(value) for value in re.findall(pattern,text,re.MULTILINE))

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_zstd_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        prediction_structure_path=temporary_dir/"prediction_structure.bin"
        for rel_eb in rel_eb_vec:
            config_path=temporary_dir/f"point_sets_{rel_eb}.json"
            with open(config_path,"w",encoding="utf-8") as file:
                json.dump({"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]},file,ensure_ascii=False)
            common_arguments=["-data-dir",test_dir,"-variable","attr0","-time-step","[0,161]","-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",matrix_path,"-dag-cache",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",4,"-actual-ref-num",4,"-inverse-distance-power",2,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",3,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental","-cross-time-label-predictor-experimental","previous","-fused-two-point-set-prediction-experimental","-normalized-weight-experimental","-compression-stage-time"]
            for repeat_id in range(repeat_num):
                level_order=zstd_level_vec[repeat_id:]+zstd_level_vec[:repeat_id]
                for zstd_level in level_order:
                    print(f"测试REL={rel_eb:g},zstd={zstd_level},repeat={repeat_id}",flush=True)
                    output=run_command([compressor,*common_arguments,"-zstd-level",zstd_level])
                    compression_time=read_float(r"^total compression time= ([^ ]+) s$",output,"压缩时间")
                    result_vec.append({"rel_eb":rel_eb,"zstd_level":zstd_level,"repeat":repeat_id,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",output,"总BR"),"psnr":read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",output,"总PSNR"),"compression_time":compression_time,"compression_mb_s":total_mb/compression_time,"lossless_encoding_time":sum_float(r"^compression stage time: layout_reorder=[^ ]+ s, point_set=[^ ]+ s, label_prediction=[^ ]+ s, value_prediction=[^ ]+ s, lossless_encoding=([^ ]+) s,",output)})
                    for time_step in range(time_step_num):
                        (test_dir/f"attr0.dat.{time_step}.unsz").unlink(missing_ok=True)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_vec)
    for zstd_level in zstd_level_vec:
        results=[result for result in result_vec if result["zstd_level"]==zstd_level]
        print(f"zstd={zstd_level}: BR={geometric_mean([result['br'] for result in results]):.6f},compression={geometric_mean([result['compression_mb_s'] for result in results]):.3f} MB/s,lossless={geometric_mean([result['lossless_encoding_time'] for result in results]):.6f} s",flush=True)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
