import csv,hashlib,json,math,os,re,subprocess,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
result_path=project_root/"py_codes/trash/results/katrina_fused_prediction_time.csv"
time_step_num=162
point_num=417642
total_mb=point_num*time_step_num*4/1e6
rel_eb=1e-3
repeat_num=3

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

def sum_stage_time(text,name):
    return sum(float(value) for value in re.findall(rf"compression stage time: .*?{name}=([0-9.eE+-]+) s",text))

def write_point_set_config(path):
    config={"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]}
    with open(path,"w",encoding="utf-8") as file:
        json.dump(config,file,ensure_ascii=False,indent=2)

def compressed_sha(test_dir):
    digest=hashlib.sha256()
    for time_step in range(time_step_num):
        with open(test_dir/f"attr0.dat.{time_step}.unsz","rb") as file:
            while True:
                data=file.read(1024*1024)
                if not data:
                    break
                digest.update(data)
    return digest.hexdigest()

def cleanup_decompressed(test_dir):
    for time_step in range(time_step_num):
        path=test_dir/f"attr0.dat.{time_step}.unsz.bin"
        if path.exists():
            path.unlink()

def geometric_mean(value_vec):
    return math.exp(sum(math.log(value) for value in value_vec)/len(value_vec))

def main():
    result_vec=[]
    baseline_sha_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_fused_prediction_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        point_set_config_path=temporary_dir/"point_sets.json"
        prediction_structure_path=temporary_dir/"prediction_structure.bin"
        write_point_set_config(point_set_config_path)
        common_arguments=["-data-dir",test_dir,"-variable","attr0","-time-step","[0,161]","-dim",3,"-point-set-config",point_set_config_path,"-adjust-coord-matrix-file",matrix_path,"-dag-cache",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",4,"-actual-ref-num",4,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",4,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental"]
        for repeat_id in range(repeat_num):
            for name,fused in (("separate",False),("fused",True)):
                print(f"测试{name}，repeat={repeat_id}",flush=True)
                arguments=[*common_arguments]
                if fused:
                    arguments.append("-fused-two-point-set-prediction-experimental")
                compression_output=run_command([compressor,*arguments,"-compression-stage-time"])
                current_sha=compressed_sha(test_dir)
                if fused:
                    if current_sha!=baseline_sha_vec[repeat_id]:
                        raise RuntimeError(f"第{repeat_id}次融合码流与原路径不同")
                else:
                    baseline_sha_vec.append(current_sha)
                decompression_output=run_command([decompressor,*arguments])
                psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
                compression_time=read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间")
                decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")
                result={"name":name,"repeat":repeat_id,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"compression_time":compression_time,"decompression_time":decompression_time,"compression_mb_s":total_mb/compression_time,"decompression_mb_s":total_mb/decompression_time,"point_set_time":sum_stage_time(compression_output,"point_set"),"label_prediction_time":sum_stage_time(compression_output,"label_prediction"),"value_prediction_time":sum_stage_time(compression_output,"value_prediction"),"lossless_encoding_time":sum_stage_time(compression_output,"lossless_encoding"),"compressed_sha256":current_sha}
                result_vec.append(result)
                print(f"  compression={result['compression_mb_s']:.3f} MB/s,decompression={result['decompression_mb_s']:.3f} MB/s",flush=True)
                cleanup_decompressed(test_dir)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    result_map={name:[result for result in result_vec if result["name"]==name] for name in ("separate","fused")}
    for name,results in result_map.items():
        compression_mb_s=geometric_mean([result["compression_mb_s"] for result in results])
        decompression_mb_s=geometric_mean([result["decompression_mb_s"] for result in results])
        print(f"{name}：compression={compression_mb_s:.3f} MB/s,decompression={decompression_mb_s:.3f} MB/s",flush=True)
    separate_compression=geometric_mean([result["compression_mb_s"] for result in result_map["separate"]])
    fused_compression=geometric_mean([result["compression_mb_s"] for result in result_map["fused"]])
    separate_decompression=geometric_mean([result["decompression_mb_s"] for result in result_map["separate"]])
    fused_decompression=geometric_mean([result["decompression_mb_s"] for result in result_map["fused"]])
    print(f"融合压缩速度变化={(fused_compression/separate_compression-1)*100:+.3f}%",flush=True)
    print(f"融合解压速度变化={(fused_decompression/separate_decompression-1)*100:+.3f}%",flush=True)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
