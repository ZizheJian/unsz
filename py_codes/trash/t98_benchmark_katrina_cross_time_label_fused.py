import csv,hashlib,json,math,os,re,subprocess,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
result_path=project_root/"py_codes/trash/results/katrina_cross_time_label_fused_time.csv"
time_step_num=162
point_num=417642
total_mb=point_num*time_step_num*4/1e6
rel_eb=1e-3
repeat_num=int(os.environ.get("UNSZ_BENCHMARK_REPEAT_NUM","3"))

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

def compressed_sha(test_dir):
    digest=hashlib.sha256()
    for time_step in range(time_step_num):
        with open(test_dir/f"attr0.dat.{time_step}.unsz","rb") as file:
            while data:=file.read(1024*1024):
                digest.update(data)
    return digest.hexdigest()

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def main():
    result_vec=[]
    reference_sha=None
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_cross_label_fused_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        config_path=temporary_dir/"point_sets.json"
        with open(config_path,"w",encoding="utf-8") as file:
            json.dump({"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]},file,ensure_ascii=False)
        prediction_structure_path=temporary_dir/"prediction_structure.bin"
        common_arguments=["-data-dir",test_dir,"-variable","attr0","-time-step","[0,161]","-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",matrix_path,"-dag-cache",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",4,"-actual-ref-num",4,"-inverse-distance-power",2,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",3,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental","-cross-time-label-predictor-experimental","previous"]
        mode_vec=[("staged",False,False,False),("fused_raw",True,False,False),("fused_normalized",True,True,False),("fused_linear_label",True,False,True),("fused_linear_normalized",True,True,True)]
        for repeat_id in range(repeat_num):
            ordered_modes=mode_vec[repeat_id:]+mode_vec[:repeat_id]
            for name,use_fused,use_normalized,use_linear_label in ordered_modes:
                print(f"测试{name}，repeat={repeat_id}",flush=True)
                arguments=[*common_arguments]
                if use_fused:
                    arguments.append("-fused-two-point-set-prediction-experimental")
                if use_normalized:
                    arguments.append("-normalized-weight-experimental")
                if use_linear_label:
                    arguments.append("-linear-previous-label-experimental")
                compression_output=run_command([compressor,*arguments,"-compression-stage-time"])
                current_sha=compressed_sha(test_dir)
                if reference_sha is None:
                    reference_sha=current_sha
                elif current_sha!=reference_sha:
                    raise RuntimeError(f"{name}的压缩码流与分阶段路径不同")
                decompression_output=run_command([decompressor,*arguments])
                compression_time=read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间")
                decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")
                psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                result={"name":name,"repeat":repeat_id,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"compression_time":compression_time,"decompression_time":decompression_time,"compression_mb_s":total_mb/compression_time,"decompression_mb_s":total_mb/decompression_time,"point_set_time":sum_float(r"^compression stage time: layout_reorder=[^ ]+ s, point_set=([^ ]+) s,",compression_output),"label_prediction_time":sum_float(r"^compression stage time: layout_reorder=[^ ]+ s, point_set=[^ ]+ s, label_prediction=([^ ]+) s,",compression_output),"value_prediction_time":sum_float(r"^compression stage time: layout_reorder=[^ ]+ s, point_set=[^ ]+ s, label_prediction=[^ ]+ s, value_prediction=([^ ]+) s,",compression_output),"lossless_encoding_time":sum_float(r"^compression stage time: layout_reorder=[^ ]+ s, point_set=[^ ]+ s, label_prediction=[^ ]+ s, value_prediction=[^ ]+ s, lossless_encoding=([^ ]+) s,",compression_output),"compressed_sha256":current_sha}
                result_vec.append(result)
                print(f"  BR={result['br']:.6f},PSNR={psnr:.6f},compression={result['compression_mb_s']:.3f} MB/s,decompression={result['decompression_mb_s']:.3f} MB/s",flush=True)
                for time_step in range(time_step_num):
                    decompressed_path=test_dir/f"attr0.dat.{time_step}.unsz.bin"
                    if decompressed_path.exists():
                        decompressed_path.unlink()
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_vec)
    result_map={name:[result for result in result_vec if result["name"]==name] for name,_,_,_ in mode_vec}
    for name,results in result_map.items():
        print(f"{name}：compression={geometric_mean([result['compression_mb_s'] for result in results]):.3f} MB/s,decompression={geometric_mean([result['decompression_mb_s'] for result in results]):.3f} MB/s",flush=True)
    for name in ("fused_raw","fused_normalized","fused_linear_label","fused_linear_normalized"):
        compression_change=geometric_mean([result["compression_mb_s"] for result in result_map[name]])/geometric_mean([result["compression_mb_s"] for result in result_map["staged"]])-1
        decompression_change=geometric_mean([result["decompression_mb_s"] for result in result_map[name]])/geometric_mean([result["decompression_mb_s"] for result in result_map["staged"]])-1
        print(f"{name}相对staged：compression={compression_change*100:+.3f}%,decompression={decompression_change*100:+.3f}%",flush=True)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
