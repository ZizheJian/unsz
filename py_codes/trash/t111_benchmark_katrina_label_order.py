import csv,hashlib,json,math,os,re,subprocess,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
result_path=project_root/"py_codes/trash/results/katrina_label_order.csv"
time_step_num=162
point_num=417642
total_mb=point_num*time_step_num*4/1e6
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

def combined_sha(test_dir):
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
    mode_vec=(("original",["-normalized-weight-experimental"],["-linear-previous-label-experimental"]),("dag",["-normalized-weight-experimental","-prediction-order-label-experimental"],["-prediction-order-label-experimental"]))
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_label_order_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        config_path=temporary_dir/"point_sets.json"
        with open(config_path,"w",encoding="utf-8") as file:
            json.dump({"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":1e-3,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]},file,ensure_ascii=False)
        prediction_structure_path=temporary_dir/"prediction_structure.bin"
        common_arguments=["-data-dir",test_dir,"-variable","attr0","-time-step","[0,161]","-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",matrix_path,"-dag-cache",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",4,"-actual-ref-num",4,"-inverse-distance-power",2,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",3,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental","-cross-time-label-predictor-experimental","previous","-fused-two-point-set-prediction-experimental"]
        for repeat_id in range(repeat_num):
            mode_order=mode_vec[repeat_id:]+mode_vec[:repeat_id]
            for name,compress_flags,decompress_flags in mode_order:
                print(f"测试{name},repeat={repeat_id}",flush=True)
                compression_output=run_command([compressor,*common_arguments,*compress_flags])
                sha256=combined_sha(test_dir)
                decompression_output=run_command([decompressor,*common_arguments,*decompress_flags])
                compression_time=read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间")
                decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")
                psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                result_vec.append({"name":name,"repeat":repeat_id,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"label_zstd_bytes":sum(int(value) for value in re.findall(r"label_zstd_bytes=([0-9]+)",compression_output)),"compression_time":compression_time,"decompression_time":decompression_time,"compression_mb_s":total_mb/compression_time,"decompression_mb_s":total_mb/decompression_time,"compressed_sha256":sha256})
                for time_step in range(time_step_num):
                    (test_dir/f"attr0.dat.{time_step}.unsz.bin").unlink(missing_ok=True)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_vec)
    for name,_,_ in mode_vec:
        results=[result for result in result_vec if result["name"]==name]
        print(f"{name}: BR={results[0]['br']:.6f},label_bytes={results[0]['label_zstd_bytes']},compression={geometric_mean([result['compression_mb_s'] for result in results]):.3f} MB/s,decompression={geometric_mean([result['decompression_mb_s'] for result in results]):.3f} MB/s",flush=True)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
