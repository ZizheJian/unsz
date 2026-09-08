import csv,json,math,os,re,subprocess,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
result_path=project_root/"py_codes/trash/results/katrina_rbf_ref10_reference_order_candidate48_nearest_fallback_rd.csv"
training_time_step_vec=(40,80,120)
test_time_step_vec=tuple(time_step for time_step in range(162) if time_step not in training_time_step_vec)
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)

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

def write_point_set_config(path,rel_eb):
    with path.open("w",encoding="utf-8") as file:
        json.dump({"version":1,"rule_type":"ordered","labels":[{"method":"rbf_fitting","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]},file,ensure_ascii=False)

def cleanup(test_dir):
    for time_step in test_time_step_vec:
        for suffix in (".unsz",".unsz.bin"):
            path=test_dir/f"attr0.dat.{time_step}{suffix}"
            if path.exists():
                path.unlink()

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_rbf_nearest_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in set(test_time_step_vec)|set(training_time_step_vec):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        time_step_expression="{"+",".join(str(time_step) for time_step in test_time_step_vec)+"}"
        training_expression="{"+",".join(str(time_step) for time_step in training_time_step_vec)+"}"
        prediction_structure_path=temporary_dir/"nearest_ref10_candidate48.pred"
        for rel_eb in rel_eb_vec:
            config_path=temporary_dir/f"point_sets_{rel_eb:g}.json"
            write_point_set_config(config_path,rel_eb)
            common_arguments=["-data-dir",test_dir,"-variable","attr0","-time-step",time_step_expression,"-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",matrix_path,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",10,"-actual-ref-num",10,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",10,"-order-candidate-num-experimental",48,"-eb-schedule-experimental","round_point_ratio",0.8,"-cross-time-label-predictor-experimental","previous","-rbf-kernel","matern32","-rbf-shape",0.01,"-rbf-nugget",1e-6]
            compression_output=run_command([compressor,*common_arguments,"-reference-data-order-experimental",training_expression,"-reference-data-rbf-score-experimental","-rbf-filtered-fallback-nearest-experimental","-fused-two-point-set-prediction-experimental","-linear-previous-label-experimental"])
            decompression_output=run_command([decompressor,*common_arguments,"-rbf-filtered-fallback-nearest-experimental","-fused-two-point-set-prediction-experimental","-linear-previous-label-experimental"])
            psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
            decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
            if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
            result={"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
            result_vec.append(result)
            result_path.parent.mkdir(parents=True,exist_ok=True)
            with result_path.open("w",newline="") as file:
                writer=csv.DictWriter(file,fieldnames=result.keys(),lineterminator="\n")
                writer.writeheader()
                writer.writerows(result_vec)
            print(f"REL={rel_eb:g}:BR={result['br']:.6f},PSNR={result['psnr']:.6f},preprocessing={result['preprocessing_time']:.6f}s",flush=True)
            cleanup(test_dir)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
