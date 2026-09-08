import csv,os,tempfile
from pathlib import Path
import t117_screen_katrina_rbf_reference_order_parameters as benchmark

result_path=benchmark.benchmark.project_root/"py_codes/trash/results/katrina_reference_order_training_num_screen.csv"
test_time_step_vec=tuple(range(20))
rel_eb_vec=(1e-2,1e-3,1e-4)
training_time_step_vec_vec=((81,),(40,80,120),(27,54,81,108,135),(24,40,56,72,88,104,120,136,152))

def write_results(result_vec):
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with result_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=("training_num","training_time_steps","rel_eb","br","psnr","preprocessing_time","compression_time"),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_vec)

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_training_num_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(benchmark.benchmark.data_dir/"coords.dat")
        all_training_time_steps=set().union(*training_time_step_vec_vec)
        for time_step in set(test_time_step_vec)|all_training_time_steps:
            (test_dir/f"attr0.dat.{time_step}").symlink_to(benchmark.benchmark.data_dir/f"attr0.dat.{time_step}")
        time_step_expression="{"+",".join(str(time_step) for time_step in test_time_step_vec)+"}"
        for training_time_step_vec in training_time_step_vec_vec:
            training_num=len(training_time_step_vec)
            training_text=",".join(str(time_step) for time_step in training_time_step_vec)
            training_expression="{"+training_text+"}"
            prediction_structure_path=temporary_dir/f"training{training_num}.pred"
            current=[]
            for rel_eb in rel_eb_vec:
                config_path=temporary_dir/f"point_sets_{training_num}_{rel_eb:g}.json"
                benchmark.benchmark.write_point_set_config(config_path,rel_eb)
                arguments=["-data-dir",test_dir,"-variable","attr0","-time-step",time_step_expression,"-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",benchmark.benchmark.matrix_path,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",10,"-actual-ref-num",10,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",10,"-order-candidate-num-experimental",48,"-eb-schedule-experimental","round_point_ratio",0.8,"-cross-time-label-predictor-experimental","previous","-rbf-kernel","matern32","-rbf-shape",0.01,"-rbf-nugget",1e-6,"-reference-data-order-experimental",training_expression,"-reference-data-rbf-score-experimental","-rbf-filtered-fallback-nearest-experimental","-fused-two-point-set-prediction-experimental","-linear-previous-label-experimental"]
                output=benchmark.benchmark.run_command([benchmark.benchmark.compressor,*arguments])
                result={"training_num":training_num,"training_time_steps":training_text,"rel_eb":rel_eb,"br":benchmark.benchmark.read_float(r"^total cr= [^,]+, total br= ([^,]+),",output,"总BR"),"psnr":benchmark.benchmark.read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",output,"总PSNR"),"preprocessing_time":benchmark.benchmark.read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",output,"预处理时间"),"compression_time":benchmark.benchmark.read_float(r"^total compression time= ([^ ]+) s$",output,"压缩时间")}
                result_vec.append(result)
                current.append(result)
                write_results(result_vec)
                benchmark.benchmark.cleanup(test_dir)
            reference=[row for row in result_vec if row["training_num"]==3]
            if reference:
                print(f"training={training_num}:BD-rate={benchmark.calculate_bd_rate(reference,current):+.6f}%",flush=True)
            else:
                print(f"training={training_num}完成",flush=True)
    reference=[row for row in result_vec if row["training_num"]==3]
    for training_time_step_vec in training_time_step_vec_vec:
        training_num=len(training_time_step_vec)
        current=[row for row in result_vec if row["training_num"]==training_num]
        print(f"training={training_num}相对3的BD-rate={benchmark.calculate_bd_rate(reference,current):+.6f}%",flush=True)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
