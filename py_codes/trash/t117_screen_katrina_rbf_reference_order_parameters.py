import csv,math,os,tempfile
from pathlib import Path
import numpy as np
import t116_screen_katrina_rbf_reference_order_nearest_fallback as benchmark

result_path=benchmark.project_root/"py_codes/trash/results/katrina_rbf_reference_order_parameter_screen.csv"
training_time_step_vec=(40,80,120)
test_time_step_vec=tuple(range(20))
rel_eb_vec=(1e-2,1e-3,1e-4)
ref_num_vec=(4,5,6,8,10)
shape_vec=(0.003,0.01,0.03)

def calculate_bd_rate(reference,test):
    reference_psnr=np.asarray([row["psnr"] for row in reference])
    reference_br=np.asarray([row["br"] for row in reference])
    test_psnr=np.asarray([row["psnr"] for row in test])
    test_br=np.asarray([row["br"] for row in test])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    reference_integral=np.polyint(np.polyfit(reference_psnr,np.log(reference_br),2))
    test_integral=np.polyint(np.polyfit(test_psnr,np.log(test_br),2))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (np.exp((test_area-reference_area)/(upper-lower))-1)*100

def write_results(result_vec):
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with result_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=("ref_num","shape","rel_eb","br","psnr","preprocessing_time","compression_time"),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_vec)

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_rbf_parameters_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(benchmark.data_dir/"coords.dat")
        for time_step in set(test_time_step_vec)|set(training_time_step_vec):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(benchmark.data_dir/f"attr0.dat.{time_step}")
        time_step_expression="{"+",".join(str(time_step) for time_step in test_time_step_vec)+"}"
        training_expression="{"+",".join(str(time_step) for time_step in training_time_step_vec)+"}"
        for ref_num in ref_num_vec:
            for shape in shape_vec:
                prediction_structure_path=temporary_dir/f"ref{ref_num}_shape{shape:g}.pred"
                current=[]
                for rel_eb in rel_eb_vec:
                    config_path=temporary_dir/f"point_sets_{ref_num}_{shape:g}_{rel_eb:g}.json"
                    benchmark.write_point_set_config(config_path,rel_eb)
                    arguments=["-data-dir",test_dir,"-variable","attr0","-time-step",time_step_expression,"-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",benchmark.matrix_path,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",ref_num,"-actual-ref-num",ref_num,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",ref_num,"-eb-schedule-experimental","round_point_ratio",0.8,"-cross-time-label-predictor-experimental","previous","-rbf-kernel","matern32","-rbf-shape",shape,"-rbf-nugget",1e-6,"-reference-data-order-experimental",training_expression,"-reference-data-rbf-score-experimental","-rbf-filtered-fallback-nearest-experimental","-fused-two-point-set-prediction-experimental","-linear-previous-label-experimental"]
                    output=benchmark.run_command([benchmark.compressor,*arguments])
                    result={"ref_num":ref_num,"shape":shape,"rel_eb":rel_eb,"br":benchmark.read_float(r"^total cr= [^,]+, total br= ([^,]+),",output,"总BR"),"psnr":benchmark.read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",output,"总PSNR"),"preprocessing_time":benchmark.read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",output,"预处理时间"),"compression_time":benchmark.read_float(r"^total compression time= ([^ ]+) s$",output,"压缩时间")}
                    result_vec.append(result)
                    current.append(result)
                    write_results(result_vec)
                    benchmark.cleanup(test_dir)
                br_text=",".join(f"{row['br']:.6f}" for row in current)
                print(f"ref={ref_num},shape={shape:g}:BR={br_text}",flush=True)
    reference=[row for row in result_vec if row["ref_num"]==8 and row["shape"]==0.01]
    for ref_num in ref_num_vec:
        for shape in shape_vec:
            current=[row for row in result_vec if row["ref_num"]==ref_num and row["shape"]==shape]
            print(f"ref={ref_num},shape={shape:g}相对ref=8,shape=0.01的BD-rate={calculate_bd_rate(reference,current):+.6f}%",flush=True)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
