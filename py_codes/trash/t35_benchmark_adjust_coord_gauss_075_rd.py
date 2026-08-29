import csv,os,tempfile
from t14_benchmark_eb_rd import calculate_bd_rate
from t29_benchmark_adjust_coord_rd import data_dir,time_step_num,rel_eb_vec,csv_path,history_dir,load_history_matrix,benchmark_matrix

config_name="second_gauss_075_4"
matrix=load_history_matrix(os.path.join(history_dir,"second_gauss_second_0_075.npz"),4)

def read_result_map():
    result_map={}
    with open(csv_path,newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            result_map.setdefault(row["CONFIG"],[]).append({"br":float(row["BR"]),"psnr":float(row["PSNR"])})
    return result_map

def main():
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_gauss_075_rd_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for time_step in range(time_step_num):
            os.symlink(os.path.join(data_dir,f"attr0.dat.{time_step}"),os.path.join(test_dir,f"attr0.dat.{time_step}"))
        for rel_eb in rel_eb_vec:
            benchmark_matrix(test_dir,config_name,matrix,rel_eb)
    result_map=read_result_map()
    for base_name in ("fourth_power_100","second_effect","second_gauss_newton_4","prediction_ggd_bit_4"):
        bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map[base_name],result_map[config_name])
        print(f"BD-rate({config_name} vs {base_name})={bd_rate:.6f}%,overlap_psnr=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)

if __name__=="__main__":
    main()
