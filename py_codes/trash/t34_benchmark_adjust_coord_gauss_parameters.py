import os,tempfile
import numpy as np
from third_party.csv_tools.py_codes.write_csv_item import write_csv_item
from t27_benchmark_adjust_coord_reduction import data_dir,history_dir,run_matrix,time_step_num,target_psnr,matched_psnr_search_num

history_config_vec=[
    ("gauss_second_0.075","second_gauss_second_0_075.npz"),
    ("gauss_weight_0.6","second_gauss_weight_0_6.npz"),
]
csv_path=os.path.join(history_dir,"katrina_gauss_parameter_benchmark.csv")

def main():
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_gauss_parameter_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for time_step in range(time_step_num):
            os.symlink(os.path.join(data_dir,f"attr0.dat.{time_step}"),os.path.join(test_dir,f"attr0.dat.{time_step}"))
        for config_name,history_name in history_config_vec:
            history=np.load(os.path.join(history_dir,history_name))
            for history_id,iteration in enumerate(history["iteration"]):
                row_info={"CONFIG":config_name,"ITERATION":int(iteration)}
                br,psnr,_,_=run_matrix(test_dir,history["matrix"][history_id],0.01)
                write_csv_item(csv_path,row_info,"BR",br)
                write_csv_item(csv_path,row_info,"PSNR",psnr)
                write_csv_item(csv_path,row_info,"TRAIN_SECONDS",float(history["elapsed_time"][history_id]))
                write_csv_item(csv_path,row_info,"PAIR_NUM",int(history["pair_num"][history_id]))
                write_csv_item(csv_path,row_info,"NFEV",int(history["nfev"][history_id]))
                lower_rel_eb=0.005
                upper_rel_eb=0.012
                best_result=None
                for _ in range(matched_psnr_search_num):
                    rel_eb=(lower_rel_eb+upper_rel_eb)/2
                    matched_br,matched_psnr,label_bytes,value_bytes=run_matrix(test_dir,history["matrix"][history_id],rel_eb)
                    if best_result is None or abs(matched_psnr-target_psnr)<abs(best_result[2]-target_psnr):
                        best_result=(rel_eb,matched_br,matched_psnr,label_bytes,value_bytes)
                    if matched_psnr>target_psnr:
                        lower_rel_eb=rel_eb
                    else:
                        upper_rel_eb=rel_eb
                write_csv_item(csv_path,row_info,"MATCHED_REL_EB",best_result[0])
                write_csv_item(csv_path,row_info,"MATCHED_BR",best_result[1])
                write_csv_item(csv_path,row_info,"MATCHED_PSNR",best_result[2])
                write_csv_item(csv_path,row_info,"MATCHED_LABEL_ZSTD_BYTES",best_result[3])
                write_csv_item(csv_path,row_info,"MATCHED_VALUE_ZSTD_BYTES",best_result[4])
                print(f"config={config_name},iteration={iteration},seconds={history['elapsed_time'][history_id]},nfev={history['nfev'][history_id]},loss={history['loss'][history_id]},matched_br={best_result[1]},matched_psnr={best_result[2]},label_bytes={best_result[3]},value_bytes={best_result[4]}",flush=True)

if __name__=="__main__":
    main()
