import os,re,subprocess,tempfile
import numpy as np
from third_party.csv_tools.py_codes.write_csv_item import write_csv_item

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina"
compressor=os.path.join(project_root,"install/bin/s1_compress")
history_dir=os.path.join(project_root,"adjust_coord_history")
weight_vec=[0.3,0.5,0.7,1.0]
time_step_num=3
target_psnr=59.868332
matched_psnr_search_num=7
csv_path=os.path.join(history_dir,"katrina_weight_short_benchmark.csv")

def parse_result(output):
    match=re.search(r"total cr=\s*[^,]+,\s*total br=\s*([^,]+),\s*total psnr=\s*([^\s]+)",output)
    if match is None:
        raise RuntimeError("压缩输出中没有总BR和总PSNR")
    return float(match.group(1)),float(match.group(2))

def run_matrix(test_dir,matrix,rel_eb):
    parameter=matrix[np.tril_indices(3)]
    command=[compressor,"-data-dir",test_dir,"-variable","attr0","-time-step",f"[0,{time_step_num-1}]","-M","REL",str(rel_eb),"-adjust-coord-matrix",*[f"{value:.17g}" for value in parameter]]
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"压缩命令失败，退出码={result.returncode}")
    return parse_result(result.stdout)

def main():
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_adjust_weight_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for time_step in range(time_step_num):
            os.symlink(os.path.join(data_dir,f"attr0.dat.{time_step}"),os.path.join(test_dir,f"attr0.dat.{time_step}"))
        for weight in weight_vec:
            history=np.load(os.path.join(history_dir,f"weight_short_{str(weight).replace('.','_')}.npz"))
            for history_id,iteration in enumerate(history["iteration"]):
                row_info={"WEIGHT":weight,"ITERATION":int(iteration)}
                br,psnr=run_matrix(test_dir,history["matrix"][history_id],0.01)
                write_csv_item(csv_path,row_info,"BR",br)
                write_csv_item(csv_path,row_info,"PSNR",psnr)
                write_csv_item(csv_path,row_info,"TRAIN_SECONDS",float(history["elapsed_time"][history_id]))
                write_csv_item(csv_path,row_info,"NFEV",int(history["nfev"][history_id]))
                lower_rel_eb=0.005
                upper_rel_eb=0.012
                best_result=None
                for _ in range(matched_psnr_search_num):
                    rel_eb=(lower_rel_eb+upper_rel_eb)/2
                    matched_br,matched_psnr=run_matrix(test_dir,history["matrix"][history_id],rel_eb)
                    if best_result is None or abs(matched_psnr-target_psnr)<abs(best_result[2]-target_psnr):
                        best_result=(rel_eb,matched_br,matched_psnr)
                    if matched_psnr>target_psnr:
                        lower_rel_eb=rel_eb
                    else:
                        upper_rel_eb=rel_eb
                write_csv_item(csv_path,row_info,"MATCHED_REL_EB",best_result[0])
                write_csv_item(csv_path,row_info,"MATCHED_BR",best_result[1])
                write_csv_item(csv_path,row_info,"MATCHED_PSNR",best_result[2])
                print(f"weight={weight},iteration={iteration},seconds={history['elapsed_time'][history_id]},nfev={history['nfev'][history_id]},matched_br={best_result[1]},matched_psnr={best_result[2]}",flush=True)

if __name__=="__main__":
    main()
