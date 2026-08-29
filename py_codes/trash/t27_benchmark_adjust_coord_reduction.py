import os,re,subprocess,tempfile
import numpy as np
from third_party.csv_tools.py_codes.write_csv_item import write_csv_item

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina"
compressor=os.path.join(project_root,"install/bin/s1_compress")
history_dir=os.path.join(project_root,"adjust_coord_history")
history_config_vec=[
    ("second_0.10_files_3","weight_short_0_5.npz"),
    ("second_0.05_files_3","reduction_second_0_05_files_3.npz"),
    ("second_0.10_files_2","reduction_second_0_10_files_2.npz"),
    ("second_0.10_files_1","reduction_second_0_10_files_1.npz"),
]
time_step_num=3
target_psnr=59.868332
matched_psnr_search_num=7
csv_path=os.path.join(history_dir,"katrina_reduction_benchmark.csv")

def parse_result(output):
    match=re.search(r"total cr=\s*[^,]+,\s*total br=\s*([^,]+),\s*total psnr=\s*([^\s]+)",output)
    if match is None:
        raise RuntimeError("压缩输出中没有总BR和总PSNR")
    label_byte_vec=[int(value) for value in re.findall(r"label_zstd_bytes=([0-9]+)",output)]
    value_byte_vec=[int(value) for value in re.findall(r"value_huffman_bytes=\d+, value_zstd_bytes=(\d+),",output)]
    if len(label_byte_vec)!=time_step_num or len(value_byte_vec)!=time_step_num:
        raise RuntimeError(f"逐文件流长度数量错误：label={len(label_byte_vec)},value={len(value_byte_vec)}")
    return float(match.group(1)),float(match.group(2)),sum(label_byte_vec),sum(value_byte_vec)

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
    with tempfile.TemporaryDirectory(prefix="unsz_adjust_reduction_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for time_step in range(time_step_num):
            os.symlink(os.path.join(data_dir,f"attr0.dat.{time_step}"),os.path.join(test_dir,f"attr0.dat.{time_step}"))
        for config_name,history_name in history_config_vec:
            history=np.load(os.path.join(history_dir,history_name))
            for history_id,iteration in enumerate(history["iteration"]):
                if int(iteration)>4:
                    continue
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
                print(f"config={config_name},iteration={iteration},seconds={history['elapsed_time'][history_id]},pair_num={history['pair_num'][history_id]},nfev={history['nfev'][history_id]},matched_br={best_result[1]},matched_psnr={best_result[2]},label_bytes={best_result[3]},value_bytes={best_result[4]}",flush=True)

if __name__=="__main__":
    main()
