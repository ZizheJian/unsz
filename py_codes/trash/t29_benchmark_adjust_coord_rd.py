import math,os,re,subprocess,tempfile
import numpy as np
from third_party.csv_tools.py_codes.write_csv_item import write_csv_item
from t14_benchmark_eb_rd import calculate_bd_rate

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina"
compressor=os.path.join(project_root,"install/bin/s1_compress")
decompressor=os.path.join(project_root,"install/bin/s1_decompress")
history_dir=os.path.join(project_root,"adjust_coord_history")
time_step_num=162
rel_eb_vec=(0.02,0.01,0.005,0.0025,0.00125)
csv_path=os.path.join(history_dir,"katrina_adjust_coord_rd.csv")

def load_history_matrix(path,iteration):
    history=np.load(path)
    history_id=np.flatnonzero(history["iteration"]==iteration)
    if history_id.size!=1:
        raise RuntimeError(f"{path}中没有唯一的第{iteration}轮矩阵")
    return history["matrix"][history_id[0]]

matrix_config_vec=[
    ("fourth_power_100",np.array([[1.12670515,0,0],[0.270622568,1.28736120,0],[-1.67488548e-4,-1.83679099e-4,3.29459469e-6]],dtype=np.float64)),
    ("second_effect",load_history_matrix(os.path.join(history_dir,"weight_short_0_5.npz"),4)),
    ("second_speed",load_history_matrix(os.path.join(history_dir,"reduction_second_0_05_files_3.npz"),4)),
]

def run_command(command):
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败，退出码={result.returncode}")
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def cleanup(test_dir):
    for time_step in range(time_step_num):
        for suffix in (".unsz",".unsz.bin"):
            path=os.path.join(test_dir,f"attr0.dat.{time_step}{suffix}")
            if os.path.isfile(path):
                os.remove(path)

def benchmark_matrix(test_dir,config_name,matrix,rel_eb):
    parameter=matrix[np.tril_indices(3)]
    common_argument=["-data-dir",test_dir,"-variable","attr0","-time-step",f"[0,{time_step_num-1}]","-M","REL",str(rel_eb),"-adjust-coord-matrix",*[f"{value:.17g}" for value in parameter],"-reverse-mean-distance-only-order-experimental"]
    compression_output=run_command([compressor,*common_argument])
    br=read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR")
    psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR")
    label_bytes=sum(int(value) for value in re.findall(r"label_zstd_bytes=([0-9]+)",compression_output))
    value_bytes=sum(int(value) for value in re.findall(r"value_huffman_bytes=\d+, value_zstd_bytes=(\d+),",compression_output))
    if len(re.findall(r"label_zstd_bytes=([0-9]+)",compression_output))!=time_step_num or len(re.findall(r"value_huffman_bytes=\d+, value_zstd_bytes=(\d+),",compression_output))!=time_step_num:
        raise RuntimeError("逐文件标签流或数值流长度数量不正确")
    decompression_output=run_command([decompressor,*common_argument])
    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
    cleanup(test_dir)
    row_info={"CONFIG":config_name,"REL_EB":rel_eb}
    write_csv_item(csv_path,row_info,"BR",br)
    write_csv_item(csv_path,row_info,"PSNR",psnr)
    write_csv_item(csv_path,row_info,"LABEL_ZSTD_BYTES",label_bytes)
    write_csv_item(csv_path,row_info,"VALUE_ZSTD_BYTES",value_bytes)
    print(f"config={config_name},rel_eb={rel_eb},br={br},psnr={psnr},label_bytes={label_bytes},value_bytes={value_bytes}",flush=True)
    return {"br":br,"psnr":psnr}

def main():
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_adjust_rd_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for time_step in range(time_step_num):
            os.symlink(os.path.join(data_dir,f"attr0.dat.{time_step}"),os.path.join(test_dir,f"attr0.dat.{time_step}"))
        result_map={}
        for config_name,matrix in matrix_config_vec:
            result_map[config_name]=[benchmark_matrix(test_dir,config_name,matrix,rel_eb) for rel_eb in rel_eb_vec]
        base_name=matrix_config_vec[0][0]
        for config_name,_ in matrix_config_vec[1:]:
            bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map[base_name],result_map[config_name])
            print(f"BD-rate({config_name} vs {base_name})={bd_rate:.6f}%,overlap_psnr=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)

if __name__=="__main__":
    main()
