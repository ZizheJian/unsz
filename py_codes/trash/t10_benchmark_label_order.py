import filecmp
import os
import re
import statistics
import subprocess
import tempfile

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina"
point_num=417642
time_step_num=162
repeat_num=5
total_mb=point_num*time_step_num*4/1e6
compressor=os.path.join(project_root,"install/bin/s1_compress")
decompressor=os.path.join(project_root,"install/bin/s1_decompress")
configs=[("base",[]),("prediction_label",["-prediction-order-label-experimental"]),("reorder_data",["-reorder-data-by-prediction-order-experimental"]),("reorder_data_prediction_label",["-reorder-data-by-prediction-order-experimental","-prediction-order-label-experimental"])]

def run_command(command):
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(command)}")
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if not match:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def append_compression_result(result,output):
    result["preprocessing"].append(read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",output,"预处理时间"))
    result["compression"].append(read_float(r"^total compression time= ([^ ]+) s$",output,"总压缩时间"))
    label_bytes=[int(value) for value in re.findall(r"label_zstd_bytes=([0-9]+)",output)]
    file_bytes=[int(value) for value in re.findall(r"file_bytes=([0-9]+)",output)]
    if len(label_bytes)!=time_step_num or len(file_bytes)!=time_step_num:
        raise RuntimeError(f"逐文件字节数数量错误：label={len(label_bytes)}，file={len(file_bytes)}")
    result["label_bytes"].append(sum(label_bytes))
    result["file_bytes"].append(sum(file_bytes))

def append_decompression_result(result,output):
    result["decompression"].append(read_float(r"^total decompression time= ([^ ]+) s$",output,"总解压时间"))

def mean_std(values):
    return statistics.mean(values),statistics.stdev(values) if len(values)>1 else 0

def print_result(name,result):
    preprocessing_mean,preprocessing_std=mean_std(result["preprocessing"])
    compression_mean,compression_std=mean_std(result["compression"])
    decompression_mean,decompression_std=mean_std(result["decompression"])
    label_bytes=round(statistics.mean(result["label_bytes"]))
    file_bytes=round(statistics.mean(result["file_bytes"]))
    br=file_bytes*8/(point_num*time_step_num)
    print(f"{name}: label_bytes={label_bytes}, file_bytes={file_bytes}, br={br:.9f}, preprocessing={preprocessing_mean:.6f}±{preprocessing_std:.6f} s, compression={compression_mean:.6f}±{compression_std:.6f} s ({total_mb/compression_mean:.3f} MB/s), decompression={decompression_mean:.6f}±{decompression_std:.6f} s ({total_mb/decompression_mean:.3f} MB/s)")

def main():
    result_map={name:{key:[] for key in ("preprocessing","compression","decompression","label_bytes","file_bytes")} for name,_ in configs}
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_label_order_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for name,_ in configs:
            for time_step in range(time_step_num):
                os.symlink(os.path.join(data_dir,f"attr0.dat.{time_step}"),os.path.join(test_dir,f"{name}.dat.{time_step}"))
        common_arguments=["-data-dir",test_dir,"-time-step","[0,161]","-M","REL","0.01","-adjust-coord"]
        for repeat_id in range(repeat_num):
            ordered_configs=configs[repeat_id%len(configs):]+configs[:repeat_id%len(configs)]
            print(f"repeat {repeat_id+1}/{repeat_num}: compression",flush=True)
            for name,flags in ordered_configs:
                output=run_command([compressor,"-variable",name,*common_arguments,*flags])
                append_compression_result(result_map[name],output)
            print(f"repeat {repeat_id+1}/{repeat_num}: decompression",flush=True)
            for name,flags in reversed(ordered_configs):
                output=run_command([decompressor,"-variable",name,*common_arguments,*flags])
                append_decompression_result(result_map[name],output)
        for time_step in range(time_step_num):
            base_decompressed=os.path.join(test_dir,f"base.dat.{time_step}.unsz.bin")
            for name,_ in configs[1:]:
                if not filecmp.cmp(base_decompressed,os.path.join(test_dir,f"{name}.dat.{time_step}.unsz.bin"),shallow=False):
                    raise RuntimeError(f"{name}的time step {time_step}解压结果与base不同")
            if not filecmp.cmp(os.path.join(test_dir,f"base.dat.{time_step}.unsz"),os.path.join(test_dir,f"layout.dat.{time_step}.unsz"),shallow=False):
                raise RuntimeError(f"layout的time step {time_step}码流与base不同")
            if not filecmp.cmp(os.path.join(test_dir,f"prediction_label.dat.{time_step}.unsz"),os.path.join(test_dir,f"layout_prediction_label.dat.{time_step}.unsz"),shallow=False):
                raise RuntimeError(f"layout_prediction_label的time step {time_step}码流与prediction_label不同")
        print("全部162个time step的解压结果逐字节相同；layout不改变对应标签顺序的码流")
        for name,_ in configs:
            print_result(name,result_map[name])

if __name__=="__main__":
    main()
