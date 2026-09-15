import filecmp
import os
import re
import statistics
import subprocess
import tempfile

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean"
point_num=235160
time_step_num=31
repeat_num=5
total_mb=point_num*time_step_num*4/1e6
compressor=os.path.join(project_root,"install/bin/s1_compress")
decompressor=os.path.join(project_root,"install/bin/s1_decompress")
configs=[("base",[]),("reorder_data",["-reorder-data-by-prediction-order-experimental"]),("normalized",["-normalized-weight-experimental"]),("both",["-reorder-data-by-prediction-order-experimental","-normalized-weight-experimental"])]

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
    stage_matches=re.findall(r"^compression stage time: layout_reorder=([^ ]+) s, point_set=([^ ]+) s, label_prediction=([^ ]+) s, value_prediction=([^ ]+) s, lossless_encoding=([^ ]+) s, validation=([^ ]+) s$",output,re.MULTILINE)
    if len(stage_matches)!=time_step_num:
        raise RuntimeError(f"压缩阶段计时数量应为{time_step_num}，实际为{len(stage_matches)}")
    for stage_id,stage_name in enumerate(("layout_reorder","point_set","label_prediction","value_prediction","lossless_encoding","validation")):
        result[stage_name].append(sum(float(match[stage_id]) for match in stage_matches))

def append_decompression_result(result,output):
    result["decompression"].append(read_float(r"^total decompression time= ([^ ]+) s$",output,"总解压时间"))

def mean_std(values):
    return statistics.mean(values),statistics.stdev(values) if len(values)>1 else 0

def print_result(name,result):
    preprocessing_mean,preprocessing_std=mean_std(result["preprocessing"])
    compression_mean,compression_std=mean_std(result["compression"])
    decompression_mean,decompression_std=mean_std(result["decompression"])
    print(f"{name}: preprocessing={preprocessing_mean:.6f}±{preprocessing_std:.6f} s, compression={compression_mean:.6f}±{compression_std:.6f} s ({total_mb/compression_mean:.3f} MB/s), decompression={decompression_mean:.6f}±{decompression_std:.6f} s ({total_mb/decompression_mean:.3f} MB/s)")
    print("  compression stages: "+", ".join(f"{stage_name}={statistics.mean(result[stage_name]):.6f} s" for stage_name in ("layout_reorder","point_set","label_prediction","value_prediction","lossless_encoding","validation")))

def main():
    result_map={name:{key:[] for key in ("preprocessing","compression","decompression","layout_reorder","point_set","label_prediction","value_prediction","lossless_encoding","validation")} for name,_ in configs}
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_prediction_layout_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for name,_ in configs:
            for time_step in range(time_step_num):
                os.symlink(os.path.join(data_dir,f"temperature.dat.{time_step}"),os.path.join(test_dir,f"{name}.dat.{time_step}"))
        common_arguments=["-data-dir",test_dir,"-time-step","[0,30]","-M","REL","0.01"]
        for repeat_id in range(repeat_num):
            ordered_configs=configs[repeat_id%len(configs):]+configs[:repeat_id%len(configs)]
            print(f"repeat {repeat_id+1}/{repeat_num}: compression",flush=True)
            for name,flags in ordered_configs:
                output=run_command([compressor,"-variable",name,*common_arguments,"-compression-stage-time",*flags])
                append_compression_result(result_map[name],output)
            print(f"repeat {repeat_id+1}/{repeat_num}: decompression",flush=True)
            for name,flags in reversed(ordered_configs):
                output=run_command([decompressor,"-variable",name,*common_arguments,*flags])
                append_decompression_result(result_map[name],output)
        for time_step in range(time_step_num):
            base_compressed=os.path.join(test_dir,f"base.dat.{time_step}.unsz")
            base_decompressed=os.path.join(test_dir,f"base.dat.{time_step}.unsz.bin")
            for name,_ in configs[1:]:
                if not filecmp.cmp(base_compressed,os.path.join(test_dir,f"{name}.dat.{time_step}.unsz"),shallow=False):
                    raise RuntimeError(f"{name}的time step {time_step}压缩文件与base不同")
                if not filecmp.cmp(base_decompressed,os.path.join(test_dir,f"{name}.dat.{time_step}.unsz.bin"),shallow=False):
                    raise RuntimeError(f"{name}的time step {time_step}解压结果与base不同")
        print("全部31个time step的压缩文件和解压结果逐字节相同")
        for name,_ in configs:
            print_result(name,result_map[name])

if __name__=="__main__":
    main()
