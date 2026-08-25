import math,os,re,shlex
from third_party.csv_tools.py_codes.write_csv_item import write_csv_item
from third_party.get_max_cpus.py_codes.get_max_cpus import get_max_cpus
from third_party.run_command.run_command import run_command

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
umc_root="/home/x-zjian1/umc"
# unsz：每个点完成后立即加入可用参考点集合
# umc：UMC的NBP-IDW预测器
compressor_list=["unsz","umc"]
dataset_list=[
    {"name":"Katrina-attr0","folder":"Katrina","variable":"attr0","time_step_num":162,"adjust_coord":True},
    # {"name":"Ocean-layerThickness","folder":"Ocean","variable":"layerThickness","time_step_num":31},
    # {"name":"Ocean-salinity","folder":"Ocean","variable":"salinity","time_step_num":31},
    {"name":"Ocean-temperature","folder":"Ocean","variable":"temperature","time_step_num":31,"adjust_coord":False},
    # {"name":"Ocean-velocityMeridional","folder":"Ocean","variable":"velocityMeridional","time_step_num":31},
    # {"name":"Ocean-velocityX","folder":"Ocean","variable":"velocityX","time_step_num":31},
    # {"name":"Ocean-velocityY","folder":"Ocean","variable":"velocityY","time_step_num":31},
    # {"name":"Ocean-velocityZ","folder":"Ocean","variable":"velocityZ","time_step_num":31},
    # {"name":"Ocean-velocityZonal","folder":"Ocean","variable":"velocityZonal","time_step_num":31},
    # {"name":"Ocean-zTop","folder":"Ocean","variable":"zTop","time_step_num":31},
    # {"name":"LES-velocity_x","folder":"LES","variable":"velocity_x","time_step_num":1},
    # {"name":"LES-velocity_y","folder":"LES","variable":"velocity_y","time_step_num":1},
    # {"name":"LES-velocity_z","folder":"LES","variable":"velocity_z","time_step_num":1},
    # {"name":"Tsunami-momentum_x","folder":"Tsunami","variable":"momentum_x","time_step_num":1},
    # {"name":"Tsunami-momentum_y","folder":"Tsunami","variable":"momentum_y","time_step_num":1},
    # {"name":"Tsunami-velocity_x","folder":"Tsunami","variable":"velocity_x","time_step_num":1},
    # {"name":"Tsunami-velocity_y","folder":"Tsunami","variable":"velocity_y","time_step_num":1},
]
rel_eb_list=[1e-2,1e-3,1e-4]
build_max_cpus=get_max_cpus(8)
silent_build=False
build_before_test=True
write_to_csv=True
cleanup_after_each_test=True
continue_after_error=True
verify_unsz_decompression=True

def build_unsz():
    unsz_path=os.path.join(project_root,"install/bin/s1_compress")
    use_unsz=any(compressor.startswith("unsz") for compressor in compressor_list)
    if build_before_test and use_unsz:
        os.makedirs(os.path.join(project_root,"build"),exist_ok=True)
        run_command(f"cmake -S {project_root}/c_codes -B {project_root}/build -DCMAKE_INSTALL_PREFIX={project_root}/install -DCODE_NAME=s1_compress",silent=silent_build)
        run_command(f"cmake --build {project_root}/build --parallel {build_max_cpus}",silent=silent_build)
        run_command(f"cmake --install {project_root}/build",silent=silent_build)
    if use_unsz and not os.path.isfile(unsz_path):
        raise FileNotFoundError(f"没有找到UNSZ可执行文件：{unsz_path}")
    if "umc" in compressor_list and not os.path.isfile(os.path.join(umc_root,"build/test/test_compress")):
        raise FileNotFoundError("没有找到UMC可执行文件，请先按照UMC自己的构建流程生成/home/x-zjian1/umc/build/test/test_compress")

def calculate_throughput(data_paths,seconds,name):
    if not seconds>0:
        raise RuntimeError(f"{name}必须大于0：{seconds}")
    return sum(os.path.getsize(path) for path in data_paths)/seconds/1e6

def parse_unsz_result(output,data_paths):
    match=re.search(r"total cr=\s*([^,]+),\s*total br=\s*([^,]+),\s*total psnr=\s*([^\s]+)",output)
    if match is None:
        raise RuntimeError("没有在UNSZ输出中找到total br和total psnr")
    preprocessing_match=re.search(r"preprocessing time excluding adjust coord=\s*([^\s]+)\s*s",output)
    compression_match=re.search(r"total compression time=\s*([^\s]+)\s*s",output)
    if preprocessing_match is None or compression_match is None:
        raise RuntimeError("没有在UNSZ输出中找到预处理时间或总压缩时间")
    preprocessing_time=float(preprocessing_match.group(1))
    compression_time=float(compression_match.group(1))
    preprocessing_throughput=calculate_throughput(data_paths,preprocessing_time,"UNSZ预处理时间")
    compression_throughput=calculate_throughput(data_paths,compression_time,"UNSZ压缩时间")
    return float(match.group(2)),float(match.group(3)),preprocessing_time,compression_time,preprocessing_throughput,compression_throughput

def parse_unsz_decompression_result(output):
    match=re.search(r"total decompressed psnr=\s*([^\s]+)",output)
    if match is None:
        raise RuntimeError("没有在UNSZ解压输出中找到total decompressed psnr")
    return float(match.group(1))

def parse_umc_result(output,data_paths):
    compressed_sizes=[int(value) for value in re.findall(r"compressed_size\s*=\s*(\d+)",output)]
    mse_list=[float(value) for value in re.findall(r"^MSE\s*=\s*([^\s]+)",output,re.MULTILINE)]
    range_matches=re.findall(r"Max value\s*=\s*([^,]+),\s*min value\s*=\s*([^,]+),",output)
    if len(compressed_sizes)!=len(data_paths) or len(mse_list)!=len(data_paths) or len(range_matches)!=len(data_paths):
        raise RuntimeError(f"UMC输出数量不一致：文件{len(data_paths)}，压缩大小{len(compressed_sizes)}，MSE {len(mse_list)}，范围{len(range_matches)}")
    total_original_bytes=sum(os.path.getsize(path) for path in data_paths)
    total_compressed_bytes=sum(compressed_sizes)
    total_point_num=sum(os.path.getsize(path)//4 for path in data_paths)
    total_squared_error=sum(mse*(os.path.getsize(path)//4) for mse,path in zip(mse_list,data_paths))
    total_min_data=min(float(minimum) for _,minimum in range_matches)
    total_max_data=max(float(maximum) for maximum,_ in range_matches)
    total_mse=total_squared_error/total_point_num
    total_cr=total_original_bytes/total_compressed_bytes
    total_br=32/total_cr
    total_psnr=10*math.log10((total_max_data-total_min_data)**2/total_mse)
    preprocessing_times=[float(value) for value in re.findall(r"^preprocessing time\s*=\s*([^\s]+)",output,re.MULTILINE)]
    reorder_times=[float(value) for value in re.findall(r"^reorder time\s*=\s*([^\s]+)",output,re.MULTILINE)]
    compression_times=[float(value) for value in re.findall(r"^compression time\s*=\s*([^\s]+)",output,re.MULTILINE)]
    if len(preprocessing_times)!=1:
        raise RuntimeError(f"UMC预处理时间数量应为1，实际为{len(preprocessing_times)}")
    if len(reorder_times)!=len(data_paths) or len(compression_times)!=len(data_paths):
        raise RuntimeError(f"UMC逐文件计时数量不一致：文件{len(data_paths)}，reorder {len(reorder_times)}，compression {len(compression_times)}")
    core_compression_time=sum(compression_time-reorder_time for compression_time,reorder_time in zip(compression_times,reorder_times))
    preprocessing_time=preprocessing_times[0]
    preprocessing_throughput=calculate_throughput(data_paths,preprocessing_time,"UMC预处理时间")
    compression_throughput=calculate_throughput(data_paths,core_compression_time,"UMC核心压缩时间")
    return total_br,total_psnr,preprocessing_time,core_compression_time,preprocessing_throughput,compression_throughput

def cleanup(data_paths):
    for data_path in data_paths:
        for suffix in [".unsz",".unsz.bin",".umc",".umc.out"]:
            output_path=data_path+suffix
            if os.path.isfile(output_path):
                os.remove(output_path)

def write_result(dataset,rel_eb,compressor,br,psnr,preprocessing_throughput,compression_throughput):
    csv_path=os.path.join(data_root,dataset["folder"],f"{dataset['variable']}.csv")
    row_info={"REL_EB":rel_eb,"TIME_STEP_NUM":dataset["time_step_num"]}
    method_name=compressor.upper()
    write_csv_item(csv_path,row_info,f"{method_name}_BR",br)
    write_csv_item(csv_path,row_info,f"{method_name}_PSNR",psnr)
    write_csv_item(csv_path,row_info,f"{method_name}_PREPROCESSING_MB_S",preprocessing_throughput)
    write_csv_item(csv_path,row_info,f"{method_name}_COMPRESSION_MB_S",compression_throughput)

def test(compressor,dataset,rel_eb):
    data_dir=os.path.join(data_root,dataset["folder"])
    variable=dataset["variable"]
    time_step_num=dataset["time_step_num"]
    data_paths=[os.path.join(data_dir,f"{variable}.dat.{time_step}") for time_step in range(time_step_num)]
    required_paths=[os.path.join(data_dir,"coords.dat"),*data_paths]
    if compressor=="umc":
        required_paths.append(os.path.join(data_dir,"conn.dat"))
    missing_paths=[path for path in required_paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError("缺少测试文件："+", ".join(missing_paths))
    if compressor=="unsz":
        time_step_expression=f"[0,{time_step_num-1}]"
        common_arguments=f"-data-dir {shlex.quote(data_dir)} -variable {shlex.quote(variable)} -time-step {shlex.quote(time_step_expression)} -M REL {rel_eb}"
        if dataset.get("adjust_coord",False):
            common_arguments+=" -adjust-coord"
        compress_command=f"{shlex.quote(project_root+'/install/bin/s1_compress')} {common_arguments}"
        compress_output=run_command(compress_command)
        br,psnr,preprocessing_time,compression_time,preprocessing_throughput,compression_throughput=parse_unsz_result(compress_output,data_paths)
        if verify_unsz_decompression:
            decompress_command=f"{shlex.quote(project_root+'/install/bin/s1_decompress')} {common_arguments}"
            decompress_output=run_command(decompress_command)
            decompressed_psnr=parse_unsz_decompression_result(decompress_output)
            if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                raise RuntimeError(f"压缩端和解压端的总PSNR不一致：{psnr} != {decompressed_psnr}")
            print(f"磁盘压缩—解压检查通过：total psnr={decompressed_psnr}")
    elif compressor=="umc":
        command=f"{umc_root}/build/test/test_compress {data_dir}/ {variable} coords.dat conn.dat 0 3 2 1 {time_step_num} 0 {rel_eb} 10"
        output=run_command(command)
        br,psnr,preprocessing_time,compression_time,preprocessing_throughput,compression_throughput=parse_umc_result(output,data_paths)
    else:
        raise ValueError(f"未知压缩器：{compressor}")
    print(f"总结果：compressor={compressor}, dataset={dataset['name']}, variable={variable}, rel_eb={rel_eb}, br={br}, psnr={psnr}")
    print(f"计时：preprocessing={preprocessing_time} s, compression={compression_time} s")
    print(f"吞吐率：preprocessing={preprocessing_throughput} MB/s, compression={compression_throughput} MB/s")
    return br,psnr,preprocessing_throughput,compression_throughput

def main():
    build_unsz()
    for dataset in dataset_list:
        for rel_eb in rel_eb_list:
            for compressor in compressor_list:
                print(f"\n开始测试：compressor={compressor}, dataset={dataset['name']}, variable={dataset['variable']}, rel_eb={rel_eb}")
                data_dir=os.path.join(data_root,dataset["folder"])
                data_paths=[os.path.join(data_dir,f"{dataset['variable']}.dat.{time_step}") for time_step in range(dataset["time_step_num"])]
                try:
                    br,psnr,preprocessing_throughput,compression_throughput=test(compressor,dataset,rel_eb)
                    if write_to_csv:
                        write_result(dataset,rel_eb,compressor,br,psnr,preprocessing_throughput,compression_throughput)
                except Exception as error:
                    print(f"failed: {error}")
                    if not continue_after_error:
                        raise
                finally:
                    if cleanup_after_each_test:
                        cleanup(data_paths)

if __name__=="__main__":
    main()
