import math,os,re
from third_party.csv_tools.py_codes.write_csv_item import write_csv_item
from third_party.get_max_cpus.py_codes.get_max_cpus import get_max_cpus
from third_party.run_command.run_command import run_command

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
umc_root="/home/x-zjian1/umc"
# unsz：每个点完成后立即加入可用参考点集合
# umc：UMC的NBP-IDW预测器
compressor_list=["unsz"]
dataset_list=[
    {"name":"Katrina-attr0","folder":"Katrina","variable":"attr0","time_step_num":3},
    # {"name":"Ocean-layerThickness","folder":"Ocean","variable":"layerThickness","time_step_num":31},
    # {"name":"Ocean-salinity","folder":"Ocean","variable":"salinity","time_step_num":31},
    # {"name":"Ocean-temperature","folder":"Ocean","variable":"temperature","time_step_num":31},
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
rel_eb_list=[1e-2]
build_max_cpus=get_max_cpus(8)
silent_build=False
build_before_test=True
write_to_csv=True
cleanup_after_each_test=True
continue_after_error=True

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

def parse_unsz_result(output):
    match=re.search(r"total cr=\s*([^,]+),\s*total br=\s*([^,]+),\s*total psnr=\s*([^\s]+)",output)
    if match is None:
        raise RuntimeError("没有在UNSZ输出中找到total br和total psnr")
    return float(match.group(2)),float(match.group(3))

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
    return total_br,total_psnr

def cleanup(data_paths):
    for data_path in data_paths:
        for suffix in [".unsz",".umc",".umc.out"]:
            output_path=data_path+suffix
            if os.path.isfile(output_path):
                os.remove(output_path)

def write_result(dataset,rel_eb,compressor,br,psnr):
    csv_path=os.path.join(data_root,dataset["folder"],f"{dataset['variable']}.csv")
    row_info={"REL_EB":rel_eb,"TIME_STEP_NUM":dataset["time_step_num"]}
    method_name=compressor.upper()
    write_csv_item(csv_path,row_info,f"{method_name}_BR",br)
    write_csv_item(csv_path,row_info,f"{method_name}_PSNR",psnr)

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
        command=f"{project_root}/install/bin/s1_compress {data_dir} {variable} {time_step_num} {rel_eb}"
        output=run_command(command)
        br,psnr=parse_unsz_result(output)
    elif compressor=="umc":
        command=f"{umc_root}/build/test/test_compress {data_dir}/ {variable} coords.dat conn.dat 0 3 2 1 {time_step_num} 0 {rel_eb} 10"
        output=run_command(command)
        br,psnr=parse_umc_result(output,data_paths)
    else:
        raise ValueError(f"未知压缩器：{compressor}")
    print(f"总结果：compressor={compressor}, dataset={dataset['name']}, variable={variable}, rel_eb={rel_eb}, br={br}, psnr={psnr}")
    return br,psnr

def main():
    build_unsz()
    for dataset in dataset_list:
        for rel_eb in rel_eb_list:
            for compressor in compressor_list:
                print(f"\n开始测试：compressor={compressor}, dataset={dataset['name']}, variable={dataset['variable']}, rel_eb={rel_eb}")
                data_dir=os.path.join(data_root,dataset["folder"])
                data_paths=[os.path.join(data_dir,f"{dataset['variable']}.dat.{time_step}") for time_step in range(dataset["time_step_num"])]
                try:
                    br,psnr=test(compressor,dataset,rel_eb)
                    if write_to_csv:
                        write_result(dataset,rel_eb,compressor,br,psnr)
                except Exception as error:
                    print(f"failed: {error}")
                    if not continue_after_error:
                        raise
                finally:
                    if cleanup_after_each_test:
                        cleanup(data_paths)

if __name__=="__main__":
    main()
