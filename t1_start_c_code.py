import argparse,array,copy,csv,json,math,os,re,shlex,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from third_party.get_max_cpus.py_codes.get_max_cpus import get_max_cpus
from third_party.run_command.run_command import run_command as run_shell_command

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
umc_root="/home/x-zjian1/umc"
config_directory=os.path.join(project_root,"configs")
def resolve_json_path(path_or_name):
    path_or_name=os.path.expanduser(path_or_name)
    if os.path.isabs(path_or_name) or os.path.dirname(path_or_name):
        json_path=os.path.abspath(path_or_name)
    else:
        config_filename=path_or_name if path_or_name.endswith(".json") else path_or_name+".json"
        json_path=os.path.join(config_directory,config_filename)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"找不到JSON配置：{json_path}")
    return json_path

def load_json_tasks(json_path):
    with open(json_path,encoding="utf-8") as config_file:
        tasks=json.load(config_file)
    if not isinstance(tasks,list):
        raise TypeError(f"JSON顶层必须是task list：{json_path}")
    if not tasks:
        raise ValueError(f"JSON中没有task：{json_path}")
    required_keys=("name","dataset","folder","field_list","rel_eb_list","method_list","steps","dim","config")
    task_names=set()
    for task_index,task in enumerate(tasks):
        if not isinstance(task,dict):
            raise TypeError(f"第{task_index}个task必须是object")
        missing_keys=[key for key in required_keys if key not in task]
        if missing_keys:
            raise ValueError(f"第{task_index}个task缺少参数：{','.join(missing_keys)}")
        for key in ("name","dataset","folder","config"):
            if not isinstance(task[key],str) or not task[key]:
                raise ValueError(f"第{task_index}个task的{key}必须是非空字符串")
        if task["name"] in task_names:
            raise ValueError(f"task name重复：{task['name']}")
        task_names.add(task["name"])
        for key in ("field_list","rel_eb_list","method_list"):
            if not isinstance(task[key],list) or not task[key]:
                raise TypeError(f"第{task_index}个task的{key}必须是非空list")
        if any(not isinstance(field,str) or not field for field in task["field_list"]):
            raise TypeError(f"第{task_index}个task的field_list只能包含非空字符串")
        if len(set(task["field_list"]))!=len(task["field_list"]):
            raise ValueError(f"第{task_index}个task的field_list包含重复field")
        if any(not isinstance(rel_eb,(int,float)) or rel_eb<=0 for rel_eb in task["rel_eb_list"]):
            raise TypeError(f"第{task_index}个task的rel_eb_list只能包含正数")
        if any(not isinstance(method,str) or not method for method in task["method_list"]):
            raise TypeError(f"第{task_index}个task的method_list只能包含非空字符串")
        if len(set(task["method_list"]))!=len(task["method_list"]):
            raise ValueError(f"第{task_index}个task的method_list包含重复method")
        if "umc" in task["method_list"] and "umc" not in task:
            raise ValueError(f"第{task_index}个task启用了umc但没有umc配置")
        if not isinstance(task["steps"],int) or task["steps"]<=0:
            raise ValueError(f"第{task_index}个task的steps必须是正整数")
        if not isinstance(task["dim"],int) or task["dim"]<=0:
            raise ValueError(f"第{task_index}个task的dim必须是正整数")
        if "key" in task and len(task["field_list"])!=1:
            raise ValueError(f"第{task_index}个task只有包含一个field时才能指定key")
    return tasks

def expand_json_tasks(tasks):
    field_configs=[]
    for task in tasks:
        for field in task["field_list"]:
            config=copy.deepcopy(task)
            config["task_name"]=config.pop("name")
            config.pop("field_list")
            config["variable"]=field
            if "key" not in config:
                dataset_key=re.sub(r"[^0-9A-Za-z]+","_",config["dataset"]).strip("_").lower()
                config["key"]=f"{dataset_key}_{field}"
            field_configs.append(config)
    key_set=set()
    for config in field_configs:
        if config["key"] in key_set:
            raise ValueError(f"展开后出现重复field key：{config['key']}")
        key_set.add(config["key"])
    return field_configs

def load_tasks(json_path):
    return expand_json_tasks(load_json_tasks(json_path))

def config_paths():
    return sorted(Path(config_directory).glob("*.json"))

def find_dataset_config_path(dataset):
    matches=[]
    for path in config_paths():
        try:
            tasks=load_json_tasks(path)
        except (TypeError,ValueError):
            continue
        if any(task["dataset"].lower()==dataset.lower() for task in tasks):
            matches.append(path)
    if len(matches)!=1:
        raise ValueError(f"dataset {dataset}应恰好出现在一个JSON中，实际找到{len(matches)}个")
    return matches[0]

def list_dataset_names():
    names=[]
    for path in config_paths():
        try:
            tasks=load_json_tasks(path)
        except (TypeError,ValueError):
            continue
        for task in tasks:
            if task["dataset"] not in names:
                names.append(task["dataset"])
    return tuple(names)

def write_selected_dataset_config(dataset,field,output_path):
    source_path=find_dataset_config_path(dataset)
    tasks=load_json_tasks(source_path)
    selected_tasks=[]
    for task in tasks:
        if task["dataset"].lower()!=dataset.lower() or field not in task["field_list"]:
            continue
        selected_task=copy.deepcopy(task)
        selected_task["field_list"]=[field]
        selected_tasks.append(selected_task)
    if len(selected_tasks)!=1:
        raise ValueError(f"dataset {dataset}中的field {field}应恰好对应一个task，实际找到{len(selected_tasks)}个")
    with open(output_path,"w",encoding="utf-8") as output_file:
        json.dump(selected_tasks,output_file,ensure_ascii=False,indent=2)
        output_file.write("\n")
build_max_cpus=get_max_cpus(8)
silent_build=False
verify_unsz_decompression=True
unsz_extra_arguments="-reverse-mean-distance-only-order-experimental"

def build_programs(use_umc):
    unsz_executable_paths=[os.path.join(project_root,"install/bin",name) for name in ["s1_compress","s1_decompress"]]
    if not all(os.path.isfile(path) for path in unsz_executable_paths):
        os.makedirs(os.path.join(project_root,"build"),exist_ok=True)
        run_shell_command(f"cmake -S {project_root}/c_codes -B {project_root}/build -DCMAKE_INSTALL_PREFIX={project_root}/install -DCODE_NAME=s1_compress",silent=silent_build)
        run_shell_command(f"cmake --build {project_root}/build --parallel {build_max_cpus}",silent=silent_build)
        run_shell_command(f"cmake --install {project_root}/build",silent=silent_build)
    for executable_path in unsz_executable_paths:
        if not os.path.isfile(executable_path):
            raise FileNotFoundError(f"没有找到UNSZ可执行文件：{executable_path}")
    if use_umc:
        for executable_name in ["test_compress","test_decompress"]:
            executable_path=os.path.join(umc_root,"build/test",executable_name)
            if not os.path.isfile(executable_path):
                raise FileNotFoundError(f"没有找到UMC可执行文件：{executable_path}")

def calculate_throughput(data_paths,seconds,name):
    if not seconds>0:
        raise RuntimeError(f"{name}必须大于0：{seconds}")
    return sum(os.path.getsize(path) for path in data_paths)/seconds/1e6

def calculate_data_range(data_paths):
    minimum=math.inf
    maximum=-math.inf
    for path in data_paths:
        values=array.array("f")
        with open(path,"rb") as file:
            values.fromfile(file,os.path.getsize(path)//values.itemsize)
        valid_values=(value for value in values if value!=-99999.0)
        try:
            first=next(valid_values)
        except StopIteration:
            continue
        local_minimum=first
        local_maximum=first
        for value in valid_values:
            local_minimum=min(local_minimum,value)
            local_maximum=max(local_maximum,value)
        minimum=min(minimum,local_minimum)
        maximum=max(maximum,local_maximum)
    if not minimum<=maximum:
        raise RuntimeError("无法从输入float32文件计算数据范围")
    return minimum,maximum

def calculate_umc_disk_error(data_paths):
    total_squared_error=0
    total_point_num=0
    max_error=0
    for path in data_paths:
        original=array.array("f")
        recovered=array.array("f")
        with open(path,"rb") as file:
            original.fromfile(file,os.path.getsize(path)//original.itemsize)
        output_path=path+".umc.out"
        with open(output_path,"rb") as file:
            recovered.fromfile(file,os.path.getsize(output_path)//recovered.itemsize)
        if len(original)!=len(recovered):
            raise RuntimeError(f"UMC解压输出大小错误：{output_path}有{len(recovered)}点，原文件有{len(original)}点")
        for original_value,recovered_value in zip(original,recovered):
            error=original_value-recovered_value
            total_squared_error+=error*error
            max_error=max(max_error,abs(error))
        total_point_num+=len(original)
    return total_squared_error/total_point_num,max_error

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
    time_match=re.search(r"total decompression time=\s*([^\s]+)\s*s",output)
    if match is None or time_match is None:
        raise RuntimeError("没有在UNSZ解压输出中找到total decompressed psnr或总解压时间")
    return float(match.group(1)),float(time_match.group(1))

def parse_umc_decompression_result(output,data_paths,umc_method,data_range=None):
    if umc_method=="NBP-IDW-INP":
        decompression_times=[float(value) for value in re.findall(r"^decompression time\s*=\s*([^\s]+)",output,re.MULTILINE)]
        if len(decompression_times)!=len(data_paths):
            raise RuntimeError(f"UMC动态异常点核心解压时间数量不一致：文件{len(data_paths)}，解压时间{len(decompression_times)}")
        decompression_time=sum(decompression_times)
    else:
        decompression_times=[float(value) for value in re.findall(r"^core decompression time\s*=\s*([^\s]+)",output,re.MULTILINE)]
        if len(decompression_times)!=len(data_paths):
            raise RuntimeError(f"UMC核心解压时间数量不一致：文件{len(data_paths)}，核心解压时间{len(decompression_times)}")
        decompression_time=sum(decompression_times)
    mse_list=[float(value) for value in re.findall(r"^MSE\s*=\s*([^,\s]+)",output,re.MULTILINE)]
    range_matches=re.findall(r"^Max value\s*=\s*([^,]+),\s*min value\s*=\s*([^,\s]+)",output,re.MULTILINE)
    max_error_list=[float(value) for value in re.findall(r"^Max error\s*=\s*([^,\s]+)",output,re.MULTILINE)]
    if len(mse_list)!=len(data_paths) or len(range_matches)!=len(data_paths) or len(max_error_list)!=len(data_paths):
        raise RuntimeError(f"UMC解压误差统计数量不一致：文件{len(data_paths)}，MSE {len(mse_list)}，范围{len(range_matches)}，最大误差{len(max_error_list)}")
    total_point_num=sum(os.path.getsize(path)//4 for path in data_paths)
    total_squared_error=sum(mse*(os.path.getsize(path)//4) for mse,path in zip(mse_list,data_paths))
    total_mse=total_squared_error/total_point_num
    total_min_data,total_max_data=data_range or calculate_data_range(data_paths)
    total_psnr=math.inf if total_mse==0 else 10*math.log10((total_max_data-total_min_data)**2/total_mse)
    return decompression_time,total_psnr,max(max_error_list)

def parse_umc_result(output,data_paths,data_range=None):
    compressed_sizes=[int(value) for value in re.findall(r"compressed_size\s*=\s*(\d+)",output)]
    mse_list=[float(value) for value in re.findall(r"^MSE\s*=\s*([^\s]+)",output,re.MULTILINE)]
    range_matches=re.findall(r"Max value\s*=\s*([^,]+),\s*min value\s*=\s*([^,]+),",output)
    if len(compressed_sizes)!=len(data_paths) or len(mse_list)!=len(data_paths) or len(range_matches)!=len(data_paths):
        raise RuntimeError(f"UMC输出数量不一致：文件{len(data_paths)}，压缩大小{len(compressed_sizes)}，MSE {len(mse_list)}，范围{len(range_matches)}")
    total_original_bytes=sum(os.path.getsize(path) for path in data_paths)
    total_compressed_bytes=sum(compressed_sizes)
    total_point_num=sum(os.path.getsize(path)//4 for path in data_paths)
    total_squared_error=sum(mse*(os.path.getsize(path)//4) for mse,path in zip(mse_list,data_paths))
    total_min_data,total_max_data=data_range or calculate_data_range(data_paths)
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

def test(compressor,dataset,rel_eb,dag_cache_path=None,point_set_config_path=None):
    data_dir=dataset.get("data_dir",os.path.join(data_root,dataset["folder"]))
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
        common_arguments=f"-data-dir {shlex.quote(data_dir)} -variable {shlex.quote(variable)} -time-step {shlex.quote(time_step_expression)} -dim {dataset['dim']}"
        common_arguments+=f" -point-set-config {shlex.quote(point_set_config_path)}" if point_set_config_path else f" -M REL {rel_eb}"
        if "adjust_coord_matrix" in dataset:
            matrix_path=os.path.join(project_root,"adjust_coord_matrices",dataset["adjust_coord_matrix"])
            common_arguments+=f" -adjust-coord-matrix-file {shlex.quote(matrix_path)}"
        if "time_step_coordinate_file" in dataset:
            time_step_coordinate_path=dataset["time_step_coordinate_file"]
            if not os.path.isabs(time_step_coordinate_path):
                time_step_coordinate_path=os.path.join(data_dir,time_step_coordinate_path)
            common_arguments+=f" -time-step-coordinate-file {shlex.quote(time_step_coordinate_path)}"
        if unsz_extra_arguments:
            common_arguments+=" "+unsz_extra_arguments
        if dataset.get("unsz_extra_arguments",""):
            common_arguments+=" "+dataset["unsz_extra_arguments"]
        if dag_cache_path:
            common_arguments+=f" -dag-cache {shlex.quote(dag_cache_path)}"
        compress_extra_arguments=dataset.get("unsz_compress_extra_arguments","")
        compress_command=f"{shlex.quote(project_root+'/install/bin/s1_compress')} {common_arguments} {compress_extra_arguments}"
        compress_output=run_shell_command(compress_command)
        br,psnr,preprocessing_time,compression_time,preprocessing_throughput,compression_throughput=parse_unsz_result(compress_output,data_paths)
        decompress_extra_arguments=dataset.get("unsz_decompress_extra_arguments","")
        decompress_command=f"{shlex.quote(project_root+'/install/bin/s1_decompress')} {common_arguments} {decompress_extra_arguments}"
        decompress_output=run_shell_command(decompress_command)
        decompressed_psnr,decompression_time=parse_unsz_decompression_result(decompress_output)
        if verify_unsz_decompression:
            if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                raise RuntimeError(f"压缩端和解压端的总PSNR不一致：{psnr} != {decompressed_psnr}")
            print(f"磁盘压缩—解压检查通过：total psnr={decompressed_psnr}")
    elif compressor=="umc":
        umc_from_edge=dataset["umc_from_edge"]
        umc_dim=dataset["dim"]
        umc_reorder=dataset.get("umc_reorder",1)
        umc_method=dataset.get("umc_method","NBP-IDW")
        umc_neighbor_num=dataset.get("umc_neighbor_num",4)
        if umc_method=="NBP-IDW":
            command=f"{umc_root}/build/test/test_compress {data_dir}/ {variable} coords.dat conn.dat {umc_from_edge} {umc_dim} 2 {umc_reorder} {time_step_num} 0 {rel_eb} {umc_neighbor_num}"
            decompress_command=f"{umc_root}/build/test/test_decompress {data_dir}/ {variable} coords.dat conn.dat {umc_from_edge} {umc_dim} 2 {umc_reorder} {time_step_num}"
        elif umc_method=="NBP-IDW-INP":
            umc_batch=dataset.get("umc_batch",10)
            command=f"{umc_root}/build/test/test_compress {data_dir}/ {variable} coords.dat conn.dat {umc_from_edge} {umc_dim} 4 {umc_reorder} {time_step_num} 0 {rel_eb} {umc_neighbor_num} {umc_batch}"
            decompress_command=f"{umc_root}/build/test/test_decompress {data_dir}/ {variable} coords.dat conn.dat {umc_from_edge} {umc_dim} 4 {umc_reorder} {time_step_num} {umc_batch}"
        else:
            raise ValueError(f"未知UMC压缩方法：{umc_method}")
        data_range=calculate_data_range(data_paths)
        output=run_shell_command(command)
        br,psnr,preprocessing_time,compression_time,preprocessing_throughput,compression_throughput=parse_umc_result(output,data_paths,data_range)
        decompress_output=run_shell_command(decompress_command)
        decompression_time,decompressed_psnr,decompressed_max_error=parse_umc_decompression_result(decompress_output,data_paths,umc_method,data_range)
        disk_mse,disk_max_error=calculate_umc_disk_error(data_paths)
        disk_psnr=math.inf if disk_mse==0 else 10*math.log10((data_range[1]-data_range[0])**2/disk_mse)
        if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-5) or not math.isclose(psnr,disk_psnr,rel_tol=1e-6,abs_tol=1e-5):
            raise RuntimeError(f"UMC压缩端、核心解压和磁盘输出总PSNR不一致：{psnr} != {decompressed_psnr} != {disk_psnr}")
        if not math.isclose(decompressed_max_error,disk_max_error,rel_tol=1e-5,abs_tol=1e-5):
            raise RuntimeError(f"UMC核心解压和磁盘输出最大误差不一致：{decompressed_max_error} != {disk_max_error}")
        print(f"UMC磁盘压缩—解压检查通过：total psnr={disk_psnr},max error={disk_max_error}")
    else:
        raise ValueError(f"未知压缩器：{compressor}")
    print(f"总结果：compressor={compressor}, dataset={dataset['name']}, variable={variable}, rel_eb={rel_eb}, br={br}, psnr={psnr}")
    decompression_throughput=calculate_throughput(data_paths,decompression_time,f"{compressor}解压时间")
    print(f"计时：preprocessing={preprocessing_time} s, compression={compression_time} s, decompression={decompression_time} s, compression+decompression={compression_time+decompression_time} s")
    print(f"吞吐率：preprocessing={preprocessing_throughput} MB/s, compression={compression_throughput} MB/s, decompression={decompression_throughput} MB/s")
    return br,psnr,preprocessing_throughput,compression_throughput,decompression_throughput

reference_canonical_configuration_vec=("idw_split_rounds_then_sort_mean_reference_distance","idw_sort_all_then_split_rounds_mean_reference_distance","idw_split_rounds_then_sort_reference_data_prediction_error","rbf_split_rounds_then_sort_mean_reference_distance","rbf_sort_all_then_split_rounds_mean_reference_distance","rbf_split_rounds_then_sort_reference_data_prediction_error")

def parse_reference_order_arguments(arguments):
    parser=argparse.ArgumentParser(description="比较一个field的IDW、RBF及不同空间预测顺序")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-variable",required=True)
    parser.add_argument("-time-step-num",type=int,required=True)
    parser.add_argument("-dim",type=int,required=True)
    parser.add_argument("-training-time-step",type=int,action="append")
    parser.add_argument("-test-time-step",type=int,action="append")
    parser.add_argument("-matrix-file",type=Path)
    parser.add_argument("-condition-operator",choices=("<","<=",">",">=","==","!="))
    parser.add_argument("-condition-value",type=float)
    parser.add_argument("-result",type=Path,required=True)
    parser.add_argument("-idw-ref-num",type=int,default=4)
    parser.add_argument("-idw-power",type=float,default=2)
    parser.add_argument("-rbf-ref-num",type=int,default=10)
    parser.add_argument("-ordering-score-max-ref-num",type=int)
    parser.add_argument("-ordering-score-min-ref-num",type=int)
    parser.add_argument("-rbf-kernel",default="matern32")
    parser.add_argument("-rbf-shape",type=float,default=0.01)
    parser.add_argument("-rbf-nugget",type=float,default=1e-6)
    parser.add_argument("-order-candidate-num",type=int,default=48)
    parser.add_argument("-disable-eb-schedule",action="store_true")
    parser.add_argument("-dag-cache-result",type=Path)
    parser.add_argument("-reorder-data-by-prediction-order",action="store_true")
    parser.add_argument("-decompression-stage-time",action="store_true")
    parser.add_argument("-common-extra",action="append",default=[])
    parser.add_argument("-compression-extra",action="append",default=[])
    parser.add_argument("-decompression-extra",action="append",default=[])
    parser.add_argument("-rel-eb",type=float,action="append")
    parser.add_argument("-configuration",choices=reference_canonical_configuration_vec,action="append")
    return parser.parse_args(arguments)

def run_argument_command(command):
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError("命令执行失败："+" ".join(str(value) for value in command))
    return result.stdout

def read_output_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def read_output_float_sum(pattern,text):
    return sum(float(value) for value in re.findall(pattern,text,re.MULTILINE))

def calculate_average_br_change(reference,test_rows):
    reference_points=sorted((float(row["psnr"]),float(row["br"])) for row in reference)
    test_points=sorted((float(row["psnr"]),float(row["br"])) for row in test_rows)
    reference_psnr=np.asarray([point[0] for point in reference_points])
    reference_br=np.asarray([point[1] for point in reference_points])
    test_psnr=np.asarray([point[0] for point in test_points])
    test_br=np.asarray([point[1] for point in test_points])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    if lower>=upper:
        raise ValueError("两条BR-PSNR曲线没有共同PSNR范围")
    knots=np.unique(np.concatenate(([lower,upper],reference_psnr[(reference_psnr>lower)&(reference_psnr<upper)],test_psnr[(test_psnr>lower)&(test_psnr<upper)])))
    reference_area=np.trapezoid(np.interp(knots,reference_psnr,reference_br),knots)
    test_area=np.trapezoid(np.interp(knots,test_psnr,test_br),knots)
    return (test_area/reference_area-1)*100

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def cleanup_reference_order(test_dir,variable,test_time_step_vec):
    for time_step in test_time_step_vec:
        for suffix in (".unsz",".unsz.bin"):
            path=test_dir/f"{variable}.dat.{time_step}{suffix}"
            if path.exists():
                path.unlink()

def write_reference_order_results(result_path,result_vec):
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with result_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_vec)

def write_reference_point_set_config(path,arguments,predictor,rel_eb):
    labels=[{"method":predictor,"eb_mode":"REL","eb_value":rel_eb},{"method":"constant"}]
    labels[0]["condition"]={"operator":arguments.condition_operator,"value":arguments.condition_value}
    config={"version":1,"labels":labels}
    with path.open("w",encoding="utf-8") as file:
        json.dump(config,file,ensure_ascii=False)

def reference_order_main(command_line_arguments):
    arguments=parse_reference_order_arguments(command_line_arguments)
    if(arguments.condition_operator is None)!=(arguments.condition_value is None):
        raise ValueError("condition-operator和condition-value必须同时指定")
    if arguments.idw_ref_num<=0 or arguments.rbf_ref_num<=0:
        raise ValueError("参考点数必须为正数")
    training_time_step_vec=tuple(dict.fromkeys(arguments.training_time_step or ()))
    test_time_step_vec=tuple(dict.fromkeys(arguments.test_time_step)) if arguments.test_time_step else tuple(range(arguments.time_step_num))
    rel_eb_vec=tuple(arguments.rel_eb) if arguments.rel_eb else (1e-1,1e-2,1e-3,1e-4,1e-5)
    for time_step in (*training_time_step_vec,*test_time_step_vec):
        if time_step<0 or time_step>=arguments.time_step_num:
            raise ValueError(f"time step超出范围：{time_step}")
    if not test_time_step_vec:
        raise ValueError("测试time step不能为空")
    common_optional=["-adjust-coord-matrix-file",arguments.matrix_file] if arguments.matrix_file is not None else []
    training_expression="{"+",".join(str(value) for value in training_time_step_vec)+"}"
    test_expression="{"+",".join(str(value) for value in test_time_step_vec)+"}"
    all_config_vec=(
        {"name":"idw_split_rounds_then_sort_mean_reference_distance","predictor":"inverse_proportion","round_formation":"split_rounds_then_sort","ordering_score":"mean_reference_distance","prediction_ref_num":arguments.idw_ref_num,"power":arguments.idw_power,"compress_extra":["-reverse-mean-distance-only-order-experimental"],"decompress_extra":[]},
        {"name":"idw_sort_all_then_split_rounds_mean_reference_distance","predictor":"inverse_proportion","round_formation":"sort_all_then_split_rounds","ordering_score":"mean_reference_distance","prediction_ref_num":arguments.idw_ref_num,"power":arguments.idw_power,"compress_extra":["-reverse-mean-distance-only-order-experimental","-global-order-before-round-split-experimental"],"decompress_extra":[]},
        {"name":"idw_split_rounds_then_sort_reference_data_prediction_error","predictor":"inverse_proportion","round_formation":"split_rounds_then_sort","ordering_score":"reference_data_prediction_error","prediction_ref_num":arguments.idw_ref_num,"power":arguments.idw_power,"compress_extra":["-reference-data-order-experimental",training_expression],"decompress_extra":[]},
        {"name":"rbf_split_rounds_then_sort_mean_reference_distance","predictor":"rbf_fitting","round_formation":"split_rounds_then_sort","ordering_score":"mean_reference_distance","prediction_ref_num":arguments.rbf_ref_num,"power":3.6,"compress_extra":["-reverse-mean-distance-only-order-experimental"],"decompress_extra":[]},
        {"name":"rbf_sort_all_then_split_rounds_mean_reference_distance","predictor":"rbf_fitting","round_formation":"sort_all_then_split_rounds","ordering_score":"mean_reference_distance","prediction_ref_num":arguments.rbf_ref_num,"power":3.6,"compress_extra":["-reverse-mean-distance-only-order-experimental","-global-order-before-round-split-experimental"],"decompress_extra":[]},
        {"name":"rbf_split_rounds_then_sort_reference_data_prediction_error","predictor":"rbf_fitting","round_formation":"split_rounds_then_sort","ordering_score":"reference_data_prediction_error","prediction_ref_num":arguments.rbf_ref_num,"power":3.6,"compress_extra":["-reference-data-order-experimental",training_expression,"-reference-data-rbf-score-experimental"],"decompress_extra":[]},
    )
    selected_configuration_set=set(arguments.configuration) if arguments.configuration else None
    config_vec=tuple(config for config in all_config_vec if selected_configuration_set is None or config["name"] in selected_configuration_set)
    for config in config_vec:
        config["ordering_score_max_ref_num"]=arguments.ordering_score_max_ref_num or config["prediction_ref_num"]
        config["ordering_score_min_ref_num"]=arguments.ordering_score_min_ref_num if arguments.ordering_score_min_ref_num is not None else (max(1,config["prediction_ref_num"]-1) if config["predictor"]=="inverse_proportion" else config["ordering_score_max_ref_num"])
        if config["prediction_ref_num"]>config["ordering_score_max_ref_num"] and config["ordering_score"]!="mean_reference_distance":
            raise ValueError("参考数据辅助排序的评分点数不能小于实际预测参考点数")
        if config["ordering_score_min_ref_num"]<=0 or config["ordering_score_min_ref_num"]>config["ordering_score_max_ref_num"]:
            raise ValueError("ordering score min ref num必须在1和ordering score max ref num之间")
    if arguments.dag_cache_result is not None and len(config_vec)!=1:
        raise ValueError("持久保存DAG缓存时必须只选择一个configuration")
    if any(config["ordering_score"]=="reference_data_prediction_error" for config in config_vec) and not training_time_step_vec:
        raise ValueError("参考数据辅助顺序至少需要一个training-time-step")
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix=f"unsz_{arguments.variable}_rbf_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(arguments.data_dir/"coords.dat")
        for time_step in set(training_time_step_vec)|set(test_time_step_vec):
            (test_dir/f"{arguments.variable}.dat.{time_step}").symlink_to(arguments.data_dir/f"{arguments.variable}.dat.{time_step}")
        for config in config_vec:
            dag_cache_path=arguments.dag_cache_result if arguments.dag_cache_result is not None else temporary_dir/f"{config['name']}.dag"
            dag_cache_path.parent.mkdir(parents=True,exist_ok=True)
            for rel_eb in rel_eb_vec:
                compression_error_arguments=["-M","REL",rel_eb,"-predictor",config["predictor"]]
                decompression_error_arguments=["-M","REL",rel_eb]
                if arguments.condition_operator is not None:
                    point_set_config_path=temporary_dir/f"{config['name']}_{rel_eb:g}.json"
                    write_reference_point_set_config(point_set_config_path,arguments,config["predictor"],rel_eb)
                    compression_error_arguments=["-point-set-config",point_set_config_path]
                    decompression_error_arguments=compression_error_arguments
                reorder_arguments=["-reorder-data-by-prediction-order-experimental"] if arguments.reorder_data_by_prediction_order else []
                schedule_arguments=[] if arguments.disable_eb_schedule else ["-eb-schedule-experimental","round_point_ratio",0.8]
                msd_arguments=["-ordering-msd-ref-num-experimental",config["ordering_score_max_ref_num"]] if config["ordering_score"]=="mean_reference_distance" else []
                common_arguments=["-data-dir",test_dir,"-variable",arguments.variable,"-time-step",test_expression,"-dim",arguments.dim,"-dag-cache",dag_cache_path,"-predict-round-num",8,"-max-ref-num",max(config["prediction_ref_num"],config["ordering_score_max_ref_num"]),"-actual-ref-num",config["prediction_ref_num"],"-inverse-distance-power",config["power"],"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",config["ordering_score_min_ref_num"],"-order-candidate-num-experimental",arguments.order_candidate_num,*msd_arguments,*schedule_arguments,"-rbf-kernel",arguments.rbf_kernel,"-rbf-shape",arguments.rbf_shape,"-rbf-nugget",arguments.rbf_nugget,*common_optional,*reorder_arguments,*arguments.common_extra]
                compression_output=run_argument_command([Path(project_root)/"install/bin/s1_compress",*compression_error_arguments,*common_arguments,*config["compress_extra"],*arguments.compression_extra])
                decompression_output=run_argument_command([Path(project_root)/"install/bin/s1_decompress",*decompression_error_arguments,*common_arguments,*config["decompress_extra"],*arguments.decompression_extra,*(["-decompression-stage-time"] if arguments.decompression_stage_time else [])])
                psnr=read_output_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_output_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                compression_time=read_output_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间")
                decompression_time=read_output_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")
                result={"config":config["name"],"round_formation":config["round_formation"],"ordering_score":config["ordering_score"],"ref_num":config["prediction_ref_num"],"ordering_score_max_ref_num":config["ordering_score_max_ref_num"],"ordering_score_min_ref_num":config["ordering_score_min_ref_num"],"reorder_data_by_prediction_order":arguments.reorder_data_by_prediction_order,"rel_eb":rel_eb,"br":read_output_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":read_output_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":compression_time,"decompression_time":decompression_time,"compression_decompression_time":compression_time+decompression_time,"compression_input_reorder_time":read_output_float_sum(r"^compression input reorder-by-prediction-order time excluded from compression= ([^ ]+) s$",compression_output),"decompression_output_restore_time":read_output_float_sum(r"^decompression output restore-original-order time excluded from decompression= ([^ ]+) s$",decompression_output),"decompression_stream_decode_time":read_output_float_sum(r"^decompression stage time: stream_decode=([^ ]+) s,",decompression_output),"decompression_label_decode_time":read_output_float_sum(r"^decompression stage time: stream_decode=[^ ]+ s, label_decode=([^ ]+) s,",decompression_output),"decompression_value_decode_time":read_output_float_sum(r"^decompression stage time: stream_decode=[^ ]+ s, label_decode=[^ ]+ s, value_decode=([^ ]+) s,",decompression_output)}
                result_vec.append(result)
                write_reference_order_results(arguments.result,result_vec)
                print(f"{config['name']} REL={rel_eb:g}:BR={result['br']:.6f},PSNR={psnr:.6f},preprocessing={result['preprocessing_time']:.6f}s,compression={result['compression_time']:.6f}s,decompression={result['decompression_time']:.6f}s,compression+decompression={result['compression_decompression_time']:.6f}s",flush=True)
                cleanup_reference_order(test_dir,arguments.variable,test_time_step_vec)
    result_map={name:[row for row in result_vec if row["config"]==name] for name in (config["name"] for config in config_vec)}
    baseline_name="idw_split_rounds_then_sort_mean_reference_distance"
    if baseline_name in result_map:
        baseline=result_map[baseline_name]
        for name in reference_canonical_configuration_vec:
            if name==baseline_name or name not in result_map:
                continue
            rows=result_map[name]
            print(f"{name}相对{baseline_name}：同PSNR平均BR变化={calculate_average_br_change(baseline,rows):+.6f}%，压缩时间倍率={geometric_mean([row['compression_time'] for row in rows])/geometric_mean([row['compression_time'] for row in baseline]):.6f}，解压时间倍率={geometric_mean([row['decompression_time'] for row in rows])/geometric_mean([row['decompression_time'] for row in baseline]):.6f}，压缩+解压时间倍率={geometric_mean([row['compression_decompression_time'] for row in rows])/geometric_mean([row['compression_decompression_time'] for row in baseline]):.6f}",flush=True)
    print(f"结果已写入{arguments.result}",flush=True)

def find_field_config(dataset,field):
    tasks=load_tasks(find_dataset_config_path(dataset))
    matches=[config for config in tasks if config["dataset"].lower()==dataset.lower() and config["variable"]==field]
    if len(matches)==1:
        return matches[0]
    if len(matches)>1:
        raise ValueError(f"dataset {dataset}中的field {field}展开为{len(matches)}个参数组合，无法作为唯一正式配置")
    available_fields=tuple(dict.fromkeys(config["variable"] for config in tasks))
    raise ValueError(f"dataset {dataset}中没有field {field}；可选field：{','.join(available_fields)}")

def cli(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="运行configs目录中的UNSZ实验配置")
    parser.add_argument("json",help="configs中的JSON配置名称，或JSON文件路径")
    arguments=parser.parse_args(command_line_arguments)
    try:
        json_path=resolve_json_path(arguments.json)
        selected_field_config_vec=load_tasks(json_path)
    except (FileNotFoundError,TypeError,ValueError) as error:
        parser.error(str(error))
    build_programs(any("umc" in field["method_list"] for field in selected_field_config_vec))
    sys.modules.setdefault("t1_start_c_code",sys.modules[__name__])
    from py_codes.refresh_selected_results import run_field
    for field in selected_field_config_vec:
        run_field(field,rel_values=field["rel_eb_list"],configuration_path=json_path)

if __name__=="__main__":
    cli()
