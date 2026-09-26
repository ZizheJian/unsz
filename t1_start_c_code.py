import argparse,array,copy,json,math,os,re,shlex,tempfile
from pathlib import Path
from py_codes.parse_umc_output import calculate_data_range,calculate_umc_disk_error,parse_umc_compression_result,parse_umc_decompression_result
from py_codes.parse_unsz_output import parse_unsz_output
from third_party.csv_tools.py_codes.write_csv_item import write_csv_item
from third_party.get_max_cpus.py_codes.get_max_cpus import get_max_cpus
from third_party.run_command.run_command import run_command

project_root=Path(__file__).resolve().parent
config_directory=project_root/"configs"
data_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data")
result_root=project_root/"results/current"
umc_root=Path("/home/x-zjian1/umc")
build_max_cpus=get_max_cpus(8)
silent_build=False
temporal_configurations={"spatial","constant_extrapolation","linear_interpolation","constant_extrapolation_spatial_residual","linear_interpolation_spatial_residual"}

# JSON配置
def resolve_json_path(path_or_name):
    path_or_name=os.path.expanduser(path_or_name)
    if os.path.isabs(path_or_name) or os.path.dirname(path_or_name):
        json_path=Path(path_or_name).resolve()
    else:
        config_filename=path_or_name if path_or_name.endswith(".json") else path_or_name+".json"
        json_path=config_directory/config_filename
    if not json_path.is_file():
        raise FileNotFoundError(f"找不到JSON配置：{json_path}")
    return json_path

def load_json_tasks(json_path):
    with Path(json_path).open(encoding="utf-8") as config_file:
        tasks=json.load(config_file)
    if not isinstance(tasks,list):
        raise TypeError(f"JSON顶层必须是task list：{json_path}")
    if not tasks:
        raise ValueError(f"JSON中没有task：{json_path}")
    if any(not isinstance(task,dict) for task in tasks):
        raise TypeError(f"JSON中的每个task必须是object：{json_path}")
    return tasks

def expand_json_tasks(tasks):
    prepared_tasks=[]
    for task in tasks:
        for field in task["field_list"]:
            config=copy.deepcopy(task)
            config["task_name"]=config.pop("name")
            config.pop("field_list")
            config["variable"]=field
            dataset_key=re.sub(r"[^0-9A-Za-z]+","_",config["dataset"]).strip("_").lower()
            config["key"]=f"{dataset_key}_{field}"
            prepared_tasks.append(config)
    key_set=set()
    for config in prepared_tasks:
        if config["key"] in key_set:
            raise ValueError(f"展开后出现重复field key：{config['key']}")
        key_set.add(config["key"])
    return prepared_tasks

def load_tasks(json_path):
    return expand_json_tasks(load_json_tasks(json_path))

def find_task(json_path_or_name,dataset,field):
    tasks=load_tasks(resolve_json_path(json_path_or_name))
    matches=[config for config in tasks if config["dataset"].lower()==dataset.lower() and config["variable"]==field]
    if len(matches)==1:
        return matches[0]
    if len(matches)>1:
        raise ValueError(f"dataset {dataset}中的field {field}展开为{len(matches)}个参数组合，无法作为唯一正式配置")
    available_fields=tuple(dict.fromkeys(config["variable"] for config in tasks))
    raise ValueError(f"dataset {dataset}中没有field {field}；可选field：{','.join(available_fields)}")

# 公共路径和程序
def data_directory(config):
    return Path(config["data_dir"]) if "data_dir" in config else data_root/config["folder"]

def field_data_paths(config):
    directory=data_directory(config)
    return [directory/f"{config['variable']}.dat.{time_step}" for time_step in range(config["steps"])]

def field_dag_cache_path(config):
    prediction_ref_num=config.get("prediction_ref_num",config.get("ref_num"))
    score_ref_num=config.get("ordering_score_max_ref_num",prediction_ref_num)
    score_suffix=f".msd{score_ref_num}" if score_ref_num!=prediction_ref_num else ""
    return data_directory(config)/".unsz_cache"/"benchmarks"/f"{config['key']}{score_suffix}.dag"

def build_programs(prepared_tasks):
    method_set={method for config in prepared_tasks for method in config["method_list"]}
    if "unsz" in method_set:
        executable_paths=[project_root/"install/bin"/name for name in ("s1_compress","s1_decompress")]
        if not all(path.is_file() for path in executable_paths):
            (project_root/"build").mkdir(exist_ok=True)
            run_command(["cmake","-S",project_root/"c_codes","-B",project_root/"build",f"-DCMAKE_INSTALL_PREFIX={project_root/'install'}","-DCODE_NAME=s1_compress"],silent=silent_build)
            run_command(["cmake","--build",project_root/"build","--parallel",build_max_cpus],silent=silent_build)
            run_command(["cmake","--install",project_root/"build"],silent=silent_build)
        for path in executable_paths:
            if not path.is_file():
                raise FileNotFoundError(f"没有找到UNSZ可执行文件：{path}")
    if "umc" in method_set:
        for name in ("test_compress","test_decompress"):
            path=umc_root/"build/test"/name
            if not path.is_file():
                raise FileNotFoundError(f"没有找到UMC可执行文件：{path}")

# UNSZ命令
def read_time_step_coordinates(config):
    if "time_step_coordinate_file" not in config:
        return [float(time_step) for time_step in range(config["steps"])]
    path=data_directory(config)/config["time_step_coordinate_file"]
    byte_num=path.stat().st_size
    if byte_num%8:
        raise ValueError(f"time step坐标文件大小不是float64的整数倍：{path}")
    coordinates=array.array("d")
    with path.open("rb") as file:
        coordinates.fromfile(file,byte_num//8)
    if len(coordinates)!=config["steps"]:
        raise ValueError(f"time step坐标数量应为{config['steps']}，实际为{len(coordinates)}")
    if any(not math.isfinite(value) for value in coordinates):
        raise ValueError("time step坐标只能包含有限值")
    differences=[right-left for left,right in zip(coordinates,coordinates[1:])]
    if differences and not(all(value>0 for value in differences) or all(value<0 for value in differences)):
        raise ValueError("time step坐标必须严格单调")
    return list(coordinates)

def linear_interpolation_time_order(coordinates):
    order=[0,len(coordinates)-1]
    intervals=[(0,len(coordinates)-1)]
    for left,right in intervals:
        if right-left<=1:
            continue
        physical_middle=(coordinates[left]+coordinates[right])/2
        middle=min(range(left+1,right),key=lambda index:abs(coordinates[index]-physical_middle))
        order.append(middle)
        intervals.extend(((left,middle),(middle,right)))
    if len(order)!=len(coordinates) or len(set(order))!=len(coordinates):
        raise RuntimeError("一阶内插顺序没有覆盖全部time step")
    return order

def write_point_set_config(path,config,predictor,rel_eb):
    labels=[{"method":predictor,"eb_mode":"REL","eb_value":rel_eb},{"method":"constant"}]
    labels[0]["condition"]=config["condition"]
    with path.open("w",encoding="utf-8") as file:
        json.dump({"version":1,"labels":labels},file,ensure_ascii=False)

def build_spatial_unsz_arguments(config,test_data_dir,dag_path):
    configuration=config["config"]
    if configuration.startswith("idw_"):
        predictor="inverse_proportion"
    elif configuration.startswith("rbf_"):
        predictor="rbf_fitting"
    else:
        raise ValueError(f"未知纯空间配置：{configuration}")
    prediction_ref_num=config.get("prediction_ref_num",config.get("ref_num"))
    if prediction_ref_num is None:
        raise ValueError(f"{config['key']}缺少prediction_ref_num")
    score_ref_num=config.get("ordering_score_max_ref_num",prediction_ref_num)
    score_min_ref_num=config.get("ordering_score_min_ref_num",max(1,prediction_ref_num-1) if predictor=="inverse_proportion" else score_ref_num)
    time_expression="{"+",".join(str(value) for value in range(config["steps"]))+"}"
    common=["-data-dir",test_data_dir,"-variable",config["variable"],"-time-step",time_expression,"-dim",config["dim"],"-dag-cache",dag_path,"-predict-round-num",8,"-max-ref-num",max(prediction_ref_num,score_ref_num),"-actual-ref-num",prediction_ref_num,"-inverse-distance-power",config.get("power",2),"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",score_min_ref_num,"-order-candidate-num-experimental",config.get("ordering_candidate_num",48),"-rbf-kernel",config.get("rbf_kernel","matern32"),"-rbf-shape",config.get("shape",0.01),"-rbf-nugget",config.get("rbf_nugget",1e-6)]
    compression_only=[]
    if configuration.endswith("mean_reference_distance"):
        common.extend(("-ordering-msd-ref-num-experimental",score_ref_num))
        compression_only.append("-reverse-mean-distance-only-order-experimental")
    elif configuration.endswith("reference_data_prediction_error"):
        training_expression="{"+",".join(str(value) for value in config.get("training",[]))+"}"
        if training_expression=="{}":
            raise ValueError("参考数据辅助顺序至少需要一个training time step")
        compression_only.extend(("-reference-data-order-experimental",training_expression))
        if predictor=="rbf_fitting":
            compression_only.append("-reference-data-rbf-score-experimental")
    else:
        raise ValueError(f"未知空间排序标准：{configuration}")
    if "sort_all_then_split_rounds" in configuration:
        compression_only.append("-global-order-before-round-split-experimental")
    if not config.get("disable_eb_schedule"):
        common.extend(("-eb-schedule-experimental","round_point_ratio",0.8))
    if "matrix" in config:
        common.extend(("-adjust-coord-matrix-file",project_root/"adjust_coord_matrices"/config["matrix"]))
    if config.get("reorder_by_prediction_order"):
        common.append("-reorder-data-by-prediction-order-experimental")
    common.extend(config.get("common_extra",[]))
    compression_only.extend(config.get("compression_extra",[]))
    return predictor,common,compression_only,list(config.get("decompression_extra",[]))

def build_temporal_unsz_arguments(config,test_data_dir,dag_path):
    if config["steps"]<2:
        raise ValueError("时间预测至少需要两个time step")
    predictor=config.get("predictor","inverse_proportion")
    coordinates=read_time_step_coordinates(config)
    if config["config"] in ("linear_interpolation","linear_interpolation_spatial_residual"):
        time_order=linear_interpolation_time_order(coordinates)
        time_expression="{"+",".join(str(value) for value in time_order)+"}"
    else:
        time_expression=f"[0,{config['steps']-1}]"
    ref_num=config["ref_num"]
    common=["-data-dir",test_data_dir,"-variable",config["variable"],"-time-step",time_expression,"-dim",config["dim"],"-dag-cache",dag_path,"-predict-round-num",8,"-max-ref-num",ref_num,"-actual-ref-num",ref_num,"-inverse-distance-power",config.get("power",2),"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",ref_num if predictor=="rbf_fitting" else 3,"-reverse-mean-distance-only-order-experimental"]
    if not config.get("disable_eb_schedule"):
        common.extend(("-eb-schedule-experimental","round_point_ratio",0.8))
    if predictor=="rbf_fitting":
        common.extend(("-rbf-kernel",config.get("rbf_kernel","matern32"),"-rbf-shape",config.get("shape",0.03),"-rbf-nugget",config.get("rbf_nugget",1e-6),"-rbf-polynomial-degree",0))
    if "matrix" in config:
        common.extend(("-adjust-coord-matrix-file",project_root/"adjust_coord_matrices"/config["matrix"]))
    if "time_step_coordinate_file" in config:
        common.extend(("-time-step-coordinate-file",data_directory(config)/config["time_step_coordinate_file"]))
    if config["config"]!="spatial":
        common.extend(("-cross-time-predictor-experimental",config["config"]))
    common.extend(config.get("common_extra",[]))
    return predictor,common,list(config.get("compression_extra",[])),list(config.get("decompression_extra",[]))

def build_unsz_commands(config,rel_eb,temporary_dir,test_data_dir):
    dag_path=field_dag_cache_path(config) if config.get("persist_dag") else temporary_dir/"prediction.dag"
    if config["config"] in temporal_configurations:
        predictor,common,compression_only,decompression_only=build_temporal_unsz_arguments(config,test_data_dir,dag_path)
    else:
        predictor,common,compression_only,decompression_only=build_spatial_unsz_arguments(config,test_data_dir,dag_path)
    if "condition" in config:
        point_set_config_path=temporary_dir/"point_sets.json"
        write_point_set_config(point_set_config_path,config,predictor,rel_eb)
        compression_error=["-point-set-config",point_set_config_path]
        decompression_error=compression_error
    else:
        compression_error=["-M","REL",rel_eb,"-predictor",predictor]
        decompression_error=["-M","REL",rel_eb]
    compression_command=[project_root/"install/bin/s1_compress",*compression_error,*common,*compression_only]
    decompression_command=[project_root/"install/bin/s1_decompress",*decompression_error,*common,*decompression_only]
    return compression_command,decompression_command

# 压缩器执行
def run_unsz(config,rel_eb):
    source_data_dir=data_directory(config)
    source_data_paths=field_data_paths(config)
    required_paths=[source_data_dir/"coords.dat",*source_data_paths]
    missing_paths=[str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("缺少测试文件："+", ".join(missing_paths))
    if config.get("persist_dag"):
        field_dag_cache_path(config).parent.mkdir(parents=True,exist_ok=True)
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix=f"unsz_{config['key']}_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_data_dir=temporary_dir/"data"
        test_data_dir.mkdir()
        (test_data_dir/"coords.dat").symlink_to(source_data_dir/"coords.dat")
        for time_step,source_path in enumerate(source_data_paths):
            (test_data_dir/f"{config['variable']}.dat.{time_step}").symlink_to(source_path)
        compression_command,decompression_command=build_unsz_commands(config,rel_eb,temporary_dir,test_data_dir)
        compression_output=run_command(compression_command,silent=True)
        decompression_output=run_command(decompression_command,silent=True)
        result=parse_unsz_output(compression_output,decompression_output)
    print(f"总结果：compressor=unsz,dataset={config['key']},variable={config['variable']},rel_eb={rel_eb},br={result['br']},psnr={result['psnr']}")
    print(f"计时：preprocessing={result['preprocessing_time']} s,compression={result['compression_time']} s,decompression={result['decompression_time']} s,compression+decompression={result['compression_time']+result['decompression_time']} s")
    return {"compressor":"unsz","rel_eb":rel_eb,**result,"command":"compression: "+shlex.join(str(value) for value in compression_command)+"; decompression: "+shlex.join(str(value) for value in decompression_command)}

def run_umc(config,rel_eb):
    source_data_dir=data_directory(config)
    variable=config["variable"]
    time_step_num=config["steps"]
    source_data_paths=field_data_paths(config)
    required_paths=[source_data_dir/"coords.dat",source_data_dir/"conn.dat",*source_data_paths]
    missing_paths=[str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("缺少测试文件："+", ".join(missing_paths))
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix=f"umc_{config['key']}_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        (temporary_dir/"coords.dat").symlink_to(source_data_dir/"coords.dat")
        (temporary_dir/"conn.dat").symlink_to(source_data_dir/"conn.dat")
        for time_step,source_path in enumerate(source_data_paths):
            (temporary_dir/f"{variable}.dat.{time_step}").symlink_to(source_path)
        data_paths=[str(temporary_dir/f"{variable}.dat.{time_step}") for time_step in range(time_step_num)]
        umc_config=config["umc"]
        umc_from_edge=umc_config["umc_from_edge"]
        umc_dim=config["dim"]
        umc_reorder=umc_config.get("umc_reorder",1)
        umc_method=umc_config.get("umc_method","NBP-IDW")
        umc_neighbor_num=umc_config.get("umc_neighbor_num",4)
        data_dir=str(temporary_dir)+"/"
        if umc_method=="NBP-IDW":
            compression_command=[umc_root/"build/test/test_compress",data_dir,variable,"coords.dat","conn.dat",umc_from_edge,umc_dim,2,umc_reorder,time_step_num,0,rel_eb,umc_neighbor_num]
            decompression_command=[umc_root/"build/test/test_decompress",data_dir,variable,"coords.dat","conn.dat",umc_from_edge,umc_dim,2,umc_reorder,time_step_num]
        elif umc_method=="NBP-IDW-INP":
            umc_batch=umc_config.get("umc_batch",10)
            compression_command=[umc_root/"build/test/test_compress",data_dir,variable,"coords.dat","conn.dat",umc_from_edge,umc_dim,4,umc_reorder,time_step_num,0,rel_eb,umc_neighbor_num,umc_batch]
            decompression_command=[umc_root/"build/test/test_decompress",data_dir,variable,"coords.dat","conn.dat",umc_from_edge,umc_dim,4,umc_reorder,time_step_num,umc_batch]
        else:
            raise ValueError(f"未知UMC压缩方法：{umc_method}")
        data_range=calculate_data_range(data_paths)
        compression_output=run_command(compression_command)
        br,psnr,preprocessing_time,compression_time=parse_umc_compression_result(compression_output,data_paths,data_range)
        decompression_output=run_command(decompression_command)
        decompression_time,decompressed_psnr,decompressed_max_error=parse_umc_decompression_result(decompression_output,data_paths,umc_method,data_range)
        if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-5):
            raise RuntimeError(f"UMC压缩端和独立解压端总PSNR不一致：{psnr} != {decompressed_psnr}")
        print(f"UMC独立解压检查通过：total psnr={decompressed_psnr},max error={decompressed_max_error}")
        if config.get("verify_umc_disk_output",False):
            disk_mse,disk_max_error=calculate_umc_disk_error(data_paths)
            disk_psnr=math.inf if disk_mse==0 else 10*math.log10((data_range[1]-data_range[0])**2/disk_mse)
            if not math.isclose(psnr,disk_psnr,rel_tol=1e-6,abs_tol=1e-5):
                raise RuntimeError(f"UMC压缩端和磁盘输出总PSNR不一致：{psnr} != {disk_psnr}")
            if not math.isclose(decompressed_max_error,disk_max_error,rel_tol=1e-5,abs_tol=1e-5):
                raise RuntimeError(f"UMC独立解压和磁盘输出最大误差不一致：{decompressed_max_error} != {disk_max_error}")
            print(f"UMC磁盘输出检查通过：total psnr={disk_psnr},max error={disk_max_error}")
    print(f"总结果：compressor=umc,dataset={config['key']},variable={variable},rel_eb={rel_eb},br={br},psnr={psnr}")
    print(f"计时：preprocessing={preprocessing_time} s,compression={compression_time} s,decompression={decompression_time} s,compression+decompression={compression_time+decompression_time} s")
    return {"compressor":"umc","rel_eb":rel_eb,"br":br,"psnr":psnr,"preprocessing_time":preprocessing_time,"compression_time":compression_time,"decompression_time":decompression_time,"command":"compression: "+shlex.join(str(value) for value in compression_command)+"; decompression: "+shlex.join(str(value) for value in decompression_command)}

# 任务循环和命令行入口
def run_task(config):
    result_root.mkdir(parents=True,exist_ok=True)
    total_bytes=sum(path.stat().st_size for path in field_data_paths(config))
    result_path=result_root/"fields"/f"{config['key']}.csv"
    result_path.parent.mkdir(parents=True,exist_ok=True)
    for rel_eb in config["rel_eb_list"]:
        for method in config["method_list"]:
            print(f"开始：field={config['key']},compressor={method},REL={rel_eb:g}",flush=True)
            if method=="unsz":
                result=run_unsz(config,rel_eb)
            elif method=="umc":
                result=run_umc(config,rel_eb)
            else:
                raise ValueError(f"尚未实现baseline：{method}")
            item={"compressor":method,"rel_eb":rel_eb}
            for column_name in ("br","psnr","preprocessing_time","compression_time","decompression_time","command"):
                write_csv_item(str(result_path),item,column_name,result[column_name])
            for column_name,value in (("field",config["key"]),("dataset",config["dataset"]),("variable",config["variable"]),("time_step_num",config["steps"]),("input_bytes",total_bytes),("compression_decompression_time",result["compression_time"]+result["decompression_time"]),("preprocessing_MB_s",total_bytes/result["preprocessing_time"]/1e6),("compression_MB_s",total_bytes/result["compression_time"]/1e6),("decompression_MB_s",total_bytes/result["decompression_time"]/1e6)):
                write_csv_item(str(result_path),item,column_name,value)
    print(f"完成：{config['key']}，结果{result_path}",flush=True)

def cli(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="运行configs目录中的UNSZ实验配置")
    parser.add_argument("json",help="configs中的JSON配置名称，或JSON文件路径")
    arguments=parser.parse_args(command_line_arguments)
    try:
        prepared_tasks=load_tasks(resolve_json_path(arguments.json))
    except (FileNotFoundError,TypeError,ValueError) as error:
        parser.error(str(error))
    build_programs(prepared_tasks)
    for config in prepared_tasks:
        run_task(config)

if __name__=="__main__":
    cli()
