import argparse,array,csv,json,math,os,shlex
from pathlib import Path
from py_codes.parse_umc_output import calculate_umc_disk_error,parse_umc_compression_result,parse_umc_decompression_result
from py_codes.parse_unsz_output import parse_unsz_compression_output,parse_unsz_output
from third_party.csv_tools.py_codes.write_csv_item import write_csv_item
from third_party.get_max_cpus.py_codes.get_max_cpus import get_max_cpus
from third_party.run_command.run_command import run_command

project_root=Path(__file__).resolve().parent
config_directory=project_root/"configs"
umc_root=Path("/home/x-zjian1/umc")
umc_inp_batch=10
build_max_cpus=get_max_cpus(8)
silent_build=False

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

def resolve_point_set_config_path(path):
    path=Path(path)
    return path if path.is_absolute() else project_root/path

def read_point_set_rel_eb_list(path):
    path=resolve_point_set_config_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到点集配置文件：{path}")
    with path.open(encoding="utf-8") as file:
        document=json.load(file)
    rel_eb_vec=[label["eb_value"] for label in document["labels"] if label["method"]=="predict" and label["eb_mode"]=="REL"]
    return list(dict.fromkeys(rel_eb_vec))

# 公共路径和程序
def generate_input_file_paths(config):
    directory=Path(config["data_dir"])
    return [directory/f"{config['field']}.dat.{time_step}" for time_step in range(config["time_step_num"])]

def calculate_data_range(data_paths):
    minimum=math.inf
    maximum=-math.inf
    for path in data_paths:
        byte_num=os.path.getsize(path)
        if byte_num%4:
            raise RuntimeError(f"输入文件不是完整的float32数组：{path}，字节数={byte_num}")
        values=array.array("f")
        with open(path,"rb") as file:
            values.fromfile(file,byte_num//values.itemsize)
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

def calculate_throughput(byte_num,elapsed_time,name):
    if not math.isfinite(elapsed_time) or elapsed_time<=0:
        raise RuntimeError(f"{name}必须是有限正数，实际为{elapsed_time}；测试规模可能过小，无法用六位小数计时")
    return byte_num/elapsed_time/1e6

def remove_file_if_exists(path):
    path=Path(path)
    if path.is_file():
        path.unlink()

def cleanup_test_outputs(config,compressor):
    if compressor=="unsz":
        for input_path in generate_input_file_paths(config):
            remove_file_if_exists(str(input_path)+".unsz")
            remove_file_if_exists(str(input_path)+".unsz.bin")
        remove_file_if_exists(Path(config["data_dir"])/f"{config['field']}.joint.unsz")
        remove_file_if_exists(Path(config["data_dir"])/f"{config['field']}.unsz.coord_transform.txt")
    elif compressor=="umc":
        for input_path in generate_input_file_paths(config):
            remove_file_if_exists(str(input_path)+".umc")
            remove_file_if_exists(str(input_path)+".umc.out")

result_identity_columns=("task","field","compressor","label_flow","value_flow","label_value_execution","rel_eb","point_set_config")

def normalize_result_csv_columns(path):
    path=Path(path)
    if not path.is_file():
        return
    with path.open(newline="",encoding="utf-8") as file:
        rows=list(csv.reader(file))
    if not rows:
        return
    header=rows[0]
    missing_columns=[column for column in result_identity_columns if column not in header]
    if missing_columns:
        raise ValueError(f"结果CSV缺少身份列：{path}：{','.join(missing_columns)}")
    if tuple(header[:len(result_identity_columns)])==result_identity_columns:
        return
    column_ids=[header.index(column) for column in result_identity_columns]
    column_ids.extend(column_id for column_id,column in enumerate(header) if column not in result_identity_columns)
    with path.open("w",newline="",encoding="utf-8") as file:
        csv.writer(file,lineterminator="\n").writerows([[row[column_id] if column_id<len(row) else "" for column_id in column_ids] for row in rows])

def build_project(build_max_cpus,silent_build):
    source_path=project_root/"c_codes"
    build_path=project_root/"build"
    install_path=project_root/"install"
    build_path.mkdir(exist_ok=True)
    run_command(["cmake","-S",source_path,"-B",build_path,f"-DCMAKE_INSTALL_PREFIX={install_path}","-DCODE_NAME=s1_compress"],silent=silent_build)
    run_command(["cmake","--build",build_path,"--parallel",build_max_cpus],silent=silent_build)
    run_command(["cmake","--install",build_path],silent=silent_build)

def check_baseline_programs(tasks):
    if any("umc" in task["compressor_list"] for task in tasks):
        for name in ("test_compress","test_decompress"):
            path=umc_root/"build/test"/name
            if not path.is_file():
                raise FileNotFoundError(f"没有找到UMC可执行文件：{path}")

# UNSZ命令
def read_time_step_coordinates(config):
    if "time_step_coordinate_file" not in config:
        return [float(time_step) for time_step in range(config["time_step_num"])]
    path=Path(config["data_dir"])/config["time_step_coordinate_file"]
    byte_num=path.stat().st_size
    if byte_num%8:
        raise ValueError(f"time step坐标文件大小不是float64的整数倍：{path}")
    coordinates=array.array("d")
    with path.open("rb") as file:
        coordinates.fromfile(file,byte_num//8)
    if len(coordinates)!=config["time_step_num"]:
        raise ValueError(f"time step坐标数量应为{config['time_step_num']}，实际为{len(coordinates)}")
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

def is_anchor_interpolation_stride_8_flow(flow):
    return flow in ("anchor_interpolation_stride_8","anchor_interpolation_stride_8_spatial_residual")

def build_predictor_arguments(predictor):
    predictor_name=predictor["name"]
    if predictor_name=="idw":
        return predictor_name,["-idw-power",predictor["idw_power"]]
    if predictor_name!="rbf":
        raise ValueError(f"未知预测器：{predictor_name}")
    arguments=["-rbf-kernel",predictor["rbf_kernel"],"-rbf-shape",predictor["rbf_shape"],"-rbf-nugget",predictor["rbf_nugget"]]
    return predictor_name,arguments

def build_eb_schedule_arguments(eb_schedule):
    if eb_schedule["name"] not in ("none","inverse_rank_log2"):
        raise ValueError(f"未知EB序列模式：{eb_schedule['name']}")
    arguments=["-eb-schedule",eb_schedule["name"],eb_schedule["base"]]
    if eb_schedule["name"]=="inverse_rank_log2":
        arguments.extend(("-eb-schedule-log2-shift",eb_schedule["log2_shift"],"-eb-schedule-max-log2-exponent",eb_schedule["max_log2_exponent"]))
    return arguments

def build_spatial_prediction_arguments(config,data_dir,dag_cache_path,time_expression):
    predictor,predictor_arguments=build_predictor_arguments(config["predictor"])
    if config["prediction_min_ref_num"]>config["prediction_max_ref_num"]:
        raise ValueError("prediction_min_ref_num不能大于prediction_max_ref_num")
    if predictor=="rbf" and config["prediction_min_ref_num"]!=config["prediction_max_ref_num"]:
        raise ValueError("RBF要求prediction_min_ref_num与prediction_max_ref_num相等")
    if config["ordering_score_min_ref_num"]>config["ordering_score_max_ref_num"]:
        raise ValueError("ordering_score_min_ref_num不能大于ordering_score_max_ref_num")
    if config["ordering_scope"] not in ("local","global"):
        raise ValueError(f"未知排序规模：{config['ordering_scope']}")
    arguments=["-data-dir",data_dir,"-field",config["field"],"-time-step",time_expression,"-coordinate-dim",config["coordinate_dim"],
               "-prediction-min-ref-num",config["prediction_min_ref_num"],"-prediction-max-ref-num",config["prediction_max_ref_num"],
               "-ordering-scope",config["ordering_scope"],"-ordering-score-max-ref-num",config["ordering_score_max_ref_num"],
               "-ordering-score-min-ref-num",config["ordering_score_min_ref_num"],
               "-ordering-candidate-num",config["ordering_candidate_num"],*predictor_arguments,
               *build_eb_schedule_arguments(config["eb_schedule"])]
    if config["ordering_scope"]=="local":
        arguments.extend(("-prediction-round-num",config["prediction_round_num"]))
    elif "prediction_round_num" in config:
        raise ValueError("global排序不使用prediction_round_num")
    if dag_cache_path is not None:
        arguments.extend(("-dag-cache",dag_cache_path))
    coordinate_transform=config.get("coordinate_transform","off")
    if coordinate_transform not in ("off","on","auto"):
        raise ValueError(f"未知坐标变换模式：{coordinate_transform}")
    if coordinate_transform!="off":
        arguments.extend(("-coordinate-transform",coordinate_transform))
    if "coordinate_transform_first_sample_ratio" in config:
        arguments.extend(("-coordinate-transform-first-sample-ratio",config["coordinate_transform_first_sample_ratio"]))
    if "coordinate_transform_second_sample_num" in config:
        arguments.extend(("-coordinate-transform-second-sample-num",config["coordinate_transform_second_sample_num"]))
    if "coordinate_transform_sample_file_num" in config:
        arguments.extend(("-coordinate-transform-sample-file-num",config["coordinate_transform_sample_file_num"]))
    has_partition_num="partition_num" in config
    has_partition_size="partition_size" in config
    if has_partition_num and has_partition_size:
        raise ValueError("partition_num和partition_size不能同时设置")
    if has_partition_num or has_partition_size:
        if "thread_num" not in config:
            raise ValueError("启用partition时必须显式设置thread_num")
        partition_name="partition_num" if has_partition_num else "partition_size"
        if not isinstance(config[partition_name],int) or config[partition_name]<=0:
            raise ValueError(f"{partition_name}必须是正整数")
        if not isinstance(config["thread_num"],int) or config["thread_num"]<=0:
            raise ValueError("thread_num必须是正整数")
        arguments.extend(("-"+partition_name.replace("_","-"),config[partition_name],"-thread-num",config["thread_num"]))
    elif "thread_num" in config:
        raise ValueError("thread_num只能和partition_num或partition_size一起设置")
    if config["label_flow"] is not None:
        if config["label_value_execution"] not in ("staged","fused"):
            raise ValueError(f"未知标签与数值执行方式：{config['label_value_execution']}")
        arguments.extend(("-label-value-execution",config["label_value_execution"]))
    return predictor,arguments

def build_spatial_unsz_arguments(config,data_dir,dag_cache_path):
    return build_spatial_prediction_arguments(config,data_dir,dag_cache_path,"{"+",".join(str(value) for value in range(config["time_step_num"]))+"}")

def build_temporal_unsz_arguments(config,data_dir,dag_cache_path):
    if config["time_step_num"]<2:
        raise ValueError("时间预测至少需要两个time step")
    coordinates=read_time_step_coordinates(config)
    value_flow=config["value_flow"]
    label_flow=config["label_flow"]
    use_anchor_order=is_anchor_interpolation_stride_8_flow(value_flow) or label_flow=="anchor_interpolation_stride_8"
    if not use_anchor_order and (value_flow in ("linear_interpolation","linear_interpolation_spatial_residual") or label_flow=="linear_interpolation_spatial_residual"):
        time_order=linear_interpolation_time_order(coordinates)
        time_expression="{"+",".join(str(value) for value in time_order)+"}"
    else:
        time_expression=f"[0,{config['time_step_num']-1}]"
    predictor,arguments=build_spatial_prediction_arguments(config,data_dir,dag_cache_path,time_expression)
    if "time_step_coordinate_file" in config:
        arguments.extend(("-time-step-coordinate-file",Path(config["data_dir"])/config["time_step_coordinate_file"]))
    if value_flow!="spatial":
        arguments.extend(("-value-flow",value_flow))
    if label_flow=="constant_extrapolation":
        arguments.extend(("-label-flow","constant_extrapolation"))
    elif label_flow=="linear_interpolation_spatial_residual":
        arguments.extend(("-label-flow","linear_interpolation_spatial_residual"))
    elif label_flow=="anchor_interpolation_stride_8":
        arguments.extend(("-label-flow",label_flow))
    elif label_flow not in (None,"spatial"):
        raise ValueError(f"未知标签流程：{label_flow}")
    return predictor,arguments

def build_unsz_commands(config,data_dir,dag_cache_path):
    if config["value_flow"]!="spatial" or config["label_flow"] not in (None,"spatial"):
        predictor,arguments=build_temporal_unsz_arguments(config,data_dir,dag_cache_path)
    else:
        predictor,arguments=build_spatial_unsz_arguments(config,data_dir,dag_cache_path)
    if "point_set_config_path" in config:
        point_set_config_path=resolve_point_set_config_path(config["point_set_config_path"])
        if not point_set_config_path.is_file():
            raise FileNotFoundError(f"找不到点集配置文件：{point_set_config_path}")
        compression_error=["-point-set-config",point_set_config_path,"-predictor",predictor]
        decompression_error=compression_error
    else:
        rel_eb=config["rel_eb"]
        compression_error=["-M","REL",rel_eb,"-predictor",predictor]
        decompression_error=["-M","REL",rel_eb,"-predictor",predictor]
    compression_command=[project_root/"install/bin/s1_compress",*compression_error,*arguments]
    decompression_command=[project_root/"install/bin/s1_decompress",*decompression_error,*arguments]
    return compression_command,decompression_command

def expand_unsz_flows(config):
    flow=config["flow"]
    return [config|{"label_flow":label_flow,"value_flow":value_flow} for label_flow in flow.get("label",[None]) for value_flow in flow["value"]]

# 压缩器执行
def run_unsz(config):
    source_data_dir=Path(config["data_dir"])
    source_data_paths=generate_input_file_paths(config)
    required_paths=[source_data_dir/"coords.dat",*source_data_paths]
    missing_paths=[str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("缺少测试文件："+", ".join(missing_paths))
    dag_cache_path=None
    if "dag_cache_path" in config:
        if not config.get("allow_dag_cache_for_screening",False):
            raise ValueError("正式实验禁止使用DAG缓存；只有筛选实验可以显式设置allow_dag_cache_for_screening=true")
        dag_cache_path=Path(config["dag_cache_path"].format(field=config["field"]))
        if not dag_cache_path.is_absolute():
            dag_cache_path=source_data_dir/dag_cache_path
        dag_cache_path.parent.mkdir(parents=True,exist_ok=True)
    compression_command,decompression_command=build_unsz_commands(config,source_data_dir,dag_cache_path)
    compression_output=run_command(compression_command,silent=True)
    staged=config.get("label_value_execution")=="staged"
    if staged:
        result={**parse_unsz_compression_output(compression_output),"decompression_preprocessing_time":0,"decompression_time":0,"staged":True}
    else:
        decompression_output=run_command(decompression_command,silent=True)
        result={**parse_unsz_output(compression_output,decompression_output),"staged":False}
    error_configuration=config["point_set_config_path"] if "point_set_config_path" in config else f"REL {config['rel_eb']}"
    print(f"总结果：compressor=unsz,task={config['name']},data_dir={config['data_dir']},field={config['field']},error_configuration={error_configuration},br={result['br']},psnr={result['psnr']}")
    if staged:
        print("staged只估计压缩效果，不记录或比较速度")
    else:
        print(f"计时：compression_preprocessing={result['compression_preprocessing_time']} s,decompression_preprocessing={result['decompression_preprocessing_time']} s,compression={result['compression_time']} s,decompression={result['decompression_time']} s,compression+decompression={result['compression_time']+result['decompression_time']} s")
    command="compression: "+shlex.join(str(value) for value in compression_command)
    if not staged:
        command+="; decompression: "+shlex.join(str(value) for value in decompression_command)
    return {"compressor":"unsz",**result,"command":command}

def run_umc(config,rel_eb,data_range):
    source_data_dir=Path(config["data_dir"])
    field=config["field"]
    time_step_num=config["time_step_num"]
    source_data_paths=generate_input_file_paths(config)
    required_paths=[source_data_dir/"coords.dat",source_data_dir/"conn.dat",*source_data_paths]
    missing_paths=[str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("缺少测试文件："+", ".join(missing_paths))
    data_paths=[str(path) for path in source_data_paths]
    umc=config["umc"]
    from_edge=umc["from_edge"]
    coordinate_dim=config["coordinate_dim"]
    reorder=umc["reorder"]
    if reorder!=1:
        raise ValueError(f"正式UMC实验必须使用reorder=1，当前为{reorder}")
    method=umc["method"]
    neighbor_num=umc["neighbor_num"]
    data_dir=str(source_data_dir)+"/"
    if method=="NBP-IDW":
        compression_command=[umc_root/"build/test/test_compress",data_dir,field,"coords.dat","conn.dat",from_edge,coordinate_dim,2,reorder,time_step_num,0,rel_eb,neighbor_num]
        decompression_command=[umc_root/"build/test/test_decompress",data_dir,field,"coords.dat","conn.dat",from_edge,coordinate_dim,2,reorder,time_step_num]
    elif method=="NBP-IDW-INP":
        compression_command=[umc_root/"build/test/test_compress",data_dir,field,"coords.dat","conn.dat",from_edge,coordinate_dim,4,reorder,time_step_num,0,rel_eb,neighbor_num,umc_inp_batch]
        decompression_command=[umc_root/"build/test/test_decompress",data_dir,field,"coords.dat","conn.dat",from_edge,coordinate_dim,4,reorder,time_step_num,umc_inp_batch]
    else:
        raise ValueError(f"未知UMC压缩方法：{method}")
    compression_output=run_command(compression_command,silent=True)
    br,psnr,compression_preprocessing_time,compression_time=parse_umc_compression_result(compression_output,data_paths,data_range)
    decompression_output=run_command(decompression_command,silent=True)
    decompression_preprocessing_time,decompression_time,decompressed_psnr,decompressed_max_error=parse_umc_decompression_result(decompression_output,data_paths,data_range)
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-5):
        raise RuntimeError(f"UMC压缩端和独立解压端总PSNR不一致：{psnr} != {decompressed_psnr}")
    print(f"UMC独立解压检查通过：total psnr={decompressed_psnr},max error={decompressed_max_error}")
    disk_mse,disk_max_error=calculate_umc_disk_error(data_paths,rel_eb)
    disk_psnr=math.inf if disk_mse==0 else 10*math.log10((data_range[1]-data_range[0])**2/disk_mse)
    if not math.isclose(psnr,disk_psnr,rel_tol=1e-6,abs_tol=1e-5):
        raise RuntimeError(f"UMC压缩端和磁盘输出总PSNR不一致：{psnr} != {disk_psnr}")
    if not math.isclose(decompressed_max_error,disk_max_error,rel_tol=1e-5,abs_tol=1e-5):
        raise RuntimeError(f"UMC独立解压和磁盘输出最大误差不一致：{decompressed_max_error} != {disk_max_error}")
    print(f"UMC逐点磁盘输出与误差界检查通过：total psnr={disk_psnr},max error={disk_max_error}")
    print(f"总结果：compressor=umc,task={config['name']},data_dir={config['data_dir']},field={field},rel_eb={rel_eb},br={br},psnr={psnr}")
    print(f"计时：compression_preprocessing={compression_preprocessing_time} s,decompression_preprocessing={decompression_preprocessing_time} s,compression={compression_time} s,decompression={decompression_time} s,compression+decompression={compression_time+decompression_time} s")
    return {"compressor":"umc","rel_eb":rel_eb,"br":br,"psnr":psnr,"compression_preprocessing_time":compression_preprocessing_time,"decompression_preprocessing_time":decompression_preprocessing_time,"compression_time":compression_time,"decompression_time":decompression_time,"command":"compression: "+shlex.join(str(value) for value in compression_command)+"; decompression: "+shlex.join(str(value) for value in decompression_command)}

# 任务循环和命令行入口
def run_task(task):
    for field in task["field_list"]:
        field_config=task|{"field":field}
        input_paths=generate_input_file_paths(field_config)
        total_bytes=sum(path.stat().st_size for path in input_paths)
        data_range=calculate_data_range(input_paths) if "umc" in task["compressor_list"] else None
        result_path=Path(task["csv_path"].format(field=field))
        if not result_path.is_absolute():
            result_path=project_root/result_path
        result_path.parent.mkdir(parents=True,exist_ok=True)
        normalize_result_csv_columns(result_path)
        unsz_configs=expand_unsz_flows(field_config|task["unsz"]) if "unsz" in task["compressor_list"] else []
        point_set_config_path_list=task.get("unsz",{}).get("point_set_config_path_list")
        test_vec=[]
        if point_set_config_path_list is None:
            for rel_eb in task["rel_eb_list"]:
                if "unsz" in task["compressor_list"]:
                    test_vec.extend(("unsz",config|{"rel_eb":rel_eb}) for config in unsz_configs)
                if "umc" in task["compressor_list"]:
                    test_vec.append(("umc",field_config|{"umc":task["umc"],"rel_eb":rel_eb}))
        else:
            scheduled_umc_rel_eb=set()
            for path in point_set_config_path_list:
                if "unsz" in task["compressor_list"]:
                    test_vec.extend(("unsz",config|{"point_set_config_path":path}) for config in unsz_configs)
                if "umc" in task["compressor_list"]:
                    for rel_eb in read_point_set_rel_eb_list(path):
                        if rel_eb not in scheduled_umc_rel_eb:
                            test_vec.append(("umc",field_config|{"umc":task["umc"],"rel_eb":rel_eb}))
                            scheduled_umc_rel_eb.add(rel_eb)
            if "umc" in task["compressor_list"] and not scheduled_umc_rel_eb:
                raise ValueError(f"task {task['name']}的点集配置中没有可供UMC使用的REL EB")
        for compressor,config in test_vec:
            label_flow=config.get("label_flow")
            value_flow=config.get("value_flow")
            rel_eb=config.get("rel_eb")
            point_set_config_path=config.get("point_set_config_path")
            error_configuration=f"REL={rel_eb:g}" if rel_eb is not None else f"point_set_config={point_set_config_path}"
            print(f"开始：task={task['name']},field={field},compressor={compressor},label_flow={label_flow},value_flow={value_flow},{error_configuration}",flush=True)
            if compressor=="unsz":
                result=run_unsz(config)
            else:
                result=run_umc(config,rel_eb,data_range)
            label_value_execution=config.get("label_value_execution","not_applicable") if compressor=="unsz" else "not_applicable"
            item={"task":task["name"],"field":field,"compressor":compressor,"label_flow":label_flow or "not_applicable","value_flow":value_flow or "not_applicable","label_value_execution":label_value_execution,"rel_eb":rel_eb if rel_eb is not None else "not_applicable","point_set_config":point_set_config_path or "not_applicable"}
            for column_name in ("br","psnr","command"):
                write_csv_item(str(result_path),item,column_name,result[column_name])
            staged=compressor=="unsz" and result.get("staged",False)
            timing_columns=("compression_preprocessing_time","decompression_preprocessing_time","compression_time","decompression_time","compression_decompression_time","compression_preprocessing_MB_s","decompression_preprocessing_MB_s","compression_MB_s","decompression_MB_s")
            if staged:
                for column_name in timing_columns:
                    write_csv_item(str(result_path),item,column_name,"not_applicable")
            else:
                for column_name in ("compression_preprocessing_time","decompression_preprocessing_time","compression_time","decompression_time"):
                    write_csv_item(str(result_path),item,column_name,result[column_name])
                for column_name,value in (("compression_decompression_time",result["compression_time"]+result["decompression_time"]),("compression_preprocessing_MB_s",calculate_throughput(total_bytes,result["compression_preprocessing_time"],"压缩预处理时间")),("decompression_preprocessing_MB_s",calculate_throughput(total_bytes,result["decompression_preprocessing_time"],"解压预处理时间")),("compression_MB_s",calculate_throughput(total_bytes,result["compression_time"],"压缩时间")),("decompression_MB_s",calculate_throughput(total_bytes,result["decompression_time"],"解压时间"))):
                    write_csv_item(str(result_path),item,column_name,value)
            for column_name,value in (("data_dir",task["data_dir"]),("time_step_num",task["time_step_num"]),("input_bytes",total_bytes)):
                write_csv_item(str(result_path),item,column_name,value)
            if task.get("cleanup_after_each_test",False):
                cleanup_test_outputs(field_config,compressor)
        print(f"完成：task={task['name']}，field={field}，结果{result_path}",flush=True)

def cli(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="运行configs目录中的UNSZ实验配置")
    parser.add_argument("json",help="configs中的JSON配置名称，或JSON文件路径")
    parser.add_argument("--skip-build",action="store_true",help="使用现有UNSZ安装而不重新构建")
    arguments=parser.parse_args(command_line_arguments)
    try:
        tasks=load_json_tasks(resolve_json_path(arguments.json))
    except (FileNotFoundError,TypeError,ValueError) as error:
        parser.error(str(error))
    if not arguments.skip_build and any("unsz" in task["compressor_list"] for task in tasks):
        build_project(build_max_cpus,silent_build)
    check_baseline_programs(tasks)
    for task in tasks:
        run_task(task)

if __name__=="__main__":
    cli()
