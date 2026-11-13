import argparse,csv,json,math,os,re
from pathlib import Path
from py_codes.parse_umc_output import calculate_umc_disk_error,parse_umc_compression_result,parse_umc_decompression_result
from py_codes.parse_unsz_output import parse_unsz_compression_output,parse_unsz_output
from third_party.csv_tools.py_codes.write_csv_item import write_csv_item
from third_party.get_max_cpus.py_codes.get_max_cpus import get_max_cpus
from third_party.run_command.run_command import run_command

project_root=Path(__file__).resolve().parent
config_directory=project_root/"configs"
umc_root=project_root/"third_party/umc"
umc_inp_batch=10
build_max_cpus=get_max_cpus(8)
silent_build=False
allowed_unsz_config_keys={"label_config_path_list","partition_num","thread_num","pipeline","coordinate_rescaling","extra_arguments"}

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
    for task in tasks:
        unsz_config=task.get("unsz",{})
        if not isinstance(unsz_config,dict):
            raise TypeError(f"task {task.get('name','<unnamed>')}的unsz必须是object")
        unsupported_keys=sorted(set(unsz_config)-allowed_unsz_config_keys)
        if unsupported_keys:
            raise ValueError(f"task {task.get('name','<unnamed>')}包含已删除的UNSZ参数：{','.join(unsupported_keys)}")
    return tasks

def select_tasks(tasks,task_names,fields,compressors):
    selected_tasks=[]
    for task in tasks:
        if task_names and task["name"] not in task_names:
            continue
        selected_fields=[field for field in task["field_list"] if not fields or field in fields]
        selected_compressors=[compressor for compressor in task["compressor_list"] if not compressors or compressor in compressors]
        if selected_fields and selected_compressors:
            selected_tasks.append(task|{"field_list":selected_fields,"compressor_list":selected_compressors})
    if not selected_tasks:
        raise ValueError("task、field和compressor筛选后没有实验")
    return selected_tasks

def resolve_label_config_path(path):
    path=Path(path)
    return path if path.is_absolute() else project_root/path

def read_label_rel_eb_list(path):
    path=resolve_label_config_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到标签配置文件：{path}")
    with path.open(encoding="utf-8") as file:
        document=json.load(file)
    rel_eb_vec=[label["eb_value"] for label in document["labels"] if label["method"]=="predict" and label["eb_mode"]=="REL"]
    return list(dict.fromkeys(rel_eb_vec))

# 公共路径和程序
def resolve_tensor_metadata_path(config):
    path=Path(config["tensor_metadata"])
    return path if path.is_absolute() else Path(config["data_dir"])/path

def load_tensor_field_config(config,field=None):
    path=resolve_tensor_metadata_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"找不到tensor metadata：{path}")
    with path.open(encoding="utf-8") as file:
        metadata=json.load(file)
    field=field or config["field"]
    try:
        return metadata["fields"][field]
    except KeyError as error:
        raise ValueError(f"tensor metadata中没有field：{field}") from error

def field_slice_num(config,field=None):
    if "tensor_metadata" not in config:
        return config["slice_num"]
    shape=load_tensor_field_config(config,field)["shape"]
    if not isinstance(shape,list) or len(shape)!=2 or any(not isinstance(value,int)or value<=0 for value in shape):
        raise ValueError("tensor metadata的shape必须包含两个正整数")
    return math.prod(shape)

def generate_input_file_paths(config):
    directory=Path(config["data_dir"])
    return [directory/f"{config['field']}.dat.{slice_id}" for slice_id in range(field_slice_num(config))]

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
        output_dir=Path(config.get("output_dir",config["data_dir"]))
        for input_path in generate_input_file_paths(config):
            output_path=output_dir/input_path.name
            remove_file_if_exists(str(output_path)+".unsz")
            remove_file_if_exists(str(output_path)+".unsz.bin")
        remove_file_if_exists(output_dir/f"{config['field']}.joint.unsz")
        remove_file_if_exists(output_dir/f"{config['field']}.unsz.coordinate_rescaling.txt")
    elif compressor=="umc":
        for input_path in generate_input_file_paths(config):
            remove_file_if_exists(str(input_path)+".umc")
            remove_file_if_exists(str(input_path)+".umc.out")

result_identity_columns=("task","field","compressor","label_pipeline","value_pipeline","label_value_execution","rel_eb","label_config")

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
def build_spatial_prediction_arguments(config,data_dir,time_expression):
    if "field_list" in config:
        field_arguments=["-field-list","{"+",".join(config["field_list"])+"}"]
    else:
        field_arguments=["-field",config["field"]]
    arguments=["-data-dir",data_dir,*field_arguments]
    if "output_dir" in config:
        arguments.extend(("-output-dir",config["output_dir"]))
    if "tensor_metadata" in config:
        arguments.extend(("-tensor-metadata",config["tensor_metadata"]))
    else:
        arguments.extend(("-slice",time_expression))
    arguments.extend(("-coordinate-dim",config["coordinate_dim"]))
    if "coordinate_rescaling" in config:
        coordinate_rescaling=config["coordinate_rescaling"]
        if coordinate_rescaling not in ("off","on","auto"):
            raise ValueError(f"未知坐标缩放模式：{coordinate_rescaling}")
        arguments.extend(("-coordinate-rescaling",coordinate_rescaling))
    if "partition_num" in config:
        if "thread_num" not in config:
            raise ValueError("启用partition时必须显式设置thread_num")
        if not isinstance(config["partition_num"],int) or config["partition_num"]<=0:
            raise ValueError("partition_num必须是正整数")
        if not isinstance(config["thread_num"],int) or config["thread_num"]<=0:
            raise ValueError("thread_num必须是正整数")
        arguments.extend(("-partition-num",config["partition_num"],"-thread-num",config["thread_num"]))
    elif "thread_num" in config:
        raise ValueError("thread_num只能和partition_num一起设置")
    # 消融实验追加的选项，压缩和解压都会带上
    arguments.extend(config.get("extra_arguments",[]))
    return arguments

def build_spatial_unsz_arguments(config,data_dir):
    return build_spatial_prediction_arguments(config,data_dir,"{"+",".join(str(value) for value in range(config["slice_num"]))+"}")

def build_tensor_unsz_arguments(config,data_dir):
    if config["slice_num"]<2:
        raise ValueError("跨slice预测至少需要两个slice")
    value_pipeline=config["value_pipeline"]
    label_pipeline=config["label_pipeline"]
    time_expression=f"[0,{config['slice_num']-1}]"
    arguments=build_spatial_prediction_arguments(config,data_dir,time_expression)
    if "slice_coordinate_file" in config:
        arguments.extend(("-slice-coordinate-file",Path(config["data_dir"])/config["slice_coordinate_file"]))
    if config.get("force_pipeline",False):
        arguments.extend(("-value-pipeline",value_pipeline))
    if not config.get("force_pipeline",False):
        return arguments
    if label_pipeline=="extrapolation":
        arguments.extend(("-label-pipeline","extrapolation"))
    elif label_pipeline=="interpolation_spatial_residual":
        arguments.extend(("-label-pipeline","interpolation_spatial_residual"))
    elif label_pipeline=="anchor_interpolation":
        arguments.extend(("-label-pipeline",label_pipeline))
    elif label_pipeline=="auto":
        arguments.extend(("-label-pipeline","auto"))
    elif label_pipeline=="independent":
        arguments.extend(("-label-pipeline","independent"))
    elif label_pipeline is not None:
        raise ValueError(f"未知标签流程：{label_pipeline}")
    return arguments

def build_unsz_commands(config,data_dir):
    if "tensor_metadata" in config:
        if config.get("force_pipeline",False):
            raise ValueError("二维tensor暂不支持通过实验配置强制指定pipeline")
        arguments=build_spatial_prediction_arguments(config,data_dir,None)
    elif config["slice_num"]>1 and (config.get("force_pipeline",False) or config["value_pipeline"]!="independent" or config["label_pipeline"] not in (None,"independent")):
        arguments=build_tensor_unsz_arguments(config,data_dir)
    else:
        arguments=build_spatial_unsz_arguments(config,data_dir)
    if "label_config_path" in config:
        label_config_path=resolve_label_config_path(config["label_config_path"])
        if not label_config_path.is_file():
            raise FileNotFoundError(f"找不到标签配置文件：{label_config_path}")
        compression_error=["-label-config",label_config_path]
        decompression_error=compression_error
    else:
        rel_eb=config["rel_eb"]
        compression_error=["-M","REL",rel_eb]
        decompression_error=["-M","REL",rel_eb]
    # 环境变量UNSZ_INSTALL_DIR可以指定另一份安装，供与正在运行的作业隔离的实验使用
    install_dir=Path(os.environ.get("UNSZ_INSTALL_DIR",project_root/"install"))
    compression_command=[install_dir/"bin/s1_compress",*compression_error,*arguments]
    decompression_command=[install_dir/"bin/s1_decompress",*decompression_error,*arguments,"-skip-decompressed-output"]
    return compression_command,decompression_command

def expand_unsz_pipelines(config):
    if "pipeline" not in config:
        has_multiple_slices=any(field_slice_num(config,field)>1 for field in config["field_list"])
        value_pipeline="auto" if has_multiple_slices else "independent"
        has_label="label_config_path_list" in config or "label_config_path" in config
        label_pipeline="auto" if has_label and has_multiple_slices else None
        return [config|{"label_pipeline":label_pipeline,"value_pipeline":value_pipeline,"force_pipeline":False}]
    pipeline=config["pipeline"]
    if "pair" in pipeline:
        if "label" in pipeline or "value" in pipeline:
            raise ValueError("pipeline.pair不能与pipeline.label或pipeline.value同时使用")
        return [config|{"label_pipeline":label_pipeline,"value_pipeline":value_pipeline,"force_pipeline":True} for label_pipeline,value_pipeline in pipeline["pair"]]
    return [config|{"label_pipeline":label_pipeline,"value_pipeline":value_pipeline,"force_pipeline":True} for label_pipeline in pipeline.get("label",[None]) for value_pipeline in pipeline["value"]]

# 压缩器执行
def extract_unsz_field_output(output,field,include_preamble):
    pattern=rf"^BEGIN UNSZ FIELD {re.escape(field)}\n(.*?)^END UNSZ FIELD {re.escape(field)}$"
    match=re.search(pattern,output,re.MULTILINE|re.DOTALL)
    if match is None:
        raise RuntimeError(f"UNSZ输出中缺少field边界：{field}")
    first_field_begin=output.find("BEGIN UNSZ FIELD ")
    preamble=output[:first_field_begin] if first_field_begin>=0 else ""
    return (preamble+match.group(1)) if include_preamble else match.group(1)

def run_unsz_fields(config):
    source_data_dir=Path(config["data_dir"])
    field_list=config["field_list"] if "field_list" in config else [config["field"]]
    source_data_paths=[path for field in field_list for path in generate_input_file_paths(config|{"field":field})]
    required_paths=[source_data_dir/"coords.dat",*source_data_paths]
    if "tensor_metadata" in config:
        required_paths.append(resolve_tensor_metadata_path(config))
    missing_paths=[str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("缺少测试文件："+", ".join(missing_paths))
    compression_command,decompression_command=build_unsz_commands(config,source_data_dir)
    compression_output=run_command(compression_command)
    decompression_output=run_command(decompression_command)
    result_by_field={}
    for field in field_list:
        if len(field_list)==1:
            field_compression_output=compression_output
            field_decompression_output=decompression_output
        else:
            field_compression_output=extract_unsz_field_output(compression_output,field,field==field_list[0])
            field_decompression_output=extract_unsz_field_output(decompression_output,field,field==field_list[0])
        result=parse_unsz_output(field_compression_output,field_decompression_output)
        error_configuration=config["label_config_path"] if "label_config_path" in config else f"REL {config['rel_eb']}"
        selected_label_pipeline=result["selected_label_pipeline"] or config.get("label_pipeline") or "not_applicable"
        selected_value_pipeline=result["selected_value_pipeline"] or config["value_pipeline"]
        print(f"总结果：compressor=unsz,task={config['name']},data_dir={config['data_dir']},field={field},error_configuration={error_configuration},selected_label_pipeline={selected_label_pipeline},selected_value_pipeline={selected_value_pipeline},br={result['br']},psnr={result['psnr']}")
        print(f"计时：compression_preprocessing={result['compression_preprocessing_time']} s,decompression_preprocessing={result['decompression_preprocessing_time']} s,compression={result['compression_time']} s,decompression={result['decompression_time']} s,compression+decompression={result['compression_time']+result['decompression_time']} s")
        result_by_field[field]={"compressor":"unsz",**result}
    return result_by_field

def run_umc(config,rel_eb):
    if "tensor_metadata" in config:
        raise ValueError("UMC实验入口不接受UNSZ二维tensor metadata；需要使用单独的UMC数据适配器")
    source_data_dir=Path(config["data_dir"])
    field=config["field"]
    slice_num=config["slice_num"]
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
        compression_command=[umc_root/"build/test/test_compress",data_dir,field,"coords.dat","conn.dat",from_edge,coordinate_dim,2,reorder,slice_num,0,rel_eb,neighbor_num]
        decompression_command=[umc_root/"build/test/test_decompress",data_dir,field,"coords.dat","conn.dat",from_edge,coordinate_dim,2,reorder,slice_num]
    elif method=="NBP-IDW-INP":
        compression_command=[umc_root/"build/test/test_compress",data_dir,field,"coords.dat","conn.dat",from_edge,coordinate_dim,4,reorder,slice_num,0,rel_eb,neighbor_num,umc_inp_batch]
        decompression_command=[umc_root/"build/test/test_decompress",data_dir,field,"coords.dat","conn.dat",from_edge,coordinate_dim,4,reorder,slice_num,umc_inp_batch]
    else:
        raise ValueError(f"未知UMC压缩方法：{method}")
    compression_output=run_command(compression_command)
    br,compression_preprocessing_time,compression_time=parse_umc_compression_result(compression_output,data_paths)
    decompression_output=run_command(decompression_command)
    decompression_preprocessing_time,decompression_time=parse_umc_decompression_result(decompression_output,data_paths)
    _,disk_max_error,slice_statistics=calculate_umc_disk_error(data_paths,rel_eb)
    print(f"UMC逐点误差界检查通过：max error={disk_max_error}")
    print(f"总结果：compressor=umc,task={config['name']},data_dir={config['data_dir']},field={field},rel_eb={rel_eb},br={br}")
    print(f"计时：compression_preprocessing={compression_preprocessing_time} s,decompression_preprocessing={decompression_preprocessing_time} s,compression={compression_time} s,decompression={decompression_time} s,compression+decompression={compression_time+decompression_time} s")
    return {**slice_statistics,"compressor":"umc","rel_eb":rel_eb,"br":br,"compression_preprocessing_time":compression_preprocessing_time,"decompression_preprocessing_time":decompression_preprocessing_time,"compression_time":compression_time,"decompression_time":decompression_time}

# 任务循环和命令行入口
def prepare_field_result_metadata(task):
    result={}
    for field in task["field_list"]:
        field_config=task|{"field":field}
        input_paths=generate_input_file_paths(field_config)
        result_path=Path(task["csv_path"].format(field=field))
        if not result_path.is_absolute():
            result_path=project_root/result_path
        result_path.parent.mkdir(parents=True,exist_ok=True)
        normalize_result_csv_columns(result_path)
        result[field]={"config":field_config,"input_paths":input_paths,"total_bytes":sum(path.stat().st_size for path in input_paths),"result_path":result_path}
    return result

def slice_valid_psnr(result):
    # 逐slice有效值PSNR：每个有效值的误差除以所在slice有效值range，再在全部有效值上平均
    error=float(result["slice_normalized_valid_squared_error"])
    return -10*math.log10(error/int(result["valid_value_num"])) if error>0 else math.inf

# 自动选择流程时评估过的候选流程，每个候选占一行，写在结果CSV所在目录下的automatic_pipeline_candidates目录里
def write_automatic_pipeline_candidates(result_path,item,candidates):
    candidate_path=result_path.parent/"automatic_pipeline_candidates"/result_path.name
    candidate_path.parent.mkdir(exist_ok=True)
    # 实验配置里的label_pipeline和value_pipeline改名为configured_*，这两个列名留给候选流程
    experiment={("configured_"+key if key in ("label_pipeline","value_pipeline") else key):value for key,value in item.items()}
    for candidate in candidates:
        row_info={**experiment,"label_pipeline":candidate["label_pipeline"],"value_pipeline":candidate["value_pipeline"]}
        for column_name,value in candidate.items():
            if column_name not in ("label_pipeline","value_pipeline"):
                write_csv_item(str(candidate_path),row_info,column_name,value)

def write_benchmark_result(task,field,compressor,config,result,metadata):
    label_pipeline=config.get("label_pipeline")
    value_pipeline=config.get("value_pipeline")
    rel_eb=config.get("rel_eb")
    label_config_path=config.get("label_config_path")
    label_value_execution="fused" if compressor=="unsz" and label_pipeline is not None else "not_applicable"
    item={"task":task["name"],"field":field,"compressor":compressor,"label_pipeline":label_pipeline or "not_applicable","value_pipeline":value_pipeline or "not_applicable","label_value_execution":label_value_execution,"rel_eb":rel_eb if rel_eb is not None else "not_applicable","label_config":label_config_path or "not_applicable"}
    result_path=metadata["result_path"]
    total_bytes=metadata["total_bytes"]
    for column_name in ("br","compression_preprocessing_time","decompression_preprocessing_time","compression_time","decompression_time"):
        write_csv_item(str(result_path),item,column_name,result[column_name])
    write_csv_item(str(result_path),item,"psnr",slice_valid_psnr(result))
    for column_name,value in (("compression_decompression_time",result["compression_time"]+result["decompression_time"]),("compression_preprocessing_MB_s",calculate_throughput(total_bytes,result["compression_preprocessing_time"],"压缩预处理时间")),("decompression_preprocessing_MB_s",calculate_throughput(total_bytes,result["decompression_preprocessing_time"],"解压预处理时间")),("compression_MB_s",calculate_throughput(total_bytes,result["compression_time"],"压缩时间")),("decompression_MB_s",calculate_throughput(total_bytes,result["decompression_time"],"解压时间"))):
        write_csv_item(str(result_path),item,column_name,value)
    if compressor=="unsz":
        for column_name in ("max_error_over_eb","coordinate_rescaling_time","prediction_structure_time","automatic_pipeline_selection_time","label_stream_br","value_stream_br","shared_metadata_br"):
            write_csv_item(str(result_path),item,column_name,result[column_name])
        for column_name in ("coordinate_rescaling_original_flatness","coordinate_rescaling_rescaled_flatness","coordinate_rescaling_flatness_ratio","coordinate_rescaling_enabled","coordinate_rescaling_untrained_group_mse_ratio","coordinate_rescaling_selected_group_loss_power"):
            write_csv_item(str(result_path),item,column_name,result[column_name] if result[column_name] is not None else "not_applicable")
        # 每个坐标方向的缩放系数各占一列，没有缩放或维数不到3时填not_applicable
        diagonal=result["coordinate_rescaling_diagonal"]
        for dimension in range(max(3,len(diagonal))):
            write_csv_item(str(result_path),item,f"coordinate_rescaling_diagonal_{dimension}",diagonal[dimension] if dimension<len(diagonal) else "not_applicable")
        write_csv_item(str(result_path),item,"selected_label_pipeline",result["selected_label_pipeline"] or label_pipeline or "not_applicable")
        write_csv_item(str(result_path),item,"selected_value_pipeline",result["selected_value_pipeline"] or value_pipeline)
        if result["automatic_pipeline_candidates"]:
            write_automatic_pipeline_candidates(result_path,item,result["automatic_pipeline_candidates"])
    # psnr列由这些统计计算：每个slice有效值的平方误差除以该slice有效值range的平方
    for column_name in ("value_num","valid_value_num","slice_normalized_valid_squared_error","slice_normalized_squared_error"):
        write_csv_item(str(result_path),item,column_name,result[column_name])
    for column_name,value in (("data_dir",task["data_dir"]),("slice_num",field_slice_num(config,field)),("input_bytes",total_bytes)):
        write_csv_item(str(result_path),item,column_name,value)

def run_task(task):
    field_metadata=prepare_field_result_metadata(task)
    label_config_path_list=task.get("unsz",{}).get("label_config_path_list")
    if "unsz" in task["compressor_list"]:
        grouped_config=task|task["unsz"]
        unsz_configs=expand_unsz_pipelines(grouped_config)
        if label_config_path_list is None:
            unsz_configs=[config|{"rel_eb":rel_eb} for rel_eb in task["rel_eb_list"] for config in unsz_configs]
        else:
            unsz_configs=[config|{"label_config_path":path} for path in label_config_path_list for config in unsz_configs]
        for config in unsz_configs:
            rel_eb=config.get("rel_eb")
            label_config_path=config.get("label_config_path")
            error_configuration=f"REL={rel_eb:g}" if rel_eb is not None else f"label_config={label_config_path}"
            print(f"开始：task={task['name']},fields={{{','.join(task['field_list'])}}},compressor=unsz,label_pipeline={config.get('label_pipeline')},value_pipeline={config.get('value_pipeline')},{error_configuration}",flush=True)
            result_by_field=run_unsz_fields(config)
            for field,result in result_by_field.items():
                write_benchmark_result(task,field,"unsz",config,result,field_metadata[field])
                if task.get("cleanup_after_each_test",False):
                    cleanup_test_outputs(field_metadata[field]["config"],"unsz")
    if "umc" in task["compressor_list"]:
        if label_config_path_list is None:
            umc_rel_eb_list=task["rel_eb_list"]
        else:
            umc_rel_eb_list=list(dict.fromkeys(rel_eb for path in label_config_path_list for rel_eb in read_label_rel_eb_list(path)))
            if not umc_rel_eb_list:
                raise ValueError(f"task {task['name']}的标签配置中没有可供UMC使用的REL EB")
        for field in task["field_list"]:
            metadata=field_metadata[field]
            field_config=metadata["config"]
            for rel_eb in umc_rel_eb_list:
                config=field_config|{"umc":task["umc"],"rel_eb":rel_eb}
                print(f"开始：task={task['name']},field={field},compressor=umc,REL={rel_eb:g}",flush=True)
                result=run_umc(config,rel_eb)
                write_benchmark_result(task,field,"umc",config,result,metadata)
                if task.get("cleanup_after_each_test",False):
                    cleanup_test_outputs(field_config,"umc")
    for field,metadata in field_metadata.items():
        print(f"完成：task={task['name']}，field={field}，结果{metadata['result_path']}",flush=True)

def cli(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="运行configs目录中的UNSZ实验配置")
    parser.add_argument("json",help="configs中的JSON配置名称，或JSON文件路径")
    parser.add_argument("--skip-build",action="store_true",help="使用现有UNSZ安装而不重新构建")
    parser.add_argument("--task",action="append",dest="task_names",help="只运行指定task，可重复")
    parser.add_argument("--field",action="append",dest="fields",help="只运行指定field，可重复")
    parser.add_argument("--compressor",action="append",choices=("unsz","umc"),dest="compressors",help="只运行指定压缩器，可重复")
    arguments=parser.parse_args(command_line_arguments)
    try:
        tasks=load_json_tasks(resolve_json_path(arguments.json))
        tasks=select_tasks(tasks,arguments.task_names,arguments.fields,arguments.compressors)
    except (FileNotFoundError,TypeError,ValueError) as error:
        parser.error(str(error))
    if not arguments.skip_build and any("unsz" in task["compressor_list"] for task in tasks):
        build_project(build_max_cpus,silent_build)
    check_baseline_programs(tasks)
    for task in tasks:
        run_task(task)

if __name__=="__main__":
    cli()
