import math,re

# UNSZ命令由t1执行，这里只解析并核对输出
def read_output_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def read_optional_output_float(pattern,text):
    match=re.search(pattern,text,re.MULTILINE)
    return float(match.group(1)) if match else None

def parse_unsz_compression_output(compression_output):
    selected_pipeline_match=re.search(r"^selected pipeline: label=([^,\r\n]+),value=([^\r\n]+)$",compression_output,re.MULTILINE)
    if selected_pipeline_match is None:
        selected_pipeline_match=re.search(r"^automatic pipeline selected: label=([^,]+),value=([^,]+),",compression_output,re.MULTILINE)
    tensor_pipeline_match=re.search(r"^selected tensor pipeline: ([^\r\n]+)$",compression_output,re.MULTILINE)
    candidate_pattern=r"^automatic pipeline candidate: label=([^,]+),value=([^,]+),estimated_label_bytes=([^,]+),estimated_value_bytes=([^,]+),estimated_total_bytes=([^,]+),selection_score=([^,]+),estimated_label_br=([^,]+),estimated_value_br=([^,]+),evaluation_time=([^ ]+) s$"
    candidates=[{"label_pipeline":match[0],"value_pipeline":match[1],"estimated_label_bytes":float(match[2]),"estimated_value_bytes":float(match[3]),"estimated_total_bytes":float(match[4]),"selection_score":float(match[5]),"estimated_label_br":float(match[6]),"estimated_value_br":float(match[7]),"evaluation_time":float(match[8])} for match in re.findall(candidate_pattern,compression_output,re.MULTILINE)]
    transform_match=re.search(r"^coordinate rescaling rescaled flatness=([^,]+),ratio=([^,]+),use=([01])$",compression_output,re.MULTILINE)
    diagonal_match=re.search(r"^coordinate rescaling diagonal=\[([^]]+)\]$",compression_output,re.MULTILINE)
    slice_dag_time=read_optional_output_float(r"^slice DAG preprocessing time= ([^ ]+) s$",compression_output) or 0
    compression_preprocessing_time=read_optional_output_float(r"^total compression preprocessing time excluding [^=]+= ([^ ]+) s$",compression_output)
    if compression_preprocessing_time is None:
        compression_preprocessing_time=read_output_float(r"^preprocessing time excluding [^=]+= ([^ ]+) s$",compression_output,"压缩端预处理时间")+slice_dag_time
    return {"br":read_output_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":read_output_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR"),"max_error_over_eb":read_output_float(r"^normalized squared error sum= [^,]+, normalized value count= [^,]+, normalized psnr= [^,]+, maximum error over eb= ([^\s]+)$",compression_output,"压缩最大误差/EB"),"label_stream_br":read_output_float(r"^total label stream br= ([^,]+),",compression_output,"标签码流BR"),"value_stream_br":read_output_float(r"^total label stream br= [^,]+, total value stream br= ([^,]+),",compression_output,"数值码流BR"),"shared_metadata_br":read_output_float(r"^total label stream br= [^,]+, total value stream br= [^,]+, total shared metadata br= ([^ ]+)$",compression_output,"共享元数据BR"),"compression_preprocessing_time":compression_preprocessing_time,"coordinate_rescaling_time":read_output_float(r"^coordinate rescaling preprocessing time excluding training-data input= ([^ ]+) s$",compression_output,"坐标缩放时间"),"prediction_structure_time":read_output_float(r"^prediction DAG and weights preprocessing time= ([^ ]+) s$",compression_output,"DAG和权重时间"),"automatic_pipeline_selection_time":read_output_float(r"^automatic pipeline preprocessing time excluding field-data input= ([^ ]+) s$",compression_output,"自动pipeline时间"),"slice_dag_time":slice_dag_time,"reference_slices_time":read_optional_output_float(r"^reference slice preprocessing time= ([^ ]+) s$",compression_output) or 0,"compression_time":read_output_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"selected_label_pipeline":tensor_pipeline_match.group(1) if tensor_pipeline_match else selected_pipeline_match.group(1) if selected_pipeline_match else None,"selected_value_pipeline":tensor_pipeline_match.group(1) if tensor_pipeline_match else selected_pipeline_match.group(2) if selected_pipeline_match else None,"automatic_pipeline_candidates":candidates,"coordinate_rescaling_original_flatness":read_optional_output_float(r"^coordinate rescaling original flatness=([^ ]+)$",compression_output),"coordinate_rescaling_rescaled_flatness":float(transform_match.group(1)) if transform_match else None,"coordinate_rescaling_flatness_ratio":float(transform_match.group(2)) if transform_match else None,"coordinate_rescaling_enabled":int(transform_match.group(3)) if transform_match else 0,"coordinate_rescaling_diagonal":[float(value) for value in diagonal_match.group(1).split(",")] if diagonal_match else [],"coordinate_rescaling_untrained_group_mse_ratio":None,"coordinate_rescaling_selected_group_loss_power":3 if diagonal_match else None}

def parse_unsz_output(compression_output,decompression_output):
    result=parse_unsz_compression_output(compression_output)
    psnr=result["psnr"]
    decompressed_psnr=read_output_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"UNSZ压缩端和解压端总PSNR不一致：{psnr} != {decompressed_psnr}")
    decompressed_max_error_over_eb=read_output_float(r"^normalized squared error sum= [^,]+, normalized value count= [^,]+, normalized psnr= [^,]+, maximum error over eb= ([^\s]+)$",decompression_output,"解压最大误差/EB")
    if not math.isclose(result["max_error_over_eb"],decompressed_max_error_over_eb,rel_tol=1e-9,abs_tol=1e-9):
        raise RuntimeError(f"UNSZ压缩端和解压端最大误差/EB不一致：{result['max_error_over_eb']} != {decompressed_max_error_over_eb}")
    slice_dag_time=read_optional_output_float(r"^slice DAG preprocessing time= ([^ ]+) s$",decompression_output) or 0
    decompression_preprocessing_time=read_optional_output_float(r"^total decompression preprocessing time excluding [^=]+= ([^ ]+) s$",decompression_output)
    if decompression_preprocessing_time is None:
        decompression_preprocessing_time=read_output_float(r"^preprocessing time excluding [^=]+= ([^ ]+) s$",decompression_output,"解压端预处理时间")+slice_dag_time
    # per-slice协议的PSNR统计：每个slice普通值的平方误差除以该slice普通值range的平方，见UNSZ FIELD RESULT行
    field_results=re.findall(r"^UNSZ FIELD RESULT side=decompression field=\S+ slices=\d+ values=(\d+) valid_values=(\d+) compressed_bytes=\d+ squared_error=\S+ valid_squared_error=\S+ slice_normalized_valid_squared_error=(\S+) slice_normalized_squared_error=(\S+) ",decompression_output,re.MULTILINE)
    if len(field_results)!=1:
        raise RuntimeError(f"UNSZ解压输出应包含一行field统计，实际为{len(field_results)}行")
    result["value_num"]=int(field_results[0][0])
    result["valid_value_num"]=int(field_results[0][1])
    result["slice_normalized_valid_squared_error"]=float(field_results[0][2])
    result["slice_normalized_squared_error"]=float(field_results[0][3])
    return {**result,"decompression_preprocessing_time":decompression_preprocessing_time,"decompression_slice_dag_time":slice_dag_time,"decompression_time":read_output_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
