import math,re

# UNSZ命令由t1执行，这里只解析并核对输出
def read_output_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def parse_unsz_compression_output(compression_output):
    selected_flow_match=re.search(r"^automatic flow selected: label=([^,]+),value=([^,]+),",compression_output,re.MULTILINE)
    candidate_pattern=r"^automatic flow candidate: label=([^,]+),value=([^,]+),estimated_label_bytes=([^,]+),estimated_value_bytes=([^,]+),estimated_total_bytes=([^,]+),selection_score=([^,]+),estimated_label_br=([^,]+),estimated_value_br=([^,]+),evaluation_time=([^ ]+) s$"
    candidates=[{"label_flow":match[0],"value_flow":match[1],"estimated_label_bytes":float(match[2]),"estimated_value_bytes":float(match[3]),"estimated_total_bytes":float(match[4]),"selection_score":float(match[5]),"estimated_label_br":float(match[6]),"estimated_value_br":float(match[7]),"evaluation_time":float(match[8])} for match in re.findall(candidate_pattern,compression_output,re.MULTILINE)]
    return {"br":read_output_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":read_output_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR"),"label_stream_br":read_output_float(r"^total label stream br= ([^,]+),",compression_output,"标签码流BR"),"value_stream_br":read_output_float(r"^total label stream br= [^,]+, total value stream br= ([^,]+),",compression_output,"数值码流BR"),"shared_metadata_br":read_output_float(r"^total label stream br= [^,]+, total value stream br= [^,]+, total shared metadata br= ([^ ]+)$",compression_output,"共享元数据BR"),"compression_preprocessing_time":read_output_float(r"^preprocessing time excluding [^=]+= ([^ ]+) s$",compression_output,"压缩端预处理时间"),"coordinate_transform_time":read_output_float(r"^coordinate transform preprocessing time excluding training-data input= ([^ ]+) s$",compression_output,"坐标变换时间"),"prediction_structure_time":read_output_float(r"^prediction DAG and weights preprocessing time= ([^ ]+) s$",compression_output,"DAG和权重时间"),"automatic_flow_selection_time":read_output_float(r"^automatic flow preprocessing time excluding field-data input= ([^ ]+) s$",compression_output,"自动flow时间"),"compression_time":read_output_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"selected_label_flow":selected_flow_match.group(1) if selected_flow_match else None,"selected_value_flow":selected_flow_match.group(2) if selected_flow_match else None,"automatic_flow_candidates":candidates}

def parse_unsz_output(compression_output,decompression_output):
    result=parse_unsz_compression_output(compression_output)
    psnr=result["psnr"]
    decompressed_psnr=read_output_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"UNSZ压缩端和解压端总PSNR不一致：{psnr} != {decompressed_psnr}")
    return {**result,"decompression_preprocessing_time":read_output_float(r"^preprocessing time excluding [^=]+= ([^ ]+) s$",decompression_output,"解压端预处理时间"),"decompression_time":read_output_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
