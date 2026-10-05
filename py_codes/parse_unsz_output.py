import math,re

# UNSZ命令由t1执行，这里只解析并核对输出
def read_output_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def parse_unsz_compression_output(compression_output):
    selected_flow_match=re.search(r"^automatic flow selected: label=([^,]+),value=([^,]+),",compression_output,re.MULTILINE)
    return {"br":read_output_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":read_output_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR"),"label_stream_br":read_output_float(r"^total label stream br= ([^,]+),",compression_output,"标签码流BR"),"value_stream_br":read_output_float(r"^total label stream br= [^,]+, total value stream br= ([^,]+),",compression_output,"数值码流BR"),"shared_metadata_br":read_output_float(r"^total label stream br= [^,]+, total value stream br= [^,]+, total shared metadata br= ([^ ]+)$",compression_output,"共享元数据BR"),"compression_preprocessing_time":read_output_float(r"^preprocessing time excluding [^=]+= ([^ ]+) s$",compression_output,"压缩端预处理时间"),"compression_time":read_output_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"selected_label_flow":selected_flow_match.group(1) if selected_flow_match else None,"selected_value_flow":selected_flow_match.group(2) if selected_flow_match else None}

def parse_unsz_output(compression_output,decompression_output):
    result=parse_unsz_compression_output(compression_output)
    psnr=result["psnr"]
    decompressed_psnr=read_output_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"UNSZ压缩端和解压端总PSNR不一致：{psnr} != {decompressed_psnr}")
    return {**result,"decompression_preprocessing_time":read_output_float(r"^preprocessing time excluding [^=]+= ([^ ]+) s$",decompression_output,"解压端预处理时间"),"decompression_time":read_output_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
