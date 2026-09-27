import math,re

# UNSZ命令由t1执行，这里只解析并核对输出
def read_output_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def parse_unsz_output(compression_output,decompression_output):
    psnr=read_output_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
    decompressed_psnr=read_output_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"UNSZ压缩端和解压端总PSNR不一致：{psnr} != {decompressed_psnr}")
    return {"br":read_output_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"compression_preprocessing_time":read_output_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"压缩端预处理时间"),"decompression_preprocessing_time":read_output_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",decompression_output,"解压端预处理时间"),"compression_time":read_output_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_output_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
