import math,os,re
import numpy as np

# UMC命令由t1执行，这里只解析和核对输出

def calculate_umc_disk_error(data_paths,rel_eb,chunk_point_num=1<<20):
    total_squared_error=0
    total_point_num=0
    max_error=0
    # per-slice协议的PSNR：每个slice的误差除以该slice普通值range的平方，普通值排除-99999
    slice_statistics={'valid_value_num':0,'slice_normalized_valid_squared_error':0.0,'slice_normalized_squared_error':0.0}
    for path in data_paths:
        output_path=path+".umc.out"
        original_bytes=os.path.getsize(path)
        recovered_bytes=os.path.getsize(output_path)
        if original_bytes!=recovered_bytes or original_bytes%4:
            raise RuntimeError(f"UMC解压输出大小错误：{output_path}有{recovered_bytes}B，原文件有{original_bytes}B")
        file_min=math.inf
        file_max=-math.inf
        file_max_error=0
        file_max_error_point_id=0
        file_point_offset=0
        file_squared_error=0.0
        file_valid_squared_error=0.0
        file_valid_num=0
        with open(path,"rb") as original_file,open(output_path,"rb") as recovered_file:
            while True:
                original=np.fromfile(original_file,dtype=np.float32,count=chunk_point_num)
                if original.size==0:
                    break
                recovered=np.fromfile(recovered_file,dtype=np.float32,count=original.size)
                if recovered.size!=original.size:
                    raise RuntimeError(f"UMC解压输出提前结束：{output_path}")
                valid=original!=-99999.0
                if np.any(valid):
                    file_min=min(file_min,float(np.min(original[valid])))
                    file_max=max(file_max,float(np.max(original[valid])))
                difference=original.astype(np.float64)
                difference-=recovered
                absolute_difference=np.abs(difference)
                chunk_max_error_index=int(np.argmax(absolute_difference))
                chunk_max_error=float(absolute_difference[chunk_max_error_index])
                if chunk_max_error>file_max_error:
                    file_max_error=chunk_max_error
                    file_max_error_point_id=file_point_offset+chunk_max_error_index
                chunk_squared_error=float(np.dot(difference,difference))
                total_squared_error+=chunk_squared_error
                file_squared_error+=chunk_squared_error
                valid_difference=difference[valid]
                file_valid_squared_error+=float(np.dot(valid_difference,valid_difference))
                file_valid_num+=int(np.count_nonzero(valid))
                max_error=max(max_error,chunk_max_error)
                total_point_num+=original.size
                file_point_offset+=original.size
        file_range=0.0 if file_min==math.inf else file_max-file_min
        if file_range>0:
            slice_statistics['slice_normalized_valid_squared_error']+=file_valid_squared_error/file_range**2
            slice_statistics['slice_normalized_squared_error']+=file_squared_error/file_range**2
        elif file_valid_squared_error!=0:
            raise RuntimeError(f"UMC没有精确恢复普通值为常量的文件：{path}")
        slice_statistics['valid_value_num']+=file_valid_num
        file_abs_eb=0 if file_min==math.inf else rel_eb*(file_max-file_min)
        if file_max_error>1.01*file_abs_eb:
            ratio=math.inf if file_abs_eb==0 else file_max_error/file_abs_eb
            raise RuntimeError(f"UMC逐点误差检查失败：文件={path}，点={file_max_error_point_id}，误差={file_max_error}，绝对误差上限={file_abs_eb}，误差/上限={ratio}")
    if total_point_num==0:
        raise RuntimeError("UMC磁盘输出检查没有读到数据")
    slice_statistics['value_num']=total_point_num
    return total_squared_error/total_point_num,max_error,slice_statistics

def parse_umc_decompression_result(output,data_paths):
    preprocessing_times=[float(value) for value in re.findall(r"^preprocessing time\s*=\s*([^\s]+)",output,re.MULTILINE)]
    if len(preprocessing_times)!=1:
        raise RuntimeError(f"UMC解压端预处理时间数量应为1，实际为{len(preprocessing_times)}")
    decompression_times=[float(value) for value in re.findall(r"^core decompression time\s*=\s*([^\s]+)",output,re.MULTILINE)]
    if len(decompression_times)!=len(data_paths):
        raise RuntimeError(f"UMC核心解压时间数量不一致：文件{len(data_paths)}，核心解压时间{len(decompression_times)}")
    decompression_time=sum(decompression_times)
    restore_times=[float(value) for value in re.findall(r"^reorder time\s*=\s*([^\s]+)",output,re.MULTILINE)]
    if len(restore_times)!=len(data_paths):
        raise RuntimeError(f"UMC输出恢复时间数量不一致：文件{len(data_paths)}，恢复时间{len(restore_times)}")
    decompression_time+=sum(restore_times)
    return preprocessing_times[0],decompression_time

def parse_umc_compression_result(output,data_paths):
    compressed_sizes=[int(value) for value in re.findall(r"compressed_size\s*=\s*(\d+)",output)]
    if len(compressed_sizes)!=len(data_paths):
        raise RuntimeError(f"UMC压缩大小数量不一致：文件{len(data_paths)}，压缩大小{len(compressed_sizes)}")
    total_original_bytes=sum(os.path.getsize(path) for path in data_paths)
    total_br=32*sum(compressed_sizes)/total_original_bytes
    preprocessing_times=[float(value) for value in re.findall(r"^preprocessing time\s*=\s*([^\s]+)",output,re.MULTILINE)]
    core_compression_times=[float(value) for value in re.findall(r"^core compression time\s*=\s*([^\s]+)",output,re.MULTILINE)]
    reorder_times=[float(value) for value in re.findall(r"^data reorder time\s*=\s*([^\s]+)",output,re.MULTILINE)]
    if len(preprocessing_times)!=1:
        raise RuntimeError(f"UMC压缩端预处理时间数量应为1，实际为{len(preprocessing_times)}")
    if len(core_compression_times)!=len(data_paths) or len(reorder_times)!=len(data_paths):
        raise RuntimeError(f"UMC逐文件计时数量不一致：文件{len(data_paths)}，核心压缩时间{len(core_compression_times)}，重排时间{len(reorder_times)}")
    return total_br,preprocessing_times[0],sum(core_compression_times)+sum(reorder_times)
