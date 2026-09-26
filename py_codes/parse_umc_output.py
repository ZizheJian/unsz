import array,math,os,re
import numpy as np

# UMC命令由t1执行，这里只解析和核对输出

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

def calculate_umc_disk_error(data_paths,chunk_point_num=1<<20):
    total_squared_error=0
    total_point_num=0
    max_error=0
    for path in data_paths:
        output_path=path+".umc.out"
        original_bytes=os.path.getsize(path)
        recovered_bytes=os.path.getsize(output_path)
        if original_bytes!=recovered_bytes or original_bytes%4:
            raise RuntimeError(f"UMC解压输出大小错误：{output_path}有{recovered_bytes}B，原文件有{original_bytes}B")
        with open(path,"rb") as original_file,open(output_path,"rb") as recovered_file:
            while True:
                original=np.fromfile(original_file,dtype=np.float32,count=chunk_point_num)
                if original.size==0:
                    break
                recovered=np.fromfile(recovered_file,dtype=np.float32,count=original.size)
                if recovered.size!=original.size:
                    raise RuntimeError(f"UMC解压输出提前结束：{output_path}")
                difference=original.astype(np.float64)
                difference-=recovered
                total_squared_error+=float(np.dot(difference,difference))
                max_error=max(max_error,float(np.max(np.abs(difference))))
                total_point_num+=original.size
    if total_point_num==0:
        raise RuntimeError("UMC磁盘输出检查没有读到数据")
    return total_squared_error/total_point_num,max_error

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

def parse_umc_compression_result(output,data_paths,data_range=None):
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
    return total_br,total_psnr,preprocessing_times[0],core_compression_time
