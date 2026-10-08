import argparse,json,math,re
from pathlib import Path

import numpy as np


point_num=3693225
time_step_num=12
source_data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data")
umc_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/umc_parallel")
fields=(umc_root/"varnames_91.txt").read_text().splitlines()
original_bytes=point_num*time_step_num*len(fields)*4


def value(path,pattern):
    match=re.search(pattern,path.read_text())
    if not match:
        raise RuntimeError(f"日志缺少结果：{path}：{pattern}")
    return float(match.group(1))


def umc_error_metrics(rel_eb):
    normalized_sse=0.0
    maximum_error_ratio=0.0
    file_num=0
    chunk_size=1<<20
    for time_step in range(time_step_num):
        for field in fields:
            original_path=source_data_dir/f"{field}.dat.{time_step}"
            umc_path=umc_root/f"step_{time_step}/{field}.merge.umc.out"
            original=np.memmap(original_path,dtype=np.float32,mode="r",shape=(point_num,))
            reconstructed=np.memmap(umc_path,dtype=np.float32,mode="r",shape=(point_num,))
            data_min=math.inf
            data_max=-math.inf
            squared_error=0.0
            maximum_error=0.0
            for begin in range(0,point_num,chunk_size):
                end=min(begin+chunk_size,point_num)
                original_chunk=np.asarray(original[begin:end],dtype=np.float64)
                data_min=min(data_min,float(np.min(original_chunk)))
                data_max=max(data_max,float(np.max(original_chunk)))
                difference=np.asarray(reconstructed[begin:end],dtype=np.float64)-original_chunk
                squared_error+=float(np.dot(difference,difference))
                maximum_error=max(maximum_error,float(np.max(np.abs(difference))))
            data_range=data_max-data_min
            if data_range==0:
                if squared_error!=0:
                    raise RuntimeError(f"UMC常量文件未精确恢复：{field}:{time_step}")
            else:
                normalized_sse+=squared_error/(data_range*data_range)
                maximum_error_ratio=max(maximum_error_ratio,maximum_error/(rel_eb*data_range))
            file_num+=1
    value_num=point_num*file_num
    normalized_mse=normalized_sse/value_num
    if maximum_error_ratio>1.01:
        raise RuntimeError(f"UMC超过误差上限：{maximum_error_ratio}")
    return {"normalized_psnr":math.inf if normalized_mse==0 else -10*math.log10(normalized_mse),"maximum_error_over_eb":maximum_error_ratio}


def logged_unsz_error_metrics(texts):
    pattern=r"^normalized squared error sum= ([^,]+), normalized value count= ([^,]+), normalized psnr= ([^,]+), maximum error over eb= ([^\r\n]+)$"
    matches=[match for text in texts for match in re.findall(pattern,text,re.MULTILINE)]
    if len(matches)!=len(fields):
        raise RuntimeError(f"UNSZ日志中的归一化误差统计数量不正确：{len(matches)} != {len(fields)}")
    normalized_squared_error=sum(float(match[0]) for match in matches)
    normalized_value_num=sum(int(match[1]) for match in matches)
    maximum_error_ratio=max(float(match[3]) for match in matches)
    normalized_psnr=math.inf if normalized_squared_error==0 else -10*math.log10(normalized_squared_error/normalized_value_num)
    if maximum_error_ratio>1.01:
        raise RuntimeError(f"UNSZ超过误差上限：{maximum_error_ratio}")
    return {"normalized_squared_error":normalized_squared_error,"normalized_value_num":normalized_value_num,"normalized_psnr":normalized_psnr,"maximum_error_over_eb":maximum_error_ratio}


def collect_umc(log_dir,rel_eb):
    compression_times=[value(log_dir/f"umc_compress_{time_step}.log",r"compression_time = ([0-9.eE+-]+)") for time_step in range(time_step_num)]
    decompression_times=[value(log_dir/f"umc_decompress_{time_step}.log",r"decomprression_time = ([0-9.eE+-]+)") for time_step in range(time_step_num)]
    compression_preprocessing_times=[value(log_dir/f"umc_compress_{time_step}.log",r"preprocessing_time = ([0-9.eE+-]+)") for time_step in range(time_step_num)]
    decompression_preprocessing_times=[value(log_dir/f"umc_decompress_{time_step}.log",r"preprocessing_time = ([0-9.eE+-]+)") for time_step in range(time_step_num)]
    suffix=f"{rel_eb:.6f}"
    compressed_bytes=sum((umc_root/f"compressed/idw/step{time_step}_32_{suffix}").stat().st_size for time_step in range(time_step_num))
    return {"compressed_bytes":compressed_bytes,"bit_rate":compressed_bytes*8/(point_num*time_step_num*len(fields)),"compression_preprocessing_time":sum(compression_preprocessing_times),"compression_preprocessing_count":len(compression_preprocessing_times),"compression_time":sum(compression_times),"decompression_preprocessing_time":sum(decompression_preprocessing_times),"decompression_preprocessing_count":len(decompression_preprocessing_times),"decompression_time":sum(decompression_times)}


def collect_unsz(log_dir,data_dir):
    compress_texts=[(log_dir/"unsz_compress.log").read_text()]
    decompress_texts=[(log_dir/"unsz_decompress.log").read_text()]
    coordinate_times=[float(item) for text in compress_texts for item in re.findall(r"coordinate transform preprocessing time excluding training-data input= ([0-9.eE+-]+)",text)]
    dag_times=[float(item) for text in compress_texts for item in re.findall(r"prediction DAG and weights preprocessing time= ([0-9.eE+-]+)",text)]
    flow_times=[float(item) for text in compress_texts for item in re.findall(r"automatic flow preprocessing time excluding field-data input= ([0-9.eE+-]+)",text)]
    compression_temporal_plan_times=[float(item) for text in compress_texts for item in re.findall(r"temporal dependency plan preprocessing time= ([0-9.eE+-]+)",text)]
    compression_temporal_dependency_times=[float(item) for text in compress_texts for item in re.findall(r"temporal dependency preprocessing time= ([0-9.eE+-]+)",text)]
    compression_times=[float(item) for text in compress_texts for item in re.findall(r"total compression time= ([0-9.eE+-]+)",text)]
    decompression_preprocessing_times=[float(item) for text in decompress_texts for item in re.findall(r"preprocessing time excluding coordinate and transform-file input= ([0-9.eE+-]+)",text)]
    decompression_temporal_plan_times=[float(item) for text in decompress_texts for item in re.findall(r"temporal dependency plan preprocessing time= ([0-9.eE+-]+)",text)]
    decompression_times=[float(item) for text in decompress_texts for item in re.findall(r"total decompression time= ([0-9.eE+-]+)",text)]
    compression_errors=logged_unsz_error_metrics(compress_texts)
    decompression_errors=logged_unsz_error_metrics(decompress_texts)
    if compression_errors["normalized_value_num"]!=decompression_errors["normalized_value_num"] or not math.isclose(compression_errors["normalized_squared_error"],decompression_errors["normalized_squared_error"],rel_tol=1e-9,abs_tol=0) or not math.isclose(compression_errors["maximum_error_over_eb"],decompression_errors["maximum_error_over_eb"],rel_tol=1e-12,abs_tol=0):
        raise RuntimeError("UNSZ压缩端与解压端的归一化误差统计不一致")
    compressed_bytes=sum((data_dir/f"{field}.dat.{time_step}.unsz").stat().st_size for field in fields for time_step in range(time_step_num))
    return {"compressed_bytes":compressed_bytes,"bit_rate":compressed_bytes*8/(point_num*time_step_num*len(fields)),"coordinate_transform_time":sum(coordinate_times),"coordinate_transform_count":len(coordinate_times),"dag_and_weight_time":sum(dag_times),"dag_and_weight_count":len(dag_times),"automatic_flow_time":sum(flow_times),"automatic_flow_count":len(flow_times),"temporal_dependency_plan_time":sum(compression_temporal_plan_times),"temporal_dependency_plan_count":len(compression_temporal_plan_times),"temporal_dependency_time":sum(compression_temporal_dependency_times),"temporal_dependency_count":len(compression_temporal_dependency_times),"compression_preprocessing_time":sum(coordinate_times)+sum(dag_times)+sum(flow_times)+sum(compression_temporal_plan_times)+sum(compression_temporal_dependency_times),"compression_time":sum(compression_times),"decompression_temporal_dependency_plan_time":sum(decompression_temporal_plan_times),"decompression_temporal_dependency_plan_count":len(decompression_temporal_plan_times),"decompression_preprocessing_time":sum(decompression_preprocessing_times)+sum(decompression_temporal_plan_times),"decompression_preprocessing_count":len(decompression_preprocessing_times),"decompression_time":sum(decompression_times),"normalized_psnr":decompression_errors["normalized_psnr"],"maximum_error_over_eb":decompression_errors["maximum_error_over_eb"]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--rel-eb",type=float,required=True)
    parser.add_argument("--log-dir",type=Path,required=True)
    parser.add_argument("--unsz-data-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reuse-umc-metrics",type=Path)
    arguments=parser.parse_args()
    reuse_umc=arguments.reuse_umc_metrics is not None
    if reuse_umc:
        previous=json.loads(arguments.reuse_umc_metrics.read_text())
        if previous["rel_eb"]!=arguments.rel_eb or previous["point_num"]!=point_num or previous["field_num"]!=len(fields) or previous["time_step_num"]!=time_step_num:
            raise RuntimeError("复用的UMC指标与当前Ocean-large配置不一致")
        umc=previous["umc"]
    else:
        umc=collect_umc(arguments.log_dir,arguments.rel_eb)|umc_error_metrics(arguments.rel_eb)
    result={"collector_version":4,"rel_eb":arguments.rel_eb,"point_num":point_num,"field_num":len(fields),"time_step_num":time_step_num,"original_bytes":original_bytes,"unsz":collect_unsz(arguments.log_dir,arguments.unsz_data_dir),"umc":umc}
    for compressor in("unsz","umc"):
        result[compressor]["compression_MB_s"]=original_bytes/1e6/result[compressor]["compression_time"]
        result[compressor]["decompression_MB_s"]=original_bytes/1e6/result[compressor]["decompression_time"]
    arguments.output.parent.mkdir(parents=True,exist_ok=True)
    arguments.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
