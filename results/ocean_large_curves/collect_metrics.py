import argparse,json,math,re
from pathlib import Path

import numpy as np


point_num=3693225
time_step_num=12
invalid_field="timeMonthly_avg_bruntVaisalaFreqML"
source_data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data")
umc_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/umc_parallel")
fields=(umc_root/"varnames_91.txt").read_text().splitlines()
original_bytes=point_num*time_step_num*len(fields)*4


def value(path,pattern):
    match=re.search(pattern,path.read_text())
    if not match:
        raise RuntimeError(f"日志缺少结果：{path}：{pattern}")
    return float(match.group(1))


def normalized_error_metrics(unsz_data_dir,rel_eb):
    normalized_sse={"unsz":0.0,"umc":0.0}
    maximum_error_ratio={"unsz":0.0,"umc":0.0}
    file_num=0
    chunk_size=1<<20
    for time_step in range(time_step_num):
        for field in fields:
            original_path=source_data_dir/f"{field}.dat.{time_step}"
            unsz_path=unsz_data_dir/f"{field}.dat.{time_step}.unsz.bin"
            umc_path=umc_root/f"step_{time_step}/{field}.merge.umc.out"
            original=np.memmap(original_path,dtype=np.float32,mode="r",shape=(point_num,))
            reconstructed={"unsz":np.memmap(unsz_path,dtype=np.float32,mode="r",shape=(point_num,)),"umc":np.memmap(umc_path,dtype=np.float32,mode="r",shape=(point_num,))}
            data_min=math.inf
            data_max=-math.inf
            squared_error={"unsz":0.0,"umc":0.0}
            maximum_error={"unsz":0.0,"umc":0.0}
            for begin in range(0,point_num,chunk_size):
                end=min(begin+chunk_size,point_num)
                original_chunk=np.asarray(original[begin:end],dtype=np.float64)
                data_min=min(data_min,float(np.min(original_chunk)))
                data_max=max(data_max,float(np.max(original_chunk)))
                for compressor in reconstructed:
                    difference=np.asarray(reconstructed[compressor][begin:end],dtype=np.float64)-original_chunk
                    squared_error[compressor]+=float(np.dot(difference,difference))
                    maximum_error[compressor]=max(maximum_error[compressor],float(np.max(np.abs(difference))))
            data_range=data_max-data_min
            for compressor in reconstructed:
                if data_range==0:
                    if squared_error[compressor]!=0:
                        raise RuntimeError(f"常量文件未精确恢复：{compressor}:{field}:{time_step}")
                else:
                    normalized_sse[compressor]+=squared_error[compressor]/(data_range*data_range)
                    maximum_error_ratio[compressor]=max(maximum_error_ratio[compressor],maximum_error[compressor]/(rel_eb*data_range))
            del reconstructed
            del original
            file_num+=1
    result={}
    value_num=point_num*file_num
    for compressor in normalized_sse:
        normalized_mse=normalized_sse[compressor]/value_num
        result[compressor]={"normalized_psnr":math.inf if normalized_mse==0 else -10*math.log10(normalized_mse),"maximum_error_over_eb":maximum_error_ratio[compressor]}
        if maximum_error_ratio[compressor]>1.01:
            raise RuntimeError(f"{compressor}超过误差上限：{maximum_error_ratio[compressor]}")
    return result


def collect_umc(log_dir,rel_eb):
    compression_times=[value(log_dir/f"umc_compress_{time_step}.log",r"compression_time = ([0-9.eE+-]+)") for time_step in range(time_step_num)]
    decompression_times=[value(log_dir/f"umc_decompress_{time_step}.log",r"decomprression_time = ([0-9.eE+-]+)") for time_step in range(time_step_num)]
    compression_preprocessing_times=[value(log_dir/f"umc_compress_{time_step}.log",r"preprocessing_time = ([0-9.eE+-]+)") for time_step in range(time_step_num)]
    decompression_preprocessing_times=[value(log_dir/f"umc_decompress_{time_step}.log",r"preprocessing_time = ([0-9.eE+-]+)") for time_step in range(time_step_num)]
    suffix=f"{rel_eb:.6f}"
    compressed_bytes=sum((umc_root/f"compressed/idw/step{time_step}_32_{suffix}").stat().st_size for time_step in range(time_step_num))
    return {"compressed_bytes":compressed_bytes,"bit_rate":compressed_bytes*8/(point_num*time_step_num*len(fields)),"compression_preprocessing_time":sum(compression_preprocessing_times),"compression_preprocessing_count":len(compression_preprocessing_times),"compression_time":sum(compression_times),"decompression_preprocessing_time":sum(decompression_preprocessing_times),"decompression_preprocessing_count":len(decompression_preprocessing_times),"decompression_time":sum(decompression_times)}


def collect_unsz(log_dir,data_dir):
    compress_texts=[(log_dir/f"unsz_{group}_compress.log").read_text() for group in("regular","invalid")]
    decompress_texts=[(log_dir/f"unsz_{group}_decompress.log").read_text() for group in("regular","invalid")]
    coordinate_times=[float(re.search(r"(?:shared )?coordinate transform preprocessing time excluding training-data input= ([0-9.eE+-]+)",text).group(1)) for text in compress_texts]
    dag_times=[float(re.search(r"(?:shared )?prediction DAG and weights preprocessing time= ([0-9.eE+-]+)",text).group(1)) for text in compress_texts]
    flow_times=[float(item) for text in compress_texts for item in re.findall(r"automatic flow preprocessing time excluding field-data input= ([0-9.eE+-]+)",text)]
    compression_times=[float(item) for text in compress_texts for item in re.findall(r"total compression time= ([0-9.eE+-]+)",text)]
    decompression_preprocessing_times=[float(re.search(r"shared decompression preprocessing time excluding coordinate and transform-file input= ([0-9.eE+-]+)",text).group(1)) for text in decompress_texts]
    decompression_times=[float(item) for text in decompress_texts for item in re.findall(r"total decompression time= ([0-9.eE+-]+)",text)]
    compressed_bytes=sum((data_dir/f"{field}.dat.{time_step}.unsz").stat().st_size for field in fields for time_step in range(time_step_num))
    return {"compressed_bytes":compressed_bytes,"bit_rate":compressed_bytes*8/(point_num*time_step_num*len(fields)),"coordinate_transform_time":sum(coordinate_times),"coordinate_transform_count":len(coordinate_times),"dag_and_weight_time":sum(dag_times),"dag_and_weight_count":len(dag_times),"automatic_flow_time":sum(flow_times),"automatic_flow_count":len(flow_times),"compression_preprocessing_time":sum(coordinate_times)+sum(dag_times)+sum(flow_times),"compression_time":sum(compression_times),"decompression_preprocessing_time":sum(decompression_preprocessing_times),"decompression_preprocessing_count":len(decompression_preprocessing_times),"decompression_time":sum(decompression_times)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--rel-eb",type=float,required=True)
    parser.add_argument("--log-dir",type=Path,required=True)
    parser.add_argument("--unsz-data-dir",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    arguments=parser.parse_args()
    errors=normalized_error_metrics(arguments.unsz_data_dir,arguments.rel_eb)
    result={"rel_eb":arguments.rel_eb,"point_num":point_num,"field_num":len(fields),"time_step_num":time_step_num,"original_bytes":original_bytes,"unsz":collect_unsz(arguments.log_dir,arguments.unsz_data_dir)|errors["unsz"],"umc":collect_umc(arguments.log_dir,arguments.rel_eb)|errors["umc"]}
    for compressor in("unsz","umc"):
        result[compressor]["compression_MB_s"]=original_bytes/1e6/result[compressor]["compression_time"]
        result[compressor]["decompression_MB_s"]=original_bytes/1e6/result[compressor]["decompression_time"]
    arguments.output.parent.mkdir(parents=True,exist_ok=True)
    arguments.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
