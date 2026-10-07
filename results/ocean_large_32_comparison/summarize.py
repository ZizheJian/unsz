import json,re
from pathlib import Path


result_dir=Path(__file__).resolve().parent
log_dir=result_dir/"logs"
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data")
umc_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/umc_parallel")
fields=(umc_root/"varnames_91.txt").read_text().splitlines()
point_num=3693225
time_step_num=12
original_bytes=point_num*time_step_num*len(fields)*4


def value(path,pattern):
    match=re.search(pattern,path.read_text())
    if not match:
        raise RuntimeError(f"日志缺少结果：{path}：{pattern}")
    return float(match.group(1))


def timing_statistics(values):
    return {"count":len(values),"total_time":sum(values),"mean_time":sum(values)/len(values),"min_time":min(values),"max_time":max(values)}


umc_compression_time_vec=[value(log_dir/f"umc_compress_{time_step}.log",r"compression_time = ([0-9.eE+-]+)") for time_step in range(time_step_num)]
umc_decompression_time_vec=[value(log_dir/f"umc_decompress_{time_step}.log",r"decomprression_time = ([0-9.eE+-]+)") for time_step in range(time_step_num)]
umc_compression_preprocessing_time_vec=[value(log_dir/f"umc_compress_{time_step}.log",r"preprocessing_time = ([0-9.eE+-]+)") for time_step in range(time_step_num)]
umc_decompression_preprocessing_time_vec=[value(log_dir/f"umc_decompress_{time_step}.log",r"preprocessing_time = ([0-9.eE+-]+)") for time_step in range(time_step_num)]
umc_compression_time=sum(umc_compression_time_vec)
umc_decompression_time=sum(umc_decompression_time_vec)
umc_compression_preprocessing_time=sum(umc_compression_preprocessing_time_vec)
umc_decompression_preprocessing_time=sum(umc_decompression_preprocessing_time_vec)
umc_compressed_bytes=sum((umc_root/"compressed/idw"/f"step{time_step}_32_0.001000").stat().st_size for time_step in range(time_step_num))

unsz_compress_text_vec=[(log_dir/f"unsz_{group}_compress.log").read_text() for group in("regular","invalid")]
unsz_decompress_text_vec=[(log_dir/f"unsz_{group}_decompress.log").read_text() for group in("regular","invalid")]
unsz_coordinate_transform_time_vec=[float(re.search(r"(?:shared )?coordinate transform preprocessing time excluding training-data input= ([0-9.eE+-]+)",text).group(1)) for text in unsz_compress_text_vec]
unsz_dag_and_weight_time_vec=[float(re.search(r"(?:shared )?prediction DAG and weights preprocessing time= ([0-9.eE+-]+)",text).group(1)) for text in unsz_compress_text_vec]
unsz_automatic_flow_time_vec=[float(value) for text in unsz_compress_text_vec for value in re.findall(r"automatic flow preprocessing time excluding field-data input= ([0-9.eE+-]+)",text)]
unsz_coordinate_transform_time=sum(unsz_coordinate_transform_time_vec)
unsz_dag_and_weight_time=sum(unsz_dag_and_weight_time_vec)
unsz_automatic_flow_time=sum(unsz_automatic_flow_time_vec)
unsz_compression_time=sum(float(value) for text in unsz_compress_text_vec for value in re.findall(r"total compression time= ([0-9.eE+-]+)",text))
unsz_decompression_preprocessing_time_vec=[float(re.search(r"shared decompression preprocessing time excluding coordinate and transform-file input= ([0-9.eE+-]+)",text).group(1)) for text in unsz_decompress_text_vec]
unsz_decompression_preprocessing_time=sum(unsz_decompression_preprocessing_time_vec)
unsz_decompression_time=sum(float(value) for text in unsz_decompress_text_vec for value in re.findall(r"total decompression time= ([0-9.eE+-]+)",text))
unsz_compressed_bytes=0
for field in fields:
    unsz_compressed_bytes+=sum((data_dir/f"{field}.dat.{time_step}.unsz").stat().st_size for time_step in range(time_step_num))

summary={
    "point_num":point_num,
    "field_num":len(fields),
    "time_step_num":time_step_num,
    "original_bytes":original_bytes,
    "umc":{
        "compressed_bytes":umc_compressed_bytes,
        "bit_rate":umc_compressed_bytes*8/(point_num*time_step_num*len(fields)),
        "compression_preprocessing_time":umc_compression_preprocessing_time,
        "decompression_preprocessing_time":umc_decompression_preprocessing_time,
        "compression_preprocessing_statistics":timing_statistics(umc_compression_preprocessing_time_vec),
        "decompression_preprocessing_statistics":timing_statistics(umc_decompression_preprocessing_time_vec),
        "compression_time":umc_compression_time,
        "decompression_time":umc_decompression_time,
    },
    "unsz":{
        "compressed_bytes":unsz_compressed_bytes,
        "bit_rate":unsz_compressed_bytes*8/(point_num*time_step_num*len(fields)),
        "coordinate_transform_time":unsz_coordinate_transform_time,
        "dag_and_weight_time":unsz_dag_and_weight_time,
        "automatic_flow_time":unsz_automatic_flow_time,
        "compression_preprocessing_time":unsz_coordinate_transform_time+unsz_dag_and_weight_time+unsz_automatic_flow_time,
        "decompression_preprocessing_time":unsz_decompression_preprocessing_time,
        "coordinate_transform_statistics":timing_statistics(unsz_coordinate_transform_time_vec),
        "dag_and_weight_statistics":timing_statistics(unsz_dag_and_weight_time_vec),
        "automatic_flow_statistics":timing_statistics(unsz_automatic_flow_time_vec),
        "decompression_preprocessing_statistics":timing_statistics(unsz_decompression_preprocessing_time_vec),
        "compression_time":unsz_compression_time,
        "decompression_time":unsz_decompression_time,
    },
}
print(json.dumps(summary,indent=2))
