import json,re
from pathlib import Path


result_dir=Path(__file__).resolve().parent
log_dir=result_dir/"logs"
source_data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data")
off_data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data_adjust_coord_off")
umc_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/umc_parallel")
invalid_field="timeMonthly_avg_bruntVaisalaFreqML"
fields=[field for field in (umc_root/"varnames_91.txt").read_text().splitlines() if field!=invalid_field]
point_num=3693225
time_step_num=12


def compressed_bytes(data_dir):
    return sum((data_dir/f"{field}.dat.{time_step}.unsz").stat().st_size for field in fields for time_step in range(time_step_num))


def log_statistics(path):
    text=path.read_text()
    compression_times=[float(value) for value in re.findall(r"total compression time= ([0-9.eE+-]+)",text)]
    automatic_flow_times=[float(value) for value in re.findall(r"automatic flow preprocessing time excluding field-data input= ([0-9.eE+-]+)",text)]
    coordinate_match=re.search(r"shared coordinate transform preprocessing time excluding training-data input= ([0-9.eE+-]+)",text)
    dag_match=re.search(r"shared prediction DAG and weights preprocessing time= ([0-9.eE+-]+)",text)
    return {"field_count":len(compression_times),"compression_time":sum(compression_times),"coordinate_transform_time":float(coordinate_match.group(1)) if coordinate_match else 0,"dag_and_weight_time":float(dag_match.group(1)),"automatic_flow_time":sum(automatic_flow_times)}


on_bytes=compressed_bytes(source_data_dir)
off_bytes=compressed_bytes(off_data_dir)
on=log_statistics(log_dir/"unsz_regular_compress.log")
off=log_statistics(log_dir/"unsz_regular_adjust_coord_off_compress.log")
on["compressed_bytes"]=on_bytes
off["compressed_bytes"]=off_bytes
on["bit_rate"]=on_bytes*8/(point_num*time_step_num*len(fields))
off["bit_rate"]=off_bytes*8/(point_num*time_step_num*len(fields))
print(json.dumps({"auto_on":on,"forced_off":off,"off_over_on_bit_rate":off["bit_rate"]/on["bit_rate"],"on_bit_rate_reduction":1-on["bit_rate"]/off["bit_rate"]},indent=2))
