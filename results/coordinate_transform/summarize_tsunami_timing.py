import csv,re,statistics
from collections import defaultdict
from pathlib import Path


project_root=Path(__file__).resolve().parents[2]
result_root=project_root/"results/coordinate_transform"
source_path=result_root/"tsunami_off_on_timing_20261004.csv"
output_path=result_root/"tsunami_off_on_timing_median.csv"
log_paths=[
    project_root/"results/current/formal_runs/tsunami_coordinate_timing_21067449.log",
    project_root/"results/current/formal_runs/tsunami_coordinate_timing_21067589.log",
]
start_pattern=re.compile(r"^开始：task=([^,]+),field=([^,]+).*,REL=([^ ]+)$")
timing_pattern=re.compile(r"^计时：compression_preprocessing=([^ ]+) s,decompression_preprocessing=([^ ]+) s,compression=([^ ]+) s,decompression=([^ ]+) s")
sample_vec=defaultdict(list)
for log_path in log_paths:
    current_key=None
    for line in log_path.read_text().splitlines():
        match=start_pattern.match(line)
        if match:
            current_key=(match.group(1),match.group(2),float(match.group(3)))
            continue
        match=timing_pattern.match(line)
        if match and current_key is not None:
            sample_vec[current_key].append(tuple(float(match.group(i)) for i in range(1,5)))
            current_key=None

with source_path.open(newline="") as file:
    rows=list(csv.DictReader(file))
    fieldnames=file.seek(0) or next(csv.reader(file))
for row in rows:
    key=(row["task"],row["field"],float(row["rel_eb"]))
    samples=sample_vec[key]
    if len(samples)!=7:
        raise RuntimeError(f"{key}应有7个计时样本，实际为{len(samples)}")
    medians=[statistics.median(sample[i] for sample in samples) for i in range(4)]
    for name,value in zip(("compression_preprocessing_time","decompression_preprocessing_time","compression_time","decompression_time"),medians):
        row[name]=format(value,".17g")
    row["compression_decompression_time"]=format(medians[2]+medians[3],".17g")
    input_mb=float(row["input_bytes"])/1e6
    for name,value in zip(("compression_preprocessing_MB_s","decompression_preprocessing_MB_s","compression_MB_s","decompression_MB_s"),medians):
        row[name]=format(input_mb/value,".17g")
with output_path.open("w",newline="") as file:
    writer=csv.DictWriter(file,fieldnames=fieldnames,lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(output_path)
