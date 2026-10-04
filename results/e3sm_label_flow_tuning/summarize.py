import csv,math,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate

result_root=project_root/"results/e3sm_label_flow_tuning/curves"

def read_curve(name):
    with (result_root/f"{name}.csv").open(newline="") as file:
        rows=sorted(csv.DictReader(file),key=lambda row:float(row["psnr"]))
    if len(rows)!=5:
        raise RuntimeError(f"{name}应有5个RD点，实际为{len(rows)}")
    return rows

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def main():
    spatial=read_curve("label_spatial")
    anchor=read_curve("label_anchor8")
    bd_rate=calculate_bd_rate(np.asarray([float(row["br"]) for row in spatial]),np.asarray([float(row["psnr"]) for row in spatial]),np.asarray([float(row["br"]) for row in anchor]),np.asarray([float(row["psnr"]) for row in anchor]))
    compression_ratio=geometric_mean([float(test["compression_time"])/float(reference["compression_time"]) for reference,test in zip(spatial,anchor)])
    decompression_ratio=geometric_mean([float(test["decompression_time"])/float(reference["decompression_time"]) for reference,test in zip(spatial,anchor)])
    total_ratio=geometric_mean([float(test["compression_decompression_time"])/float(reference["compression_decompression_time"]) for reference,test in zip(spatial,anchor)])
    print("label_anchor8_vs_spatial")
    print(f"bd_rate_percent={bd_rate:+.9f}")
    print(f"compression_time_ratio={compression_ratio:.9f}")
    print(f"decompression_time_ratio={decompression_ratio:.9f}")
    print(f"total_time_ratio={total_ratio:.9f}")

if __name__=="__main__":
    main()
