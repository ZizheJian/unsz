import csv,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate
from py_codes.summarize_current_results import geometric_metric_ratio_at_same_psnr
from benchmark import sample_num_vec,task_name

result_path=project_root/"results/coordinate_transform_second_sample_num_tuning/results.csv"

def main():
    with result_path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    curve_map={sample_num:sorted((row for row in rows if row["task"]==task_name(sample_num)),key=lambda row:float(row["psnr"])) for sample_num in sample_num_vec}
    for sample_num,curve in curve_map.items():
        if len(curve)!=5:
            raise RuntimeError(f"sample_num={sample_num}应有5个点，实际为{len(curve)}")
    reference=curve_map[3000]
    reference_br=np.asarray([float(row["br"]) for row in reference])
    reference_psnr=np.asarray([float(row["psnr"]) for row in reference])
    print("sample_num,bd_rate_vs_3000_percent,compression_time_ratio,decompression_time_ratio,total_time_ratio")
    for sample_num in sample_num_vec:
        curve=curve_map[sample_num]
        br=np.asarray([float(row["br"]) for row in curve])
        psnr=np.asarray([float(row["psnr"]) for row in curve])
        bd_rate=calculate_bd_rate(reference_br,reference_psnr,br,psnr)
        compression_ratio=geometric_metric_ratio_at_same_psnr(reference,curve,"compression_time")
        decompression_ratio=geometric_metric_ratio_at_same_psnr(reference,curve,"decompression_time")
        total_ratio=geometric_metric_ratio_at_same_psnr(reference,curve,"compression_decompression_time")
        print(f"{sample_num},{bd_rate:+.9f},{compression_ratio:.9f},{decompression_ratio:.9f},{total_ratio:.9f}")

if __name__=="__main__":
    main()
