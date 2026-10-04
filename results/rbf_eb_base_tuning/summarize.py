import argparse,csv,math,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate
from benchmark import base_name,dataset_config_map,result_root

shape_result_root=project_root/"results/rbf_shape_tuning"
base_vec=(0.8,0.85,0.9)

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def read_curve(path):
    with path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    rows.sort(key=lambda row:float(row["psnr"]))
    if len(rows)!=5:
        raise RuntimeError(f"{path}应有5个RD点，实际为{len(rows)}")
    return rows

def candidate_path(dataset,field,base):
    if base==0.85:
        return shape_result_root/"curves"/dataset/f"exponential_shape_0p01_{field}.csv"
    return result_root/"curves"/dataset/f"exponential_shape_0p01_base_{base_name(base)}_{field}.csv"

def compare_curves(reference,test):
    reference_br=np.asarray([float(row["br"]) for row in reference])
    reference_psnr=np.asarray([float(row["psnr"]) for row in reference])
    test_br=np.asarray([float(row["br"]) for row in test])
    test_psnr=np.asarray([float(row["psnr"]) for row in test])
    bd_rate=calculate_bd_rate(reference_br,reference_psnr,test_br,test_psnr)
    compression_ratio=geometric_mean([float(test_row["compression_time"])/float(reference_row["compression_time"]) for reference_row,test_row in zip(reference,test)])
    decompression_ratio=geometric_mean([float(test_row["decompression_time"])/float(reference_row["decompression_time"]) for reference_row,test_row in zip(reference,test)])
    return bd_rate,compression_ratio,decompression_ratio

def main():
    parser=argparse.ArgumentParser(description="汇总inverse_rank_log2 EB base曲线")
    parser.add_argument("--dataset",choices=tuple(dataset_config_map))
    arguments=parser.parse_args()
    selected_datasets=(arguments.dataset,) if arguments.dataset else tuple(dataset_config_map)
    global_summary={base:[] for base in base_vec}
    print("dataset,field,base,bd_rate_vs_base_0.85_percent,compression_time_ratio,decompression_time_ratio")
    for dataset in selected_datasets:
        baseline_paths=sorted((shape_result_root/"curves"/dataset).glob("exponential_shape_0p01_*.csv"))
        dataset_summary={base:[] for base in base_vec}
        for baseline_path in baseline_paths:
            field=baseline_path.stem.removeprefix("exponential_shape_0p01_")
            reference=read_curve(baseline_path)
            for base in base_vec:
                result=compare_curves(reference,read_curve(candidate_path(dataset,field,base)))
                dataset_summary[base].append(result)
                print(f"{dataset},{field},{base:g},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")
        for base in base_vec:
            values=dataset_summary[base]
            result=(100*(geometric_mean([1+value[0]/100 for value in values])-1),geometric_mean([value[1] for value in values]),geometric_mean([value[2] for value in values]))
            global_summary[base].append(result)
            print(f"DATASET,{dataset},{base:g},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")
    print("GLOBAL_EQUAL_DATASET_WEIGHT")
    for base in base_vec:
        values=global_summary[base]
        if len(values)!=len(selected_datasets):
            continue
        result=(100*(geometric_mean([1+value[0]/100 for value in values])-1),geometric_mean([value[1] for value in values]),geometric_mean([value[2] for value in values]))
        print(f"{base:g},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")

if __name__=="__main__":
    main()
