import argparse,csv,math,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate
from benchmark import dataset_config_map,nugget_name,result_root

shape_result_root=project_root/"results/rbf_shape_tuning"
nugget_vec=(1e-7,1e-6,1e-5)

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def read_curve(path):
    with path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    rows.sort(key=lambda row:float(row["psnr"]))
    if len(rows)!=5:
        raise RuntimeError(f"{path}应有5个RD点，实际为{len(rows)}")
    return rows

def candidate_path(dataset,field,nugget):
    if nugget==1e-6:
        return shape_result_root/"curves"/dataset/f"exponential_shape_0p01_{field}.csv"
    return result_root/"curves"/dataset/f"exponential_shape_0p01_nugget_{nugget_name(nugget)}_{field}.csv"

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
    parser=argparse.ArgumentParser(description="汇总Exponential shape=0.01的RBF nugget曲线")
    parser.add_argument("--dataset",choices=tuple(dataset_config_map))
    arguments=parser.parse_args()
    selected_datasets=(arguments.dataset,) if arguments.dataset else tuple(dataset_config_map)
    global_summary={nugget:[] for nugget in nugget_vec}
    print("dataset,field,nugget,bd_rate_vs_nugget_1e-6_percent,compression_time_ratio,decompression_time_ratio")
    for dataset in selected_datasets:
        baseline_paths=sorted((shape_result_root/"curves"/dataset).glob("exponential_shape_0p01_*.csv"))
        dataset_summary={nugget:[] for nugget in nugget_vec}
        for baseline_path in baseline_paths:
            field=baseline_path.stem.removeprefix("exponential_shape_0p01_")
            reference=read_curve(baseline_path)
            for nugget in nugget_vec:
                result=compare_curves(reference,read_curve(candidate_path(dataset,field,nugget)))
                dataset_summary[nugget].append(result)
                print(f"{dataset},{field},{nugget:g},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")
        for nugget in nugget_vec:
            values=dataset_summary[nugget]
            result=(100*(geometric_mean([1+value[0]/100 for value in values])-1),geometric_mean([value[1] for value in values]),geometric_mean([value[2] for value in values]))
            global_summary[nugget].append(result)
            print(f"DATASET,{dataset},{nugget:g},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")
    print("GLOBAL_EQUAL_DATASET_WEIGHT")
    for nugget in nugget_vec:
        values=global_summary[nugget]
        if len(values)!=len(selected_datasets):
            continue
        result=(100*(geometric_mean([1+value[0]/100 for value in values])-1),geometric_mean([value[1] for value in values]),geometric_mean([value[2] for value in values]))
        print(f"{nugget:g},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")

if __name__=="__main__":
    main()
