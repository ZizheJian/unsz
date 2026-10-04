import argparse,csv,math,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate
from benchmark import dataset_config_map,kernel_vec,result_root

baseline_kernel="matern32"

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def read_curve(path):
    with path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    rows.sort(key=lambda row:float(row["psnr"]))
    if len(rows)!=5:
        raise RuntimeError(f"{path}应有5个RD点，实际为{len(rows)}")
    return rows

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
    parser=argparse.ArgumentParser(description="汇总RBF kernel完整RD曲线")
    parser.add_argument("--dataset",choices=tuple(dataset_config_map))
    arguments=parser.parse_args()
    selected_datasets=(arguments.dataset,) if arguments.dataset else tuple(dataset_config_map)
    dataset_summary={kernel:[] for kernel in kernel_vec}
    print("dataset,field,kernel,bd_rate_vs_matern32_percent,compression_time_ratio,decompression_time_ratio")
    for dataset in selected_datasets:
        field_paths=sorted((result_root/"curves"/dataset).glob(f"{baseline_kernel}_*.csv"))
        if not field_paths:
            print(f"{dataset}:结果缺失",file=sys.stderr)
            continue
        field_summary={kernel:[] for kernel in kernel_vec}
        for baseline_path in field_paths:
            field=baseline_path.stem.removeprefix(baseline_kernel+"_")
            reference=read_curve(baseline_path)
            for kernel in kernel_vec:
                test_path=baseline_path.with_name(f"{kernel}_{field}.csv")
                if not test_path.is_file():
                    raise RuntimeError(f"缺少{test_path}")
                result=compare_curves(reference,read_curve(test_path))
                field_summary[kernel].append(result)
                print(f"{dataset},{field},{kernel},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")
        for kernel in kernel_vec:
            values=field_summary[kernel]
            dataset_result=(100*(geometric_mean([1+value[0]/100 for value in values])-1),geometric_mean([value[1] for value in values]),geometric_mean([value[2] for value in values]))
            dataset_summary[kernel].append(dataset_result)
            print(f"DATASET,{dataset},{kernel},{dataset_result[0]:+.9f},{dataset_result[1]:.9f},{dataset_result[2]:.9f}")
    print("GLOBAL_EQUAL_DATASET_WEIGHT")
    for kernel in kernel_vec:
        values=dataset_summary[kernel]
        if len(values)!=len(selected_datasets):
            continue
        result=(100*(geometric_mean([1+value[0]/100 for value in values])-1),geometric_mean([value[1] for value in values]),geometric_mean([value[2] for value in values]))
        print(f"{kernel},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")

if __name__=="__main__":
    main()
