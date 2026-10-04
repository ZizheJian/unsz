import argparse,csv,math,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate
from benchmark import dataset_config_map,result_root

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def read_curve(path):
    with path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    rows.sort(key=lambda row:float(row["psnr"]))
    if len(rows)!=5:
        raise RuntimeError(f"{path}应有5个RD点，实际为{len(rows)}")
    return rows

def compare_curves(local,global_):
    local_br=np.asarray([float(row["br"]) for row in local])
    local_psnr=np.asarray([float(row["psnr"]) for row in local])
    global_br=np.asarray([float(row["br"]) for row in global_])
    global_psnr=np.asarray([float(row["psnr"]) for row in global_])
    bd_rate=calculate_bd_rate(local_br,local_psnr,global_br,global_psnr)
    compression_ratio=geometric_mean([float(row["compression_time"]) for row in global_])/geometric_mean([float(row["compression_time"]) for row in local])
    decompression_ratio=geometric_mean([float(row["decompression_time"]) for row in global_])/geometric_mean([float(row["decompression_time"]) for row in local])
    preprocessing_ratio=float(global_[0]["compression_preprocessing_time"])/float(local[0]["compression_preprocessing_time"])
    return bd_rate,compression_ratio,decompression_ratio,preprocessing_ratio

def main():
    parser=argparse.ArgumentParser(description="汇总local与global完整RD曲线")
    parser.add_argument("--dataset",choices=tuple(dataset_config_map))
    arguments=parser.parse_args()
    selected_datasets=(arguments.dataset,) if arguments.dataset else tuple(dataset_config_map)
    global_results=[]
    print("dataset,field,global_bd_rate_vs_local_percent,global_compression_time_ratio,global_decompression_time_ratio,global_fresh_preprocessing_ratio")
    for dataset in selected_datasets:
        local_paths=sorted((result_root/"curves"/dataset).glob("local_*.csv"))
        dataset_results=[]
        for local_path in local_paths:
            field=local_path.stem.removeprefix("local_")
            result=compare_curves(read_curve(local_path),read_curve(result_root/"curves"/dataset/f"global_{field}.csv"))
            dataset_results.append(result)
            print(f"{dataset},{field},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f},{result[3]:.9f}")
        if not dataset_results:
            continue
        result=tuple(100*(geometric_mean([1+value[0]/100 for value in dataset_results])-1) if column==0 else geometric_mean([value[column] for value in dataset_results]) for column in range(4))
        print(f"DATASET,{dataset},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f},{result[3]:.9f}")
        global_results.append(result)
    if not arguments.dataset and len(global_results)==len(dataset_config_map):
        result=tuple(100*(geometric_mean([1+value[0]/100 for value in global_results])-1) if column==0 else geometric_mean([value[column] for value in global_results]) for column in range(4))
        print("GLOBAL_EQUAL_DATASET_WEIGHT")
        print(f"global,{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f},{result[3]:.9f}")

if __name__=="__main__":
    main()
