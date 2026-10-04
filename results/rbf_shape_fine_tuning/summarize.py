import argparse,csv,math,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate
from benchmark import dataset_config_map,result_root,shape_name

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
    parser=argparse.ArgumentParser(description="汇总Exponential shape细筛")
    parser.add_argument("--dataset",choices=tuple(dataset_config_map))
    arguments=parser.parse_args()
    selected_datasets=(arguments.dataset,) if arguments.dataset else tuple(dataset_config_map)
    global_results=[]
    print("dataset,field,shape_0.003_bd_rate_vs_0.01_percent,compression_time_ratio,decompression_time_ratio")
    for dataset in selected_datasets:
        dataset_results=[]
        for reference_path in sorted((result_root/"curves"/dataset).glob(f"shape_{shape_name(0.01)}_*.csv")):
            field=reference_path.stem.removeprefix(f"shape_{shape_name(0.01)}_")
            test_path=result_root/"curves"/dataset/f"shape_{shape_name(0.003)}_{field}.csv"
            result=compare_curves(read_curve(reference_path),read_curve(test_path))
            dataset_results.append(result)
            print(f"{dataset},{field},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")
        result=(100*(geometric_mean([1+value[0]/100 for value in dataset_results])-1),geometric_mean([value[1] for value in dataset_results]),geometric_mean([value[2] for value in dataset_results]))
        global_results.append(result)
        print(f"DATASET,{dataset},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")
    if len(global_results)==len(dataset_config_map):
        result=(100*(geometric_mean([1+value[0]/100 for value in global_results])-1),geometric_mean([value[1] for value in global_results]),geometric_mean([value[2] for value in global_results]))
        print(f"GLOBAL_EQUAL_DATASET_WEIGHT,{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")

if __name__=="__main__":
    main()
