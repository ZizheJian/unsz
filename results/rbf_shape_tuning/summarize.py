import argparse,csv,math,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate
from benchmark import dataset_config_map,kernel_result_root,result_root,shape_name

candidate_vec=(("exponential",0.003),("exponential",0.01),("exponential",0.03),("exponential",0.1),("inverse_multiquadric",0.01),("inverse_multiquadric",0.03),("inverse_multiquadric",0.1))

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def read_curve(path):
    with path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    rows.sort(key=lambda row:float(row["psnr"]))
    if len(rows)!=5:
        raise RuntimeError(f"{path}应有5个RD点，实际为{len(rows)}")
    return rows

def candidate_path(dataset,field,kernel,shape):
    if shape==0.03:
        return kernel_result_root/"curves"/dataset/f"{kernel}_{field}.csv"
    return result_root/"curves"/dataset/f"{kernel}_shape_{shape_name(shape)}_{field}.csv"

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
    parser=argparse.ArgumentParser(description="汇总Exponential和IMQ的RBF shape曲线")
    parser.add_argument("--dataset",choices=tuple(dataset_config_map))
    arguments=parser.parse_args()
    selected_datasets=(arguments.dataset,) if arguments.dataset else tuple(dataset_config_map)
    global_summary={candidate:[] for candidate in candidate_vec}
    print("dataset,field,kernel,shape,bd_rate_vs_matern32_shape_0.03_percent,compression_time_ratio,decompression_time_ratio")
    for dataset in selected_datasets:
        baseline_paths=sorted((kernel_result_root/"curves"/dataset).glob("matern32_*.csv"))
        dataset_summary={candidate:[] for candidate in candidate_vec}
        for baseline_path in baseline_paths:
            field=baseline_path.stem.removeprefix("matern32_")
            reference=read_curve(baseline_path)
            for candidate in candidate_vec:
                kernel,shape=candidate
                result=compare_curves(reference,read_curve(candidate_path(dataset,field,kernel,shape)))
                dataset_summary[candidate].append(result)
                print(f"{dataset},{field},{kernel},{shape:g},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")
        for candidate in candidate_vec:
            values=dataset_summary[candidate]
            result=(100*(geometric_mean([1+value[0]/100 for value in values])-1),geometric_mean([value[1] for value in values]),geometric_mean([value[2] for value in values]))
            global_summary[candidate].append(result)
            print(f"DATASET,{dataset},{candidate[0]},{candidate[1]:g},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")
    print("GLOBAL_EQUAL_DATASET_WEIGHT")
    for candidate in candidate_vec:
        values=global_summary[candidate]
        if len(values)!=len(selected_datasets):
            continue
        result=(100*(geometric_mean([1+value[0]/100 for value in values])-1),geometric_mean([value[1] for value in values]),geometric_mean([value[2] for value in values]))
        print(f"{candidate[0]},{candidate[1]:g},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")

if __name__=="__main__":
    main()
