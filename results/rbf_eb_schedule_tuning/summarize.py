import argparse,csv,math,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate

result_root=project_root/"results/rbf_eb_schedule_tuning"
base_result_root=project_root/"results/rbf_eb_base_tuning"
dataset_vec=("tsunami","les","e3sm","haiyan","katrina")
shift_vec=(0.25,0.5,0.75)
exponent_vec=(6,7,8)

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def read_curve(path):
    with path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    rows.sort(key=lambda row:float(row["psnr"]))
    if len(rows)!=5:
        raise RuntimeError(f"{path}应有5个RD点，实际为{len(rows)}")
    return rows

def number_name(value):
    return f"{value:g}".replace(".","p")

def baseline_paths(dataset):
    return sorted((base_result_root/"curves"/dataset).glob("exponential_shape_0p01_base_0p8_*.csv"))

def candidate_path(dataset,field,parameter,value):
    baseline_value=0.5 if parameter=="shift" else 7
    if value==baseline_value:
        return base_result_root/"curves"/dataset/f"exponential_shape_0p01_base_0p8_{field}.csv"
    return result_root/"curves"/dataset/f"{parameter}_{number_name(value)}_{field}.csv"

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
    parser=argparse.ArgumentParser(description="汇总inverse_rank_log2的log2_shift曲线")
    parser.add_argument("--dataset",choices=dataset_vec)
    parser.add_argument("--parameter",choices=("shift","exponent"),default="shift")
    arguments=parser.parse_args()
    selected_datasets=(arguments.dataset,) if arguments.dataset else dataset_vec
    candidate_vec=shift_vec if arguments.parameter=="shift" else exponent_vec
    baseline_value=0.5 if arguments.parameter=="shift" else 7
    global_summary={value:[] for value in candidate_vec}
    print(f"dataset,field,{arguments.parameter},bd_rate_vs_{baseline_value:g}_percent,compression_time_ratio,decompression_time_ratio")
    for dataset in selected_datasets:
        dataset_summary={value:[] for value in candidate_vec}
        for baseline_path in baseline_paths(dataset):
            field=baseline_path.stem.removeprefix("exponential_shape_0p01_base_0p8_")
            reference=read_curve(baseline_path)
            for value in candidate_vec:
                result=compare_curves(reference,read_curve(candidate_path(dataset,field,arguments.parameter,value)))
                dataset_summary[value].append(result)
                print(f"{dataset},{field},{value:g},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")
        for value in candidate_vec:
            values=dataset_summary[value]
            result=(100*(geometric_mean([1+value[0]/100 for value in values])-1),geometric_mean([value[1] for value in values]),geometric_mean([value[2] for value in values]))
            global_summary[value].append(result)
            print(f"DATASET,{dataset},{value:g},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")
    print("GLOBAL_EQUAL_DATASET_WEIGHT")
    for value in candidate_vec:
        values=global_summary[value]
        if len(values)!=len(selected_datasets):
            continue
        result=(100*(geometric_mean([1+value[0]/100 for value in values])-1),geometric_mean([value[1] for value in values]),geometric_mean([value[2] for value in values]))
        print(f"{value:g},{result[0]:+.9f},{result[1]:.9f},{result[2]:.9f}")

if __name__=="__main__":
    main()
