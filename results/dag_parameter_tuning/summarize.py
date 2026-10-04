import argparse,csv,math,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate

result_root=project_root/"results/dag_parameter_tuning"
baseline_root=project_root/"results/rbf_eb_base_tuning/curves"
dataset_vec=("tsunami","les","e3sm","haiyan","katrina")
variant_vec=((16,1),(12,1),(16,2),(12,2))

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def read_curve(path):
    with path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    rows.sort(key=lambda row:float(row["psnr"]))
    if len(rows)!=5:
        raise RuntimeError(f"{path}应有5个RD点，实际为{len(rows)}")
    return rows

def baseline_paths(dataset):
    return sorted((baseline_root/dataset).glob("exponential_shape_0p01_base_0p8_*.csv"))

def candidate_path(dataset,field,candidate_num,msd_ref_num):
    if candidate_num==16 and msd_ref_num==1:
        return baseline_root/dataset/f"exponential_shape_0p01_base_0p8_{field}.csv"
    return result_root/"curves"/dataset/f"candidate_{candidate_num}_msd_{msd_ref_num}_{field}.csv"

def compare_curves(reference,test):
    reference_br=np.asarray([float(row["br"]) for row in reference])
    reference_psnr=np.asarray([float(row["psnr"]) for row in reference])
    test_br=np.asarray([float(row["br"]) for row in test])
    test_psnr=np.asarray([float(row["psnr"]) for row in test])
    return calculate_bd_rate(reference_br,reference_psnr,test_br,test_psnr)

def main():
    parser=argparse.ArgumentParser(description="汇总DAG候选数与MSD评分参考点数实验")
    parser.add_argument("--dataset",choices=dataset_vec)
    arguments=parser.parse_args()
    selected_datasets=(arguments.dataset,) if arguments.dataset else dataset_vec
    global_summary={variant:[] for variant in variant_vec}
    print("dataset,field,candidate_num,msd_ref_num,bd_rate_vs_candidate16_msd1_percent")
    for dataset in selected_datasets:
        dataset_summary={variant:[] for variant in variant_vec}
        for baseline_path in baseline_paths(dataset):
            field=baseline_path.stem.removeprefix("exponential_shape_0p01_base_0p8_")
            reference=read_curve(baseline_path)
            for variant in variant_vec:
                result=compare_curves(reference,read_curve(candidate_path(dataset,field,*variant)))
                dataset_summary[variant].append(result)
                print(f"{dataset},{field},{variant[0]},{variant[1]},{result:+.9f}")
        for variant in variant_vec:
            result=100*(geometric_mean([1+value/100 for value in dataset_summary[variant]])-1)
            global_summary[variant].append(result)
            print(f"DATASET,{dataset},{variant[0]},{variant[1]},{result:+.9f}")
    print("GLOBAL_EQUAL_DATASET_WEIGHT")
    for variant in variant_vec:
        if len(global_summary[variant])==len(selected_datasets):
            result=100*(geometric_mean([1+value/100 for value in global_summary[variant]])-1)
            print(f"{variant[0]},{variant[1]},{result:+.9f}")

if __name__=="__main__":
    main()
