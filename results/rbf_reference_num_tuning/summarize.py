import argparse,csv,math,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate
from benchmark import dataset_config_map,reference_num_vec,result_root,variant_name

reference_pair=(10,10)

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def read_curve(path):
    with path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    rows.sort(key=lambda row:float(row["psnr"]))
    if len(rows)!=5:
        raise RuntimeError(f"{path}应有5个RD点，实际为{len(rows)}")
    return np.asarray([float(row["br"]) for row in rows]),np.asarray([float(row["psnr"]) for row in rows])

def compare_curves(reference,test):
    reference_br,reference_psnr=reference
    test_br,test_psnr=test
    return calculate_bd_rate(reference_br,reference_psnr,test_br,test_psnr)

def main():
    parser=argparse.ArgumentParser(description="汇总RBF参考点数测试的同PSNR压缩效果")
    parser.add_argument("--dataset",choices=tuple(dataset_config_map))
    arguments=parser.parse_args()
    selected_datasets=(arguments.dataset,) if arguments.dataset else tuple(dataset_config_map)
    global_results={pair:[] for pair in reference_num_vec if pair!=reference_pair}
    print("dataset,field,candidate,bd_rate_vs_min_10_max_10_percent")
    for dataset in selected_datasets:
        dataset_results={pair:[] for pair in global_results}
        reference_prefix=variant_name(*reference_pair)+"_"
        for reference_path in sorted((result_root/"curves"/dataset).glob(reference_prefix+"*.csv")):
            field=reference_path.stem.removeprefix(reference_prefix)
            reference=read_curve(reference_path)
            for pair in dataset_results:
                test_path=result_root/"curves"/dataset/f"{variant_name(*pair)}_{field}.csv"
                result=compare_curves(reference,read_curve(test_path))
                dataset_results[pair].append(result)
                print(f"{dataset},{field},{variant_name(*pair)},{result:+.9f}")
        for pair,results in dataset_results.items():
            result=100*(geometric_mean([1+value/100 for value in results])-1)
            global_results[pair].append(result)
            print(f"DATASET,{dataset},{variant_name(*pair)},{result:+.9f}")
    if len(selected_datasets)==len(dataset_config_map):
        for pair,results in global_results.items():
            result=100*(geometric_mean([1+value/100 for value in results])-1)
            print(f"GLOBAL_EQUAL_DATASET_WEIGHT,{variant_name(*pair)},{result:+.9f}")

if __name__=="__main__":
    main()
