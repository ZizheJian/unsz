import csv,math

from benchmark import dataset_config_map,result_root,round_orders,variant_name

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def read_rows(path):
    with path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    if len(rows)!=5:
        raise RuntimeError(f"{path}应有5个结果，实际为{len(rows)}")
    return {row["point_set_config"]:row for row in rows}

def time_ratios(reference,test):
    if reference.keys()!=test.keys():
        raise RuntimeError("成对测速使用了不同的误差配置")
    compression_ratios=[]
    decompression_ratios=[]
    total_ratios=[]
    for key in reference:
        reference_row=reference[key]
        test_row=test[key]
        compression_ratios.append(float(test_row["compression_time"])/float(reference_row["compression_time"]))
        decompression_ratios.append(float(test_row["decompression_time"])/float(reference_row["decompression_time"]))
        total_ratios.append(float(test_row["compression_decompression_time"])/float(reference_row["compression_decompression_time"]))
    return geometric_mean(compression_ratios),geometric_mean(decompression_ratios),geometric_mean(total_ratios)

def main():
    print("dataset,round,candidate,compression_time_ratio,decompression_time_ratio,total_time_ratio")
    combined={(dataset,min_ref_num):[] for dataset in dataset_config_map for min_ref_num in (7,8)}
    for dataset in dataset_config_map:
        paths=sorted((result_root/"curves"/dataset).glob("round_1_min_10_max_10_*.csv"))
        if len(paths)!=1:
            raise RuntimeError(f"{dataset}的field数量不是1")
        field=paths[0].stem.removeprefix("round_1_min_10_max_10_")
        for round_id,_ in enumerate(round_orders,1):
            reference=read_rows(result_root/"curves"/dataset/f"round_{round_id}_{variant_name(10)}_{field}.csv")
            for min_ref_num in (7,8):
                result=time_ratios(reference,read_rows(result_root/"curves"/dataset/f"round_{round_id}_{variant_name(min_ref_num)}_{field}.csv"))
                combined[dataset,min_ref_num].append(result)
                print(f"{dataset},{round_id},{variant_name(min_ref_num)},{result[0]:.9f},{result[1]:.9f},{result[2]:.9f}")
        for min_ref_num in (7,8):
            results=combined[dataset,min_ref_num]
            result=tuple(geometric_mean([item[index] for item in results]) for index in range(3))
            print(f"DATASET,{dataset},{variant_name(min_ref_num)},{result[0]:.9f},{result[1]:.9f},{result[2]:.9f}")
    for min_ref_num in (7,8):
        dataset_results=[]
        for dataset in dataset_config_map:
            rounds=combined[dataset,min_ref_num]
            dataset_results.append(tuple(geometric_mean([item[index] for item in rounds]) for index in range(3)))
        result=tuple(geometric_mean([item[index] for item in dataset_results]) for index in range(3))
        print(f"GLOBAL_EQUAL_DATASET_WEIGHT,{variant_name(min_ref_num)},{result[0]:.9f},{result[1]:.9f},{result[2]:.9f}")

if __name__=="__main__":
    main()
