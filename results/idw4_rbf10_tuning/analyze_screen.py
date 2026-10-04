import argparse,csv,math
from pathlib import Path

result_root=Path(__file__).resolve().parent
all_datasets=("tsunami","les","e3sm","haiyan","katrina")

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def read_dataset(dataset):
    path=result_root/f"{dataset}.csv"
    with path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    prefix=dataset+"_"
    selected=[]
    for row in rows:
        if row["compressor"]!="unsz" or not row["task"].startswith(prefix):
            continue
        selected.append({**row,"variant":row["task"][len(prefix):],"br":float(row["br"]),"psnr":float(row["psnr"]),"compression_time":float(row["compression_time"]),"decompression_time":float(row["decompression_time"])})
    return selected

def summarize_dataset(dataset):
    rows=read_dataset(dataset)
    expected_fields={row["field"] for row in rows}
    grouped={}
    for row in rows:
        grouped.setdefault(row["variant"],[]).append(row)
    summary=[]
    for variant,variant_rows in grouped.items():
        if {row["field"] for row in variant_rows}!=expected_fields:
            raise RuntimeError(f"{dataset}的{variant}缺少字段结果")
        compression_times=[row["compression_time"] for row in variant_rows]
        decompression_times=[row["decompression_time"] for row in variant_rows]
        summary.append({"dataset":dataset,"variant":variant,"family":"idw4" if variant.startswith("idw4_") else "rbf10","field_num":len(variant_rows),"geomean_br":geometric_mean([row["br"] for row in variant_rows]),"mean_psnr_db":sum(row["psnr"] for row in variant_rows)/len(variant_rows),"geomean_compression_time":geometric_mean(compression_times),"geomean_decompression_time":geometric_mean(decompression_times),"geomean_total_time":geometric_mean([compression+decompression for compression,decompression in zip(compression_times,decompression_times)])})
    return summary

def is_pareto(candidate,candidates):
    for other in candidates:
        no_worse=other["geomean_br"]<=candidate["geomean_br"] and other["mean_psnr_db"]>=candidate["mean_psnr_db"]
        strictly_better=other["geomean_br"]<candidate["geomean_br"] or other["mean_psnr_db"]>candidate["mean_psnr_db"]
        if no_worse and strictly_better:
            return False
    return True

def main():
    parser=argparse.ArgumentParser(description="汇总单个REL点的宽筛结果；最终选择仍需完整RD曲线")
    parser.add_argument("datasets",nargs="*",choices=all_datasets,default=list(all_datasets))
    arguments=parser.parse_args()
    rows=[]
    for dataset in arguments.datasets:
        rows.extend(summarize_dataset(dataset))
    output_path=result_root/"screen_summary.csv"
    with output_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for dataset in arguments.datasets:
        print(dataset)
        for family in ("idw4","rbf10"):
            candidates=[row for row in rows if row["dataset"]==dataset and row["family"]==family]
            candidates.sort(key=lambda row:row["geomean_br"])
            for row in candidates:
                pareto="*" if is_pareto(row,candidates) else " "
                print(f" {pareto}{row['variant']}:BR={row['geomean_br']:.6f},PSNR={row['mean_psnr_db']:.6f} dB,compression={row['geomean_compression_time']:.6f} s,decompression={row['geomean_decompression_time']:.6f} s")
    print("*表示单点宽筛中的BR/PSNR Pareto候选；最终参数必须由完整RD曲线决定")
    print(f"汇总={output_path}")

if __name__=="__main__":
    main()
