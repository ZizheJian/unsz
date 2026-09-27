import argparse,csv,math,sys
from collections import defaultdict
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project_root))
from py_codes.reference_order import main as run_reference_order

def parse_arguments():
    parser=argparse.ArgumentParser(description="用五点曲线复核空间配置初筛中排名最前的候选")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-field",required=True)
    parser.add_argument("-coordinate-dim",type=int,required=True)
    parser.add_argument("-adjust-coord-matrix-file",type=Path)
    parser.add_argument("-tune-result",type=Path,required=True)
    parser.add_argument("-output-prefix",type=Path,required=True)
    parser.add_argument("-umc-csv",type=Path)
    parser.add_argument("-top-num",type=int,default=5)
    return parser.parse_args()

def calculate_average_br_change(reference,test):
    reference_psnr=np.asarray([float(row["psnr"]) for row in reference])
    reference_br=np.asarray([float(row["br"]) for row in reference])
    test_psnr=np.asarray([float(row["psnr"]) for row in test])
    test_br=np.asarray([float(row["br"]) for row in test])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    degree=min(3,len(reference)-1,len(test)-1)
    reference_integral=np.polyint(np.polyfit(reference_psnr,reference_br,degree))
    test_integral=np.polyint(np.polyfit(test_psnr,test_br,degree))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (test_area/reference_area-1)*100

def key_from_row(row):
    return row["coordinate"],row["predictor"],row["ordering_scope"],row["ordering_score"],int(row["prediction_max_ref_num"]),float(row["idw_power"]) if row["idw_power"] else "",float(row["rbf_shape"]) if row["rbf_shape"] else ""

def read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))

def main():
    arguments=parse_arguments()
    tune_rows=read_rows(arguments.tune_result)
    group_map=defaultdict(list)
    for row in tune_rows:
        group_map[key_from_row(row)].append(row)
    baseline_key=("original","idw","local","msd",4,2.0,"")
    baseline=group_map[baseline_key]
    if len(baseline)<3:
        raise RuntimeError("初筛结果缺少完整baseline")
    ranking=sorted((calculate_average_br_change(baseline,rows),key) for key,rows in group_map.items() if len(rows)>=3)
    candidate_key_vec=[key for _,key in ranking[:arguments.top_num]]
    run_key_vec=list(dict.fromkeys([baseline_key,*candidate_key_vec]))
    full_result_map={}
    arguments.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    for index,key in enumerate(run_key_vec):
        coordinate,predictor,ordering_scope,ordering_score,prediction_max_ref_num,idw_power,rbf_shape=key
        result_path=Path(f"{arguments.output_prefix}_{index}.csv")
        if result_path.exists() and len(read_rows(result_path))==5:
            full_result_map[key]=read_rows(result_path)
            continue
        configuration=f"{predictor}_{ordering_scope}_{ordering_score}"
        command=["-data-dir",arguments.data_dir,"-field",arguments.field,"-time-step-num",1,"-coordinate-dim",arguments.coordinate_dim,"-test-time-step",0,"-configuration",configuration,"-result",result_path]
        if predictor=="idw":
            command.extend(("-idw-ref-num",prediction_max_ref_num,"-idw-power",idw_power))
        else:
            command.extend(("-rbf-ref-num",prediction_max_ref_num,"-rbf-shape",rbf_shape))
        if coordinate=="adjusted":
            if arguments.adjust_coord_matrix_file is None:
                raise RuntimeError("adjusted候选缺少matrix-file")
            command.extend(("-adjust-coord-matrix-file",arguments.adjust_coord_matrix_file))
        run_reference_order(command)
        full_result_map[key]=read_rows(result_path)
    baseline_full=full_result_map[baseline_key]
    final_ranking=sorted((calculate_average_br_change(baseline_full,full_result_map[key]),key) for key in candidate_key_vec)
    best_average_br_change,best_key=final_ranking[0]
    summary={"field":arguments.field,"coordinate":best_key[0],"predictor":best_key[1],"ordering_scope":best_key[2],"ordering_score":best_key[3],"prediction_max_ref_num":best_key[4],"idw_power":best_key[5],"rbf_shape":best_key[6],"average_br_change_vs_default":best_average_br_change,"average_br_change_vs_umc":"","compression_ratio_vs_umc":""}
    if arguments.umc_csv is not None:
        umc_source=read_rows(arguments.umc_csv)
        if umc_source and {"UMC_BR","UMC_PSNR"}<=set(umc_source[0]):
            umc_rows=[{"br":row["UMC_BR"],"psnr":row["UMC_PSNR"]} for row in umc_source]
            umc_average_br_change=calculate_average_br_change(umc_rows,full_result_map[best_key])
            summary["average_br_change_vs_umc"]=umc_average_br_change
            summary["compression_ratio_vs_umc"]=1/(1+umc_average_br_change/100)
        else:
            print(f"{arguments.umc_csv}没有UMC_BR/UMC_PSNR列，跳过UMC比较",flush=True)
    summary_path=Path(f"{arguments.output_prefix}_summary.csv")
    with summary_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=summary.keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)
    print(summary,flush=True)

if __name__=="__main__":
    main()
