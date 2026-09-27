import argparse,csv,math,os,sys,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project_root))
from py_codes.reference_order import main as run_reference_order
default_rel_eb_vec=(1e-2,1e-3,1e-4)

def parse_arguments():
    parser=argparse.ArgumentParser(description="在冻结的有限候选集中筛选单个field的空间configuration")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-field",required=True)
    parser.add_argument("-time-step-num",type=int,required=True)
    parser.add_argument("-coordinate-dim",type=int,required=True)
    parser.add_argument("-adjust-coord-matrix-file",type=Path)
    parser.add_argument("-training-time-step",type=int,action="append")
    parser.add_argument("-test-time-step",type=int,action="append")
    parser.add_argument("-condition-operator",choices=("<","<=",">",">=","==","!="))
    parser.add_argument("-condition-value",type=float)
    parser.add_argument("-candidate-set",choices=("compact","full"),default="compact")
    parser.add_argument("-rel-eb",type=float,action="append")
    parser.add_argument("-result",type=Path,required=True)
    return parser.parse_args()

def write_results(path,row_vec):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=row_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(row_vec)

def read_results(path):
    if not path.exists():
        return []
    result_vec=[]
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            result_vec.append({"coordinate":row["coordinate"],"predictor":row["predictor"],"ordering_scope":row["ordering_scope"],"ordering_score":row["ordering_score"],"prediction_max_ref_num":int(row["prediction_max_ref_num"]),"idw_power":float(row["idw_power"]) if row["idw_power"] else "","rbf_shape":float(row["rbf_shape"]) if row["rbf_shape"] else "","rel_eb":float(row["rel_eb"]),"br":float(row["br"]),"psnr":float(row["psnr"]),"preprocessing_time":float(row["preprocessing_time"]),"compression_time":float(row["compression_time"]),"decompression_time":float(row["decompression_time"])})
    return result_vec

def configuration_key(row):
    return row["coordinate"],row["predictor"],row["ordering_scope"],row["ordering_score"],row["prediction_max_ref_num"],row["idw_power"],row["rbf_shape"]

def calculate_average_br_change(reference,test):
    reference_psnr=np.asarray([row["psnr"] for row in reference])
    reference_br=np.asarray([row["br"] for row in reference])
    test_psnr=np.asarray([row["psnr"] for row in test])
    test_br=np.asarray([row["br"] for row in test])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    degree=min(3,len(reference)-1,len(test)-1)
    reference_integral=np.polyint(np.polyfit(reference_psnr,reference_br,degree))
    test_integral=np.polyint(np.polyfit(test_psnr,test_br,degree))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (test_area/reference_area-1)*100

def main():
    arguments=parse_arguments()
    rel_eb_vec=tuple(arguments.rel_eb) if arguments.rel_eb else default_rel_eb_vec
    order_dimension_vec=(("local","msd"),("local","reference_data_prediction_error")) if arguments.training_time_step else (("local","msd"),("global","msd"))
    predictor_vec=[("idw",4,2,float("nan"))]
    predictor_vec.extend(("rbf",10,3.6,shape) for shape in (0.01,0.03))
    if arguments.candidate_set=="full":
        predictor_vec[1:1]=[("idw",5,2.5,float("nan")),("idw",10,3.6,float("nan"))]
        predictor_vec.extend(("rbf",8,3.6,shape) for shape in (0.01,0.03))
    coordinate_vec=[("original",None)]
    if arguments.adjust_coord_matrix_file is not None:
        coordinate_vec.append(("adjusted",arguments.adjust_coord_matrix_file))
    result_vec=read_results(arguments.result)
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix=f"unsz_tune_{arguments.field}_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        for coordinate,matrix_file in coordinate_vec:
            for predictor,prediction_max_ref_num,power,shape in predictor_vec:
                for ordering_scope,ordering_score in order_dimension_vec:
                    key=(coordinate,predictor,ordering_scope,ordering_score,prediction_max_ref_num,power if predictor=="idw" else "",shape if predictor=="rbf" else "")
                    existing_rel_eb_set={row["rel_eb"] for row in result_vec if configuration_key(row)==key}
                    if all(rel_eb in existing_rel_eb_set for rel_eb in rel_eb_vec):
                        print(f"跳过已完成configuration：{key}",flush=True)
                        continue
                    configuration=f"{predictor}_{ordering_scope}_{ordering_score}"
                    output_path=temporary_dir/f"{coordinate}_{predictor}_r{prediction_max_ref_num}_{ordering_scope}_{ordering_score}_{shape}.csv"
                    command=["-data-dir",arguments.data_dir,"-field",arguments.field,"-time-step-num",arguments.time_step_num,"-coordinate-dim",arguments.coordinate_dim,"-configuration",configuration,"-result",output_path]
                    for rel_eb in rel_eb_vec:
                        command.extend(("-rel-eb",rel_eb))
                    for time_step in arguments.training_time_step or ():
                        command.extend(("-training-time-step",time_step))
                    for time_step in arguments.test_time_step or ():
                        command.extend(("-test-time-step",time_step))
                    if matrix_file is not None:
                        command.extend(("-adjust-coord-matrix-file",matrix_file))
                    if arguments.condition_operator is not None:
                        command.extend(("-condition-operator",arguments.condition_operator,"-condition-value",arguments.condition_value))
                    if predictor=="idw":
                        command.extend(("-idw-ref-num",prediction_max_ref_num,"-idw-power",power))
                    else:
                        command.extend(("-rbf-ref-num",prediction_max_ref_num,"-rbf-shape",shape))
                    run_reference_order(command)
                    result_vec=[row for row in result_vec if configuration_key(row)!=key]
                    with output_path.open(newline="") as file:
                        for row in csv.DictReader(file):
                            result_vec.append({"coordinate":coordinate,"predictor":predictor,"ordering_scope":ordering_scope,"ordering_score":ordering_score,"prediction_max_ref_num":prediction_max_ref_num,"idw_power":power if predictor=="idw" else "","rbf_shape":shape if predictor=="rbf" else "","rel_eb":float(row["rel_eb"]),"br":float(row["br"]),"psnr":float(row["psnr"]),"preprocessing_time":float(row["preprocessing_time"]),"compression_time":float(row["compression_time"]),"decompression_time":float(row["decompression_time"])})
                    write_results(arguments.result,result_vec)
    baseline=[row for row in result_vec if row["coordinate"]=="original" and row["predictor"]=="idw" and row["ordering_scope"]=="local" and row["ordering_score"]=="msd" and row["prediction_max_ref_num"]==4]
    summary=[]
    key_vec=[]
    for row in result_vec:
        key=configuration_key(row)
        if key not in key_vec:
            key_vec.append(key)
    for key in key_vec:
        rows=[row for row in result_vec if configuration_key(row)==key]
        summary.append((calculate_average_br_change(baseline,rows),key))
    print(f"相对原坐标4点IDW local/msd的{len(rel_eb_vec)}点同PSNR平均BR变化排名：")
    for average_br_change,key in sorted(summary):
        print(f"  {key}: {average_br_change:+.6f}%")
    print(f"结果已写入{arguments.result}")

if __name__=="__main__":
    main()
