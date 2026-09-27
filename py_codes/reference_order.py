import argparse,json,math,os,re,sys,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project_root))
import numpy as np
from third_party.csv_tools.py_codes.write_csv_item import write_csv_item
from third_party.run_command.run_command import run_command

reference_canonical_configurations=("idw_local_msd","idw_global_msd","idw_local_reference_data_prediction_error","rbf_local_msd","rbf_global_msd","rbf_local_reference_data_prediction_error")

def parse_arguments(arguments):
    parser=argparse.ArgumentParser(description="比较一个field的IDW、RBF及不同空间预测顺序")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-field",required=True)
    parser.add_argument("-time-step-num",type=int,required=True)
    parser.add_argument("-coordinate-dim",type=int,required=True)
    parser.add_argument("-training-time-step",type=int,action="append")
    parser.add_argument("-test-time-step",type=int,action="append")
    parser.add_argument("-adjust-coord-matrix-file",type=Path)
    parser.add_argument("-condition-operator",choices=("<","<=",">",">=","==","!="))
    parser.add_argument("-condition-value",type=float)
    parser.add_argument("-result",type=Path,required=True)
    parser.add_argument("-idw-ref-num",type=int,default=4)
    parser.add_argument("-idw-power",type=float,default=2)
    parser.add_argument("-rbf-ref-num",type=int,default=10)
    parser.add_argument("-ordering-score-max-ref-num",type=int)
    parser.add_argument("-ordering-score-min-ref-num",type=int)
    parser.add_argument("-rbf-kernel",default="matern32")
    parser.add_argument("-rbf-shape",type=float,default=0.01)
    parser.add_argument("-rbf-nugget",type=float,default=1e-6)
    parser.add_argument("-ordering-candidate-num",type=int,default=48)
    parser.add_argument("-dag-cache-path",type=Path)
    parser.add_argument("-decompression-stage-time",action="store_true")
    parser.add_argument("-rel-eb",type=float,action="append")
    parser.add_argument("-configuration",choices=reference_canonical_configurations,action="append")
    return parser.parse_args(arguments)

def read_output_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def read_output_float_sum(pattern,text):
    return sum(float(value) for value in re.findall(pattern,text,re.MULTILINE))

def calculate_average_br_change(reference,test_rows):
    reference_points=sorted((float(row["psnr"]),float(row["br"])) for row in reference)
    test_points=sorted((float(row["psnr"]),float(row["br"])) for row in test_rows)
    reference_psnr=np.asarray([point[0] for point in reference_points])
    reference_br=np.asarray([point[1] for point in reference_points])
    test_psnr=np.asarray([point[0] for point in test_points])
    test_br=np.asarray([point[1] for point in test_points])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    if lower>=upper:
        raise ValueError("两条BR-PSNR曲线没有共同PSNR范围")
    knots=np.unique(np.concatenate(([lower,upper],reference_psnr[(reference_psnr>lower)&(reference_psnr<upper)],test_psnr[(test_psnr>lower)&(test_psnr<upper)])))
    reference_area=np.trapezoid(np.interp(knots,reference_psnr,reference_br),knots)
    test_area=np.trapezoid(np.interp(knots,test_psnr,test_br),knots)
    return (test_area/reference_area-1)*100

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def cleanup_outputs(test_dir,field,test_time_steps):
    for time_step in test_time_steps:
        for suffix in (".unsz",".unsz.bin"):
            path=test_dir/f"{field}.dat.{time_step}{suffix}"
            if path.exists():
                path.unlink()

def write_point_set_config(path,arguments,predictor,rel_eb):
    labels=[{"predictor":predictor,"eb_mode":"REL","eb_value":rel_eb},{"predictor":"constant"}]
    labels[0]["condition"]={"operator":arguments.condition_operator,"value":arguments.condition_value}
    config={"version":2,"labels":labels}
    with path.open("w",encoding="utf-8") as file:
        json.dump(config,file,ensure_ascii=False)

def main(command_line_arguments):
    arguments=parse_arguments(command_line_arguments)
    if(arguments.condition_operator is None)!=(arguments.condition_value is None):
        raise ValueError("condition-operator和condition-value必须同时指定")
    if arguments.idw_ref_num<=0 or arguments.rbf_ref_num<=0:
        raise ValueError("参考点数必须为正数")
    training_time_steps=tuple(dict.fromkeys(arguments.training_time_step or ()))
    test_time_steps=tuple(dict.fromkeys(arguments.test_time_step)) if arguments.test_time_step else tuple(range(arguments.time_step_num))
    rel_eb_values=tuple(arguments.rel_eb) if arguments.rel_eb else (1e-1,1e-2,1e-3,1e-4,1e-5)
    for time_step in (*training_time_steps,*test_time_steps):
        if time_step<0 or time_step>=arguments.time_step_num:
            raise ValueError(f"time step超出范围：{time_step}")
    if not test_time_steps:
        raise ValueError("测试time step不能为空")
    common_optional=["-adjust-coord-matrix-file",arguments.adjust_coord_matrix_file] if arguments.adjust_coord_matrix_file is not None else []
    training_expression="{"+",".join(str(value) for value in training_time_steps)+"}"
    test_expression="{"+",".join(str(value) for value in test_time_steps)+"}"
    all_configs=(
        {"name":"idw_local_msd","predictor":"idw","ordering_scope":"local","prediction_round_num":8,"ordering_score":"msd","prediction_min_ref_num":1,"prediction_max_ref_num":arguments.idw_ref_num,"idw_power":arguments.idw_power,"compression_arguments":["-reverse-msd-order"]},
        {"name":"idw_global_msd","predictor":"idw","ordering_scope":"global","prediction_round_num":1,"ordering_score":"msd","prediction_min_ref_num":1,"prediction_max_ref_num":arguments.idw_ref_num,"idw_power":arguments.idw_power,"compression_arguments":["-reverse-msd-order"]},
        {"name":"idw_local_reference_data_prediction_error","predictor":"idw","ordering_scope":"local","prediction_round_num":8,"ordering_score":"reference_data_prediction_error","prediction_min_ref_num":1,"prediction_max_ref_num":arguments.idw_ref_num,"idw_power":arguments.idw_power,"compression_arguments":["-reference-data-order",training_expression]},
        {"name":"rbf_local_msd","predictor":"rbf","ordering_scope":"local","prediction_round_num":8,"ordering_score":"msd","prediction_min_ref_num":arguments.rbf_ref_num,"prediction_max_ref_num":arguments.rbf_ref_num,"idw_power":3.6,"compression_arguments":["-reverse-msd-order"]},
        {"name":"rbf_global_msd","predictor":"rbf","ordering_scope":"global","prediction_round_num":1,"ordering_score":"msd","prediction_min_ref_num":arguments.rbf_ref_num,"prediction_max_ref_num":arguments.rbf_ref_num,"idw_power":3.6,"compression_arguments":["-reverse-msd-order"]},
        {"name":"rbf_local_reference_data_prediction_error","predictor":"rbf","ordering_scope":"local","prediction_round_num":8,"ordering_score":"reference_data_prediction_error","prediction_min_ref_num":arguments.rbf_ref_num,"prediction_max_ref_num":arguments.rbf_ref_num,"idw_power":3.6,"compression_arguments":["-reference-data-order",training_expression,"-reference-data-rbf-score"]},
    )
    selected_configuration_set=set(arguments.configuration) if arguments.configuration else None
    configs=tuple(config for config in all_configs if selected_configuration_set is None or config["name"] in selected_configuration_set)
    for config in configs:
        config["ordering_score_max_ref_num"]=arguments.ordering_score_max_ref_num or config["prediction_max_ref_num"]
        config["ordering_score_min_ref_num"]=arguments.ordering_score_min_ref_num if arguments.ordering_score_min_ref_num is not None else config["ordering_score_max_ref_num"]
        if config["ordering_score_min_ref_num"]<=0 or config["ordering_score_min_ref_num"]>config["ordering_score_max_ref_num"]:
            raise ValueError("ordering score min ref num必须在1和ordering score max ref num之间")
    if arguments.dag_cache_result is not None and len(configs)!=1:
        raise ValueError("持久保存DAG缓存时必须只选择一个configuration")
    if any(config["ordering_score"]=="reference_data_prediction_error" for config in configs) and not training_time_steps:
        raise ValueError("参考数据辅助顺序至少需要一个training-time-step")
    results=[]
    arguments.result.parent.mkdir(parents=True,exist_ok=True)
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix=f"unsz_{arguments.field}_rbf_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(arguments.data_dir/"coords.dat")
        for time_step in set(training_time_steps)|set(test_time_steps):
            (test_dir/f"{arguments.field}.dat.{time_step}").symlink_to(arguments.data_dir/f"{arguments.field}.dat.{time_step}")
        for config in configs:
            dag_cache_path=arguments.dag_cache_path if arguments.dag_cache_path is not None else temporary_dir/f"{config['name']}.dag"
            dag_cache_path.parent.mkdir(parents=True,exist_ok=True)
            for rel_eb in rel_eb_values:
                compression_error_arguments=["-M","REL",rel_eb,"-predictor",config["predictor"]]
                decompression_error_arguments=["-M","REL",rel_eb]
                if arguments.condition_operator is not None:
                    point_set_config_path=temporary_dir/f"{config['name']}_{rel_eb:g}.json"
                    write_point_set_config(point_set_config_path,arguments,config["predictor"],rel_eb)
                    compression_error_arguments=["-point-set-config",point_set_config_path]
                    decompression_error_arguments=compression_error_arguments
                schedule_arguments=["-eb-schedule","round_point_ratio",0.8]
                common_arguments=["-data-dir",test_dir,"-field",arguments.field,"-time-step",test_expression,"-coordinate-dim",arguments.coordinate_dim,"-dag-cache",dag_cache_path,"-prediction-round-num",config["prediction_round_num"],"-prediction-min-ref-num",config["prediction_min_ref_num"],"-prediction-max-ref-num",config["prediction_max_ref_num"],"-ordering-scope",config["ordering_scope"],"-idw-power",config["idw_power"],"-kd-leaf-size",32,"-ordering-score-max-ref-num",config["ordering_score_max_ref_num"],"-ordering-score-min-ref-num",config["ordering_score_min_ref_num"],"-ordering-candidate-num",arguments.ordering_candidate_num,*schedule_arguments,"-rbf-kernel",arguments.rbf_kernel,"-rbf-shape",arguments.rbf_shape,"-rbf-nugget",arguments.rbf_nugget,*common_optional]
                compression_output=run_command([project_root/"install/bin/s1_compress",*compression_error_arguments,*common_arguments,*config["compression_arguments"]],silent=True)
                decompression_output=run_command([project_root/"install/bin/s1_decompress",*decompression_error_arguments,*common_arguments,*(["-decompression-stage-time"] if arguments.decompression_stage_time else [])],silent=True)
                psnr=read_output_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_output_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                compression_time=read_output_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间")
                decompression_time=read_output_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")
                result={"configuration_name":config["name"],"ordering_scope":config["ordering_scope"],"prediction_round_num":config["prediction_round_num"],"ordering_score":config["ordering_score"],"prediction_max_ref_num":config["prediction_max_ref_num"],"ordering_score_max_ref_num":config["ordering_score_max_ref_num"],"ordering_score_min_ref_num":config["ordering_score_min_ref_num"],"rel_eb":rel_eb,"br":read_output_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":read_output_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":compression_time,"decompression_time":decompression_time,"compression_decompression_time":compression_time+decompression_time,"decompression_stream_decode_time":read_output_float_sum(r"^decompression stage time: stream_decode=([^ ]+) s,",decompression_output),"decompression_label_decode_time":read_output_float_sum(r"^decompression stage time: stream_decode=[^ ]+ s, label_decode=([^ ]+) s,",decompression_output),"decompression_value_decode_time":read_output_float_sum(r"^decompression stage time: stream_decode=[^ ]+ s, label_decode=[^ ]+ s, value_decode=([^ ]+) s,",decompression_output)}
                results.append(result)
                row_info={name:result[name] for name in ("configuration_name","ordering_scope","prediction_round_num","ordering_score","prediction_max_ref_num","ordering_score_max_ref_num","ordering_score_min_ref_num","rel_eb")}
                for column_name,value in result.items():
                    if column_name not in row_info:
                        write_csv_item(str(arguments.result),row_info,column_name,value)
                print(f"{config['name']} REL={rel_eb:g}:BR={result['br']:.6f},PSNR={psnr:.6f},preprocessing={result['preprocessing_time']:.6f}s,compression={result['compression_time']:.6f}s,decompression={result['decompression_time']:.6f}s,compression+decompression={result['compression_decompression_time']:.6f}s",flush=True)
                cleanup_outputs(test_dir,arguments.field,test_time_steps)
    result_map={name:[row for row in results if row["configuration_name"]==name] for name in (config["name"] for config in configs)}
    baseline_name="idw_local_msd"
    if baseline_name in result_map:
        baseline=result_map[baseline_name]
        for name in reference_canonical_configurations:
            if name==baseline_name or name not in result_map:
                continue
            rows=result_map[name]
            print(f"{name}相对{baseline_name}：同PSNR平均BR变化={calculate_average_br_change(baseline,rows):+.6f}%，压缩时间倍率={geometric_mean([row['compression_time'] for row in rows])/geometric_mean([row['compression_time'] for row in baseline]):.6f}，解压时间倍率={geometric_mean([row['decompression_time'] for row in rows])/geometric_mean([row['decompression_time'] for row in baseline]):.6f}，压缩+解压时间倍率={geometric_mean([row['compression_decompression_time'] for row in rows])/geometric_mean([row['compression_decompression_time'] for row in baseline]):.6f}",flush=True)
    print(f"结果已写入{arguments.result}",flush=True)

if __name__=="__main__":
    main(sys.argv[1:])
