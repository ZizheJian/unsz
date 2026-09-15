import argparse,csv,json,math,os,re,subprocess,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
default_rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
configuration_alias_map={
    "idw_d_only":"idw_split_rounds_then_sort_mean_reference_distance",
    "idw_global":"idw_sort_all_then_split_rounds_mean_reference_distance",
    "idw_reference_data":"idw_split_rounds_then_sort_reference_data_prediction_error",
    "rbf_d_only":"rbf_split_rounds_then_sort_mean_reference_distance",
    "rbf_global":"rbf_sort_all_then_split_rounds_mean_reference_distance",
    "rbf_reference_data":"rbf_split_rounds_then_sort_reference_data_prediction_error",
}
canonical_configuration_vec=tuple(configuration_alias_map.values())

def parse_arguments():
    parser=argparse.ArgumentParser(description="比较一个field的IDW、RBF及不同空间预测顺序")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-variable",required=True)
    parser.add_argument("-time-step-num",type=int,required=True)
    parser.add_argument("-dim",type=int,required=True)
    parser.add_argument("-training-time-step",type=int,action="append")
    parser.add_argument("-test-time-step",type=int,action="append")
    parser.add_argument("-matrix-file",type=Path)
    parser.add_argument("-condition-operator",choices=("<","<=",">",">=","==","!="))
    parser.add_argument("-condition-value",type=float)
    parser.add_argument("-label-file-pattern")
    parser.add_argument("-label-file-dtype",default="uint8")
    parser.add_argument("-result",type=Path,required=True)
    parser.add_argument("-idw-ref-num",type=int,default=4)
    parser.add_argument("-idw-power",type=float,default=2)
    parser.add_argument("-idw-cache-threshold",type=int)
    parser.add_argument("-rbf-ref-num",type=int,default=10)
    parser.add_argument("-rbf-cache-threshold",type=int)
    parser.add_argument("-rbf-kernel",default="matern32")
    parser.add_argument("-rbf-shape",type=float,default=0.01)
    parser.add_argument("-rbf-nugget",type=float,default=1e-6)
    parser.add_argument("-order-candidate-num",type=int,default=48)
    parser.add_argument("-disable-eb-schedule",action="store_true")
    parser.add_argument("-dag-cache-result",type=Path)
    parser.add_argument("-reorder-data-by-prediction-order",action="store_true")
    parser.add_argument("-decompression-stage-time",action="store_true")
    parser.add_argument("-common-extra",action="append",default=[])
    parser.add_argument("-compression-extra",action="append",default=[])
    parser.add_argument("-decompression-extra",action="append",default=[])
    parser.add_argument("-rel-eb",type=float,action="append")
    parser.add_argument("-configuration",choices=(*configuration_alias_map,*canonical_configuration_vec),action="append")
    return parser.parse_args()

def run_command(command):
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError("命令执行失败："+" ".join(str(value) for value in command))
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def read_float_sum(pattern,text):
    return sum(float(value) for value in re.findall(pattern,text,re.MULTILINE))

def calculate_bd_rate(reference,test):
    reference_psnr=np.asarray([row["psnr"] for row in reference])
    reference_br=np.asarray([row["br"] for row in reference])
    test_psnr=np.asarray([row["psnr"] for row in test])
    test_br=np.asarray([row["br"] for row in test])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    reference_integral=np.polyint(np.polyfit(reference_psnr,np.log(reference_br),3))
    test_integral=np.polyint(np.polyfit(test_psnr,np.log(test_br),3))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (np.exp((test_area-reference_area)/(upper-lower))-1)*100

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def cleanup(test_dir,variable,test_time_step_vec):
    for time_step in test_time_step_vec:
        for suffix in (".unsz",".unsz.bin"):
            path=test_dir/f"{variable}.dat.{time_step}{suffix}"
            if path.exists():
                path.unlink()

def write_results(result_path,result_vec):
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with result_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_vec)

def write_point_set_config(path,arguments,predictor,rel_eb):
    labels=[{"method":predictor,"eb_mode":"REL","eb_value":rel_eb},{"method":"constant"}]
    if arguments.label_file_pattern is not None:
        config={"version":1,"rule_type":"label_file","label_file":{"path":arguments.label_file_pattern,"dtype":arguments.label_file_dtype},"labels":labels}
    else:
        labels[0]["condition"]={"operator":arguments.condition_operator,"value":arguments.condition_value}
        config={"version":1,"rule_type":"ordered","labels":labels}
    with path.open("w",encoding="utf-8") as file:
        json.dump(config,file,ensure_ascii=False)

def main():
    arguments=parse_arguments()
    if(arguments.condition_operator is None)!=(arguments.condition_value is None):
        raise ValueError("condition-operator和condition-value必须同时指定")
    if arguments.label_file_pattern is not None and arguments.condition_operator is not None:
        raise ValueError("label-file-pattern不能和condition-operator同时指定")
    if arguments.idw_ref_num<=0 or arguments.rbf_ref_num<=0:
        raise ValueError("参考点数必须为正数")
    idw_cache_threshold=arguments.idw_cache_threshold if arguments.idw_cache_threshold is not None else max(1,arguments.idw_ref_num-1)
    rbf_cache_threshold=arguments.rbf_cache_threshold if arguments.rbf_cache_threshold is not None else arguments.rbf_ref_num
    if idw_cache_threshold<0 or idw_cache_threshold>arguments.idw_ref_num:
        raise ValueError("idw-cache-threshold必须在0和idw-ref-num之间")
    training_time_step_vec=tuple(dict.fromkeys(arguments.training_time_step or ()))
    if arguments.test_time_step:
        test_time_step_vec=tuple(dict.fromkeys(arguments.test_time_step))
    else:
        test_time_step_vec=tuple(range(arguments.time_step_num))
    rel_eb_vec=tuple(arguments.rel_eb) if arguments.rel_eb else default_rel_eb_vec
    for time_step in (*training_time_step_vec,*test_time_step_vec):
        if time_step<0 or time_step>=arguments.time_step_num:
            raise ValueError(f"time step超出范围：{time_step}")
    if not test_time_step_vec:
        raise ValueError("测试time step不能为空")
    common_optional=[]
    if arguments.matrix_file is not None:
        common_optional=["-adjust-coord-matrix-file",arguments.matrix_file]
    training_expression="{"+",".join(str(value) for value in training_time_step_vec)+"}"
    test_expression="{"+",".join(str(value) for value in test_time_step_vec)+"}"
    all_config_vec=(
        {"name":"idw_split_rounds_then_sort_mean_reference_distance","predictor":"inverse_proportion","round_formation":"split_rounds_then_sort","ordering_score":"mean_reference_distance","ref_num":arguments.idw_ref_num,"cache_threshold":idw_cache_threshold,"power":arguments.idw_power,"compress_extra":["-reverse-mean-distance-only-order-experimental"],"decompress_extra":[]},
        {"name":"idw_sort_all_then_split_rounds_mean_reference_distance","predictor":"inverse_proportion","round_formation":"sort_all_then_split_rounds","ordering_score":"mean_reference_distance","ref_num":arguments.idw_ref_num,"cache_threshold":idw_cache_threshold,"power":arguments.idw_power,"compress_extra":["-reverse-mean-distance-only-order-experimental","-global-order-before-round-split-experimental"],"decompress_extra":[]},
        {"name":"idw_split_rounds_then_sort_reference_data_prediction_error","predictor":"inverse_proportion","round_formation":"split_rounds_then_sort","ordering_score":"reference_data_prediction_error","ref_num":arguments.idw_ref_num,"cache_threshold":idw_cache_threshold,"power":arguments.idw_power,"compress_extra":["-reference-data-order-experimental",training_expression,"-order-candidate-num-experimental",arguments.order_candidate_num],"decompress_extra":[]},
        {"name":"rbf_split_rounds_then_sort_mean_reference_distance","predictor":"rbf_fitting","round_formation":"split_rounds_then_sort","ordering_score":"mean_reference_distance","ref_num":arguments.rbf_ref_num,"cache_threshold":rbf_cache_threshold,"power":3.6,"compress_extra":["-reverse-mean-distance-only-order-experimental"],"decompress_extra":[]},
        {"name":"rbf_sort_all_then_split_rounds_mean_reference_distance","predictor":"rbf_fitting","round_formation":"sort_all_then_split_rounds","ordering_score":"mean_reference_distance","ref_num":arguments.rbf_ref_num,"cache_threshold":rbf_cache_threshold,"power":3.6,"compress_extra":["-reverse-mean-distance-only-order-experimental","-global-order-before-round-split-experimental"],"decompress_extra":[]},
        {"name":"rbf_split_rounds_then_sort_reference_data_prediction_error","predictor":"rbf_fitting","round_formation":"split_rounds_then_sort","ordering_score":"reference_data_prediction_error","ref_num":arguments.rbf_ref_num,"cache_threshold":rbf_cache_threshold,"power":3.6,"compress_extra":["-reference-data-order-experimental",training_expression,"-reference-data-rbf-score-experimental","-order-candidate-num-experimental",arguments.order_candidate_num],"decompress_extra":[]},
    )
    selected_configuration_set={configuration_alias_map.get(name,name) for name in arguments.configuration} if arguments.configuration else None
    config_vec=tuple(config for config in all_config_vec if selected_configuration_set is None or config["name"] in selected_configuration_set)
    if arguments.dag_cache_result is not None and len(config_vec)!=1:
        raise ValueError("持久保存预测DAG缓存时必须只选择一个configuration")
    if any(config["ordering_score"]=="reference_data_prediction_error" for config in config_vec) and not training_time_step_vec:
        raise ValueError("参考数据辅助顺序至少需要一个training-time-step")
    if any(config["ordering_score"]=="reference_data_prediction_error" for config in config_vec) and arguments.label_file_pattern is not None:
        raise ValueError("label_file分类暂时不能用于参考数据辅助顺序")
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix=f"unsz_{arguments.variable}_rbf_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(arguments.data_dir/"coords.dat")
        for time_step in set(training_time_step_vec)|set(test_time_step_vec):
            (test_dir/f"{arguments.variable}.dat.{time_step}").symlink_to(arguments.data_dir/f"{arguments.variable}.dat.{time_step}")
        for config in config_vec:
            prediction_structure_path=arguments.dag_cache_result if arguments.dag_cache_result is not None else temporary_dir/f"{config['name']}.dag"
            prediction_structure_path.parent.mkdir(parents=True,exist_ok=True)
            for rel_eb in rel_eb_vec:
                compression_error_arguments=["-M","REL",rel_eb,"-predictor",config["predictor"]]
                decompression_error_arguments=["-M","REL",rel_eb]
                if arguments.condition_operator is not None or arguments.label_file_pattern is not None:
                    point_set_config_path=temporary_dir/f"{config['name']}_{rel_eb:g}.json"
                    write_point_set_config(point_set_config_path,arguments,config["predictor"],rel_eb)
                    compression_error_arguments=["-point-set-config",point_set_config_path]
                    decompression_error_arguments=compression_error_arguments
                reorder_arguments=["-reorder-data-by-prediction-order-experimental"] if arguments.reorder_data_by_prediction_order else []
                schedule_arguments=[] if arguments.disable_eb_schedule else ["-eb-schedule-experimental","round_point_ratio",0.8]
                common_arguments=["-data-dir",test_dir,"-variable",arguments.variable,"-time-step",test_expression,"-dim",arguments.dim,"-dag-cache",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",config["ref_num"],"-actual-ref-num",config["ref_num"],"-inverse-distance-power",config["power"],"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",config["cache_threshold"],*schedule_arguments,"-rbf-kernel",arguments.rbf_kernel,"-rbf-shape",arguments.rbf_shape,"-rbf-nugget",arguments.rbf_nugget,*common_optional,*reorder_arguments,*arguments.common_extra]
                compression_output=run_command([compressor,*compression_error_arguments,*common_arguments,*config["compress_extra"],*arguments.compression_extra])
                decompression_output=run_command([decompressor,*decompression_error_arguments,*common_arguments,*config["decompress_extra"],*arguments.decompression_extra,*(["-decompression-stage-time"] if arguments.decompression_stage_time else [])])
                psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                compression_time=read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间")
                decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")
                result={"config":config["name"],"round_formation":config["round_formation"],"ordering_score":config["ordering_score"],"ref_num":config["ref_num"],"reorder_data_by_prediction_order":arguments.reorder_data_by_prediction_order,"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":compression_time,"decompression_time":decompression_time,"compression_decompression_time":compression_time+decompression_time,"compression_input_reorder_time":read_float_sum(r"^compression input reorder-by-prediction-order time excluded from compression= ([^ ]+) s$",compression_output),"decompression_output_restore_time":read_float_sum(r"^decompression output restore-original-order time excluded from decompression= ([^ ]+) s$",decompression_output),"decompression_stream_decode_time":read_float_sum(r"^decompression stage time: stream_decode=([^ ]+) s,",decompression_output),"decompression_label_decode_time":read_float_sum(r"^decompression stage time: stream_decode=[^ ]+ s, label_decode=([^ ]+) s,",decompression_output),"decompression_value_decode_time":read_float_sum(r"^decompression stage time: stream_decode=[^ ]+ s, label_decode=[^ ]+ s, value_decode=([^ ]+) s,",decompression_output)}
                result_vec.append(result)
                write_results(arguments.result,result_vec)
                print(f"{config['name']} REL={rel_eb:g}:BR={result['br']:.6f},PSNR={psnr:.6f},preprocessing={result['preprocessing_time']:.6f}s,compression={result['compression_time']:.6f}s,decompression={result['decompression_time']:.6f}s,compression+decompression={result['compression_decompression_time']:.6f}s",flush=True)
                cleanup(test_dir,arguments.variable,test_time_step_vec)
    result_map={name:[row for row in result_vec if row["config"]==name] for name in (config["name"] for config in config_vec)}
    baseline_name="idw_split_rounds_then_sort_mean_reference_distance"
    if baseline_name in result_map:
        baseline=result_map[baseline_name]
        for name in canonical_configuration_vec:
            if name==baseline_name:
                continue
            if name not in result_map:
                continue
            rows=result_map[name]
            print(f"{name}相对{baseline_name}：BD-rate={calculate_bd_rate(baseline,rows):+.6f}%，压缩时间倍率={geometric_mean([row['compression_time'] for row in rows])/geometric_mean([row['compression_time'] for row in baseline]):.6f}，解压时间倍率={geometric_mean([row['decompression_time'] for row in rows])/geometric_mean([row['decompression_time'] for row in baseline]):.6f}，压缩+解压时间倍率={geometric_mean([row['compression_decompression_time'] for row in rows])/geometric_mean([row['compression_decompression_time'] for row in baseline]):.6f}",flush=True)
    print(f"结果已写入{arguments.result}",flush=True)

if __name__=="__main__":
    main()
