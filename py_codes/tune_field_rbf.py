import argparse,csv,math,os,tempfile
from pathlib import Path
from screen_field_geometric_order import calculate_average_br_change,cleanup,decompressor,geometric_mean,project_root,read_float,run_command,write_results,compressor

default_rel_eb_vec=(1e-2,1e-3,1e-4)
ordering_map={
    "reverse_mean_reference_distance":("reverse_elimination","mean_reference_distance",["-reverse-mean-distance-only-order-experimental"]),
    "reverse_nearest_reference_distance_direction_concentration":("reverse_elimination","nearest_reference_distance+direction_concentration",["-reverse-direction-order-experimental"]),
    "global_sort_msd":("reverse_elimination","msd",["-reverse-mean-distance-only-order-experimental","-global-order-before-round-split-experimental"]),
}

def parse_arguments():
    parser=argparse.ArgumentParser(description="分阶段筛选单个field的RBF参数")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-variable",required=True)
    parser.add_argument("-time-step",type=int,default=0)
    parser.add_argument("-dim",type=int,required=True)
    parser.add_argument("-matrix-file",type=Path)
    parser.add_argument("-stage",choices=("kernel_shape","shape","ref_num","regularization"),required=True)
    parser.add_argument("-kernel",default="matern32")
    parser.add_argument("-shape",type=float,default=0.03)
    parser.add_argument("-prediction-ref-num",type=int,default=10)
    parser.add_argument("-ordering",choices=tuple(ordering_map),default="global_sort_msd")
    parser.add_argument("-ordering-candidate-num",type=int,default=16)
    parser.add_argument("-ordering-score-max-ref-num",type=int,default=10)
    parser.add_argument("-ordering-score-min-ref-num",type=int)
    parser.add_argument("-eb-schedule-base",type=float,default=0.85)
    parser.add_argument("-eb-schedule-log2-shift",type=float,default=0.5)
    parser.add_argument("-rel-eb",type=float,action="append")
    parser.add_argument("-result",type=Path,required=True)
    return parser.parse_args()

def get_config_vec(arguments):
    if arguments.stage=="kernel_shape":
        kernel_shape_map={
            "gaussian":(0.1,0.25,0.5,1),
            "exponential":(0.05,0.1,0.25,0.5,1),
            "matern32":(0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.2),
            "matern52":(0.01,0.03,0.05,0.1,0.25,0.5),
            "inverse_multiquadric":(0.1,0.25,0.5,1,2),
        }
        return tuple({"kernel":kernel,"shape":shape,"prediction_ref_num":arguments.prediction_ref_num,"nugget":1e-6,"degree":0} for kernel,shape_vec in kernel_shape_map.items() for shape in shape_vec)
    if arguments.stage=="shape":
        return tuple({"kernel":arguments.kernel,"shape":shape,"prediction_ref_num":arguments.prediction_ref_num,"nugget":1e-6,"degree":0} for shape in (0.01,0.03,0.1,0.3))
    if arguments.stage=="ref_num":
        return tuple({"kernel":arguments.kernel,"shape":arguments.shape,"prediction_ref_num":ref_num,"nugget":1e-6,"degree":0} for ref_num in (4,6,8,10))
    return tuple([{"kernel":arguments.kernel,"shape":arguments.shape,"prediction_ref_num":arguments.prediction_ref_num,"nugget":nugget,"degree":0} for nugget in (1e-10,1e-8,1e-6,1e-4,1e-2)]+[{"kernel":arguments.kernel,"shape":arguments.shape,"prediction_ref_num":arguments.prediction_ref_num,"nugget":nugget,"degree":degree} for degree in (1,2) for nugget in (1e-8,1e-6,1e-4)])

def configuration_name(config):
    return f"{config['kernel']}_shape_{config['shape']:g}_ref_{config['prediction_ref_num']}_nugget_{config['nugget']:g}_degree_{config['degree']}"

def configuration_key(row):
    return row["kernel"],row["shape"],row["prediction_ref_num"],row["nugget"],row["polynomial_degree"],row["rel_eb"]

def read_results(path):
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return [{"config":row["config"],"ordering_scope":row["ordering_scope"],"ordering_construction":row["ordering_construction"],"ordering_score":row["ordering_score"],"ordering_candidate_num":int(row["ordering_candidate_num"]),"ordering_score_max_ref_num":int(row["ordering_score_max_ref_num"]),"ordering_score_min_ref_num":int(row["ordering_score_min_ref_num"]),"kernel":row["kernel"],"shape":float(row["shape"]),"prediction_ref_num":int(row["prediction_ref_num"]),"nugget":float(row["nugget"]),"polynomial_degree":int(row["polynomial_degree"]),"rel_eb":float(row["rel_eb"]),"br":float(row["br"]),"psnr":float(row["psnr"]),"preprocessing_time":float(row["preprocessing_time"]),"compression_time":float(row["compression_time"]),"decompression_time":float(row["decompression_time"])} for row in csv.DictReader(file)]

def main():
    arguments=parse_arguments()
    if arguments.prediction_ref_num<=0:
        raise ValueError("prediction-ref-num必须为正数")
    if arguments.ordering_candidate_num<=0 or arguments.ordering_score_max_ref_num<=0:
        raise ValueError("排序候选数和排序评分最大参考点数必须为正数")
    ordering_score_min_ref_num=arguments.ordering_score_min_ref_num or arguments.ordering_score_max_ref_num
    if ordering_score_min_ref_num<=0 or ordering_score_min_ref_num>arguments.ordering_score_max_ref_num:
        raise ValueError("ordering-score-min-ref-num必须在[1,ordering-score-max-ref-num]内")
    if not 0<arguments.eb_schedule_base<=1 or arguments.eb_schedule_log2_shift<=0:
        raise ValueError("EB底数必须在(0,1]内，shift必须为正数")
    config_vec=get_config_vec(arguments)
    rel_eb_vec=tuple(arguments.rel_eb) if arguments.rel_eb else default_rel_eb_vec
    ordering_construction,ordering_score,ordering_flags=ordering_map[arguments.ordering]
    ordering_scope="global_sort" if arguments.ordering=="global_sort_msd" else "layered_sort"
    result_vec=read_results(arguments.result)
    completed_key_set={configuration_key(row) for row in result_vec}
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix=f"unsz_rbf_{arguments.variable}_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        data_dir=temporary_dir/"data"
        data_dir.mkdir()
        (data_dir/"coords.dat").symlink_to(arguments.data_dir/"coords.dat")
        (data_dir/f"{arguments.variable}.dat.{arguments.time_step}").symlink_to(arguments.data_dir/f"{arguments.variable}.dat.{arguments.time_step}")
        for config in config_vec:
            name=configuration_name(config)
            if config["prediction_ref_num"]>arguments.ordering_score_max_ref_num:
                raise ValueError("当前C++接口要求prediction-ref-num不超过ordering-score-max-ref-num")
            prediction_structure_path=temporary_dir/f"{arguments.ordering}_score_ref_{arguments.ordering_score_max_ref_num}.dag"
            for rel_eb in rel_eb_vec:
                key=config["kernel"],config["shape"],config["prediction_ref_num"],config["nugget"],config["degree"],rel_eb
                if key in completed_key_set:
                    continue
                common_arguments=["-data-dir",data_dir,"-variable",arguments.variable,"-time-step",f"{{{arguments.time_step}}}","-dim",arguments.dim,"-dag-cache",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",arguments.ordering_score_max_ref_num,"-actual-ref-num",config["prediction_ref_num"],"-kd-leaf-size",32,"-order-candidate-num-experimental",arguments.ordering_candidate_num,"-d-only-cache-refill-threshold-experimental",ordering_score_min_ref_num,"-eb-schedule-experimental","point_log2_ratio",arguments.eb_schedule_base,"-eb-schedule-log2-shift-experimental",arguments.eb_schedule_log2_shift,"-rbf-kernel",config["kernel"],"-rbf-shape",config["shape"],"-rbf-nugget",config["nugget"],"-rbf-polynomial-degree",config["degree"]]
                if arguments.matrix_file is not None:
                    common_arguments.extend(("-adjust-coord-matrix-file",arguments.matrix_file))
                compression_output=run_command([compressor,"-M","REL",rel_eb,"-predictor","rbf_fitting",*common_arguments,*ordering_flags])
                decompression_output=run_command([decompressor,"-M","REL",rel_eb,*common_arguments])
                psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                row={"config":name,"ordering_scope":ordering_scope,"ordering_construction":ordering_construction,"ordering_score":ordering_score,"ordering_candidate_num":arguments.ordering_candidate_num,"ordering_score_max_ref_num":arguments.ordering_score_max_ref_num,"ordering_score_min_ref_num":ordering_score_min_ref_num,"kernel":config["kernel"],"shape":config["shape"],"prediction_ref_num":config["prediction_ref_num"],"nugget":config["nugget"],"polynomial_degree":config["degree"],"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
                result_vec.append(row)
                completed_key_set.add(key)
                write_results(arguments.result,result_vec)
                print(f"{name} REL={rel_eb:g}:BR={row['br']:.6f},PSNR={psnr:.6f},preprocessing={row['preprocessing_time']:.6f}s,compression={row['compression_time']:.6f}s,decompression={row['decompression_time']:.6f}s",flush=True)
                cleanup(data_dir,arguments.variable,arguments.time_step)
    ranked=[]
    for config in config_vec:
        name=configuration_name(config)
        rows=[row for row in result_vec if row["config"]==name and row["rel_eb"] in rel_eb_vec]
        if len(rows)==len(rel_eb_vec):
            ranked.append((name,rows))
    baseline_name=configuration_name({"kernel":arguments.kernel,"shape":arguments.shape,"prediction_ref_num":arguments.prediction_ref_num,"nugget":1e-6,"degree":0})
    baseline=next((rows for name,rows in ranked if name==baseline_name),None)
    if baseline is None:
        baseline=min(ranked,key=lambda item:geometric_mean([row["br"] for row in item[1]]))[1]
        baseline_name=baseline[0]["config"]
    baseline_compression_time=geometric_mean([row["compression_time"] for row in baseline])
    baseline_decompression_time=geometric_mean([row["decompression_time"] for row in baseline])
    baseline_total_time=geometric_mean([row["compression_time"]+row["decompression_time"] for row in baseline])
    summary=sorted((calculate_average_br_change(baseline,rows),name,geometric_mean([row["compression_time"] for row in rows])/baseline_compression_time,geometric_mean([row["decompression_time"] for row in rows])/baseline_decompression_time,geometric_mean([row["compression_time"]+row["decompression_time"] for row in rows])/baseline_total_time) for name,rows in ranked)
    print(f"相对{baseline_name}的同PSNR平均BR变化排名：")
    for average_br_change,name,compression_ratio,decompression_ratio,total_ratio in summary:
        print(f"  {name}:average_BR_change={average_br_change:+.6f}%,compression_time={compression_ratio:.6f}x,decompression_time={decompression_ratio:.6f}x,total_time={total_ratio:.6f}x")
    print(f"结果已写入{arguments.result}")

if __name__=="__main__":
    main()
