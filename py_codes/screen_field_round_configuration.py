import argparse,csv,math,os,tempfile
from pathlib import Path
from screen_field_geometric_order import calculate_average_br_change,cleanup,decompressor,project_root,read_float,run_command,write_results,compressor

default_rel_eb_vec=(1e-2,1e-3,1e-4)

def parse_arguments():
    parser=argparse.ArgumentParser(description="筛选单个field的空间预测轮结构")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-field",required=True)
    parser.add_argument("-time-step",type=int,default=0)
    parser.add_argument("-coordinate-dim",type=int,required=True)
    parser.add_argument("-prediction-max-ref-num",type=int,required=True)
    parser.add_argument("-rbf-kernel",default="matern32")
    parser.add_argument("-rbf-shape",type=float,default=0.03)
    parser.add_argument("-rbf-nugget",type=float,default=1e-6)
    parser.add_argument("-prediction-round-num",type=int,action="append")
    parser.add_argument("-eb-ratio",type=float,action="append")
    parser.add_argument("-rel-eb",type=float,action="append")
    parser.add_argument("-result",type=Path,required=True)
    return parser.parse_args()

def main():
    arguments=parse_arguments()
    rel_eb_vec=tuple(arguments.rel_eb) if arguments.rel_eb else default_rel_eb_vec
    round_num_vec=tuple(arguments.prediction_round_num) if arguments.prediction_round_num else (6,8,10,12)
    eb_ratio_vec=tuple(arguments.eb_ratio) if arguments.eb_ratio else (0.7,0.8,0.9)
    config_vec=tuple((round_num,eb_ratio) for round_num in round_num_vec for eb_ratio in eb_ratio_vec)
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix=f"unsz_round_{arguments.field}_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        data_dir=temporary_dir/"data"
        data_dir.mkdir()
        (data_dir/"coords.dat").symlink_to(arguments.data_dir/"coords.dat")
        (data_dir/f"{arguments.field}.dat.{arguments.time_step}").symlink_to(arguments.data_dir/f"{arguments.field}.dat.{arguments.time_step}")
        for round_num,eb_ratio in config_vec:
            config=f"round_{round_num}_eb_ratio_{eb_ratio:g}"
            dag_cache_path=temporary_dir/f"round_{round_num}.dag"
            for rel_eb in rel_eb_vec:
                common_arguments=["-data-dir",data_dir,"-field",arguments.field,"-time-step",f"{{{arguments.time_step}}}","-coordinate-dim",arguments.coordinate_dim,"-dag-cache",dag_cache_path,"-prediction-round-num",round_num,"-prediction-min-ref-num",arguments.prediction_max_ref_num,"-prediction-max-ref-num",arguments.prediction_max_ref_num,"-ordering-scope","local","-kd-leaf-size",32,"-ordering-candidate-num",arguments.prediction_max_ref_num,"-ordering-score-max-ref-num",arguments.prediction_max_ref_num,"-ordering-score-min-ref-num",arguments.prediction_max_ref_num,"-eb-schedule","round_point_ratio",eb_ratio,"-rbf-kernel",arguments.rbf_kernel,"-rbf-shape",arguments.rbf_shape,"-rbf-nugget",arguments.rbf_nugget,"-reverse-msd-order"]
                compression_output=run_command([compressor,"-M","REL",rel_eb,"-predictor","rbf",*common_arguments])
                decompression_output=run_command([decompressor,"-M","REL",rel_eb,*common_arguments])
                psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                row={"config":config,"prediction_round_num":round_num,"eb_schedule":"round_point_ratio","eb_ratio":eb_ratio,"round_size_growth_ratio":2,"ordering_scope":"local","ordering_construction":"reverse_elimination","ordering_score":"msd","prediction_max_ref_num":arguments.prediction_max_ref_num,"rbf_kernel":arguments.rbf_kernel,"rbf_shape":arguments.rbf_shape,"rbf_nugget":arguments.rbf_nugget,"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
                result_vec.append(row)
                write_results(arguments.result,result_vec)
                print(f"{config} REL={rel_eb:g}:BR={row['br']:.6f},PSNR={psnr:.6f}",flush=True)
                cleanup(data_dir,arguments.field,arguments.time_step)
    baseline_config="round_8_eb_ratio_0.8" if any(row["config"]=="round_8_eb_ratio_0.8" for row in result_vec) else result_vec[0]["config"]
    baseline=[row for row in result_vec if row["config"]==baseline_config]
    print(f"相对{baseline_config}的同PSNR平均BR变化排名：")
    for average_br_change,config in sorted((calculate_average_br_change(baseline,[row for row in result_vec if row["config"]==config]),config) for config in {row["config"] for row in result_vec}):
        print(f"  {config}:average_BR_change={average_br_change:+.6f}%")
    print(f"结果已写入{arguments.result}")

if __name__=="__main__":
    main()
