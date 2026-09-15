import argparse,csv,math,os,re,subprocess,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[1]
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
default_rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
order_config_vec=(
    {"config":"reverse_mean_reference_distance","ordering_construction":"reverse_elimination","ordering_score":"mean_reference_distance","flags":["-reverse-mean-distance-only-order-experimental"]},
    {"config":"reverse_predictability","ordering_construction":"reverse_elimination","ordering_score":"predictability","flags":["-reverse-predictability-order-experimental"]},
    {"config":"reverse_mean_reference_distance_s2","ordering_construction":"reverse_elimination","ordering_score":"mean_reference_distance+s2","flags":["-reverse-distribution-order-experimental"]},
    {"config":"reverse_direction_concentration","ordering_construction":"reverse_elimination","ordering_score":"direction_concentration","flags":["-reverse-direction-only-order-experimental"]},
    {"config":"reverse_nearest_reference_distance_direction_concentration","ordering_construction":"reverse_elimination","ordering_score":"nearest_reference_distance+direction_concentration","flags":["-reverse-direction-order-experimental"]},
    {"config":"reverse_direction_concentration_s2","ordering_construction":"reverse_elimination","ordering_score":"direction_concentration+s2","flags":["-reverse-direction-distribution-order-experimental"]},
    {"config":"forward_predictability","ordering_construction":"forward_greedy","ordering_score":"predictability","flags":["-predictability-order-experimental"]},
    {"config":"forward_mean_reference_distance","ordering_construction":"forward_greedy","ordering_score":"mean_reference_distance","flags":["-forward-mean-distance-only-order-experimental"]},
    {"config":"forward_direction_concentration","ordering_construction":"forward_greedy","ordering_score":"direction_concentration","flags":["-forward-direction-only-order-experimental"]},
    {"config":"forward_mean_reference_distance_s2","ordering_construction":"forward_greedy","ordering_score":"mean_reference_distance+s2","flags":["-forward-mean-distance-s2-order-experimental"]},
    {"config":"forward_mean_reference_distance_direction_concentration","ordering_construction":"forward_greedy","ordering_score":"mean_reference_distance+direction_concentration","flags":["-forward-mean-distance-direction-order-experimental"]},
)

def parse_arguments():
    parser=argparse.ArgumentParser(description="固定空间预测器并筛选纯几何预测顺序")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-variable",required=True)
    parser.add_argument("-time-step",type=int,default=0)
    parser.add_argument("-dim",type=int,required=True)
    parser.add_argument("-matrix-file",type=Path)
    parser.add_argument("-ref-num",type=int,default=10)
    parser.add_argument("-rbf-kernel",default="matern32")
    parser.add_argument("-rbf-shape",type=float,default=0.03)
    parser.add_argument("-rbf-nugget",type=float,default=1e-6)
    parser.add_argument("-rbf-polynomial-degree",type=int,default=0)
    parser.add_argument("-rel-eb",type=float,action="append")
    parser.add_argument("-configuration",action="append",choices=tuple(config["config"] for config in order_config_vec))
    parser.add_argument("-round-formation",choices=("split_rounds_then_sort","sort_all_then_split_rounds"),default="split_rounds_then_sort")
    parser.add_argument("-result",type=Path,required=True)
    return parser.parse_args()

def run_command(command):
    completed=subprocess.run([str(value) for value in command],cwd=project_root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if completed.returncode!=0:
        print(completed.stdout,flush=True)
        raise RuntimeError("命令执行失败："+" ".join(str(value) for value in command))
    return completed.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

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

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def write_results(path,row_vec):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=row_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(row_vec)

def cleanup(data_dir,variable,time_step):
    for suffix in (".unsz",".unsz.bin"):
        path=data_dir/f"{variable}.dat.{time_step}{suffix}"
        if path.exists():
            path.unlink()

def main():
    arguments=parse_arguments()
    if arguments.ref_num<=0:
        raise ValueError("ref-num必须为正数")
    if arguments.rbf_polynomial_degree not in (0,1,2):
        raise ValueError("rbf-polynomial-degree必须是0、1或2")
    selected=set(arguments.configuration) if arguments.configuration else None
    config_vec=tuple(config for config in order_config_vec if selected is None or config["config"] in selected)
    if arguments.round_formation=="sort_all_then_split_rounds":
        unsupported=tuple(config["config"] for config in config_vec if config["config"]!="reverse_mean_reference_distance")
        if selected is not None and unsupported:
            raise ValueError("sort_all_then_split_rounds目前只支持reverse_mean_reference_distance")
        config_vec=tuple(config for config in config_vec if config["config"]=="reverse_mean_reference_distance")
    rel_eb_vec=tuple(arguments.rel_eb) if arguments.rel_eb else default_rel_eb_vec
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix=f"unsz_order_{arguments.variable}_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        data_dir=temporary_dir/"data"
        data_dir.mkdir()
        (data_dir/"coords.dat").symlink_to(arguments.data_dir/"coords.dat")
        (data_dir/f"{arguments.variable}.dat.{arguments.time_step}").symlink_to(arguments.data_dir/f"{arguments.variable}.dat.{arguments.time_step}")
        for config in config_vec:
            prediction_structure_path=temporary_dir/f"{config['config']}.dag"
            for rel_eb in rel_eb_vec:
                common_arguments=["-data-dir",data_dir,"-variable",arguments.variable,"-time-step",f"{{{arguments.time_step}}}","-dim",arguments.dim,"-dag-cache",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",arguments.ref_num,"-actual-ref-num",arguments.ref_num,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",arguments.ref_num,"-eb-schedule-experimental","round_point_ratio",0.8,"-rbf-kernel",arguments.rbf_kernel,"-rbf-shape",arguments.rbf_shape,"-rbf-nugget",arguments.rbf_nugget,"-rbf-polynomial-degree",arguments.rbf_polynomial_degree]
                if arguments.matrix_file is not None:
                    common_arguments.extend(("-adjust-coord-matrix-file",arguments.matrix_file))
                round_formation_flags=["-global-order-before-round-split-experimental"] if arguments.round_formation=="sort_all_then_split_rounds" else []
                compression_output=run_command([compressor,"-M","REL",rel_eb,"-predictor","rbf_fitting",*common_arguments,*config["flags"],*round_formation_flags])
                decompression_output=run_command([decompressor,"-M","REL",rel_eb,*common_arguments])
                psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                result={"config":config["config"],"coordinate":"adjusted" if arguments.matrix_file is not None else "original","predictor":"rbf_fitting","ref_num":arguments.ref_num,"rbf_kernel":arguments.rbf_kernel,"rbf_shape":arguments.rbf_shape,"rbf_nugget":arguments.rbf_nugget,"rbf_polynomial_degree":arguments.rbf_polynomial_degree,"round_formation":arguments.round_formation,"ordering_construction":config["ordering_construction"],"ordering_score":config["ordering_score"],"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
                result_vec.append(result)
                write_results(arguments.result,result_vec)
                print(f"{config['config']} REL={rel_eb:g}:BR={result['br']:.6f},PSNR={psnr:.6f},preprocessing={result['preprocessing_time']:.6f}s,compression={result['compression_time']:.6f}s,decompression={result['decompression_time']:.6f}s",flush=True)
                cleanup(data_dir,arguments.variable,arguments.time_step)
    baseline=[row for row in result_vec if row["config"]=="reverse_mean_reference_distance"]
    if baseline:
        baseline_compression_time=geometric_mean([row["compression_time"] for row in baseline])
        baseline_decompression_time=geometric_mean([row["decompression_time"] for row in baseline])
        print("相对reverse_mean_reference_distance的同PSNR平均BR变化排名：")
        ranked=[]
        for config in config_vec:
            rows=[row for row in result_vec if row["config"]==config["config"]]
            ranked.append((calculate_average_br_change(baseline,rows),config["config"],geometric_mean([row["compression_time"] for row in rows])/baseline_compression_time,geometric_mean([row["decompression_time"] for row in rows])/baseline_decompression_time))
        for average_br_change,name,compression_ratio,decompression_ratio in sorted(ranked):
            print(f"  {name}:average_BR_change={average_br_change:+.6f}%,compression_time={compression_ratio:.6f}x,decompression_time={decompression_ratio:.6f}x")
    print(f"结果已写入{arguments.result}")

if __name__=="__main__":
    main()
