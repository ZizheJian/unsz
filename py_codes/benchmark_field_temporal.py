import argparse,csv,json,math,os,re,subprocess,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[1]
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
default_rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)

def parse_arguments():
    parser=argparse.ArgumentParser(description="比较一个多time step field的逐步空间、零阶外插、一阶内插及其空间残差补偿")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-variable",required=True)
    parser.add_argument("-time-step-num",type=int,required=True)
    parser.add_argument("-dim",type=int,required=True)
    parser.add_argument("-matrix-file",type=Path)
    parser.add_argument("-time-step-coordinate-file",type=Path)
    parser.add_argument("-condition-operator",choices=("<","<=",">",">=","==","!="))
    parser.add_argument("-condition-value",type=float)
    parser.add_argument("-label-file-pattern")
    parser.add_argument("-label-file-dtype",default="uint8")
    parser.add_argument("-configuration",choices=("spatial","constant_extrapolation","linear_interpolation","constant_extrapolation_spatial_residual","linear_interpolation_spatial_residual"),action="append")
    parser.add_argument("-predictor",choices=("inverse_proportion","rbf_fitting"),default="inverse_proportion")
    parser.add_argument("-ref-num",type=int,default=4)
    parser.add_argument("-inverse-distance-power",type=float,default=2)
    parser.add_argument("-rbf-kernel",default="matern32")
    parser.add_argument("-rbf-shape",type=float,default=0.03)
    parser.add_argument("-rbf-nugget",type=float,default=1e-6)
    parser.add_argument("-rel-eb",type=float,action="append")
    parser.add_argument("-result",type=Path,required=True)
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

def read_time_step_coordinates(path,time_step_num):
    if path is None:
        return np.arange(time_step_num,dtype=np.float64)
    coordinates=np.fromfile(path,dtype=np.float64)
    if coordinates.size!=time_step_num:
        raise ValueError(f"time step坐标数量应为{time_step_num}，实际为{coordinates.size}")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("time step坐标只能包含有限值")
    differences=np.diff(coordinates)
    if not(np.all(differences>0) or np.all(differences<0)):
        raise ValueError("time step坐标必须严格单调")
    return coordinates

def linear_interpolation_time_order(time_step_coordinates):
    time_step_num=time_step_coordinates.size
    order=[0,time_step_num-1]
    interval_vec=[(0,time_step_num-1)]
    for left,right in interval_vec:
        if right-left<=1:
            continue
        physical_middle=(time_step_coordinates[left]+time_step_coordinates[right])/2
        middle=int(left+1+np.argmin(np.abs(time_step_coordinates[left+1:right]-physical_middle)))
        order.append(middle)
        interval_vec.append((left,middle))
        interval_vec.append((middle,right))
    if len(order)!=time_step_num or len(set(order))!=time_step_num:
        raise RuntimeError("一阶内插顺序没有覆盖全部time step")
    return order

def write_point_set_config(path,arguments,rel_eb):
    labels=[{"method":arguments.predictor,"eb_mode":"REL","eb_value":rel_eb},{"method":"constant"}]
    if arguments.label_file_pattern is not None:
        config={"version":1,"rule_type":"label_file","label_file":{"path":arguments.label_file_pattern,"dtype":arguments.label_file_dtype},"labels":labels}
    else:
        labels[0]["condition"]={"operator":arguments.condition_operator,"value":arguments.condition_value}
        config={"version":1,"rule_type":"ordered","labels":labels}
    with path.open("w",encoding="utf-8") as file:
        json.dump(config,file,ensure_ascii=False)

def write_results(path,row_vec):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=row_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(row_vec)

def cleanup(test_dir,variable,time_step_num):
    for time_step in range(time_step_num):
        for suffix in (".unsz",".unsz.bin"):
            path=test_dir/f"{variable}.dat.{time_step}{suffix}"
            if path.exists():
                path.unlink()

def main():
    arguments=parse_arguments()
    if arguments.time_step_num<2:
        raise ValueError("纯时间预测至少需要两个time step")
    if(arguments.condition_operator is None)!=(arguments.condition_value is None):
        raise ValueError("condition-operator和condition-value必须同时指定")
    if arguments.label_file_pattern is not None and arguments.condition_operator is not None:
        raise ValueError("label-file-pattern不能和condition-operator同时指定")
    rel_eb_vec=tuple(arguments.rel_eb) if arguments.rel_eb else default_rel_eb_vec
    time_step_coordinates=read_time_step_coordinates(arguments.time_step_coordinate_file,arguments.time_step_num)
    matrix_arguments=["-adjust-coord-matrix-file",arguments.matrix_file] if arguments.matrix_file is not None else []
    time_coordinate_arguments=["-time-step-coordinate-file",arguments.time_step_coordinate_file] if arguments.time_step_coordinate_file is not None else []
    spatial_expression=f"[0,{arguments.time_step_num-1}]"
    linear_interpolation_expression="{"+",".join(str(value) for value in linear_interpolation_time_order(time_step_coordinates))+"}"
    config_vec=(
        ("spatial",spatial_expression,()),
        ("constant_extrapolation",spatial_expression,("-cross-time-predictor-experimental","constant_extrapolation")),
        ("linear_interpolation",linear_interpolation_expression,("-cross-time-predictor-experimental","linear_interpolation")),
        ("constant_extrapolation_spatial_residual",spatial_expression,("-cross-time-predictor-experimental","constant_extrapolation_spatial_residual")),
        ("linear_interpolation_spatial_residual",linear_interpolation_expression,("-cross-time-predictor-experimental","linear_interpolation_spatial_residual")),
    )
    if arguments.configuration:
        selected=set(arguments.configuration)
        config_vec=tuple(config for config in config_vec if config[0] in selected)
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix=f"unsz_{arguments.variable}_temporal_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(arguments.data_dir/"coords.dat")
        for time_step in range(arguments.time_step_num):
            (test_dir/f"{arguments.variable}.dat.{time_step}").symlink_to(arguments.data_dir/f"{arguments.variable}.dat.{time_step}")
        prediction_structure_path=temporary_dir/"spatial.dag"
        for config_name,time_expression,extra_arguments in config_vec:
            for rel_eb in rel_eb_vec:
                compression_error_arguments=["-M","REL",rel_eb,"-predictor",arguments.predictor]
                decompression_error_arguments=["-M","REL",rel_eb]
                if arguments.condition_operator is not None or arguments.label_file_pattern is not None:
                    point_set_config_path=temporary_dir/f"{config_name}_{rel_eb:g}.json"
                    write_point_set_config(point_set_config_path,arguments,rel_eb)
                    compression_error_arguments=["-point-set-config",point_set_config_path]
                    decompression_error_arguments=compression_error_arguments
                predictor_arguments=["-rbf-kernel",arguments.rbf_kernel,"-rbf-shape",arguments.rbf_shape,"-rbf-nugget",arguments.rbf_nugget,"-rbf-polynomial-degree",0] if arguments.predictor=="rbf_fitting" else []
                common_arguments=["-data-dir",test_dir,"-variable",arguments.variable,"-time-step",time_expression,"-dim",arguments.dim,"-dag-cache",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",arguments.ref_num,"-actual-ref-num",arguments.ref_num,"-inverse-distance-power",arguments.inverse_distance_power,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",arguments.ref_num if arguments.predictor=="rbf_fitting" else 3,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental",*predictor_arguments,*matrix_arguments,*time_coordinate_arguments,*extra_arguments]
                compression_output=run_command([compressor,*compression_error_arguments,*common_arguments])
                decompression_output=run_command([decompressor,*decompression_error_arguments,*common_arguments])
                psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                row={"config":config_name,"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
                result_vec.append(row)
                write_results(arguments.result,result_vec)
                print(f"{config_name} REL={rel_eb:g}:BR={row['br']:.6f},PSNR={row['psnr']:.6f},compression={row['compression_time']:.6f}s,decompression={row['decompression_time']:.6f}s",flush=True)
                cleanup(test_dir,arguments.variable,arguments.time_step_num)
    print(f"结果已写入{arguments.result}",flush=True)

if __name__=="__main__":
    main()
