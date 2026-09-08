import argparse,csv,math,os,re,subprocess,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[1]
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
default_rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)

def parse_arguments():
    parser=argparse.ArgumentParser(description="比较一个多time step field的IDW、RBF及其参考数据辅助顺序")
    parser.add_argument("-data-dir",type=Path,required=True)
    parser.add_argument("-variable",required=True)
    parser.add_argument("-time-step-num",type=int,required=True)
    parser.add_argument("-dim",type=int,required=True)
    parser.add_argument("-training-time-step",type=int,action="append",required=True)
    parser.add_argument("-test-time-step",type=int,action="append")
    parser.add_argument("-matrix-file",type=Path)
    parser.add_argument("-result",type=Path,required=True)
    parser.add_argument("-rbf-ref-num",type=int,default=10)
    parser.add_argument("-rbf-kernel",default="matern32")
    parser.add_argument("-rbf-shape",type=float,default=0.01)
    parser.add_argument("-rbf-nugget",type=float,default=1e-6)
    parser.add_argument("-order-candidate-num",type=int,default=48)
    parser.add_argument("-rel-eb",type=float,action="append")
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

def main():
    arguments=parse_arguments()
    training_time_step_vec=tuple(dict.fromkeys(arguments.training_time_step))
    if arguments.test_time_step:
        test_time_step_vec=tuple(dict.fromkeys(arguments.test_time_step))
    else:
        training_time_step_set=set(training_time_step_vec)
        test_time_step_vec=tuple(time_step for time_step in range(arguments.time_step_num) if time_step not in training_time_step_set)
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
    config_vec=(
        {"name":"idw_d_only","predictor":"inverse_proportion","ref_num":4,"cache_threshold":3,"power":2,"compress_extra":["-reverse-mean-distance-only-order-experimental"],"decompress_extra":[]},
        {"name":"idw_reference_data","predictor":"inverse_proportion","ref_num":4,"cache_threshold":3,"power":2,"compress_extra":["-reference-data-order-experimental",training_expression,"-order-candidate-num-experimental",arguments.order_candidate_num],"decompress_extra":[]},
        {"name":"rbf_d_only","predictor":"rbf_fitting","ref_num":arguments.rbf_ref_num,"cache_threshold":arguments.rbf_ref_num,"power":3.6,"compress_extra":["-reverse-mean-distance-only-order-experimental"],"decompress_extra":[]},
        {"name":"rbf_reference_data","predictor":"rbf_fitting","ref_num":arguments.rbf_ref_num,"cache_threshold":arguments.rbf_ref_num,"power":3.6,"compress_extra":["-reference-data-order-experimental",training_expression,"-reference-data-rbf-score-experimental","-order-candidate-num-experimental",arguments.order_candidate_num],"decompress_extra":[]},
    )
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
            prediction_structure_path=temporary_dir/f"{config['name']}.pred"
            for rel_eb in rel_eb_vec:
                common_arguments=["-data-dir",test_dir,"-variable",arguments.variable,"-time-step",test_expression,"-dim",arguments.dim,"-M","REL",rel_eb,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",config["ref_num"],"-actual-ref-num",config["ref_num"],"-inverse-distance-power",config["power"],"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",config["cache_threshold"],"-eb-schedule-experimental","round_point_ratio",0.8,"-rbf-kernel",arguments.rbf_kernel,"-rbf-shape",arguments.rbf_shape,"-rbf-nugget",arguments.rbf_nugget,*common_optional]
                compression_output=run_command([compressor,*common_arguments,"-predictor",config["predictor"],*config["compress_extra"]])
                decompression_output=run_command([decompressor,*common_arguments,*config["decompress_extra"]])
                psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
                decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                result={"config":config["name"],"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"decompression_time":read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")}
                result_vec.append(result)
                write_results(arguments.result,result_vec)
                print(f"{config['name']} REL={rel_eb:g}:BR={result['br']:.6f},PSNR={psnr:.6f},preprocessing={result['preprocessing_time']:.6f}s,compression={result['compression_time']:.6f}s,decompression={result['decompression_time']:.6f}s",flush=True)
                cleanup(test_dir,arguments.variable,test_time_step_vec)
    result_map={name:[row for row in result_vec if row["config"]==name] for name in (config["name"] for config in config_vec)}
    baseline=result_map["idw_d_only"]
    for name in ("idw_reference_data","rbf_d_only","rbf_reference_data"):
        rows=result_map[name]
        print(f"{name}相对idw_d_only：BD-rate={calculate_bd_rate(baseline,rows):+.6f}%，压缩时间倍率={geometric_mean([row['compression_time'] for row in rows])/geometric_mean([row['compression_time'] for row in baseline]):.6f}，解压时间倍率={geometric_mean([row['decompression_time'] for row in rows])/geometric_mean([row['decompression_time'] for row in baseline]):.6f}",flush=True)
    print(f"结果已写入{arguments.result}",flush=True)

if __name__=="__main__":
    main()
