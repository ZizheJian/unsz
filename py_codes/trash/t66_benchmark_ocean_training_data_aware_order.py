import csv,math,os,re,subprocess,sys,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root/"py_codes"))
from py_unsz_core.adjust_coord_matrix_file import read_adjust_coord_matrix
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean")
compressor=project_root/"build/s1_compress"
decompressor=project_root/"build/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Ocean_temperature_spatial.txt"
result_path=project_root/"py_codes/trash/results/ocean_temperature_training_data_aware_order_benchmark.csv"
variable_name="temperature"
time_step_num=5
training_time_step_vec=(8,16,24)
rel_eb_vec=(0.02,0.01,0.005,0.0025,0.00125)

def run_command(command):
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(str(value) for value in command)}")
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def calculate_bd_rate(base_result_vec,test_result_vec):
    base_psnr=np.asarray([result["psnr"] for result in base_result_vec])
    test_psnr=np.asarray([result["psnr"] for result in test_result_vec])
    base_log_br=np.log(np.asarray([result["br"] for result in base_result_vec]))
    test_log_br=np.log(np.asarray([result["br"] for result in test_result_vec]))
    min_psnr=max(base_psnr.min(),test_psnr.min())
    max_psnr=min(base_psnr.max(),test_psnr.max())
    if min_psnr>=max_psnr:
        raise RuntimeError("两条RD曲线没有重叠的PSNR范围")
    base_polynomial=np.polyfit(base_psnr,base_log_br,3)
    test_polynomial=np.polyfit(test_psnr,test_log_br,3)
    base_integral=np.polyval(np.polyint(base_polynomial),max_psnr)-np.polyval(np.polyint(base_polynomial),min_psnr)
    test_integral=np.polyval(np.polyint(test_polynomial),max_psnr)-np.polyval(np.polyint(test_polynomial),min_psnr)
    return (math.exp((test_integral-base_integral)/(max_psnr-min_psnr))-1)*100,min_psnr,max_psnr

def benchmark(test_dir,config,rel_eb,matrix_parameter):
    test_variable=f"{variable_name}_{config['name']}"
    prediction_structure_path=test_dir/f"{config['name']}.pred"
    common_arguments=["-data-dir",test_dir,"-variable",test_variable,"-time-step",f"[0,{time_step_num-1}]","-M","REL",rel_eb,"-adjust-coord-matrix",*matrix_parameter,"-prediction-structure",prediction_structure_path]
    first_generation=not prediction_structure_path.is_file()
    compression_output=run_command([compressor,*common_arguments,*config["order_arguments"]])
    result={"config":config["name"],"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR"),"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),"first_generation":int(first_generation)}
    decompression_output=run_command([decompressor,*common_arguments])
    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
    if not math.isclose(result["psnr"],decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩与解压PSNR不一致：{result['psnr']}!={decompressed_psnr}")
    print(f"{config['name']}:rel_eb={rel_eb},br={result['br']},psnr={result['psnr']},preprocessing={result['preprocessing_time']} s,compression={result['compression_time']} s",flush=True)
    return result

def main():
    matrix=read_adjust_coord_matrix(matrix_path)["matrix"]
    matrix_parameter=[f"{value:.17g}" for value in matrix[np.tril_indices(matrix.shape[0])]]
    training_expression="{"+",".join(str(value) for value in training_time_step_vec)+"}"
    config_vec=(
        {"name":"reverse_d_only","order_arguments":["-reverse-mean-distance-only-order-experimental"]},
        {"name":"training_data_aware","order_arguments":["-reference-data-order-experimental",training_expression]},
    )
    result_vec=[]
    result_map={}
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_ocean_training_order_",dir=temporary_root) as temporary_directory:
        test_dir=Path(temporary_directory)
        os.symlink(data_dir/"coords.dat",test_dir/"coords.dat")
        required_time_step_vec=sorted(set(range(time_step_num))|set(training_time_step_vec))
        for config in config_vec:
            test_variable=f"{variable_name}_{config['name']}"
            for time_step in required_time_step_vec:
                os.symlink(data_dir/f"{variable_name}.dat.{time_step}",test_dir/f"{test_variable}.dat.{time_step}")
        for config in config_vec:
            result_map[config["name"]]=[]
            for rel_eb in rel_eb_vec:
                result=benchmark(test_dir,config,rel_eb,matrix_parameter)
                result_vec.append(result)
                result_map[config["name"]].append(result)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with result_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map["reverse_d_only"],result_map["training_data_aware"])
    print(f"BD-rate(training_data_aware vs reverse_d_only)={bd_rate:.6f}%,PSNR范围=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
    print(f"结果已写入：{result_path}")

if __name__=="__main__":
    main()
