import csv,math,os,re,subprocess,tempfile
import numpy as np
import t44_benchmark_ocean_idw_power as common

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean"
compressor=os.path.join(project_root,"build/s1_compress")
decompressor=os.path.join(project_root,"build/s1_decompress")
history_path=os.path.join(project_root,"adjust_coord_history/ocean_velocity_zonal_t0_second_early_4.npz")
result_path=os.path.join(project_root,"adjust_coord_history/ocean_velocity_zonal_t0_adjust_rd.csv")
rel_eb_vec=(0.01,0.003,0.001,0.0003,0.0001)

def run_command(command):
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(command)}")
    return result.stdout

def read_value(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if not match:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def benchmark(config_name,matrix_parameter,rel_eb,test_dir,prediction_structure_path):
    variable=f"velocityZonal_{config_name}_{str(rel_eb).replace('.','_')}"
    os.symlink(os.path.join(data_dir,"velocityZonal.dat.0"),os.path.join(test_dir,f"{variable}.dat.0"))
    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step","{0}","-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental","-prediction-structure",prediction_structure_path,"-predictor","inverse_proportion","-max-ref-num","10","-actual-ref-num","10","-inverse-distance-power","3.5","-d-only-cache-refill-threshold-experimental","8","-independent-block-num-experimental","4","-block-adaptive-predictor-experimental","-block-adaptive-sample-num-experimental","256","-block-fitting-candidates-experimental","mixed"]
    if matrix_parameter is not None:
        arguments.extend(["-adjust-coord-matrix",*matrix_parameter])
    compression_output=run_command([compressor,*arguments])
    decompression_arguments=[]
    skip_next=False
    for argument in arguments:
        if skip_next:
            skip_next=False
            continue
        if argument=="-predictor":
            skip_next=True
            continue
        decompression_arguments.append(argument)
    decompression_output=run_command([decompressor,*decompression_arguments])
    psnr=read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
    decompressed_psnr=read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
    return {"config":config_name,"rel_eb":rel_eb,"br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"压缩BR"),"psnr":psnr}

def main():
    history=np.load(history_path)
    config_vec=[("no_adjust",None)]
    for iteration,matrix in zip(history["iteration"],history["matrix"]):
        config_vec.append((f"second_early_{iteration}",[f"{value:.17g}" for value in matrix[np.tril_indices(3)]]))
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_velocity_zonal_adjust_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        for config_name,matrix_parameter in config_vec:
            prediction_structure_path=os.path.join(test_dir,f"{config_name}.pred")
            for rel_eb in rel_eb_vec:
                result=benchmark(config_name,matrix_parameter,rel_eb,test_dir,prediction_structure_path)
                result_vec.append(result)
                print(f"{config_name} rel_eb={rel_eb}: br={result['br']:.6f},psnr={result['psnr']:.6f}",flush=True)
    base_vec=[result for result in result_vec if result["config"]=="no_adjust"]
    for config_name,_ in config_vec[1:]:
        test_vec=[result for result in result_vec if result["config"]==config_name]
        bd_rate,min_psnr,max_psnr=common.calculate_bd_rate(base_vec,test_vec)
        print(f"{config_name}: BD-rate vs no_adjust={bd_rate:.6f}%,overlap=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
