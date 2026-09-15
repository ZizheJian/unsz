import csv,math,os,re,subprocess,tempfile
import numpy as np

project_root=os.path.dirname(os.path.abspath(__file__))
data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean"
compressor=os.path.join(project_root,"build/s1_compress")
decompressor=os.path.join(project_root,"build/s1_decompress")
history_path=os.path.join(project_root,"adjust_coord_history/ocean_temperature_second_early_4.npz")
result_path=os.path.join(project_root,"adjust_coord_history/ocean_temperature_early_adjust_rd.csv")
time_step_num=31
rel_eb_vec=(0.02,0.01,0.005,0.0025,0.00125)
history=np.load(history_path)
matrix=history["matrix"][np.flatnonzero(history["iteration"]==4)[0]]
matrix_parameter=[f"{value:.17g}" for value in matrix[np.tril_indices(3)]]
config_vec=(
    {"name":"no_adjust","matrix":None},
    {"name":"second_early_4","matrix":matrix_parameter},
)

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

def calculate_bd_rate(base_result_vec,test_result_vec):
    base_psnr=np.asarray([result["psnr"] for result in base_result_vec])
    test_psnr=np.asarray([result["psnr"] for result in test_result_vec])
    base_log_br=np.log(np.asarray([result["br"] for result in base_result_vec]))
    test_log_br=np.log(np.asarray([result["br"] for result in test_result_vec]))
    min_psnr=max(base_psnr.min(),test_psnr.min())
    max_psnr=min(base_psnr.max(),test_psnr.max())
    base_polynomial=np.polyfit(base_psnr,base_log_br,3)
    test_polynomial=np.polyfit(test_psnr,test_log_br,3)
    base_integral=np.polyval(np.polyint(base_polynomial),max_psnr)-np.polyval(np.polyint(base_polynomial),min_psnr)
    test_integral=np.polyval(np.polyint(test_polynomial),max_psnr)-np.polyval(np.polyint(test_polynomial),min_psnr)
    return (math.exp((test_integral-base_integral)/(max_psnr-min_psnr))-1)*100,min_psnr,max_psnr

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_ocean_temperature_adjust_",dir=temporary_root) as test_dir:
        os.symlink(os.path.join(data_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
        result_map={}
        for config in config_vec:
            prediction_structure_path=os.path.join(test_dir,f"{config['name']}.dag")
            config_result_vec=[]
            for rel_eb in rel_eb_vec:
                variable=f"{config['name']}_{str(rel_eb).replace('.','_')}"
                for time_step in range(time_step_num):
                    os.symlink(os.path.join(data_dir,f"temperature.dat.{time_step}"),os.path.join(test_dir,f"{variable}.dat.{time_step}"))
                arguments=["-data-dir",test_dir,"-variable",variable,"-time-step",f"[0,{time_step_num-1}]","-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental","-dag-cache",prediction_structure_path]
                if config["matrix"] is not None:
                    arguments.extend(["-adjust-coord-matrix",*config["matrix"]])
                compression_output=run_command([compressor,*arguments])
                decompression_output=run_command([decompressor,*arguments])
                result={
                    "config":config["name"],
                    "rel_eb":rel_eb,
                    "br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),
                    "psnr":read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR"),
                    "preprocessing_time":read_value(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),
                    "compression_time":read_value(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),
                    "decompression_time":read_value(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间"),
                }
                decompressed_psnr=read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                if not math.isclose(result["psnr"],decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                    raise RuntimeError("压缩和解压PSNR不一致")
                result_vec.append(result)
                config_result_vec.append(result)
                print(f"{config['name']} rel_eb={rel_eb}: br={result['br']:.6f},psnr={result['psnr']:.6f},compression={result['compression_time']:.6f} s,decompression={result['decompression_time']:.6f} s",flush=True)
            result_map[config["name"]]=config_result_vec
        bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map["no_adjust"],result_map["second_early_4"])
        print(f"BD-rate(second_early_4 vs no_adjust)={bd_rate:.6f}%,overlap_psnr=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
