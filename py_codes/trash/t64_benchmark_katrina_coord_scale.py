import csv,math,os,re,subprocess,sys,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.py_unsz_core.adjust_coord import adjust_coord
from py_codes.py_unsz_core.init import init
from py_codes.py_unsz_core.param import Param

data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"build/s1_compress"
raw_adjust_history_path=project_root/"adjust_coord_history/second_speed_least_squares.npz"
scaled_adjust_history_path=project_root/"py_codes/trash/results/adjust_coord_history/katrina_equal_range_second_early_4.npz"
result_path=project_root/"py_codes/trash/results/katrina_coord_scale_rd.csv"
time_step_num=5
all_time_step_num=162
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

def train_scaled_adjust_matrix():
    coord=np.fromfile(data_dir/"coords.dat",dtype=np.float32).reshape(-1,3)
    coord_range=np.ptp(coord.astype(np.float64),axis=0)
    equal_range_scale=coord_range[0]/coord_range
    if scaled_adjust_history_path.is_file():
        with np.load(scaled_adjust_history_path) as history:
            print(f"复用已训练的等范围坐标adjust矩阵：{scaled_adjust_history_path}",flush=True)
            return equal_range_scale,history["matrix"][-1]
    param=Param()
    param.coord_path=str(data_dir/"coords.dat")
    param.original_data_path_vec=[str(data_dir/f"attr0.dat.{time_step}") for time_step in range(all_time_step_num)]
    param.rel_eb=0.01
    param.actual_ref_num=5
    init(param)
    param.coord_array=param.coord_array.astype(np.float64)*equal_range_scale
    param.min_coord=param.coord_array.min(axis=0)
    param.max_coord=param.coord_array.max(axis=0)
    scaled_adjust_history_path.parent.mkdir(parents=True,exist_ok=True)
    print(f"等范围scale系数：{equal_range_scale}",flush=True)
    adjust_coord(param,"second_power",4,str(scaled_adjust_history_path),0.25,0.1,0.5,100,3,1,"least_squares")
    return equal_range_scale,param.coord_transform_matrix

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

def benchmark(test_dir,config_name,matrix,rel_eb):
    variable=f"attr0_{config_name}"
    parameter=matrix[np.tril_indices(3)]
    prediction_structure_path=test_dir/f"{config_name}.pred"
    arguments=[compressor,"-data-dir",test_dir,"-variable",variable,"-time-step",f"[0,{time_step_num-1}]","-M","REL",rel_eb,"-adjust-coord-matrix",*[f"{value:.17g}" for value in parameter],"-reverse-mean-distance-only-order-experimental","-prediction-structure",prediction_structure_path]
    output=run_command(arguments)
    result={"config":config_name,"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",output,"总BR"),"psnr":read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",output,"总PSNR")}
    print(f"{config_name}:rel_eb={rel_eb},br={result['br']},psnr={result['psnr']}",flush=True)
    return result

def main():
    with np.load(raw_adjust_history_path) as history:
        raw_adjust_matrix=history["matrix"][-1]
    equal_range_scale,scaled_adjust_matrix=train_scaled_adjust_matrix()
    scale_matrix=np.diag(equal_range_scale)
    combined_matrix=scale_matrix@scaled_adjust_matrix
    config_vec=(
        ("raw_adjust",raw_adjust_matrix),
        ("equal_range_scale",scale_matrix),
        ("equal_range_scale_raw_adjust",scale_matrix@raw_adjust_matrix),
        ("equal_range_scale_retrained_adjust",combined_matrix),
    )
    print(f"原始坐标adjust矩阵：\n{raw_adjust_matrix}")
    print(f"等范围scale矩阵：\n{scale_matrix}")
    print(f"等范围scale后直接使用原adjust矩阵：\n{scale_matrix@raw_adjust_matrix}")
    print(f"等范围scale后的adjust矩阵：\n{scaled_adjust_matrix}")
    print(f"最终组合矩阵scale@adjust：\n{combined_matrix}")
    result_vec=[]
    result_map={}
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_coord_scale_",dir=temporary_root) as temporary_directory:
        test_dir=Path(temporary_directory)
        os.symlink(data_dir/"coords.dat",test_dir/"coords.dat")
        for config_name,_ in config_vec:
            for time_step in range(time_step_num):
                os.symlink(data_dir/f"attr0.dat.{time_step}",test_dir/f"attr0_{config_name}.dat.{time_step}")
        for config_name,matrix in config_vec:
            result_map[config_name]=[]
            for rel_eb in rel_eb_vec:
                result=benchmark(test_dir,config_name,matrix,rel_eb)
                result_vec.append(result)
                result_map[config_name].append(result)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with result_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    for config_name,_ in config_vec[1:]:
        bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map["raw_adjust"],result_map[config_name])
        print(f"BD-rate({config_name} vs raw_adjust)={bd_rate:.6f}%,PSNR范围=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
    print(f"结果已写入：{result_path}")

if __name__=="__main__":
    main()
