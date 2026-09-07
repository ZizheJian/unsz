import csv,json,math,os,re,subprocess,sys,tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.py_unsz_core.adjust_coord_matrix_file import read_adjust_coord_matrix
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"build/s1_compress"
spatial_matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
joint_matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatiotemporal_t162_random_hyperrectangle_regions_r4.txt"
previous_spatial_csv_path=project_root/"py_codes/trash/results/katrina_current_cross_time_label_rd.csv"
result_path=project_root/"py_codes/trash/results/katrina_previous_label_joint_value_rd.csv"
figure_path=project_root/"figures/katrina_previous_label_joint_value_rd"
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
time_step_num=162
spatial_dim=3
joint_dim=4
partition_num=128
thread_num=32

def run_command(command):
    print("运行："," ".join(str(value) for value in command),flush=True)
    result=subprocess.run([str(value) for value in command],cwd=project_root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError("命令失败："+" ".join(str(value) for value in command))
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def read_sum(pattern,text):
    return sum(int(value) for value in re.findall(pattern,text,re.MULTILINE))

def calculate_bd_rate(reference_br,reference_psnr,test_br,test_psnr):
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    reference_integral=np.polyint(np.polyfit(reference_psnr,np.log(reference_br),3))
    test_integral=np.polyint(np.polyfit(test_psnr,np.log(test_br),3))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (np.exp((test_area-reference_area)/(upper-lower))-1)*100

def write_config(path,rel_eb):
    with path.open("w",encoding="utf-8") as file:
        json.dump({"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]},file,ensure_ascii=False)

def common_arguments(data_path,time_step,dim,config_path,prediction_structure_path,matrix_path):
    return ["-data-dir",data_path,"-variable","attr0","-time-step",time_step,"-dim",dim,"-point-set-config",config_path,"-adjust-coord-matrix-file",matrix_path,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",4,"-actual-ref-num",4,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",4,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental"]

def prepare_spatial_data(work_dir):
    spatial_dir=work_dir/"spatial"
    spatial_dir.mkdir()
    (spatial_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
    for time_step in range(time_step_num):
        (spatial_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
    return spatial_dir

def prepare_joint_data(work_dir):
    coord=np.fromfile(data_dir/"coords.dat",dtype=np.float32).reshape(-1,spatial_dim)
    point_num=coord.shape[0]
    nearest_distance=cKDTree(coord.astype(np.float64)).query(coord,k=2,workers=1)[0][:,1]
    time_spacing=float(np.median(nearest_distance))
    joint_dir=work_dir/"joint"
    joint_dir.mkdir()
    joint_coord=np.memmap(joint_dir/"coords.dat",mode="w+",dtype=np.float32,shape=(point_num*time_step_num,joint_dim))
    joint_data=np.memmap(joint_dir/"attr0.dat.0",mode="w+",dtype=np.float32,shape=(point_num*time_step_num,))
    for time_step in range(time_step_num):
        begin=time_step*point_num
        end=begin+point_num
        joint_coord[begin:end,:spatial_dim]=coord
        joint_coord[begin:end,spatial_dim]=time_step*time_spacing
        source=np.fromfile(data_dir/f"attr0.dat.{time_step}",dtype=np.float32)
        if source.size!=point_num:
            raise RuntimeError(f"time step {time_step}的数据点数错误")
        joint_data[begin:end]=source
    joint_coord.flush()
    joint_data.flush()
    del joint_coord,joint_data
    matrix_info=read_adjust_coord_matrix(joint_matrix_path)
    if matrix_info["training_type"]!="spatiotemporal" or matrix_info["dim"]!=joint_dim:
        raise RuntimeError("Katrina时空矩阵的类型或维数错误")
    print(f"联合数据点数={point_num*time_step_num},初始时间间距={time_spacing}",flush=True)
    return joint_dir,point_num

def main():
    if not compressor.is_file() or not spatial_matrix_path.is_file() or not joint_matrix_path.is_file():
        raise FileNotFoundError("缺少压缩程序或Katrina坐标变换矩阵")
    with previous_spatial_csv_path.open(newline="") as file:
        previous_spatial_rows=sorted(csv.DictReader(file),key=lambda row:float(row["rel_eb"]),reverse=True)
    if [float(row["rel_eb"]) for row in previous_spatial_rows]!=list(rel_eb_vec):
        raise RuntimeError("上一时刻标签＋空间数值的现有五点曲线不完整")
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_mixed_time_",dir=temporary_root) as temporary_dir_string:
        work_dir=Path(temporary_dir_string)
        spatial_dir=prepare_spatial_data(work_dir)
        joint_dir,point_num=prepare_joint_data(work_dir)
        spatial_prediction_structure_path=work_dir/"spatial_prediction_structure.bin"
        joint_prediction_structure_path=work_dir/"joint_prediction_structure.bin"
        result_vec=[]
        for rel_eb,previous_spatial_row in zip(rel_eb_vec,previous_spatial_rows):
            config_path=work_dir/f"point_sets_{rel_eb}.json"
            write_config(config_path,rel_eb)
            previous_arguments=common_arguments(spatial_dir,f"[0,{time_step_num-1}]",spatial_dim,config_path,spatial_prediction_structure_path,spatial_matrix_path)
            previous_arguments.extend(["-cross-time-label-predictor-experimental","previous","-fused-two-point-set-prediction-experimental"])
            previous_output=run_command([compressor,*previous_arguments])
            previous_label_bytes=read_sum(r"^label_prediction_error_ratio=[^\n]*label_zstd_bytes=(\d+)",previous_output)
            joint_arguments=common_arguments(joint_dir,"{0}",joint_dim,config_path,joint_prediction_structure_path,joint_matrix_path)
            joint_arguments.extend(["-cross-time-spatial-point-num-experimental",point_num,"-partition-num",partition_num,"-thread-num",thread_num])
            joint_output=run_command([compressor,*joint_arguments])
            joint_br=read_float(r"^total cr= [^,]+, total br= ([^,]+),",joint_output,"联合BR")
            joint_psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",joint_output,"联合PSNR")
            joint_label_bytes=read_sum(r"^label_prediction_error_ratio=[^\n]*label_zstd_bytes=(\d+)",joint_output)
            joint_value_bytes=read_sum(r"^value_huffman_bytes=[^\n]*value_zstd_bytes=(\d+)",joint_output)
            joint_file_bytes=read_sum(r"^value_huffman_bytes=[^\n]*file_bytes=(\d+)",joint_output)
            joint_metadata_bytes=joint_file_bytes-joint_label_bytes-joint_value_bytes
            hybrid_file_bytes=joint_metadata_bytes+previous_label_bytes+joint_value_bytes+time_step_num*8
            hybrid_br=hybrid_file_bytes*8/(point_num*time_step_num)
            result_vec.append({"rel_eb":rel_eb,"previous_label_spatial_value_br":float(previous_spatial_row["unsz_br"]),"previous_label_spatial_value_psnr":float(previous_spatial_row["unsz_psnr"]),"joint_all_br":joint_br,"joint_all_psnr":joint_psnr,"previous_label_zstd_bytes":previous_label_bytes,"joint_label_zstd_bytes":joint_label_bytes,"joint_value_zstd_bytes":joint_value_bytes,"joint_metadata_bytes":joint_metadata_bytes,"previous_label_joint_value_estimated_bytes":hybrid_file_bytes,"previous_label_joint_value_estimated_br":hybrid_br,"previous_label_joint_value_psnr":joint_psnr})
            print(f"REL={rel_eb},上一标签＋空间数值BR={float(previous_spatial_row['unsz_br']):.6f},联合全部BR={joint_br:.6f},上一标签＋联合数值估算BR={hybrid_br:.6f},PSNR={joint_psnr:.6f}",flush=True)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with result_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    spatial_br=np.asarray([row["previous_label_spatial_value_br"] for row in result_vec])
    spatial_psnr=np.asarray([row["previous_label_spatial_value_psnr"] for row in result_vec])
    hybrid_br=np.asarray([row["previous_label_joint_value_estimated_br"] for row in result_vec])
    hybrid_psnr=np.asarray([row["previous_label_joint_value_psnr"] for row in result_vec])
    bd_rate=calculate_bd_rate(spatial_br,spatial_psnr,hybrid_br,hybrid_psnr)
    figure,axis=plt.subplots(figsize=(6.4,4.6),layout="constrained")
    axis.plot(spatial_br,spatial_psnr,marker="o",linewidth=2,label="Previous-time label + spatial value")
    axis.plot(hybrid_br,hybrid_psnr,marker="s",linewidth=2,label="Previous-time label + joint value")
    axis.set_xlim(left=0)
    axis.set_xlabel("Bit rate (bit/value)")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Katrina attr0: fixed label predictor")
    axis.grid(True,linestyle=":",alpha=0.55)
    axis.legend(frameon=False)
    figure.savefig(figure_path.with_suffix(".png"),dpi=220,bbox_inches="tight",facecolor="white")
    figure.savefig(figure_path.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(figure)
    print(f"上一时刻标签＋联合数值相对上一时刻标签＋空间数值BD-rate={bd_rate:.6f}%")
    print(f"结果={result_path}")

if __name__=="__main__":
    main()
