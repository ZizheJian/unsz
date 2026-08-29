import argparse,os,re,subprocess,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from py_codes.py_unsz_core.param import Param
from py_codes.py_unsz_core.init import init
from py_codes.py_unsz_core.adjust_coord import adjust_coord

project_root=Path(__file__).resolve().parent
compress_path=project_root/"build/s1_compress"
decompress_path=project_root/"build/s1_decompress"
default_data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean"
default_spatial_matrix=[0.60533851,0.08032648,0.26534194,0.27221464,0.13329533,1.56998030]

def run_command(command):
    print("运行："," ".join(str(value) for value in command),flush=True)
    result=subprocess.run([str(value) for value in command],cwd=project_root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True)
    print(result.stdout,end="")
    return result.stdout

def parse_result(output,prefix):
    match=re.search(r"total br=\s*([0-9.eE+-]+), total psnr=\s*([0-9.eE+-]+)",output)
    if match is None:
        raise RuntimeError(f"无法从{prefix}输出中读取总BR和PSNR")
    return float(match.group(1)),float(match.group(2))

def parse_decompressed_psnr(output,prefix):
    match=re.search(r"total decompressed psnr=\s*([0-9.eE+-]+)",output)
    if match is None:
        raise RuntimeError(f"无法从{prefix}输出中读取解压PSNR")
    return float(match.group(1))

def make_command(executable,data_dir,variable,time_step,dim,rel_eb,extra_flags):
    return [executable,"-data-dir",data_dir,"-variable",variable,"-time-step",time_step,"-dim",dim,"-M","REL",rel_eb,"-inverse-distance-power",3.5,"-max-ref-num",10,"-actual-ref-num",10,"-reverse-mean-distance-only-order-experimental","-d-only-cache-refill-threshold-experimental",8,"-eb-schedule-experimental","none",1,*extra_flags]

def run_codec(data_dir,variable,time_step,dim,rel_eb,extra_flags,name):
    compression_output=run_command(make_command(compress_path,data_dir,variable,time_step,dim,rel_eb,extra_flags))
    br,psnr=parse_result(compression_output,name)
    decompression_output=run_command(make_command(decompress_path,data_dir,variable,time_step,dim,rel_eb,extra_flags))
    decompressed_psnr=parse_decompressed_psnr(decompression_output,name)
    if abs(psnr-decompressed_psnr)>1e-5:
        raise RuntimeError(f"{name}压缩端PSNR {psnr}与解压端PSNR {decompressed_psnr}不一致")
    return br,psnr

def read_input(data_dir,variable,time_step_vec):
    coord=np.fromfile(Path(data_dir)/"coords.dat",dtype=np.float32)
    if coord.size%3!=0:
        raise RuntimeError("Ocean坐标数量不能被3整除")
    coord=coord.reshape(-1,3)
    data_vec=[]
    for time_step in time_step_vec:
        path=Path(data_dir)/f"{variable}.dat.{time_step}"
        data=np.fromfile(path,dtype=np.float32)
        if data.size!=coord.shape[0]:
            raise RuntimeError(f"{path}包含{data.size}个数据，坐标有{coord.shape[0]}个点")
        data_vec.append(data)
    return coord,data_vec

def prepare_spatial_files(work_dir,coord,data_vec,variable,time_step_vec):
    spatial_dir=work_dir/"spatial"
    spatial_dir.mkdir(parents=True,exist_ok=True)
    coord.tofile(spatial_dir/"coords.dat")
    for time_step,data in zip(time_step_vec,data_vec):
        data.tofile(spatial_dir/f"{variable}.dat.{time_step}")
    return spatial_dir

def prepare_joint_files(work_dir,coord,data_vec,variable,time_spacing):
    joint_raw_dir=work_dir/"joint_raw"
    joint_raw_dir.mkdir(parents=True,exist_ok=True)
    point_num=coord.shape[0]
    combined_coord=np.empty((point_num*len(data_vec),4),dtype=np.float32)
    for time_id in range(len(data_vec)):
        begin=time_id*point_num
        end=begin+point_num
        combined_coord[begin:end,:3]=coord
        combined_coord[begin:end,3]=time_id*time_spacing
    combined_data=np.concatenate(data_vec)
    combined_coord.tofile(joint_raw_dir/"coords.dat")
    combined_data.tofile(joint_raw_dir/f"{variable}.dat.0")
    return joint_raw_dir,combined_coord,combined_data

def train_joint_matrix(joint_raw_dir,variable,rel_eb,spatial_point_num,args):
    param=Param()
    param.coord_path=str(joint_raw_dir/"coords.dat")
    param.original_data_path_vec=[str(joint_raw_dir/f"{variable}.dat.0")]
    param.dim=4
    param.rel_eb=rel_eb
    param.actual_ref_num=10
    param.cross_time_compress=True
    param.spatial_point_num=spatial_point_num
    init(param)
    adjust_coord(param,"second_power",args.iteration_num,args.matrix_history_path,args.first_sample_ratio,args.second_sample_ratio,args.new_matrix_weight,args.max_nfev,1,1,"least_squares")
    return param.coord_transform_matrix

def write_transformed_joint(work_dir,combined_coord,combined_data,variable,matrix):
    joint_dir=work_dir/"joint_transformed"
    joint_dir.mkdir(parents=True,exist_ok=True)
    transformed_coord=(combined_coord.astype(np.float64)@matrix).astype(np.float32)
    transformed_coord.tofile(joint_dir/"coords.dat")
    combined_data.tofile(joint_dir/f"{variable}.dat.0")
    return joint_dir,transformed_coord

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-data-dir",default=default_data_dir)
    parser.add_argument("-variable",default="velocityZonal")
    parser.add_argument("-time-step-num",type=int,default=2)
    parser.add_argument("-rel-eb",type=float,default=1e-3)
    parser.add_argument("-initial-time-spacing-ratio",type=float,default=1)
    parser.add_argument("-iteration-num",type=int,default=4)
    parser.add_argument("-first-sample-ratio",type=float,default=0.25)
    parser.add_argument("-second-sample-ratio",type=float,default=0.1)
    parser.add_argument("-new-matrix-weight",type=float,default=0.5)
    parser.add_argument("-max-nfev",type=int,default=100)
    parser.add_argument("-work-dir",default="/tmp/unsz_cross_time_joint")
    parser.add_argument("-matrix-history-path",default=str(project_root/"adjust_coord_history/ocean_velocity_zonal_joint_t2_second_early_4.npz"))
    parser.add_argument("-spatial-matrix",type=float,nargs=6,default=default_spatial_matrix)
    args=parser.parse_args()
    if args.time_step_num<2:
        raise ValueError("time-step-num至少为2")
    if args.rel_eb<=0 or args.initial_time_spacing_ratio<=0:
        raise ValueError("rel-eb和initial-time-spacing-ratio必须为正数")
    work_dir=Path(args.work_dir)
    work_dir.mkdir(parents=True,exist_ok=True)
    time_step_vec=list(range(args.time_step_num))

    total_start_time=time.perf_counter()
    coord,data_vec=read_input(args.data_dir,args.variable,time_step_vec)
    tree_start_time=time.perf_counter()
    tree=cKDTree(coord.astype(np.float64))
    nearest_distance,nearest_id=tree.query(coord,k=2,workers=1)
    median_spatial_distance=float(np.median(nearest_distance[:,1]))
    time_spacing=median_spatial_distance*args.initial_time_spacing_ratio
    print(f"空间最近邻距离中位数={median_spatial_distance},初始时间间距={time_spacing},计算耗时={time.perf_counter()-tree_start_time:.6f} s")

    spatial_dir=prepare_spatial_files(work_dir,coord,data_vec,args.variable,time_step_vec)
    time_expression=f"[0,{args.time_step_num-1}]"
    spatial_flags=["-adjust-coord-matrix",*args.spatial_matrix]
    spatial_br,spatial_psnr=run_codec(spatial_dir,args.variable,time_expression,3,args.rel_eb,spatial_flags,"逐时间步空间预测")
    temporal_br,temporal_psnr=run_codec(spatial_dir,args.variable,time_expression,3,args.rel_eb,[*spatial_flags,"-cross-time-predictor-experimental","previous"],"上一时间步同点预测")

    joint_raw_dir,combined_coord,combined_data=prepare_joint_files(work_dir,coord,data_vec,args.variable,time_spacing)
    training_start_time=time.perf_counter()
    matrix=train_joint_matrix(joint_raw_dir,args.variable,args.rel_eb,coord.shape[0],args)
    training_time=time.perf_counter()-training_start_time
    joint_dir,transformed_coord=write_transformed_joint(work_dir,combined_coord,combined_data,args.variable,matrix)
    spatial_diff=np.column_stack((coord-coord[nearest_id[:,1]],np.zeros(coord.shape[0])))@matrix
    transformed_spatial_distance=float(np.median(np.linalg.norm(spatial_diff,axis=1)))
    transformed_time_distance=float(time_spacing*np.linalg.norm(matrix[3]))
    print(f"四维变换矩阵：\n{matrix}")
    print(f"变换后时间相邻距离={transformed_time_distance},空间最近邻距离中位数={transformed_spatial_distance},二者比例={transformed_time_distance/transformed_spatial_distance}")
    print(f"四维训练耗时={training_time:.6f} s,变换后坐标范围={list(zip(transformed_coord.min(axis=0),transformed_coord.max(axis=0)))}")
    joint_br,joint_psnr=run_codec(joint_dir,args.variable,"{0}",4,args.rel_eb,["-cross-time-spatial-point-num-experimental",coord.shape[0]],"四维时空联合预测")

    print("\n跨time-step初步结果：")
    print(f"  逐时间步空间预测：BR={spatial_br:.6f},PSNR={spatial_psnr:.6f}")
    print(f"  上一时间步同点预测：BR={temporal_br:.6f},PSNR={temporal_psnr:.6f},相对空间BR变化={(temporal_br/spatial_br-1)*100:.3f}%")
    print(f"  四维时空联合预测：BR={joint_br:.6f},PSNR={joint_psnr:.6f},相对空间BR变化={(joint_br/spatial_br-1)*100:.3f}%")
    print(f"  总耗时={time.perf_counter()-total_start_time:.6f} s")

if __name__=="__main__":
    main()
