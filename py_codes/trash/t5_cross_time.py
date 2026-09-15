import argparse,os,re,subprocess,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from py_codes.py_unsz_core.param import Param
from py_codes.py_unsz_core.init import init
from py_codes.py_unsz_core.adjust_coord import adjust_coord
from py_codes.py_unsz_core.adjust_coord_matrix_file import write_adjust_coord_matrix

project_root=Path(__file__).resolve().parents[2]
compress_path=project_root/"build/s1_compress"
decompress_path=project_root/"build/s1_decompress"
default_data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean"
default_spatial_matrix=[0.60533851,0.08032648,0.26534194,0.27221464,0.13329533,1.56998030]

def run_command(command):
    print("运行："," ".join(str(value) for value in command),flush=True)
    result=subprocess.run([str(value) for value in command],cwd=project_root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    print(result.stdout,end="")
    if result.returncode!=0:
        raise RuntimeError("命令执行失败："+" ".join(str(value) for value in command))
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

def parse_single_time(output,pattern,prefix):
    match=re.search(pattern,output)
    if match is None:
        raise RuntimeError(f"无法从{prefix}输出中读取耗时")
    return float(match.group(1))

def make_command(executable,data_dir,variable,time_step,dim,error_arguments,extra_flags,thread_num):
    compress_only_flags=["-normalized-weight-experimental"] if Path(executable)==compress_path else []
    return [executable,"-data-dir",data_dir,"-variable",variable,"-time-step",time_step,"-dim",dim,*error_arguments,"-inverse-distance-power",3.6,"-max-ref-num",4,"-actual-ref-num",4,"-predict-round-num",8,"-reverse-mean-distance-only-order-experimental","-d-only-cache-refill-threshold-experimental",4,"-eb-schedule-experimental","round_point_ratio",0.8,*compress_only_flags,"-thread-num",thread_num,*extra_flags]

def run_codec(data_dir,variable,time_step,dim,error_arguments,extra_flags,name,thread_num,compress_only=False):
    compression_output=run_command(make_command(compress_path,data_dir,variable,time_step,dim,error_arguments,extra_flags,thread_num))
    br,psnr=parse_result(compression_output,name)
    preprocessing_time=parse_single_time(compression_output,r"preprocessing time excluding adjust coord=\s*([0-9.eE+-]+)",name)
    compression_time=parse_single_time(compression_output,r"total compression time=\s*([0-9.eE+-]+)",name)
    if compress_only:
        return br,psnr,preprocessing_time,compression_time,float("nan")
    decompression_output=run_command(make_command(decompress_path,data_dir,variable,time_step,dim,error_arguments,extra_flags,thread_num))
    decompressed_psnr=parse_decompressed_psnr(decompression_output,name)
    decompression_time=parse_single_time(decompression_output,r"total decompression time=\s*([0-9.eE+-]+)",name)
    if abs(psnr-decompressed_psnr)>1e-5:
        raise RuntimeError(f"{name}压缩端PSNR {psnr}与解压端PSNR {decompressed_psnr}不一致")
    return br,psnr,preprocessing_time,compression_time,decompression_time

def read_input(data_dir,variable,time_step_vec,spatial_dim):
    coord=np.fromfile(Path(data_dir)/"coords.dat",dtype=np.float32)
    if coord.size%spatial_dim!=0:
        raise RuntimeError(f"坐标数量不能被空间维数{spatial_dim}整除")
    coord=coord.reshape(-1,spatial_dim)
    data_vec=[]
    for time_step in time_step_vec:
        path=Path(data_dir)/f"{variable}.dat.{time_step}"
        data=np.fromfile(path,dtype=np.float32)
        if data.size!=coord.shape[0]:
            raise RuntimeError(f"{path}包含{data.size}个数据，坐标有{coord.shape[0]}个点")
        data_vec.append(data)
    return coord,data_vec

def discover_time_steps(data_dir,variable):
    pattern=re.compile(rf"^{re.escape(variable)}\.dat\.(\d+)$")
    time_step_vec=sorted(int(match.group(1)) for path in Path(data_dir).iterdir() if (match:=pattern.match(path.name)))
    if len(time_step_vec)<2:
        raise RuntimeError(f"{data_dir}中只找到了{len(time_step_vec)}个{variable}时间步")
    if time_step_vec!=list(range(time_step_vec[-1]+1)):
        raise RuntimeError(f"{variable}的时间步必须从0开始连续，实际为{time_step_vec}")
    return time_step_vec

def hierarchical_time_order(time_step_vec):
    sorted_time_step_vec=sorted(time_step_vec)
    if len(sorted_time_step_vec)<=1:
        return sorted_time_step_vec
    order=[sorted_time_step_vec[0],sorted_time_step_vec[-1]]
    interval_vec=[(0,len(sorted_time_step_vec)-1)]
    for left_id,right_id in interval_vec:
        if right_id-left_id<=1:
            continue
        middle_id=(left_id+right_id)//2
        order.append(sorted_time_step_vec[middle_id])
        interval_vec.append((left_id,middle_id))
        interval_vec.append((middle_id,right_id))
    if len(order)!=len(sorted_time_step_vec) or len(set(order))!=len(order):
        raise RuntimeError("层次时间顺序没有恰好覆盖全部time step")
    return order

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
    spatial_dim=coord.shape[1]
    combined_coord=np.empty((point_num*len(data_vec),spatial_dim+1),dtype=np.float32)
    for time_id in range(len(data_vec)):
        begin=time_id*point_num
        end=begin+point_num
        combined_coord[begin:end,:spatial_dim]=coord
        combined_coord[begin:end,spatial_dim]=time_id*time_spacing
    combined_data=np.concatenate(data_vec)
    combined_coord.tofile(joint_raw_dir/"coords.dat")
    combined_data.tofile(joint_raw_dir/f"{variable}.dat.0")
    return joint_raw_dir,combined_coord,combined_data

def train_joint_matrix(joint_raw_dir,variable,rel_eb,spatial_point_num,joint_dim,args):
    param=Param()
    param.coord_path=str(joint_raw_dir/"coords.dat")
    param.original_data_path_vec=[str(joint_raw_dir/f"{variable}.dat.0")]
    param.dim=joint_dim
    param.rel_eb=rel_eb
    param.actual_ref_num=4
    param.cross_time_compress=True
    param.spatial_point_num=spatial_point_num
    init(param)
    adjust_coord(param,"second_power",args.iteration_num,args.matrix_history_path,args.first_sample_ratio,args.second_sample_ratio,args.new_matrix_weight,args.max_nfev,1,1,"least_squares",args.cross_time_first_sample_mode,args.cross_time_first_sample_num,args.cross_time_first_sample_region_num,second_sample_num=args.second_sample_num)
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
    parser.add_argument("-spatial-dim",type=int,default=3)
    parser.add_argument("-time-step-num",type=int,default=0)
    parser.add_argument("-rel-eb",type=float,default=1e-3)
    parser.add_argument("-point-set-config",default="")
    parser.add_argument("-initial-time-spacing-ratio",type=float,default=1)
    parser.add_argument("-iteration-num",type=int,default=4)
    parser.add_argument("-first-sample-ratio",type=float,default=0.25)
    parser.add_argument("-second-sample-ratio",type=float,default=0.1)
    parser.add_argument("-second-sample-num",type=int,default=0,help="每轮第二次采样的固定目标总数；0表示使用second-sample-ratio")
    parser.add_argument("-cross-time-first-sample-mode",choices=["grouped_ratio","local_fixed","local_fixed_regions","random_hyperrectangle_regions"],default="random_hyperrectangle_regions")
    parser.add_argument("-cross-time-first-sample-num",type=int,default=117580)
    parser.add_argument("-cross-time-first-sample-region-num",type=int,choices=[1,2,4,8],default=1)
    parser.add_argument("-new-matrix-weight",type=float,default=0.5)
    parser.add_argument("-max-nfev",type=int,default=100)
    parser.add_argument("-work-dir",default="/tmp/unsz_cross_time_joint")
    parser.add_argument("-matrix-history-path",default="")
    parser.add_argument("-joint-matrix-output-path",default="")
    parser.add_argument("-joint-matrix-history-input",default="")
    parser.add_argument("-joint-matrix-iteration",type=int,default=0)
    parser.add_argument("-joint-only",action="store_true")
    parser.add_argument("-temporal-only",action="store_true")
    parser.add_argument("-train-only",action="store_true")
    parser.add_argument("-compress-only",action="store_true")
    parser.add_argument("-joint-block-num",type=int,choices=[1,2,4,8,16,32,64,128,256],default=1)
    parser.add_argument("-thread-num",type=int,default=1)
    spatial_matrix_group=parser.add_mutually_exclusive_group()
    spatial_matrix_group.add_argument("-spatial-matrix",type=float,nargs="+",default=None)
    spatial_matrix_group.add_argument("-spatial-matrix-file",default="")
    args=parser.parse_args()
    if args.joint_only and args.temporal_only:
        raise ValueError("joint-only和temporal-only不能同时启用")
    if args.time_step_num==1 or args.time_step_num<0:
        raise ValueError("time-step-num必须为0或至少为2，0表示自动使用全部可用时间步")
    if args.spatial_dim<=0:
        raise ValueError("spatial-dim必须为正数")
    if args.thread_num<=0:
        raise ValueError("thread-num必须为正数")
    if args.rel_eb<=0 or args.initial_time_spacing_ratio<=0:
        raise ValueError("rel-eb和initial-time-spacing-ratio必须为正数")
    if args.cross_time_first_sample_num<=0 or args.cross_time_first_sample_region_num<=0 or args.second_sample_num<0:
        raise ValueError("cross-time-first-sample-num和cross-time-first-sample-region-num必须为正数")
    work_dir=Path(args.work_dir)
    work_dir.mkdir(parents=True,exist_ok=True)
    if args.point_set_config:
        point_set_config_path=Path(args.point_set_config).resolve()
        if not point_set_config_path.is_file():
            raise FileNotFoundError(f"找不到点集配置文件：{point_set_config_path}")
        error_arguments=["-point-set-config",point_set_config_path]
    else:
        error_arguments=["-M","REL",args.rel_eb]
    time_step_vec=discover_time_steps(args.data_dir,args.variable) if args.time_step_num==0 else list(range(args.time_step_num))
    args.time_step_num=len(time_step_vec)
    if not args.matrix_history_path:
        args.matrix_history_path=str(project_root/f"adjust_coord_history/ocean_{args.variable}_joint_t{args.time_step_num}_{args.cross_time_first_sample_mode}_r{args.cross_time_first_sample_region_num}.npz")
    print(f"使用时间步：{time_step_vec}")

    total_start_time=time.perf_counter()
    coord,data_vec=read_input(args.data_dir,args.variable,time_step_vec,args.spatial_dim)
    total_data_mb=sum(data.nbytes for data in data_vec)/1e6
    tree_start_time=time.perf_counter()
    tree=cKDTree(coord.astype(np.float64))
    nearest_distance,nearest_id=tree.query(coord,k=2,workers=1)
    median_spatial_distance=float(np.median(nearest_distance[:,1]))
    time_spacing=median_spatial_distance*args.initial_time_spacing_ratio
    print(f"空间最近邻距离中位数={median_spatial_distance},初始时间间距={time_spacing},计算耗时={time.perf_counter()-tree_start_time:.6f} s")

    spatial_dir=prepare_spatial_files(work_dir,coord,data_vec,args.variable,time_step_vec)
    time_expression=f"[0,{args.time_step_num-1}]"
    interpolation_time_step_vec=hierarchical_time_order(time_step_vec)
    interpolation_time_expression="{"+",".join(str(time_step) for time_step in interpolation_time_step_vec)+"}"
    if args.spatial_matrix_file:
        spatial_matrix_path=Path(args.spatial_matrix_file).resolve()
        if not spatial_matrix_path.is_file():
            raise FileNotFoundError(f"找不到空间坐标变换矩阵：{spatial_matrix_path}")
        spatial_flags=["-adjust-coord-matrix-file",spatial_matrix_path]
    else:
        spatial_matrix=args.spatial_matrix if args.spatial_matrix is not None else default_spatial_matrix
        expected_parameter_num=args.spatial_dim*(args.spatial_dim+1)//2
        if len(spatial_matrix)!=expected_parameter_num:
            raise ValueError(f"spatial-matrix需要{expected_parameter_num}个下三角元素，实际为{len(spatial_matrix)}")
        spatial_flags=["-adjust-coord-matrix",*spatial_matrix]
    if not args.joint_only:
        spatial_br,spatial_psnr,spatial_preprocessing_time,spatial_compression_time,spatial_decompression_time=run_codec(spatial_dir,args.variable,time_expression,args.spatial_dim,error_arguments,spatial_flags,"逐时间步空间预测",1,args.compress_only)
        temporal_label_br,temporal_label_psnr,temporal_label_preprocessing_time,temporal_label_compression_time,temporal_label_decompression_time=run_codec(spatial_dir,args.variable,time_expression,args.spatial_dim,error_arguments,[*spatial_flags,"-cross-time-label-predictor-experimental","previous"],"上一时间步同点标签预测",1,args.compress_only)
        temporal_br,temporal_psnr,temporal_preprocessing_time,temporal_compression_time,temporal_decompression_time=run_codec(spatial_dir,args.variable,time_expression,args.spatial_dim,error_arguments,[*spatial_flags,"-cross-time-predictor-experimental","previous"],"上一时间步同点预测",1,args.compress_only)
        interpolation_br,interpolation_psnr,interpolation_preprocessing_time,interpolation_compression_time,interpolation_decompression_time=run_codec(spatial_dir,args.variable,interpolation_time_expression,args.spatial_dim,error_arguments,[*spatial_flags,"-cross-time-predictor-experimental","interpolation"],"层次时间插值预测",1,args.compress_only)
        if args.temporal_only:
            print("\n跨time-step初步结果：")
            print(f"  逐时间步空间预测：BR={spatial_br:.6f},PSNR={spatial_psnr:.6f},预处理={spatial_preprocessing_time:.6f} s,压缩={spatial_compression_time:.6f} s({total_data_mb/spatial_compression_time:.3f} MB/s),解压={spatial_decompression_time:.6f} s({total_data_mb/spatial_decompression_time:.3f} MB/s)")
            print(f"  上一时间步同点标签预测：BR={temporal_label_br:.6f},PSNR={temporal_label_psnr:.6f},预处理={temporal_label_preprocessing_time:.6f} s,压缩={temporal_label_compression_time:.6f} s({total_data_mb/temporal_label_compression_time:.3f} MB/s),解压={temporal_label_decompression_time:.6f} s({total_data_mb/temporal_label_decompression_time:.3f} MB/s),相对空间BR变化={(temporal_label_br/spatial_br-1)*100:.3f}%")
            print(f"  上一时间步同点预测：BR={temporal_br:.6f},PSNR={temporal_psnr:.6f},预处理={temporal_preprocessing_time:.6f} s,压缩={temporal_compression_time:.6f} s({total_data_mb/temporal_compression_time:.3f} MB/s),解压={temporal_decompression_time:.6f} s({total_data_mb/temporal_decompression_time:.3f} MB/s),相对空间BR变化={(temporal_br/spatial_br-1)*100:.3f}%")
            print(f"  层次时间插值预测：BR={interpolation_br:.6f},PSNR={interpolation_psnr:.6f},预处理={interpolation_preprocessing_time:.6f} s,压缩={interpolation_compression_time:.6f} s({total_data_mb/interpolation_compression_time:.3f} MB/s),解压={interpolation_decompression_time:.6f} s({total_data_mb/interpolation_decompression_time:.3f} MB/s),相对空间BR变化={(interpolation_br/spatial_br-1)*100:.3f}%")
            print(f"  总耗时={time.perf_counter()-total_start_time:.6f} s")
            return

    joint_raw_dir,combined_coord,combined_data=prepare_joint_files(work_dir,coord,data_vec,args.variable,time_spacing)
    if args.joint_matrix_history_input:
        history=np.load(args.joint_matrix_history_input)
        matrix_id=-1 if args.joint_matrix_iteration==0 else np.flatnonzero(history["iteration"]==args.joint_matrix_iteration)[0]
        matrix=history["matrix"][matrix_id]
        training_time=0
        print(f"从{args.joint_matrix_history_input}读取第{history['iteration'][matrix_id]}轮矩阵")
    else:
        training_start_time=time.perf_counter()
        matrix=train_joint_matrix(joint_raw_dir,args.variable,args.rel_eb,coord.shape[0],combined_coord.shape[1],args)
        training_time=time.perf_counter()-training_start_time
    dataset=Path(args.data_dir).resolve().name
    matrix_output_path=Path(args.joint_matrix_output_path) if args.joint_matrix_output_path else project_root/"adjust_coord_matrices"/f"{dataset}_{args.variable}_spatiotemporal_t{args.time_step_num}_{args.cross_time_first_sample_mode}_r{args.cross_time_first_sample_region_num}.txt"
    write_adjust_coord_matrix(matrix_output_path,matrix,"spatiotemporal",dataset,args.variable)
    if args.train_only:
        print(f"四维训练耗时={training_time:.6f} s")
        print(f"四维变换矩阵：\n{matrix}")
        print(f"矩阵历史文件={args.matrix_history_path}")
        return
    joint_dir,transformed_coord=write_transformed_joint(work_dir,combined_coord,combined_data,args.variable,matrix)
    spatial_diff=np.column_stack((coord-coord[nearest_id[:,1]],np.zeros(coord.shape[0])))@matrix
    transformed_spatial_distance=float(np.median(np.linalg.norm(spatial_diff,axis=1)))
    transformed_time_distance=float(time_spacing*np.linalg.norm(matrix[3]))
    print(f"四维变换矩阵：\n{matrix}")
    print(f"变换后时间相邻距离={transformed_time_distance},空间最近邻距离中位数={transformed_spatial_distance},二者比例={transformed_time_distance/transformed_spatial_distance}")
    print(f"四维训练耗时={training_time:.6f} s,变换后坐标范围={list(zip(transformed_coord.min(axis=0),transformed_coord.max(axis=0)))}")
    joint_flags=["-cross-time-spatial-point-num-experimental",coord.shape[0]]
    if args.joint_block_num>1:
        joint_flags.extend(["-partition-num",args.joint_block_num])
    joint_br,joint_psnr,joint_preprocessing_time,joint_compression_time,joint_decompression_time=run_codec(joint_dir,args.variable,"{0}",combined_coord.shape[1],error_arguments,joint_flags,"时空联合预测",args.thread_num,args.compress_only)

    print("\n跨time-step初步结果：")
    if not args.joint_only:
        print(f"  逐时间步空间预测：BR={spatial_br:.6f},PSNR={spatial_psnr:.6f},预处理={spatial_preprocessing_time:.6f} s,压缩={spatial_compression_time:.6f} s({total_data_mb/spatial_compression_time:.3f} MB/s),解压={spatial_decompression_time:.6f} s({total_data_mb/spatial_decompression_time:.3f} MB/s)")
        print(f"  上一时间步同点标签预测：BR={temporal_label_br:.6f},PSNR={temporal_label_psnr:.6f},预处理={temporal_label_preprocessing_time:.6f} s,压缩={temporal_label_compression_time:.6f} s({total_data_mb/temporal_label_compression_time:.3f} MB/s),解压={temporal_label_decompression_time:.6f} s({total_data_mb/temporal_label_decompression_time:.3f} MB/s),相对空间BR变化={(temporal_label_br/spatial_br-1)*100:.3f}%")
        print(f"  上一时间步同点预测：BR={temporal_br:.6f},PSNR={temporal_psnr:.6f},预处理={temporal_preprocessing_time:.6f} s,压缩={temporal_compression_time:.6f} s({total_data_mb/temporal_compression_time:.3f} MB/s),解压={temporal_decompression_time:.6f} s({total_data_mb/temporal_decompression_time:.3f} MB/s),相对空间BR变化={(temporal_br/spatial_br-1)*100:.3f}%")
        print(f"  层次时间插值预测：BR={interpolation_br:.6f},PSNR={interpolation_psnr:.6f},预处理={interpolation_preprocessing_time:.6f} s,压缩={interpolation_compression_time:.6f} s({total_data_mb/interpolation_compression_time:.3f} MB/s),解压={interpolation_decompression_time:.6f} s({total_data_mb/interpolation_decompression_time:.3f} MB/s),相对空间BR变化={(interpolation_br/spatial_br-1)*100:.3f}%")
        print(f"  四维时空联合预测：分区数={args.joint_block_num},线程数={args.thread_num},BR={joint_br:.6f},PSNR={joint_psnr:.6f},预处理={joint_preprocessing_time:.6f} s,压缩={joint_compression_time:.6f} s({total_data_mb/joint_compression_time:.3f} MB/s),解压={joint_decompression_time:.6f} s({total_data_mb/joint_decompression_time:.3f} MB/s),相对空间BR变化={(joint_br/spatial_br-1)*100:.3f}%")
    else:
        print(f"  四维时空联合预测：分区数={args.joint_block_num},线程数={args.thread_num},BR={joint_br:.6f},PSNR={joint_psnr:.6f},预处理={joint_preprocessing_time:.6f} s,压缩={joint_compression_time:.6f} s({total_data_mb/joint_compression_time:.3f} MB/s),解压={joint_decompression_time:.6f} s({total_data_mb/joint_decompression_time:.3f} MB/s)")
    print(f"  总耗时={time.perf_counter()-total_start_time:.6f} s")

if __name__=="__main__":
    main()
