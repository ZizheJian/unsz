import argparse
from pathlib import Path
from py_codes.py_unsz_core.param import Param
from py_codes.py_unsz_core.init import init
from py_codes.py_unsz_core.adjust_coord import adjust_coord
from py_codes.py_unsz_core.adjust_coord_matrix_file import write_adjust_coord_matrix

parser=argparse.ArgumentParser(description="训练UNSZ坐标变换矩阵")
parser.add_argument("-data-dir",default="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
parser.add_argument("-variable",default="attr0")
parser.add_argument("-time-step-num",type=int,default=162)
parser.add_argument("-dim",type=int,default=3)
parser.add_argument("-loss",choices=["fourth_power","second_power","log_distance","prediction_ggd_bit"],default="second_power")
parser.add_argument("-optimizer",choices=["least_squares","gauss_newton","lbfgs"],default="least_squares")
parser.add_argument("-iteration-num",type=int,default=4)
parser.add_argument("-matrix-history-path",default="")
parser.add_argument("-matrix-output-path",default="")
parser.add_argument("-first-sample-ratio",type=float,default=0.25)
parser.add_argument("-first-sample-num",type=int,default=0)
parser.add_argument("-first-sample-mode",choices=["random_points","random_hyperrectangle_regions","random_fixed_point_regions","random_regions_then_points","random_sampled_kd_blocks","random_sampled_kd_neighborhoods"],default="random_points")
parser.add_argument("-first-sample-region-num",type=int,choices=[1,2,4,8],default=1)
parser.add_argument("-first-sample-inner-ratio",type=float,default=0.25)
parser.add_argument("-first-sample-pool-ratio",type=float,default=0.25)
parser.add_argument("-second-sample-ratio",type=float,default=0.1)
parser.add_argument("-second-sample-num",type=int,default=0,help="每轮第二次采样的固定目标总数；0表示使用second-sample-ratio")
parser.add_argument("-new-matrix-weight",type=float,default=0.5)
parser.add_argument("-max-nfev",type=int,default=100)
parser.add_argument("-sample-file-num",type=int,default=3)
parser.add_argument("-actual-ref-num",type=int,default=4)
parser.add_argument("-matrix-save-interval",type=int,default=1)
parser.add_argument("-matrix-mix-space",choices=["factor","metric"],default="factor",help="factor直接混合C，metric混合C@C.T")
parser.add_argument("-small-eigenvalue-correction",choices=["on","off"],default="off")
parser.add_argument("-matrix-scale-mode",choices=["discard","separate"],default="separate")
parser.add_argument("-new-matrix-scale-weight",type=float,default=0.9)
args=parser.parse_args()
if args.iteration_num<=0:
    raise ValueError("iteration-num必须是正数")
if not 0<args.first_sample_ratio<=1 or not 0<args.second_sample_ratio<=1:
    raise ValueError("两次采样率必须大于0且不大于1")
if not 0<args.first_sample_inner_ratio<=1:
    raise ValueError("first-sample-inner-ratio必须大于0且不大于1")
if args.first_sample_num<0 or args.second_sample_num<0 or not 0<args.first_sample_pool_ratio<=1:
    raise ValueError("first-sample-num和second-sample-num不能为负数，first-sample-pool-ratio必须大于0且不大于1")
if not 0<args.new_matrix_weight<=1:
    raise ValueError("new-matrix-weight必须大于0且不大于1")
if not 0<args.new_matrix_scale_weight<=1:
    raise ValueError("new-matrix-scale-weight必须大于0且不大于1")
if args.max_nfev<=0 or args.sample_file_num<=0 or args.actual_ref_num<=0 or args.matrix_save_interval<=0 or args.time_step_num<=0 or args.dim<=0:
    raise ValueError("max-nfev、sample-file-num、actual-ref-num、matrix-save-interval、time-step-num和dim必须是正数")

param=Param()
param.coord_path=f"{args.data_dir}/coords.dat"
param.original_data_path_vec=[f"{args.data_dir}/{args.variable}.dat.{i}" for i in range(args.time_step_num)]
param.dim=args.dim
param.rel_eb=1e-2
param.actual_ref_num=args.actual_ref_num
param.cross_time_compress=False
param.if_adjust_coord=True

init(param)
adjust_coord(param,args.loss,args.iteration_num,args.matrix_history_path or None,args.first_sample_ratio,args.second_sample_ratio,args.new_matrix_weight,args.max_nfev,args.sample_file_num,args.matrix_save_interval,args.optimizer,spatial_first_sample_mode=args.first_sample_mode,spatial_first_sample_region_num=args.first_sample_region_num,spatial_first_sample_inner_ratio=args.first_sample_inner_ratio,spatial_first_sample_num=args.first_sample_num,spatial_pool_sample_ratio=args.first_sample_pool_ratio,matrix_mix_space=args.matrix_mix_space,correct_small_eigenvalues=args.small_eigenvalue_correction=="on",matrix_scale_mode=args.matrix_scale_mode,new_matrix_scale_weight=args.new_matrix_scale_weight,second_sample_num=args.second_sample_num)
dataset=Path(args.data_dir).resolve().name
default_output_name=f"{dataset}_{args.variable}_spatial.txt" if args.first_sample_mode=="random_points" else f"{dataset}_{args.variable}_spatial_{args.first_sample_mode}_r{args.first_sample_region_num}.txt"
matrix_output_path=Path(args.matrix_output_path) if args.matrix_output_path else Path(__file__).resolve().parent/"adjust_coord_matrices"/default_output_name
write_adjust_coord_matrix(matrix_output_path,param.coord_transform_matrix,"spatial",dataset,args.variable)
