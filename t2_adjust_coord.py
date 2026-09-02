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
parser.add_argument("-loss",choices=["fourth_power","second_power","log_distance","prediction_ggd_bit"],default="second_power")
parser.add_argument("-optimizer",choices=["least_squares","gauss_newton","lbfgs"],default="least_squares")
parser.add_argument("-iteration-num",type=int,default=4)
parser.add_argument("-matrix-history-path",default="")
parser.add_argument("-matrix-output-path",default="")
parser.add_argument("-first-sample-ratio",type=float,default=0.25)
parser.add_argument("-second-sample-ratio",type=float,default=0.1)
parser.add_argument("-new-matrix-weight",type=float,default=0.5)
parser.add_argument("-max-nfev",type=int,default=100)
parser.add_argument("-sample-file-num",type=int,default=3)
parser.add_argument("-actual-ref-num",type=int,default=10)
parser.add_argument("-matrix-save-interval",type=int,default=1)
args=parser.parse_args()
if args.iteration_num<=0:
    raise ValueError("iteration-num必须是正数")
if not 0<args.first_sample_ratio<=1 or not 0<args.second_sample_ratio<=1:
    raise ValueError("两次采样率必须大于0且不大于1")
if not 0<args.new_matrix_weight<=1:
    raise ValueError("new-matrix-weight必须大于0且不大于1")
if args.max_nfev<=0 or args.sample_file_num<=0 or args.actual_ref_num<=0 or args.matrix_save_interval<=0 or args.time_step_num<=0:
    raise ValueError("max-nfev、sample-file-num、actual-ref-num、matrix-save-interval和time-step-num必须是正数")

param=Param()
param.coord_path=f"{args.data_dir}/coords.dat"
param.original_data_path_vec=[f"{args.data_dir}/{args.variable}.dat.{i}" for i in range(args.time_step_num)]
param.dim=3
param.rel_eb=1e-2
param.actual_ref_num=args.actual_ref_num
param.cross_time_compress=False
param.if_adjust_coord=True

init(param)
adjust_coord(param,args.loss,args.iteration_num,args.matrix_history_path or None,args.first_sample_ratio,args.second_sample_ratio,args.new_matrix_weight,args.max_nfev,args.sample_file_num,args.matrix_save_interval,args.optimizer)
dataset=Path(args.data_dir).resolve().name
matrix_output_path=Path(args.matrix_output_path) if args.matrix_output_path else Path(__file__).resolve().parent/"adjust_coord_matrices"/f"{dataset}_{args.variable}_spatial.txt"
write_adjust_coord_matrix(matrix_output_path,param.coord_transform_matrix,"spatial",dataset,args.variable)
