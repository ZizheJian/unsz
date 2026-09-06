import argparse,subprocess,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data")
history_dir=project_root/"py_codes/trash/results/adjust_coord_history/spatial_parameter_screen"
matrix_dir=project_root/"adjust_coord_matrices"
dataset_config={
    "Ocean":{"dim":3,"point_num":235160,"first_sample_num":58790,"field_vec":("layerThickness","salinity","temperature","velocityMeridional","velocityX","velocityY","velocityZ","velocityZonal","zTop")},
    "LES":{"dim":3,"point_num":12737274,"first_sample_num":117580,"field_vec":("velocity_x","velocity_y","velocity_z")},
    "Tsunami":{"dim":2,"point_num":113885,"first_sample_num":28472,"field_vec":("momentum_x","momentum_y","velocity_x","velocity_y")},
}

parser=argparse.ArgumentParser()
parser.add_argument("-dataset",choices=dataset_config,required=True)
parser.add_argument("-field",action="append",default=[])
parser.add_argument("-actual-ref-num",type=int,default=4)
args=parser.parse_args()
if args.actual_ref_num<=0:
    raise ValueError("actual-ref-num必须是正数")
config=dataset_config[args.dataset]
first_sample_ratio=config["first_sample_num"]/config["point_num"]
history_dir.mkdir(parents=True,exist_ok=True)
matrix_dir.mkdir(parents=True,exist_ok=True)
field_vec=[field for field in config["field_vec"] if not args.field or field in args.field]
if not field_vec:
    raise ValueError("没有符合筛选条件的field")
for field in field_vec:
    history_path=history_dir/f"{args.dataset}_{field}_gauss_newton_ref{args.actual_ref_num}.npz"
    matrix_path=matrix_dir/f"{args.dataset}_{field}_spatial_ref{args.actual_ref_num}_candidate.txt"
    command=[sys.executable,str(project_root/"t2_adjust_coord.py"),"-data-dir",str(data_root/args.dataset),"-variable",field,"-time-step-num","1","-dim",str(config["dim"]),"-loss","second_power","-optimizer","gauss_newton","-iteration-num","4","-matrix-history-path",str(history_path),"-matrix-output-path",str(matrix_path),"-first-sample-ratio",str(first_sample_ratio),"-second-sample-ratio","0.1","-new-matrix-weight","0.5","-max-nfev","100","-sample-file-num","1","-actual-ref-num",str(args.actual_ref_num),"-matrix-save-interval","1"]
    print(f"开始训练{args.dataset}-{field}的{args.actual_ref_num}近邻空间矩阵",flush=True)
    subprocess.run(command,check=True,cwd=project_root)
