import argparse
from pathlib import Path
from py_codes.py_unsz_core.param import Param
from py_codes.py_unsz_core.init import init
from py_codes.py_unsz_core.adjust_coord import adjust_coord as optimize_adjust_coord
from py_codes.py_unsz_core.adjust_coord_matrix_file import write_adjust_coord_matrix
from t1_start_c_code import load_tasks,resolve_json_path

data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"

def parse_arguments(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="按照JSON中的field配置训练UNSZ坐标变换矩阵")
    parser.add_argument("json",help="configs中的JSON配置名称，或JSON文件路径")
    arguments=parser.parse_args(command_line_arguments)
    try:
        arguments.json_path=resolve_json_path(arguments.json)
        arguments.prepared_tasks=load_tasks(arguments.json_path)
    except (FileNotFoundError,TypeError,ValueError) as error:
        parser.error(str(error))
    return arguments

def get_predictor_config(config):
    predictor=config.get("predictor")
    if predictor is None:
        predictor="inverse_proportion" if config["config"].startswith("idw_") else "rbf_fitting"
    default_power=3.6 if predictor=="rbf_fitting" else 2
    return {"predictor_name":predictor,"inverse_distance_power":config.get("power",default_power),"rbf_kernel":config.get("kernel","matern32"),"rbf_shape":config.get("shape",0.01),"rbf_nugget":config.get("nugget",1e-6),"rbf_polynomial_degree":config.get("polynomial_degree",0)}

def train_adjust_coord(config,matrix_output_path=None,rel_eb=None,actual_ref_num=None,**training_arguments):
    if rel_eb is None:
        rel_eb=config.get("adjust_coord_rel_eb",1e-2)
    if actual_ref_num is None:
        actual_ref_num=config.get("prediction_ref_num",config.get("ref_num",4))
    param=Param()
    data_dir=Path(data_root)/config["folder"]
    param.coord_path=str(data_dir/"coords.dat")
    param.original_data_path_vec=[str(data_dir/f"{config['variable']}.dat.{time_step}") for time_step in range(config["steps"])]
    param.dim=config["dim"]
    param.rel_eb=rel_eb
    param.actual_ref_num=actual_ref_num
    param.cross_time_compress=False
    param.if_adjust_coord=True
    init(param)
    loss_function_name=config.get("adjust_coord_loss","prediction_ggd_bit")
    optimizer_name=config.get("adjust_coord_optimizer","lbfgs" if loss_function_name=="prediction_ggd_differential_entropy" else "least_squares")
    default_training_arguments={"loss_function_name":loss_function_name,"iteration_num":config.get("adjust_coord_iteration_num",4),"first_sample_ratio":config.get("adjust_coord_first_sample_ratio",0.25),"second_sample_ratio":config.get("adjust_coord_second_sample_ratio",0.1),"new_matrix_weight":config.get("adjust_coord_new_matrix_weight",0.5),"adjust_max_nfev":config.get("adjust_coord_max_nfev",100),"sample_file_num":config.get("adjust_coord_sample_file_num",3),"matrix_save_interval":config.get("adjust_coord_matrix_save_interval",1),"optimizer_name":optimizer_name,"matrix_scale_mode":config.get("adjust_coord_matrix_scale_mode","separate"),"new_matrix_scale_weight":config.get("adjust_coord_new_matrix_scale_weight",0.9),"second_sample_num":config.get("adjust_coord_second_sample_num",10000),**get_predictor_config(config)}
    optimize_adjust_coord(param,**(default_training_arguments|training_arguments))
    default_matrix_name=config.get("matrix",f"{config['dataset']}_{config['variable']}_spatial.txt")
    matrix_output_path=Path(matrix_output_path) if matrix_output_path is not None else Path(__file__).resolve().parent/"adjust_coord_matrices"/default_matrix_name
    write_adjust_coord_matrix(matrix_output_path,param.coord_transform_matrix,"spatial",config["dataset"],config["variable"])
    print(f"矩阵已写入{matrix_output_path}")
    return param.coord_transform_matrix

def main(command_line_arguments=None):
    arguments=parse_arguments(command_line_arguments)
    for config in arguments.prepared_tasks:
        train_adjust_coord(config)

if __name__=="__main__":
    main()
