import argparse
from pathlib import Path
from py_codes.py_unsz_core.param import Param
from py_codes.py_unsz_core.init import init
from py_codes.py_unsz_core.adjust_coord import adjust_coord as optimize_adjust_coord
from py_codes.py_unsz_core.adjust_coord_matrix_file import write_adjust_coord_matrix
from t1_start_c_code import data_root,find_field_config,list_dataset_names

def parse_arguments(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="按照JSON中的field配置训练UNSZ坐标变换矩阵")
    dataset_names=tuple(name.lower() for name in list_dataset_names())
    parser.add_argument("-data",required=True,choices=dataset_names)
    parser.add_argument("-field",required=True)
    arguments=parser.parse_args(command_line_arguments)
    try:
        arguments.field_config=find_field_config(arguments.data,arguments.field)
    except ValueError as error:
        parser.error(str(error))
    return arguments

def get_predictor_config(field_config):
    predictor=field_config.get("predictor")
    if predictor is None:
        predictor="inverse_proportion" if field_config["config"].startswith("idw_") else "rbf_fitting"
    default_power=3.6 if predictor=="rbf_fitting" else 2
    return {"predictor_name":predictor,"inverse_distance_power":field_config.get("power",default_power),"rbf_kernel":field_config.get("kernel","matern32"),"rbf_shape":field_config.get("shape",0.01),"rbf_nugget":field_config.get("nugget",1e-6),"rbf_polynomial_degree":field_config.get("polynomial_degree",0)}

def train_adjust_coord(data,field,matrix_output_path=None,rel_eb=None,actual_ref_num=None,**training_arguments):
    field_config=find_field_config(data,field)
    if rel_eb is None:
        rel_eb=field_config.get("adjust_coord_rel_eb",1e-2)
    if actual_ref_num is None:
        actual_ref_num=field_config.get("prediction_ref_num",field_config.get("ref_num",4))
    param=Param()
    data_dir=Path(data_root)/field_config["folder"]
    param.coord_path=str(data_dir/"coords.dat")
    param.original_data_path_vec=[str(data_dir/f"{field_config['variable']}.dat.{time_step}") for time_step in range(field_config["steps"])]
    param.dim=field_config["dim"]
    param.rel_eb=rel_eb
    param.actual_ref_num=actual_ref_num
    param.cross_time_compress=False
    param.if_adjust_coord=True
    init(param)
    loss_function_name=field_config.get("adjust_coord_loss","prediction_ggd_bit")
    optimizer_name=field_config.get("adjust_coord_optimizer","lbfgs" if loss_function_name=="prediction_ggd_differential_entropy" else "least_squares")
    default_training_arguments={"loss_function_name":loss_function_name,"iteration_num":field_config.get("adjust_coord_iteration_num",4),"first_sample_ratio":field_config.get("adjust_coord_first_sample_ratio",0.25),"second_sample_ratio":field_config.get("adjust_coord_second_sample_ratio",0.1),"new_matrix_weight":field_config.get("adjust_coord_new_matrix_weight",0.5),"adjust_max_nfev":field_config.get("adjust_coord_max_nfev",100),"sample_file_num":field_config.get("adjust_coord_sample_file_num",3),"matrix_save_interval":field_config.get("adjust_coord_matrix_save_interval",1),"optimizer_name":optimizer_name,"matrix_scale_mode":field_config.get("adjust_coord_matrix_scale_mode","separate"),"new_matrix_scale_weight":field_config.get("adjust_coord_new_matrix_scale_weight",0.9),"second_sample_num":field_config.get("adjust_coord_second_sample_num",10000),**get_predictor_config(field_config)}
    optimize_adjust_coord(param,**(default_training_arguments|training_arguments))
    default_matrix_name=field_config.get("matrix",f"{field_config['dataset']}_{field_config['variable']}_spatial.txt")
    matrix_output_path=Path(matrix_output_path) if matrix_output_path is not None else Path(__file__).resolve().parent/"adjust_coord_matrices"/default_matrix_name
    write_adjust_coord_matrix(matrix_output_path,param.coord_transform_matrix,"spatial",field_config["dataset"],field_config["variable"])
    print(f"矩阵已写入{matrix_output_path}")
    return param.coord_transform_matrix

def main(command_line_arguments=None):
    arguments=parse_arguments(command_line_arguments)
    train_adjust_coord(arguments.data,arguments.field)

if __name__=="__main__":
    main()
