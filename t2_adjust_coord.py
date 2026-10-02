import argparse
from pathlib import Path
from py_codes.py_unsz_core.param import Param
from py_codes.py_unsz_core.init import init
from py_codes.py_unsz_core.adjust_coord import adjust_coord as optimize_adjust_coord
from py_codes.py_unsz_core.adjust_coord_matrix_file import write_adjust_coord_matrix
from t1_start_c_code import load_json_tasks,resolve_json_path

def parse_arguments(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="按照JSON中的field配置训练UNSZ坐标变换矩阵")
    parser.add_argument("json",help="configs中的JSON配置名称，或JSON文件路径")
    arguments=parser.parse_args(command_line_arguments)
    try:
        arguments.json_path=resolve_json_path(arguments.json)
        arguments.tasks=load_json_tasks(arguments.json_path)
    except (FileNotFoundError,TypeError,ValueError) as error:
        parser.error(str(error))
    return arguments

def get_predictor_config(config):
    predictor=config["predictor"]
    predictor_name=predictor["name"]
    if predictor_name not in ("idw","rbf"):
        raise ValueError(f"未知预测器：{predictor_name}")
    if predictor_name=="idw":
        return {"predictor_name":"idw","idw_power":predictor["idw_power"]}
    return {"predictor_name":"rbf","rbf_kernel":predictor["rbf_kernel"],"rbf_shape":predictor["rbf_shape"],"rbf_nugget":predictor["rbf_nugget"]}

def train_adjust_coord(config,matrix_output_path=None,rel_eb=None,prediction_max_ref_num=None,**training_arguments):
    if rel_eb is None:
        rel_eb=config["adjust_coord_rel_eb"]
    if prediction_max_ref_num is None:
        prediction_max_ref_num=config["prediction_max_ref_num"]
    param=Param()
    data_dir=Path(config["data_dir"])
    param.coord_path=str(data_dir/"coords.dat")
    param.original_data_path_vec=[str(data_dir/f"{config['field']}.dat.{time_step}") for time_step in range(config["time_step_num"])]
    param.dim=config["coordinate_dim"]
    param.rel_eb=rel_eb
    param.prediction_max_ref_num=prediction_max_ref_num
    init(param)
    configured_training_arguments={"loss_function_name":config["adjust_coord_loss"],"iteration_num":config["adjust_coord_iteration_num"],"first_sample_ratio":config["adjust_coord_first_sample_ratio"],"second_sample_ratio":config["adjust_coord_second_sample_ratio"],"new_matrix_weight":config["adjust_coord_new_matrix_weight"],"adjust_max_nfev":config["adjust_coord_max_nfev"],"sample_file_num":config["adjust_coord_sample_file_num"],"matrix_save_interval":config["adjust_coord_matrix_save_interval"],"optimizer_name":config["adjust_coord_optimizer"],"matrix_scale_mode":config["adjust_coord_matrix_scale_mode"],"new_matrix_scale_weight":config["adjust_coord_new_matrix_scale_weight"],"second_sample_num":config["adjust_coord_second_sample_num"],**get_predictor_config(config)}
    optimize_adjust_coord(param,**(configured_training_arguments|training_arguments))
    matrix_output_path=Path(matrix_output_path) if matrix_output_path is not None else Path(__file__).resolve().parent/"adjust_coord_matrices"/config["adjust_coord_matrix_file"]
    write_adjust_coord_matrix(matrix_output_path,param.coord_transform_matrix,"spatial",config["field"])
    print(f"矩阵已写入{matrix_output_path}")
    return param.coord_transform_matrix

def main(command_line_arguments=None):
    arguments=parse_arguments(command_line_arguments)
    for task in arguments.tasks:
        for field in task["field_list"]:
            train_adjust_coord(task|task["unsz"]|{"field":field})

if __name__=="__main__":
    main()
