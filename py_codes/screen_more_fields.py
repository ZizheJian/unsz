import argparse,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project_root))

import t1_start_c_code as benchmark


data_root=Path(benchmark.data_root)
field_config_vec=[
    ({"name":"Tsunami-bathy","folder":"Tsunami","variable":"bathy","time_step_num":1,"umc_from_edge":1,"dim":2},["unsz","umc"]),
    ({"name":"Tsunami-height","folder":"Tsunami","variable":"height","time_step_num":1,"umc_from_edge":1,"dim":2},["unsz","umc"]),
    ({"name":"ADCIRC-Haiyan-1.5km-depth","folder":"ADCIRC-Haiyan-1.5km","variable":"depth","time_step_num":1,"umc_from_edge":1,"umc_reorder":0,"dim":3},["unsz","umc"]),
    ({"name":"E3SM-h0-PS","folder":"E3SM-h0","variable":"PS","time_step_num":1,"umc_from_edge":1,"dim":3},["unsz","umc"]),
    ({"name":"E3SM-h0-TREFHT","folder":"E3SM-h0","variable":"TREFHT","time_step_num":1,"umc_from_edge":1,"dim":3},["unsz","umc"]),
    ({"name":"E3SM-h0-PRECC","folder":"E3SM-h0","variable":"PRECC","time_step_num":1,"umc_from_edge":1,"dim":3},["unsz","umc"]),
    ({"name":"E3SM-h0-U10","folder":"E3SM-h0","variable":"U10","time_step_num":1,"umc_from_edge":1,"dim":3},["unsz","umc"]),
    ({"name":"E3SM-h0-TAUX","folder":"E3SM-h0","variable":"TAUX","time_step_num":1,"umc_from_edge":1,"dim":3},["unsz","umc"]),
    ({"name":"E3SM-h0-FLNS","folder":"E3SM-h0","variable":"FLNS","time_step_num":1,"umc_from_edge":1,"dim":3},["unsz","umc"]),
    ({"name":"E3SM-h0-AODVIS","folder":"E3SM-h0","variable":"AODVIS","time_step_num":1,"umc_from_edge":1,"dim":3,"umc_method":"NBP-IDW-INP","umc_neighbor_num":4,"umc_batch":10,"point_set_config":{"rule_type":"label_file","label_file":{"path":str(data_root/"E3SM-h0/AODVIS.label.{time_step}"),"dtype":"uint8"},"labels":[{"method":"inverse_proportion","eb_mode":"REL"},{"method":"constant"}]}},["unsz","umc"]),
    ({"name":"E3SM-h0-CLDTOT","folder":"E3SM-h0","variable":"CLDTOT","time_step_num":1,"umc_from_edge":1,"dim":3},["unsz","umc"]),
    ({"name":"TAPS-PRESSURE","folder":"TAPS","variable":"PRESSURE","time_step_num":1,"dim":3},["unsz"]),
    ({"name":"TAPS-TKE","folder":"TAPS","variable":"TKE","time_step_num":1,"dim":3},["unsz"]),
    ({"name":"TAPS-VELOCITY_X","folder":"TAPS","variable":"VELOCITY_X","time_step_num":1,"dim":3},["unsz"]),
    ({"name":"TAPS-RMS_T","folder":"TAPS","variable":"RMS_T","time_step_num":1,"dim":3},["unsz"]),
    ({"name":"TAPS-CO2","folder":"TAPS","variable":"CO2","time_step_num":1,"dim":3},["unsz"]),
    ({"name":"TAPS-C11H22","folder":"TAPS","variable":"C11H22","time_step_num":1,"dim":3},["unsz"]),
]


def main():
    parser=argparse.ArgumentParser(description="筛选尚未测试的field")
    parser.add_argument("-field",action="append",help="field配置名；可重复指定，默认测试全部")
    parser.add_argument("-rel-eb",type=float,nargs="+",default=[1e-3])
    parser.add_argument("-use-adjust-matrix",action="store_true")
    arguments=parser.parse_args()
    selected_config_vec=field_config_vec
    if arguments.field:
        selected_name_set=set(arguments.field)
        selected_config_vec=[config for config in field_config_vec if config[0]["name"] in selected_name_set]
        missing_name_set=selected_name_set-{config[0]["name"] for config in selected_config_vec}
        if missing_name_set:
            raise ValueError("未知field配置："+",".join(sorted(missing_name_set)))
    benchmark.build_before_test=False
    benchmark.rel_eb_list=arguments.rel_eb
    for dataset,compressor_vec in selected_config_vec:
        dataset=dataset.copy()
        compressor_vec=compressor_vec.copy()
        matrix_path=project_root/"adjust_coord_matrices"/f"{dataset['folder']}_{dataset['variable']}_spatial.txt"
        if arguments.use_adjust_matrix:
            if not matrix_path.is_file():
                raise FileNotFoundError(f"没有找到当前field的adjust coord矩阵：{matrix_path}")
            dataset["adjust_coord_matrix"]=matrix_path.name
            dataset["result_name_map"]={"unsz":"unsz_adjusted"}
            compressor_vec=["unsz"]
        print(f"\n新增field筛选：{dataset['name']}",flush=True)
        benchmark.dataset_list=[dataset]
        benchmark.compressor_list=compressor_vec
        benchmark.main()


if __name__=="__main__":
    main()
