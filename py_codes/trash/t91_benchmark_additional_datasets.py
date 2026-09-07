import argparse,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))

import t1_start_c_code as benchmark


data_root=Path(benchmark.data_root)
dataset_map={
    "tpost20":({"name":"Tpost20-scalar","folder":"Tpost20","variable":"scalar","time_step_num":20,"umc_from_edge":0,"dim":3},["unsz","umc"]),
    "e3sm-mass":({"name":"E3SM-h4-Mass_dst_srf","folder":"E3SM-h4","variable":"Mass_dst_srf","time_step_num":720,"umc_from_edge":1,"dim":3},["unsz","umc"]),
    "e3sm-aoddust":({"name":"E3SM-h4-AODDUST","folder":"E3SM-h4","variable":"AODDUST","time_step_num":720,"umc_from_edge":1,"dim":3,"umc_method":"NBP-IDW-INP","umc_neighbor_num":4,"umc_batch":10,"point_set_config":{"rule_type":"label_file","label_file":{"path":str(data_root/"E3SM-h4/AODDUST.label.{time_step}"),"dtype":"uint8"},"labels":[{"method":"inverse_proportion","eb_mode":"REL"},{"method":"constant"}]}},["unsz","umc"]),
    "haiyan":({"name":"ADCIRC-Haiyan-1.5km-zeta","folder":"ADCIRC-Haiyan-1.5km","variable":"zeta","time_step_num":24,"umc_from_edge":1,"umc_reorder":0,"dim":3,"umc_method":"NBP-IDW-INP","umc_neighbor_num":4,"umc_batch":10,"point_set_config":{"rule_type":"label_file","label_file":{"path":str(data_root/"ADCIRC-Haiyan-1.5km/zeta.label.{time_step}"),"dtype":"uint8"},"labels":[{"method":"inverse_proportion","eb_mode":"REL"},{"method":"constant"}]}},["unsz","umc"]),
    "taps":({"name":"TAPS-temperature","folder":"TAPS","variable":"TEMPERATURE","time_step_num":1,"dim":3},["unsz"]),
}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("-dataset",choices=tuple(dataset_map)+("all",),default="all")
    parser.add_argument("-rel-eb",type=float,nargs="+",default=[1e-3])
    arguments=parser.parse_args()
    selected=dataset_map if arguments.dataset=="all" else {arguments.dataset:dataset_map[arguments.dataset]}
    benchmark.build_before_test=False
    benchmark.rel_eb_list=arguments.rel_eb
    for name,(dataset,compressor_vec) in selected.items():
        print(f"\n新增数据筛选：{name}",flush=True)
        benchmark.dataset_list=[dataset]
        benchmark.compressor_list=compressor_vec
        benchmark.main()


if __name__=="__main__":
    main()
