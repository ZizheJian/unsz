import argparse
import sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project_root))

import t1_start_c_code as benchmark


def main():
    parser=argparse.ArgumentParser(description="运行单个field的UNSZ和UMC空间压缩实验，默认使用t1中的五个REL EB")
    parser.add_argument("-dataset",choices=["Ocean","LES","Tsunami"],required=True)
    parser.add_argument("-field",required=True)
    parser.add_argument("-rel-eb",type=float)
    args=parser.parse_args()
    selected=[dataset for dataset in benchmark.dataset_list if dataset["folder"]==args.dataset and dataset["variable"]==args.field]
    if len(selected)!=1:
        raise ValueError(f"没有找到唯一的数据配置：dataset={args.dataset},field={args.field}")
    benchmark.dataset_list=selected
    benchmark.compressor_list=["unsz","umc"]
    if args.rel_eb is not None:
        benchmark.rel_eb_list=[args.rel_eb]
    benchmark.build_before_test=False
    benchmark.main()


if __name__=="__main__":
    main()
