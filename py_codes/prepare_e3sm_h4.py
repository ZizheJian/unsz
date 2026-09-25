import argparse,sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from third_party.prepare_dataset_common.prepare_dataset_common import data_root_default
from prepare_e3sm_common import prepare_e3sm


default_fields=("AODDUST","Mass_dst_srf")


def main(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="转换E3SM-h4数据")
    parser.add_argument("-source-path",type=Path)
    parser.add_argument("-grid-path",type=Path)
    parser.add_argument("-output-dir",type=Path)
    parser.add_argument("-field",action="append")
    arguments=parser.parse_args(command_line_arguments)
    source_path=arguments.source_path or data_root_default/"yfeng/20250218.v3.LR.F2010.NUG2005-2015_MidEmi.eam.h4.2015-07-10-00000.nc"
    grid_path=arguments.grid_path or data_root_default/"yfeng/ne30pg2.nc"
    output_dir=arguments.output_dir or data_root_default/"E3SM-h4"
    prepare_e3sm(source_path,grid_path,output_dir,arguments.field or default_fields,"h4")


if __name__=="__main__":
    main()
