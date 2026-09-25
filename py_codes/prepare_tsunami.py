import argparse,sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from third_party.prepare_dataset_common.prepare_dataset_common import data_root_default,extract_hdf5_native_float32


point_num=113885
source_field_name={"bathy":"bathy","height":"height","momentum_x":"m_x","momentum_y":"m_y","velocity_x":"u_x","velocity_y":"u_y"}


def prepare_tsunami(source_path,output_dir,field_vec):
    output_dir.mkdir(parents=True,exist_ok=True)
    for field in field_vec:
        output_path=output_dir/f"{field}.dat.0"
        extract_hdf5_native_float32(source_path,f"/Mesh2_{source_field_name[field]}",output_path,"0,0",f"1,{point_num}")
        if output_path.stat().st_size!=point_num*4:
            raise RuntimeError(f"Tsunami field {field}的文件大小不正确：{output_path.stat().st_size}")
    print(f"Tsunami转换完成：point_num={point_num},field={','.join(field_vec)}")


def main(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="转换Tsunami field；coords.dat和conn.dat沿用UMC发布数据")
    parser.add_argument("-source-path",type=Path)
    parser.add_argument("-output-dir",type=Path)
    parser.add_argument("-field",choices=tuple(source_field_name),action="append")
    arguments=parser.parse_args(command_line_arguments)
    source_path=arguments.source_path or data_root_default/"Tsunami/ugrid_demo.nc"
    output_dir=arguments.output_dir or data_root_default/"Tsunami"
    prepare_tsunami(source_path,output_dir,arguments.field or tuple(source_field_name))


if __name__=="__main__":
    main()
