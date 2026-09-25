import argparse,sys,tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from third_party.prepare_dataset_common.prepare_dataset_common import data_root_default,extract_hdf5_float32,run_h5dump,write_interleaved_float32


point_num=16002271
default_fields=("TEMPERATURE","PRESSURE","TKE","VELOCITY_X","RMS_T","CO2","C11H22")


def prepare_taps(source_path,output_dir,field_vec):
    output_dir.mkdir(parents=True,exist_ok=True)
    prefix="/STREAM_00/CELL_CENTER"
    with tempfile.TemporaryDirectory(prefix="unsz_taps_",dir=output_dir) as temporary_dir:
        temporary_dir=Path(temporary_dir)
        component_vec=[]
        for name in("XCEN_X","XCEN_Y","XCEN_Z"):
            raw_path=temporary_dir/f"{name}.bin"
            run_h5dump(source_path,f"{prefix}/{name}",raw_path)
            component_vec.append(np.memmap(raw_path,dtype="<f8",mode="r",shape=(point_num,)))
        write_interleaved_float32(output_dir/"coords.dat",component_vec)
    for field in field_vec:
        extract_hdf5_float32(source_path,f"{prefix}/{field}",output_dir/f"{field}.dat.0",(point_num,))
    print(f"TAPS转换完成：point_num={point_num},field={','.join(field_vec)}")


def main(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="转换TAPS数据")
    parser.add_argument("-source-path",type=Path)
    parser.add_argument("-output-dir",type=Path)
    parser.add_argument("-field",choices=default_fields,action="append")
    arguments=parser.parse_args(command_line_arguments)
    source_path=arguments.source_path or data_root_default/"TAPS_compressor/map9_8.000038e-02.h5"
    output_dir=arguments.output_dir or data_root_default/"TAPS"
    prepare_taps(source_path,output_dir,arguments.field or default_fields)


if __name__=="__main__":
    main()
