import argparse,shutil,sys,tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from third_party.prepare_dataset_common.prepare_dataset_common import data_root_default,extract_hdf5_float32,run_h5dump,write_interleaved_float32


point_num=2187832
cell_num=4032359
time_step_num=24


def prepare_haiyan_mesh(source_path,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="unsz_haiyan_",dir=output_dir) as temporary_dir:
        temporary_dir=Path(temporary_dir)
        component_vec=[]
        for name in("x","y"):
            raw_path=temporary_dir/f"{name}.bin"
            run_h5dump(source_path,f"/{name}",raw_path)
            component_vec.append(np.memmap(raw_path,dtype="<f8",mode="r",shape=(point_num,)))
        longitude=np.deg2rad(component_vec[0])
        latitude=np.deg2rad(component_vec[1])
        cos_latitude=np.cos(latitude)
        write_interleaved_float32(output_dir/"coords.dat",(cos_latitude*np.cos(longitude),cos_latitude*np.sin(longitude),np.sin(latitude)))
        element_path=temporary_dir/"element.bin"
        run_h5dump(source_path,"/element",element_path)
        connectivity=np.memmap(element_path,dtype="<i4",mode="r+",shape=(cell_num,3))
        if int(connectivity.min())!=1 or int(connectivity.max())!=point_num:
            raise RuntimeError("Haiyan connectivity不是预期的1-based编号")
        connectivity-=1
        connectivity.flush()
        shutil.copyfile(element_path,output_dir/"element.dat")
        edge_key=np.concatenate([np.minimum(connectivity[:,0],connectivity[:,1]).astype(np.int64)*point_num+np.maximum(connectivity[:,0],connectivity[:,1]),np.minimum(connectivity[:,1],connectivity[:,2]).astype(np.int64)*point_num+np.maximum(connectivity[:,1],connectivity[:,2]),np.minimum(connectivity[:,2],connectivity[:,0]).astype(np.int64)*point_num+np.maximum(connectivity[:,2],connectivity[:,0])])
        edge_key=np.unique(edge_key)
        np.column_stack((edge_key//point_num,edge_key%point_num)).astype(np.uint32).tofile(output_dir/"conn.dat")
    print(f"ADCIRC Haiyan网格转换完成：point_num={point_num},cell_num={cell_num}")


def prepare_haiyan_zeta(source_path,output_dir):
    for time_step in range(time_step_num):
        extract_hdf5_float32(source_path,"/zeta",output_dir/f"zeta.dat.{time_step}",(point_num,),f"{time_step},0",f"1,{point_num}")
    print(f"ADCIRC Haiyan zeta转换完成：time_step_num={time_step_num}")


def prepare_haiyan_depth(source_path,output_dir):
    extract_hdf5_float32(source_path,"/depth",output_dir/"depth.dat.0",(point_num,))
    print(f"ADCIRC Haiyan depth转换完成：point_num={point_num}")


def main(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="转换ADCIRC Haiyan网格、zeta和depth")
    parser.add_argument("-source-path",type=Path)
    parser.add_argument("-output-dir",type=Path)
    parser.add_argument("-field",choices=("zeta","depth"),action="append")
    arguments=parser.parse_args(command_line_arguments)
    source_path=arguments.source_path or data_root_default/"ADCIRC-Haiyan-1.5km/Haiyan_1.5km.63.nc"
    output_dir=arguments.output_dir or data_root_default/"ADCIRC-Haiyan-1.5km"
    field_vec=arguments.field or("zeta","depth")
    prepare_haiyan_mesh(source_path,output_dir)
    if "zeta" in field_vec:
        prepare_haiyan_zeta(source_path,output_dir)
    if "depth" in field_vec:
        prepare_haiyan_depth(source_path,output_dir)


if __name__=="__main__":
    main()
