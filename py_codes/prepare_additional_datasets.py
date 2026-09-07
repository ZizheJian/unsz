import argparse,gzip,shutil,subprocess,tempfile
from pathlib import Path

import numpy as np


def write_interleaved_float32(output_path,array_vec,chunk_size=1<<20):
    point_num=array_vec[0].size
    if any(array.size!=point_num for array in array_vec):
        raise ValueError("坐标分量长度不一致")
    with open(output_path,"wb") as output_file:
        for begin in range(0,point_num,chunk_size):
            end=min(begin+chunk_size,point_num)
            np.column_stack([np.asarray(array[begin:end],dtype=np.float32) for array in array_vec]).tofile(output_file)


def prepare_tpost20(source_path,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    with gzip.open(source_path,"rt") as source_file:
        point_num=int(source_file.readline())
        time_step_num=int(source_file.readline())
        cell_num=int(source_file.readline())
        coord_array=np.empty((point_num,3),dtype=np.float32)
        data_array=np.empty((time_step_num,point_num),dtype=np.float32)
        for point_id in range(point_num):
            value=np.fromstring(source_file.readline(),sep=" ",dtype=np.float32)
            if value.size!=3+time_step_num:
                raise RuntimeError(f"Tpost20第{point_id}个点应有{3+time_step_num}个数，实际为{value.size}")
            coord_array[point_id]=value[:3]
            data_array[:,point_id]=value[3:]
        connectivity=np.empty((cell_num,4),dtype=np.uint32)
        for cell_id in range(cell_num):
            value=np.fromstring(source_file.readline(),sep=" ",dtype=np.uint32)
            if value.size!=4:
                raise RuntimeError(f"Tpost20第{cell_id}个四面体应有4个点，实际为{value.size}")
            connectivity[cell_id]=value
        if source_file.readline():
            raise RuntimeError("Tpost20文件末尾存在未解析内容")
    coord_array.tofile(output_dir/"coords.dat")
    connectivity.tofile(output_dir/"conn.dat")
    for time_step in range(time_step_num):
        data_array[time_step].tofile(output_dir/f"scalar.dat.{time_step}")
    print(f"Tpost20转换完成：point_num={point_num},time_step_num={time_step_num},cell_num={cell_num}")


def prepare_e3sm_h4(source_path,output_dir,field_vec):
    import xarray as xr
    output_dir.mkdir(parents=True,exist_ok=True)
    dataset=xr.open_dataset(source_path,engine="scipy",decode_times=False)
    try:
        latitude=np.deg2rad(np.asarray(dataset["lat"].values,dtype=np.float64))
        longitude=np.deg2rad(np.asarray(dataset["lon"].values,dtype=np.float64))
        cos_latitude=np.cos(latitude)
        write_interleaved_float32(output_dir/"coords.dat",(cos_latitude*np.cos(longitude),cos_latitude*np.sin(longitude),np.sin(latitude)))
        prepare_scrip_cell_edges(source_path.parent/"ne30pg2.nc",output_dir/"conn.dat")
        time_step_num=int(dataset.sizes["time"])
        point_num=int(dataset.sizes["ncol"])
        for field in field_vec:
            variable=dataset[field]
            if variable.dims!=("time","ncol"):
                raise ValueError(f"E3SM field {field}维度不是(time,ncol)：{variable.dims}")
            for time_step in range(time_step_num):
                data=np.asarray(variable.isel(time=time_step).values,dtype=np.float32)
                invalid=~np.isfinite(data)
                if np.any(invalid):
                    data=data.copy()
                    data[invalid]=-99999
                    invalid.astype(np.uint8).tofile(output_dir/f"{field}.label.{time_step}")
                data.tofile(output_dir/f"{field}.dat.{time_step}")
        print(f"E3SM h4转换完成：point_num={point_num},time_step_num={time_step_num},field={','.join(field_vec)}")
    finally:
        dataset.close()


def prepare_e3sm_h0(source_path,grid_path,output_dir,field_vec):
    import xarray as xr
    output_dir.mkdir(parents=True,exist_ok=True)
    dataset=xr.open_dataset(source_path,engine="scipy",decode_times=False)
    try:
        latitude=np.deg2rad(np.asarray(dataset["lat"].values,dtype=np.float64))
        longitude=np.deg2rad(np.asarray(dataset["lon"].values,dtype=np.float64))
        cos_latitude=np.cos(latitude)
        write_interleaved_float32(output_dir/"coords.dat",(cos_latitude*np.cos(longitude),cos_latitude*np.sin(longitude),np.sin(latitude)))
        prepare_scrip_cell_edges(grid_path,output_dir/"conn.dat")
        time_step_num=int(dataset.sizes["time"])
        point_num=int(dataset.sizes["ncol"])
        for field in field_vec:
            variable=dataset[field]
            if variable.dims!=("time","ncol"):
                raise ValueError(f"E3SM field {field}维度不是(time,ncol)：{variable.dims}")
            for time_step in range(time_step_num):
                data=np.asarray(variable.isel(time=time_step).values,dtype=np.float32)
                invalid=~np.isfinite(data)
                if np.any(invalid):
                    data=data.copy()
                    data[invalid]=-99999
                    invalid.astype(np.uint8).tofile(output_dir/f"{field}.label.{time_step}")
                data.tofile(output_dir/f"{field}.dat.{time_step}")
        print(f"E3SM h0转换完成：point_num={point_num},time_step_num={time_step_num},field={','.join(field_vec)}")
    finally:
        dataset.close()


def prepare_scrip_cell_edges(grid_path,output_path):
    point_num=21600
    corner_num=4
    with tempfile.TemporaryDirectory(prefix="unsz_scrip_",dir=output_path.parent) as temporary_dir:
        temporary_dir=Path(temporary_dir)
        corner_component_vec=[]
        for name in ("grid_corner_lat","grid_corner_lon"):
            raw_path=temporary_dir/f"{name}.bin"
            run_h5dump(grid_path,f"/{name}",raw_path)
            corner_component_vec.append(np.memmap(raw_path,dtype="<f8",mode="r",shape=(point_num,corner_num)))
        latitude=np.deg2rad(corner_component_vec[0])
        longitude=np.deg2rad(corner_component_vec[1])
        cos_latitude=np.cos(latitude)
        corner_xyz=np.stack((cos_latitude*np.cos(longitude),cos_latitude*np.sin(longitude),np.sin(latitude)),axis=-1)
        _,corner_id=np.unique(np.round(corner_xyz.reshape(-1,3),decimals=11),axis=0,return_inverse=True)
        corner_id=corner_id.reshape(point_num,corner_num)
        edge_min=np.minimum(corner_id,np.roll(corner_id,-1,axis=1))
        edge_max=np.maximum(corner_id,np.roll(corner_id,-1,axis=1))
        vertex_num=int(corner_id.max())+1
        edge_key=(edge_min.astype(np.int64)*vertex_num+edge_max).ravel()
        cell_id=np.repeat(np.arange(point_num,dtype=np.uint32),corner_num)
        order=np.argsort(edge_key)
        edge_key=edge_key[order]
        cell_id=cell_id[order]
        unique_key,first,count=np.unique(edge_key,return_index=True,return_counts=True)
        if np.any(count!=2):
            raise RuntimeError(f"E3SM网格存在{np.count_nonzero(count!=2)}条不恰好属于两个cell的边")
        del unique_key
        np.column_stack((cell_id[first],cell_id[first+1])).astype(np.uint32).tofile(output_path)


def run_h5dump(source_path,dataset_name,output_path,start=None,count=None):
    h5dump=shutil.which("h5dump")
    if h5dump is None:
        raise FileNotFoundError("PATH中没有h5dump")
    command=[h5dump,"-d",dataset_name]
    if start is not None:
        command.extend(["-s",start])
    if count is not None:
        command.extend(["-c",count])
    command.extend(["-b","LE","-o",str(output_path)])
    command.append(str(source_path))
    subprocess.run(command,stdout=subprocess.DEVNULL,check=True)


def extract_hdf5_float32(source_path,dataset_name,output_path,shape,start=None,count=None):
    with tempfile.TemporaryDirectory(prefix="unsz_h5dump_",dir=output_path.parent) as temporary_dir:
        raw_path=Path(temporary_dir)/"raw.bin"
        run_h5dump(source_path,dataset_name,raw_path,start,count)
        raw=np.memmap(raw_path,dtype="<f8",mode="r",shape=shape)
        with open(output_path,"wb") as output_file:
            for begin in range(0,raw.size,1<<20):
                np.asarray(raw[begin:min(begin+(1<<20),raw.size)],dtype=np.float32).tofile(output_file)


def extract_hdf5_native_float32(source_path,dataset_name,output_path,start=None,count=None):
    with tempfile.TemporaryDirectory(prefix="unsz_h5dump_",dir=output_path.parent) as temporary_dir:
        raw_path=Path(temporary_dir)/"raw.bin"
        run_h5dump(source_path,dataset_name,raw_path,start,count)
        shutil.copyfile(raw_path,output_path)


def prepare_adcirc_haiyan(source_path,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    point_num=2187832
    cell_num=4032359
    time_step_num=24
    with tempfile.TemporaryDirectory(prefix="unsz_haiyan_",dir=output_dir) as temporary_dir:
        temporary_dir=Path(temporary_dir)
        component_vec=[]
        for name in ("x","y"):
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
        edge_array=np.column_stack((edge_key//point_num,edge_key%point_num)).astype(np.uint32)
        edge_array.tofile(output_dir/"conn.dat")
    for time_step in range(time_step_num):
        extract_hdf5_float32(source_path,"/zeta",output_dir/f"zeta.dat.{time_step}",(point_num,),f"{time_step},0",f"1,{point_num}")
        data=np.fromfile(output_dir/f"zeta.dat.{time_step}",dtype=np.float32)
        (data==-99999).astype(np.uint8).tofile(output_dir/f"zeta.label.{time_step}")
    print(f"ADCIRC Haiyan转换完成：point_num={point_num},time_step_num={time_step_num},cell_num={cell_num}")


def prepare_adcirc_haiyan_depth(source_path,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    point_num=2187832
    extract_hdf5_float32(source_path,"/depth",output_dir/"depth.dat.0",(point_num,))
    print(f"ADCIRC Haiyan depth转换完成：point_num={point_num}")


def prepare_tsunami(source_path,output_dir,field_vec):
    output_dir.mkdir(parents=True,exist_ok=True)
    point_num=113885
    for field in field_vec:
        output_path=output_dir/f"{field}.dat.0"
        extract_hdf5_native_float32(source_path,f"/Mesh2_{field}",output_path,"0,0",f"1,{point_num}")
        if output_path.stat().st_size!=point_num*np.dtype(np.float32).itemsize:
            raise RuntimeError(f"Tsunami field {field}的文件大小不正确：{output_path.stat().st_size}")
    print(f"Tsunami转换完成：point_num={point_num},field={','.join(field_vec)}")


def prepare_taps(source_path,output_dir,field_vec):
    output_dir.mkdir(parents=True,exist_ok=True)
    point_num=16002271
    prefix="/STREAM_00/CELL_CENTER"
    with tempfile.TemporaryDirectory(prefix="unsz_taps_",dir=output_dir) as temporary_dir:
        temporary_dir=Path(temporary_dir)
        component_vec=[]
        for name in ("XCEN_X","XCEN_Y","XCEN_Z"):
            raw_path=temporary_dir/f"{name}.bin"
            run_h5dump(source_path,f"{prefix}/{name}",raw_path)
            component_vec.append(np.memmap(raw_path,dtype="<f8",mode="r",shape=(point_num,)))
        write_interleaved_float32(output_dir/"coords.dat",component_vec)
    for field in field_vec:
        extract_hdf5_float32(source_path,f"{prefix}/{field}",output_dir/f"{field}.dat.0",(point_num,))
    print(f"TAPS转换完成：point_num={point_num},field={','.join(field_vec)}")


def main():
    data_root=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data")
    parser=argparse.ArgumentParser()
    parser.add_argument("-dataset",required=True,choices=("tpost20","e3sm-h4","e3sm-h0","adcirc-haiyan","adcirc-haiyan-depth","tsunami","taps"))
    parser.add_argument("-data-root",type=Path,default=data_root)
    parser.add_argument("-field",action="append")
    arguments=parser.parse_args()
    if arguments.dataset=="tpost20":
        prepare_tpost20(arguments.data_root/"Tpost20/Tpost20-ascii-all.gz",arguments.data_root/"Tpost20")
    elif arguments.dataset=="e3sm-h4":
        source=arguments.data_root/"yfeng/20250218.v3.LR.F2010.NUG2005-2015_MidEmi.eam.h4.2015-07-10-00000.nc"
        prepare_e3sm_h4(source,arguments.data_root/"E3SM-h4",arguments.field or ["AODDUST","Mass_dst_srf"])
    elif arguments.dataset=="e3sm-h0":
        source=arguments.data_root/"yfeng/20250218.v3.LR.F2010.NUG2005-2015_MidEmi.eam.h0.2015-07.nc"
        prepare_e3sm_h0(source,arguments.data_root/"yfeng/ne30pg2.nc",arguments.data_root/"E3SM-h0",arguments.field or ["PS","TREFHT","PRECC","U10","TAUX","FLNS","AODVIS","CLDTOT"])
    elif arguments.dataset=="adcirc-haiyan":
        source=arguments.data_root/"ADCIRC-Haiyan-1.5km/Haiyan_1.5km.63.nc"
        prepare_adcirc_haiyan(source,arguments.data_root/"ADCIRC-Haiyan-1.5km")
    elif arguments.dataset=="adcirc-haiyan-depth":
        source=arguments.data_root/"ADCIRC-Haiyan-1.5km/Haiyan_1.5km.63.nc"
        prepare_adcirc_haiyan_depth(source,arguments.data_root/"ADCIRC-Haiyan-1.5km")
    elif arguments.dataset=="tsunami":
        prepare_tsunami(arguments.data_root/"Tsunami/ugrid_demo.nc",arguments.data_root/"Tsunami",arguments.field or ["bathy","height"])
    else:
        prepare_taps(arguments.data_root/"TAPS_compressor/map9_8.000038e-02.h5",arguments.data_root/"TAPS",arguments.field or ["TEMPERATURE"])


if __name__=="__main__":
    main()
