import sys,tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from py_codes.prepare_datasets.prepare_dataset_common import run_h5dump,write_interleaved_float32


def prepare_scrip_cell_edges(grid_path,output_path):
    point_num=21600
    corner_num=4
    with tempfile.TemporaryDirectory(prefix="unsz_scrip_",dir=output_path.parent) as temporary_dir:
        temporary_dir=Path(temporary_dir)
        corner_component_vec=[]
        for name in("grid_corner_lat","grid_corner_lon"):
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
        _,first,count=np.unique(edge_key,return_index=True,return_counts=True)
        if np.any(count!=2):
            raise RuntimeError(f"E3SM网格存在{np.count_nonzero(count!=2)}条不恰好属于两个cell的边")
        np.column_stack((cell_id[first],cell_id[first+1])).astype(np.uint32).tofile(output_path)


def prepare_e3sm(source_path,grid_path,output_dir,field_vec,stream_name):
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
                data.tofile(output_dir/f"{field}.dat.{time_step}")
        print(f"E3SM {stream_name}转换完成：point_num={point_num},time_step_num={time_step_num},field={','.join(field_vec)}")
    finally:
        dataset.close()
