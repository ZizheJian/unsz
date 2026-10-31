import argparse,json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


def write_mesh(mesh_source_path,mesh_dir,point_num):
    mesh_dir.mkdir(parents=True,exist_ok=True)
    with Dataset(mesh_source_path,"r") as dataset:
        if len(dataset.dimensions["nCells"])!=point_num:
            raise RuntimeError("UMC网格点数与月度数据不一致")
        if len(dataset.dimensions["maxEdges"])!=7:
            raise RuntimeError("UMC Ocean并行代码要求maxEdges=7")
        coordinate_path=mesh_dir/"coords.dat"
        with coordinate_path.open("wb") as output_file:
            for begin in range(0,point_num,1<<18):
                end=min(begin+(1<<18),point_num)
                coordinates=np.column_stack(tuple(np.asarray(dataset.variables[name][begin:end],dtype=np.float64) for name in("xCell","yCell","zCell")))
                coordinates.tofile(output_file)
        connection_path=mesh_dir/"conn.dat"
        with connection_path.open("wb") as output_file:
            cells_on_cell=dataset.variables["cellsOnCell"]
            for begin in range(0,point_num,1<<18):
                end=min(begin+(1<<18),point_num)
                np.asarray(cells_on_cell[begin:end,:],dtype=np.int32).tofile(output_file)


def prepare(monthly_data_dir,mesh_source_path,output_dir):
    manifest=json.loads((monthly_data_dir/"monthly_fields.json").read_text())
    point_num=int(manifest["point_num"])
    slice_num=int(manifest["slice_num"])
    fields=manifest["single_level_fields"]+manifest["multi_level_level0_fields"]
    if len(fields)!=91:
        raise RuntimeError(f"UMC Ocean-large要求91个field，实际为{len(fields)}")
    output_dir.mkdir(parents=True,exist_ok=True)
    write_mesh(mesh_source_path,output_dir/"mesh",point_num)
    (output_dir/"mesh"/"partition").mkdir(exist_ok=True)
    (output_dir/"varnames_91.txt").write_text("\n".join(fields)+"\n")
    for slice_id in range(slice_num):
        step_dir=output_dir/f"step_{slice_id}"
        step_dir.mkdir(exist_ok=True)
        for field in fields:
            source=(monthly_data_dir/f"{field}.dat.{slice_id}").resolve()
            if source.stat().st_size!=point_num*np.dtype(np.float32).itemsize:
                raise RuntimeError(f"UMC field大小不正确：{source}")
            target=step_dir/field
            if target.is_symlink() or target.exists():
                if target.resolve()!=source:
                    raise RuntimeError(f"UMC field链接指向错误：{target}")
            else:
                target.symlink_to(source)
    for name in("idw","sz3ori","sz3rod","zfpori","zfprod"):
        (output_dir/"compressed"/name).mkdir(parents=True,exist_ok=True)
    print(f"UMC Ocean-large布局完成：root={output_dir},point_num={point_num},field_num={len(fields)},slice_num={slice_num}")


def main():
    parser=argparse.ArgumentParser(description="生成UMC Ocean-large并行程序所需的数据布局")
    parser.add_argument("-monthly-data-dir",type=Path,required=True)
    parser.add_argument("-mesh-source-path",type=Path,required=True)
    parser.add_argument("-output-dir",type=Path,required=True)
    arguments=parser.parse_args()
    prepare(arguments.monthly_data_dir,arguments.mesh_source_path,arguments.output_dir)


if __name__=="__main__":
    main()
