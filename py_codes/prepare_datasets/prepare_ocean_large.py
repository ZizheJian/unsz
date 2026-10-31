import argparse,contextlib,json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from py_codes.prepare_datasets.prepare_dataset_common import data_root_default,write_interleaved_float32


monthly_single_level_field_num=58
monthly_multi_level_field_num=33
monthly_invalid_value=np.float32(-99999)


def prepare_mesh(dataset,output_dir):
    point_num=len(dataset.dimensions["nCells"])
    edge_num=len(dataset.dimensions["nEdges"])
    coord_path=output_dir/"coords.dat"
    expected_coord_bytes=point_num*3*np.dtype(np.float32).itemsize
    if coord_path.exists():
        if coord_path.stat().st_size!=expected_coord_bytes:
            raise RuntimeError(f"coords.dat大小不一致：expected={expected_coord_bytes},actual={coord_path.stat().st_size}")
    else:
        write_interleaved_float32(coord_path,tuple(dataset.variables[name] for name in("xCell","yCell","zCell")))
    conn_path=output_dir/"conn.dat"
    if conn_path.exists():
        if conn_path.stat().st_size%(2*np.dtype(np.uint32).itemsize)!=0:
            raise RuntimeError("conn.dat大小不是两端点uint32记录的整数倍")
        valid_edge_num=conn_path.stat().st_size//(2*np.dtype(np.uint32).itemsize)
    else:
        valid_edge_num=0
        cells_on_edge=dataset.variables["cellsOnEdge"]
        with conn_path.open("wb") as output_file:
            for begin in range(0,edge_num,1<<20):
                values=np.asarray(cells_on_edge[begin:min(begin+(1<<20),edge_num),:],dtype=np.int64)
                values=values[np.all(values>0,axis=1)]-1
                if values.size and(values.min()<0 or values.max()>=point_num):
                    raise RuntimeError("cellsOnEdge包含越界cell编号")
                values.astype(np.uint32,copy=False).tofile(output_file)
                valid_edge_num+=values.shape[0]
    return point_num,valid_edge_num


def monthly_field_lists(dataset):
    single_level=[]
    multi_level=[]
    for name,variable in dataset.variables.items():
        if variable.dtype!=np.dtype(np.float32):
            continue
        if variable.dimensions==("Time","nCells"):
            single_level.append(name)
        elif variable.dimensions==("Time","nCells","nVertLevels"):
            multi_level.append(name)
    if len(single_level)!=monthly_single_level_field_num or len(multi_level)!=monthly_multi_level_field_num:
        raise RuntimeError(f"Ocean-large月度field结构不一致：单层={len(single_level)}(应为{monthly_single_level_field_num}),80层={len(multi_level)}(应为{monthly_multi_level_field_num})")
    return single_level,multi_level


def read_monthly_field(variable,multi_level):
    data=np.asarray(variable[0,:,0] if multi_level else variable[0,:],dtype=np.float32)
    invalid_mask=~np.isfinite(data)
    fill_value=getattr(variable,"_FillValue",None)
    if fill_value is not None:
        invalid_mask|=data==np.float32(fill_value)
    invalid_num=int(np.count_nonzero(invalid_mask))
    if invalid_num:
        data[invalid_mask]=monthly_invalid_value
    return data,invalid_num


def prepare_ocean_large_monthly(source_dir,mesh_source_path,output_dir):
    try:
        from netCDF4 import Dataset
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("转换Ocean-large需要Python包netCDF4") from error
    source_path_vec=sorted(source_dir.glob("*.timeSeriesStatsMonthly.*.nc"))
    if len(source_path_vec)!=12:
        raise RuntimeError(f"Ocean-large月度输出必须恰好包含12个NetCDF文件，实际为{len(source_path_vec)}")
    output_dir.mkdir(parents=True,exist_ok=True)
    with Dataset(mesh_source_path,"r") as mesh_dataset:
        point_num,valid_edge_num=prepare_mesh(mesh_dataset,output_dir)
    field_manifest=None
    invalid_num_by_field={}
    for slice_id,source_path in enumerate(source_path_vec):
        with Dataset(source_path,"r") as dataset:
            dataset.set_auto_mask(False)
            if len(dataset.dimensions["nCells"])!=point_num:
                raise RuntimeError(f"{source_path.name}的nCells与网格文件不一致")
            single_level,multi_level=monthly_field_lists(dataset)
            current_manifest={"single_level":single_level,"multi_level_level0":multi_level}
            if field_manifest is None:
                field_manifest=current_manifest
            elif current_manifest!=field_manifest:
                raise RuntimeError(f"{source_path.name}的月度field列表与第一个文件不一致")
            for field in single_level:
                data,invalid_num=read_monthly_field(dataset.variables[field],False)
                data.tofile(output_dir/f"{field}.dat.{slice_id}")
                invalid_num_by_field[field]=invalid_num_by_field.get(field,0)+invalid_num
            for field in multi_level:
                data,invalid_num=read_monthly_field(dataset.variables[field],True)
                data.tofile(output_dir/f"{field}.dat.{slice_id}")
                invalid_num_by_field[field]=invalid_num_by_field.get(field,0)+invalid_num
        print(f"Ocean-large月度输出已转换：slice={slice_id},source={source_path.name}")
    manifest={
        "selection":"NetCDF中全部58个(Time,nCells)float32变量，以及全部33个(Time,nCells,nVertLevels)float32变量的第0层；UMC公开仓库未提供varnames_91.txt，因此这是根据91=58+33恢复的推定口径",
        "slice_num":len(source_path_vec),
        "point_num":point_num,
        "edge_num":valid_edge_num,
        "invalid_value":float(monthly_invalid_value),
        "invalid_num_by_field":{field:invalid_num for field,invalid_num in invalid_num_by_field.items() if invalid_num},
        "single_level_fields":field_manifest["single_level"],
        "multi_level_level0_fields":field_manifest["multi_level_level0"]
    }
    (output_dir/"monthly_fields.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(f"Ocean-large月度输出转换完成：point_num={point_num},edge_num={valid_edge_num},field_num={len(field_manifest['single_level'])+len(field_manifest['multi_level_level0'])},slice_num={len(source_path_vec)}")


def prepare_ocean_large(source_path,output_dir,field_vec,level):
    try:
        from netCDF4 import Dataset
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("转换Ocean-large需要Python包netCDF4") from error
    output_dir.mkdir(parents=True,exist_ok=True)
    with Dataset(source_path,"r") as dataset:
        dataset.set_auto_mask(False)
        point_num,valid_edge_num=prepare_mesh(dataset,output_dir)
        for field in field_vec:
            variable=dataset.variables[field]
            if variable.dimensions[:2]!=("Time","nCells") or len(variable.dimensions)not in(2,3):
                raise RuntimeError(f"field {field}必须是(Time,nCells)或(Time,nCells,level)，实际为{variable.dimensions}")
            if len(variable.dimensions)==3:
                level_num=variable.shape[2]
                level_dimension=variable.dimensions[2]
                if level is not None:
                    if level<0 or level>=level_num:
                        raise RuntimeError(f"三维field {field}的-level范围为0～{level_num-1}")
                    data=np.asarray(variable[0,:,level],dtype=np.float32)
                    if not np.all(np.isfinite(data)):
                        raise RuntimeError(f"field {field}包含{np.count_nonzero(~np.isfinite(data))}个非有限值")
                    output_name=f"{field}_level{level}"
                    data.tofile(output_dir/f"{output_name}.dat.0")
                    print(f"Ocean-large field范围：field={output_name},min={data.min()},max={data.max()}")
                    continue
                coordinate_variable_name="refBottomDepth" if level_dimension=="nVertLevels" else level_dimension
                if coordinate_variable_name not in dataset.variables:
                    raise RuntimeError(f"三维field {field}的层维度{level_dimension}没有对应坐标变量")
                level_coordinates=np.asarray(dataset.variables[coordinate_variable_name][:],dtype=np.float64)
                if level_coordinates.shape!=(level_num,) or not np.all(np.isfinite(level_coordinates)):
                    raise RuntimeError(f"层坐标{coordinate_variable_name}必须包含{level_num}个有限float64值")
                if level_num>1 and not(np.all(np.diff(level_coordinates)>0) or np.all(np.diff(level_coordinates)<0)):
                    raise RuntimeError(f"层坐标{coordinate_variable_name}必须严格单调")
                coordinate_name=f"{coordinate_variable_name}.dat"
                level_coordinates.tofile(output_dir/coordinate_name)
                minimum=np.full(level_num,np.inf,dtype=np.float32)
                maximum=np.full(level_num,-np.inf,dtype=np.float32)
                with contextlib.ExitStack() as stack:
                    output_files=[stack.enter_context((output_dir/f"{field}.dat.{level_id}").open("wb")) for level_id in range(level_num)]
                    for begin in range(0,point_num,1<<17):
                        end=min(begin+(1<<17),point_num)
                        data=np.asarray(variable[0,begin:end,:],dtype=np.float32)
                        if not np.all(np.isfinite(data)):
                            raise RuntimeError(f"field {field}的cell区间[{begin},{end})包含{np.count_nonzero(~np.isfinite(data))}个非有限值")
                        minimum=np.minimum(minimum,np.min(data,axis=0))
                        maximum=np.maximum(maximum,np.max(data,axis=0))
                        for level_id,output_file in enumerate(output_files):
                            data[:,level_id].tofile(output_file)
                for level_id in range(level_num):
                    print(f"Ocean-large field范围：field={field},slice={level_id},coordinate={level_coordinates[level_id]},min={minimum[level_id]},max={maximum[level_id]}")
                print(f"Ocean-large三维field已按层转为slice_id：field={field},slice_num={level_num},coordinate_file={coordinate_name}")
            else:
                if level is not None:
                    raise RuntimeError(f"二维field {field}不能指定-level")
                data=np.asarray(variable[0,:],dtype=np.float32)
                if not np.all(np.isfinite(data)):
                    raise RuntimeError(f"field {field}包含{np.count_nonzero(~np.isfinite(data))}个非有限值")
                data.tofile(output_dir/f"{field}.dat.0")
                print(f"Ocean-large field范围：field={field},min={data.min()},max={data.max()}")
    print(f"Ocean-large转换完成：point_num={point_num},edge_num={valid_edge_num},field={','.join(field_vec)}")


def main(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="转换Ocean-large数据")
    source_group=parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("-source-path",type=Path)
    source_group.add_argument("-monthly-source-dir",type=Path)
    parser.add_argument("-mesh-source-path",type=Path)
    parser.add_argument("-output-dir",type=Path)
    parser.add_argument("-field",action="append")
    parser.add_argument("-level",type=int)
    arguments=parser.parse_args(command_line_arguments)
    if arguments.monthly_source_dir is not None:
        if arguments.field is not None or arguments.level is not None:
            parser.error("-monthly-source-dir不能与-field或-level同时使用")
        if arguments.mesh_source_path is None:
            parser.error("-monthly-source-dir必须同时指定-mesh-source-path")
        prepare_ocean_large_monthly(arguments.monthly_source_dir,arguments.mesh_source_path,arguments.output_dir or data_root_default/"Ocean-large/monthly_data")
    else:
        if arguments.mesh_source_path is not None:
            parser.error("-mesh-source-path只能与-monthly-source-dir同时使用")
        prepare_ocean_large(arguments.source_path,arguments.output_dir or data_root_default/"Ocean-large/data",arguments.field or["temperatureSurfaceValue"],arguments.level)


if __name__=="__main__":
    main()
