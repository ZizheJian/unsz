import argparse,json,sys,time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from py_codes.prepare_datasets.prepare_dataset_common import data_root_default
from py_codes.prepare_datasets.prepare_ocean_large import monthly_invalid_value,prepare_mesh


default_mesh_path=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/source/MPAS-O_V6.0_RRS18to6/oRRS18to6v3.171116.nc")
default_monthly_source_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/mpaso")
default_output_dir=data_root_default/"Ocean-large/tensor_data"
conversion_progress_version=1


def write_json_atomic(path,value):
    temporary_path=path.with_name(path.name+".tmp")
    temporary_path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    temporary_path.replace(path)


def find_fields(dataset,selected_fields):
    result=[]
    for name,variable in dataset.variables.items():
        if variable.dtype!=np.dtype(np.float32):
            continue
        if variable.dimensions==("Time","nCells"):
            height_num=1
            level_kind="single"
        elif variable.dimensions==("Time","nCells","nVertLevels"):
            height_num=80
            level_kind="center"
        elif variable.dimensions==("Time","nCells","nVertLevelsP1"):
            height_num=81
            level_kind="interface"
        else:
            continue
        if selected_fields and name not in selected_fields:
            continue
        result.append({"name":name,"height_num":height_num,"level_kind":level_kind})
    if selected_fields:
        missing=sorted(set(selected_fields)-{field["name"] for field in result})
        if missing:
            raise RuntimeError("没有找到指定的Ocean-large field："+",".join(missing))
    elif [sum(field["level_kind"]==kind for field in result) for kind in("single","center","interface")]!=[58,33,4]:
        raise RuntimeError("Ocean-large完整field结构不是58个单层、33个中心层和4个界面层")
    return result


def fill_value(variable):
    value=getattr(variable,"_FillValue",None)
    return None if value is None else np.float32(value)


def validate_and_replace(values,active_mask,variable,field,time_index,cell_begin):
    finite=np.isfinite(values)
    fill=fill_value(variable)
    explicit_invalid=~finite
    if fill is not None:
        explicit_invalid|=values==fill
    active_finite=active_mask&~explicit_invalid
    collision=active_finite&(values==monthly_invalid_value)
    if np.any(collision):
        local=np.argwhere(collision)[0]
        raise RuntimeError(f"有效有限值与-99999哨兵冲突：field={field},time={time_index},cell={cell_begin+int(local[0])},height={int(local[1])}")
    invalid=~active_mask|explicit_invalid
    if np.any(invalid):
        values[invalid]=monthly_invalid_value
    return int(np.count_nonzero(~active_mask)),int(np.count_nonzero(explicit_invalid&active_mask))


def convert_field(dataset,field,time_index,point_num,max_level_cell,output_dir,block_point_num,completed_group):
    name=field["name"]
    height_num=field["height_num"]
    variable=dataset.variables[name]
    output_paths=[output_dir/f"{name}.dat.{time_index*height_num+height_index}" for height_index in range(height_num)]
    expected_bytes=point_num*np.dtype(np.float32).itemsize
    if completed_group is not None and all(path.is_file() and path.stat().st_size==expected_bytes for path in output_paths):
        return completed_group["structural"],completed_group["explicit"],True
    temporary_paths=[path.with_name(path.name+".tmp") for path in output_paths]
    structural_invalid_num=0
    explicit_invalid_num=0
    with ExitStack() as stack:
        output_files=[stack.enter_context(path.open("wb")) for path in temporary_paths]
        for begin in range(0,point_num,block_point_num):
            end=min(begin+block_point_num,point_num)
            if height_num==1:
                values=np.asarray(variable[0,begin:end],dtype=np.float32).reshape(end-begin,1)
                active_mask=np.ones(values.shape,dtype=bool)
            else:
                values=np.asarray(variable[0,begin:end,:],dtype=np.float32)
                level_offset=1 if field["level_kind"]=="interface" else 0
                active_mask=np.arange(height_num)[None,:]<max_level_cell[begin:end,None]+level_offset
            structural_invalid,explicit_invalid=validate_and_replace(values,active_mask,variable,name,time_index,begin)
            structural_invalid_num+=structural_invalid
            explicit_invalid_num+=explicit_invalid
            for height_index,output_file in enumerate(output_files):
                values[:,height_index].tofile(output_file)
    for path in temporary_paths:
        if path.stat().st_size!=expected_bytes:
            raise RuntimeError(f"输出切片大小不正确：{path}")
    for temporary_path,output_path in zip(temporary_paths,output_paths):
        temporary_path.replace(output_path)
    return structural_invalid_num,explicit_invalid_num,False


def prepare(arguments):
    conversion_start=time.perf_counter()
    try:
        from netCDF4 import Dataset
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("转换Ocean-large需要Python包netCDF4") from error
    monthly_paths=sorted(arguments.monthly_source_dir.glob("*.timeSeriesStatsMonthly.*.nc"))
    if len(monthly_paths)!=12:
        raise RuntimeError(f"Ocean-large月度输出必须恰好包含12个NetCDF文件，实际为{len(monthly_paths)}")
    arguments.output_dir.mkdir(parents=True,exist_ok=True)
    with Dataset(arguments.mesh_path,"r") as mesh:
        full_point_num=len(mesh.dimensions["nCells"])
        point_num=full_point_num if arguments.point_limit is None else min(arguments.point_limit,full_point_num)
        if point_num==full_point_num:
            _,edge_num=prepare_mesh(mesh,arguments.output_dir)
        else:
            coordinates=np.column_stack(tuple(np.asarray(mesh.variables[name][:point_num],dtype=np.float32) for name in("xCell","yCell","zCell")))
            coordinates.tofile(arguments.output_dir/"coords.dat")
            edge_num=0
        max_level_cell=np.asarray(mesh.variables["maxLevelCell"][:point_num],dtype=np.uint32)
        if np.any(max_level_cell>80):
            raise RuntimeError("maxLevelCell超过80")
        max_level_cell.tofile(arguments.output_dir/"active_level_count.u32")
        bottom_depth=np.asarray(mesh.variables["refBottomDepth"][:],dtype=np.float64)
    progress_path=arguments.output_dir/"conversion_progress.json"
    if progress_path.exists():
        progress=json.loads(progress_path.read_text())
        if progress.get("version")!=conversion_progress_version or progress.get("point_num")!=point_num:
            raise RuntimeError("现有Ocean-large转换进度与当前转换器或point_num不兼容，请使用新的输出目录")
    else:
        progress={"version":conversion_progress_version,"point_num":point_num,"groups":{}}
    center_depth=(np.concatenate((np.zeros(1,dtype=np.float64),bottom_depth[:-1]))+bottom_depth)/2
    interface_depth=np.concatenate((np.zeros(1,dtype=np.float64),bottom_depth))
    center_depth.tofile(arguments.output_dir/"height_center_80.f64")
    interface_depth.tofile(arguments.output_dir/"height_interface_81.f64")
    with Dataset(monthly_paths[0],"r") as dataset:
        fields=find_fields(dataset,set(arguments.field or[]))
    time_coordinates=[]
    invalid_statistics={field["name"]:{"structural":0,"explicit":0} for field in fields}
    for time_index,monthly_path in enumerate(monthly_paths):
        with Dataset(monthly_path,"r") as dataset:
            dataset.set_auto_mask(False)
            if len(dataset.dimensions["nCells"])!=full_point_num:
                raise RuntimeError(f"{monthly_path.name}的nCells与网格不一致")
            time_coordinate=float(dataset.variables["timeMonthly_avg_daysSinceStartOfSim"][0])
            if not np.isfinite(time_coordinate):
                raise RuntimeError(f"{monthly_path.name}的时间坐标不是有限值")
            time_coordinates.append(time_coordinate)
            if find_fields(dataset,set(arguments.field or[]))!=fields:
                raise RuntimeError(f"{monthly_path.name}的field结构与第一个月不一致")
            for field in fields:
                group_key=f"{time_index}:{field['name']}"
                group_start=time.perf_counter()
                structural_invalid,explicit_invalid,skipped=convert_field(dataset,field,time_index,point_num,max_level_cell,arguments.output_dir,arguments.block_point_num,progress["groups"].get(group_key))
                group_time=time.perf_counter()-group_start
                invalid_statistics[field["name"]]["structural"]+=structural_invalid
                invalid_statistics[field["name"]]["explicit"]+=explicit_invalid
                if not skipped:
                    progress["groups"][group_key]={"structural":structural_invalid,"explicit":explicit_invalid}
                    write_json_atomic(progress_path,progress)
                output_gib=point_num*field["height_num"]*np.dtype(np.float32).itemsize/2**30
                print(f"Ocean-large tensor切片{'已跳过' if skipped else '已转换'}：time={time_index},field={field['name']},height_num={field['height_num']},output={output_gib:.3f} GiB,time={group_time:.3f} s,output_rate={output_gib/group_time:.3f} GiB/s",flush=True)
    time_coordinates=np.asarray(time_coordinates,dtype=np.float64)
    if not np.all(np.diff(time_coordinates)>0):
        raise RuntimeError("月份坐标必须严格递增")
    time_coordinates.tofile(arguments.output_dir/"time_12.f64")
    metadata_fields={}
    for field in fields:
        item={"shape":[len(monthly_paths),field["height_num"]],"axis0_coordinates":"time_12.f64"}
        if field["level_kind"]=="center":
            item|={"axis1_coordinates":"height_center_80.f64","active_level_count_file":"active_level_count.u32","active_level_count_offset":0}
        elif field["level_kind"]=="interface":
            item|={"axis1_coordinates":"height_interface_81.f64","active_level_count_file":"active_level_count.u32","active_level_count_offset":1}
        metadata_fields[field["name"]]=item
    tensor_metadata={"version":1,"fields":metadata_fields}
    write_json_atomic(arguments.output_dir/"tensor_metadata.json",tensor_metadata)
    manifest={"point_num":point_num,"edge_num":edge_num,"time_num":len(monthly_paths),"slice_num":sum(len(monthly_paths)*field["height_num"] for field in fields),"invalid_value":float(monthly_invalid_value),"field_num":len(fields),"fields":fields,"invalid_statistics":invalid_statistics}
    write_json_atomic(arguments.output_dir/"ocean_large_tensor.json",manifest)
    print(f"Ocean-large tensor转换完成：point_num={point_num},field_num={len(fields)},slice_num={manifest['slice_num']},time={time.perf_counter()-conversion_start:.3f} s,output_dir={arguments.output_dir}")


def main(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="将完整Ocean-large转换为每个field独立的[time,height,point]切片")
    parser.add_argument("-mesh-path",type=Path,default=default_mesh_path)
    parser.add_argument("-monthly-source-dir",type=Path,default=default_monthly_source_dir)
    parser.add_argument("-output-dir",type=Path,default=default_output_dir)
    parser.add_argument("-field",action="append")
    parser.add_argument("-point-limit",type=int,help="仅用于小型转换回归；正式数据不要设置")
    parser.add_argument("-block-point-num",type=int,default=1<<17)
    arguments=parser.parse_args(command_line_arguments)
    if arguments.point_limit is not None and arguments.point_limit<=0:
        parser.error("-point-limit必须是正整数")
    if arguments.block_point_num<=0:
        parser.error("-block-point-num必须是正整数")
    prepare(arguments)


if __name__=="__main__":
    main()
