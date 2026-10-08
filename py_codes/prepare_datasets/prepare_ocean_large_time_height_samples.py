import argparse,json
from pathlib import Path

import numpy as np


default_mesh_path=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/source/MPAS-O_V6.0_RRS18to6/oRRS18to6v3.171116.nc")
default_monthly_source_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/mpaso")
default_output_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/tensor-data/Ocean-large-time-height-samples")


def select_full_depth_cells(max_level_cell,sample_num,region_num):
    full_depth_ids=np.flatnonzero(max_level_cell==max_level_cell.max())
    if sample_num%region_num:
        raise ValueError("sample-num必须能被region-num整除")
    quota=sample_num//region_num
    sample_ids=[]
    windows=[]
    for region_id in range(region_num):
        region_begin=len(full_depth_ids)*region_id//region_num
        region_end=len(full_depth_ids)*(region_id+1)//region_num
        region_ids=full_depth_ids[region_begin:region_end]
        if len(region_ids)<quota:
            raise RuntimeError("完整深度cell不足以完成分区采样")
        spans=region_ids[quota-1:]-region_ids[:len(region_ids)-quota+1]
        window_offset=int(np.argmin(spans))
        window_ids=region_ids[window_offset:window_offset+quota]
        sample_begin=len(sample_ids)
        sample_ids.extend(int(value) for value in window_ids)
        windows.append({
            "region_id":region_id,
            "sample_begin":sample_begin,
            "sample_end":sample_begin+quota,
            "cell_begin":int(window_ids[0]),
            "cell_end":int(window_ids[-1])+1,
        })
    result=np.asarray(sample_ids,dtype=np.uint32)
    if np.any(result[1:]<=result[:-1]):
        raise RuntimeError("采样cell ID必须严格递增")
    return result,windows


def output_name(field):
    prefix="timeMonthly_avg_"
    return field[len(prefix):] if field.startswith(prefix) else field


def prepare(arguments):
    try:
        from netCDF4 import Dataset
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("生成Ocean-large时间-高度样本需要netCDF4") from error
    monthly_paths=sorted(arguments.monthly_source_dir.glob("*.timeSeriesStatsMonthly.*.nc"))
    if len(monthly_paths)!=12:
        raise RuntimeError(f"应找到12个月度NetCDF，实际找到{len(monthly_paths)}个")
    arguments.output_dir.mkdir(parents=True,exist_ok=True)
    with Dataset(arguments.mesh_path,"r") as mesh:
        max_level_cell=np.asarray(mesh.variables["maxLevelCell"][:],dtype=np.int32)
        if max_level_cell.max()!=80:
            raise RuntimeError(f"最高垂直层数应为80，实际为{max_level_cell.max()}")
        sample_ids,windows=select_full_depth_cells(max_level_cell,arguments.sample_num,arguments.region_num)
        coordinate_indices=sample_ids.astype(np.int64)
        coordinates=np.column_stack(tuple(np.asarray(mesh.variables[name][coordinate_indices],dtype=np.float64) for name in("xCell","yCell","zCell")))
        bottom_depth=np.asarray(mesh.variables["refBottomDepth"][:],dtype=np.float64)
    center_depth=(np.concatenate((np.zeros(1,dtype=np.float64),bottom_depth[:-1]))+bottom_depth)/2
    interface_depth=np.concatenate((np.zeros(1,dtype=np.float64),bottom_depth))
    sample_ids.tofile(arguments.output_dir/"sample_cell_ids.u32")
    coordinates.tofile(arguments.output_dir/"sample_coordinates.f64")
    center_depth.tofile(arguments.output_dir/"height_center_80.f64")
    interface_depth.tofile(arguments.output_dir/"height_interface_81.f64")
    with Dataset(monthly_paths[0],"r") as dataset:
        fields=[]
        for field,variable in dataset.variables.items():
            if variable.dtype!=np.dtype(np.float32):
                continue
            if variable.dimensions in(("Time","nCells","nVertLevels"),("Time","nCells","nVertLevelsP1")):
                fields.append(field)
    if len(fields)!=37:
        raise RuntimeError(f"应找到37个垂直float32 field，实际找到{len(fields)}个")
    outputs={}
    statistics={}
    for field in fields:
        with Dataset(monthly_paths[0],"r") as dataset:
            height_num=dataset.variables[field].shape[2]
        path=arguments.output_dir/f"{output_name(field)}.f32"
        outputs[field]={
            "path":path,
            "height_num":height_num,
            "values":np.memmap(path,dtype=np.float32,mode="w+",shape=(len(monthly_paths),height_num,arguments.sample_num)),
        }
        statistics[field]={"minimum":np.inf,"maximum":-np.inf}
    for time_index,monthly_path in enumerate(monthly_paths):
        with Dataset(monthly_path,"r") as dataset:
            dataset.set_auto_mask(False)
            for field,output in outputs.items():
                variable=dataset.variables[field]
                values=output["values"]
                for window in windows:
                    block=np.asarray(variable[0,window["cell_begin"]:window["cell_end"],:],dtype=np.float32)
                    local_ids=sample_ids[window["sample_begin"]:window["sample_end"]].astype(np.int64)-window["cell_begin"]
                    sampled=block[local_ids,:].T
                    if not np.all(np.isfinite(sampled)):
                        raise RuntimeError(f"采样值包含NaN或Inf：field={field},time_index={time_index}")
                    values[time_index,:,window["sample_begin"]:window["sample_end"]]=sampled
                    statistics[field]["minimum"]=min(statistics[field]["minimum"],float(sampled.min()))
                    statistics[field]["maximum"]=max(statistics[field]["maximum"],float(sampled.max()))
        print(f"Ocean-large时间-高度样本已读取：time_index={time_index},source={monthly_path.name}",flush=True)
    manifest_lines=[]
    metadata_fields=[]
    for field,output in outputs.items():
        output["values"].flush()
        del output["values"]
        shape=[len(monthly_paths),output["height_num"],arguments.sample_num]
        expected_bytes=int(np.prod(shape))*np.dtype(np.float32).itemsize
        if output["path"].stat().st_size!=expected_bytes:
            raise RuntimeError(f"输出文件大小不正确：{output['path']}")
        height_path=arguments.output_dir/("height_center_80.f64" if output["height_num"]==80 else "height_interface_81.f64")
        manifest_lines.append(f"{field}\t{output['path']}\t{height_path}\t{shape[0]}\t{shape[1]}\t{shape[2]}")
        metadata_fields.append({
            "name":field,
            "path":str(output["path"]),
            "data_type":"f32",
            "shape":shape,
            "height_coordinates":str(height_path),
            "sample_minimum":statistics[field]["minimum"],
            "sample_maximum":statistics[field]["maximum"],
        })
    manifest_path=arguments.output_dir/"manifest.tsv"
    manifest_path.write_text("field\tpath\theight_path\ttime_num\theight_num\tsample_num\n"+"\n".join(manifest_lines)+"\n")
    metadata={
        "name":"Ocean-large时间-高度可预测性样本",
        "note":"轴顺序是[time,height,sampled_unstructured_cell]。第三维仅是一组固定非结构cell样本，不是规则空间轴。所有样本cell的maxLevelCell均为80。",
        "axis_order":["time","height","sampled_unstructured_cell"],
        "sample_num":arguments.sample_num,
        "sample_cell_ids":str(arguments.output_dir/"sample_cell_ids.u32"),
        "sample_coordinates":str(arguments.output_dir/"sample_coordinates.f64"),
        "sample_coordinate_shape":[arguments.sample_num,3],
        "selection":"将全部完整深度cell按原始ID分成8个区域，每区选择ID跨度最小的512个cell。",
        "windows":windows,
        "fields":metadata_fields,
    }
    (arguments.output_dir/"metadata.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
    print(f"Ocean-large时间-高度样本生成完成：output_dir={arguments.output_dir},field_num={len(fields)},sample_num={arguments.sample_num}")


def main():
    parser=argparse.ArgumentParser(description="生成Ocean-large时间与非均匀高度可预测性实验样本")
    parser.add_argument("-mesh-path",type=Path,default=default_mesh_path)
    parser.add_argument("-monthly-source-dir",type=Path,default=default_monthly_source_dir)
    parser.add_argument("-output-dir",type=Path,default=default_output_dir)
    parser.add_argument("-sample-num",type=int,default=4096)
    parser.add_argument("-region-num",type=int,default=8)
    arguments=parser.parse_args()
    if arguments.sample_num<=0 or arguments.region_num<=0:
        parser.error("sample-num和region-num必须为正数")
    prepare(arguments)


if __name__=="__main__":
    main()
