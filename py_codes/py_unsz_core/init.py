from pathlib import Path
import numpy as np
from py_codes.py_unsz_core.is_abnormal import is_abnormal


def expand_error_bound(error_bound,file_num,name):
    if np.isscalar(error_bound):
        result=[float(error_bound)]*file_num
    else:
        result=[float(value) for value in error_bound]
        if len(result)!=file_num:
            raise ValueError(f"{name}的数量为{len(result)}，数据文件数量为{file_num}，二者应当相同")
    if not np.isfinite(result).all() or np.any(np.asarray(result)<0):
        raise ValueError(f"{name}必须全部是有限的非负数")
    return result

def init(param)->None:# 读取坐标文件，检查数据文件是否存在
    if param.dim<=0:
        raise ValueError(f"dim必须是正数，当前值为{param.dim}")
    if not param.coord_path:
        raise ValueError("coord_path不能为空")
    if not param.original_data_path_vec:
        raise ValueError("original_data_path_vec不能为空")

    coord_path=Path(param.coord_path)
    if not coord_path.is_file():
        raise FileNotFoundError(f"找不到坐标文件：{coord_path}")

    coord_point_bytes=param.dim*param.coord_dtype.itemsize
    coord_file_bytes=coord_path.stat().st_size
    if coord_file_bytes%coord_point_bytes!=0:
        raise ValueError(f"坐标文件大小{coord_file_bytes}不能被dim和数据类型大小的乘积整除（{param.dim}*{param.coord_dtype.itemsize}）")

    param.point_num=coord_file_bytes//coord_point_bytes
    if param.point_num==0:
        raise ValueError(f"坐标文件为空：{coord_path}")
    print(f"点数：{param.point_num}")

    expected_data_bytes=param.point_num*param.data_dtype.itemsize
    for path_string in param.original_data_path_vec:
        data_path=Path(path_string)
        if not data_path.is_file():
            raise FileNotFoundError(f"找不到数据文件：{data_path}")
        if data_path.stat().st_size!=expected_data_bytes:
            raise ValueError(f"数据文件{data_path}的大小为{data_path.stat().st_size}字节，理应是{expected_data_bytes}字节")

    param.coord_array=np.fromfile(coord_path,dtype=param.coord_dtype).reshape(param.point_num,param.dim)
    param.min_coord=param.coord_array.min(axis=0)
    param.max_coord=param.coord_array.max(axis=0)
    print(f"坐标范围：[{", ".join(f"[{minimum},{maximum}]" for minimum,maximum in zip(param.min_coord,param.max_coord))}]")

    data_range_vec=[]
    for data_path in param.original_data_path_vec:
        data=np.fromfile(data_path,dtype=param.data_dtype)
        valid_data=data[~is_abnormal(data)]
        if valid_data.size==0:
            raise ValueError(f"数据文件{data_path}中没有有效数据")
        data_range_vec.append(float(valid_data.max()-valid_data.min()))

    file_num=len(param.original_data_path_vec)
    if param.rel_eb is None and param.abs_eb is None:
        raise ValueError("rel_eb和abs_eb不能同时为空")
    if param.rel_eb is not None:
        print(f"相对误差界限：{param.rel_eb}")
        param.rel_eb=[float(param.rel_eb)]*len(param.original_data_path_vec)
        param.abs_eb=[param.rel_eb[i]*data_range_vec[i] for i in range(file_num)]
    elif param.abs_eb is not None:
        print(f"绝对误差界限：{param.abs_eb}")
        param.abs_eb=expand_error_bound(param.abs_eb,file_num,"abs_eb")
        param.rel_eb=[param.abs_eb[i]/data_range_vec[i] for i in range(file_num)]