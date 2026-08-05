from pathlib import Path
import numpy as np

def init(self)->None:
    """读取坐标并检查输入数据文件。"""
    if self.dim<=0:
        raise ValueError(f"dim必须是正数，当前值为{self.dim}")
    if not self.coord_path:
        raise ValueError("coord_path不能为空")
    if not self.original_data_path_vec:
        raise ValueError("original_data_path_vec不能为空")

    coord_path=Path(self.coord_path)
    if not coord_path.is_file():
        raise FileNotFoundError(f"找不到坐标文件：{coord_path}")

    coord_dtype=np.dtype(self.coord_dtype)
    data_dtype=np.dtype(self.data_dtype)
    coord_point_bytes=self.dim*coord_dtype.itemsize
    coord_file_bytes=coord_path.stat().st_size
    if coord_file_bytes%coord_point_bytes!=0:
        raise ValueError(f"坐标文件大小{coord_file_bytes}不能被dim和数据类型大小的乘积整除（{self.dim}*{coord_dtype.itemsize}）")

    self.point_num=coord_file_bytes//coord_point_bytes
    if self.point_num==0:
        raise ValueError(f"坐标文件为空：{coord_path}")

    expected_data_bytes=self.point_num*data_dtype.itemsize
    for path_string in self.original_data_path_vec:
        data_path=Path(path_string)
        if not data_path.is_file():
            raise FileNotFoundError(f"找不到数据文件：{data_path}")
        if data_path.stat().st_size!=expected_data_bytes:
            raise ValueError(f"数据文件{data_path}的大小为{data_path.stat().st_size}字节；{self.point_num}个点应当占用{expected_data_bytes}字节")

    self.coord_array=np.fromfile(coord_path,dtype=coord_dtype).reshape(self.point_num,self.dim)
    self.min_coord=self.coord_array.min(axis=0)
    self.max_coord=self.coord_array.max(axis=0)

    print(f"点数：{self.point_num}")
    ranges=", ".join(f"[{minimum},{maximum}]" for minimum,maximum in zip(self.min_coord,self.max_coord))
    print(f"坐标范围：[{ranges}]")
