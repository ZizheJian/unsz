import argparse,gzip,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from py_codes.prepare_datasets.prepare_dataset_common import data_root_default


def prepare_tpost20(source_path,output_dir):
    output_dir.mkdir(parents=True,exist_ok=True)
    with gzip.open(source_path,"rt") as source_file:
        point_num=int(source_file.readline())
        slice_num=int(source_file.readline())
        cell_num=int(source_file.readline())
        coord_array=np.empty((point_num,3),dtype=np.float32)
        data_array=np.empty((slice_num,point_num),dtype=np.float32)
        for point_id in range(point_num):
            value=np.fromstring(source_file.readline(),sep=" ",dtype=np.float32)
            if value.size!=3+slice_num:
                raise RuntimeError(f"Tpost20第{point_id}个点应有{3+slice_num}个数，实际为{value.size}")
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
    for slice_id in range(slice_num):
        data_array[slice_id].tofile(output_dir/f"scalar.dat.{slice_id}")
    print(f"Tpost20转换完成：point_num={point_num},slice_num={slice_num},cell_num={cell_num}")


def main(command_line_arguments=None):
    parser=argparse.ArgumentParser(description="转换Tpost20数据")
    parser.add_argument("-source-path",type=Path)
    parser.add_argument("-output-dir",type=Path)
    arguments=parser.parse_args(command_line_arguments)
    prepare_tpost20(arguments.source_path or data_root_default/"Tpost20/Tpost20-ascii-all.gz",arguments.output_dir or data_root_default/"Tpost20")


if __name__=="__main__":
    main()
