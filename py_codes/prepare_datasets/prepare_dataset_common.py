import shutil,subprocess,tempfile
from pathlib import Path

import numpy as np


data_root_default=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data")


def write_interleaved_float32(output_path,array_vec,chunk_size=1<<20):
    point_num=array_vec[0].size
    if any(array.size!=point_num for array in array_vec):
        raise ValueError("坐标分量长度不一致")
    with open(output_path,"wb") as output_file:
        for begin in range(0,point_num,chunk_size):
            end=min(begin+chunk_size,point_num)
            np.column_stack([np.asarray(array[begin:end],dtype=np.float32) for array in array_vec]).tofile(output_file)


def run_h5dump(source_path,dataset_name,output_path,start=None,count=None):
    h5dump=shutil.which("h5dump")
    if h5dump is None:
        raise FileNotFoundError("PATH中没有h5dump")
    command=[h5dump,"-d",dataset_name]
    if start is not None:
        command.extend(["-s",start])
    if count is not None:
        command.extend(["-c",count])
    command.extend(["-b","LE","-o",str(output_path),str(source_path)])
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
