from dataclasses import dataclass,field
from py_codes.py_unsz_core.init import init
from py_codes.py_unsz_core.adjust_coord import adjust_coord

import numpy as np

@dataclass
class Param:
    coord_path:str=""
    original_data_path_vec:list[str]=field(default_factory=list)
    compressed_data_path_vec:list[str]=field(default_factory=list)
    decompressed_data_path_vec:list[str]=field(default_factory=list)

    dim:int=3
    coord_dtype:np.dtype=np.dtype(np.float32)
    data_dtype:np.dtype=np.dtype(np.float32)
    rel_eb:float|None=None
    abs_eb:float|None=None

    cross_time_compress:bool=False
    if_adjust_coord:bool=False
    actual_ref_num:int=5
    first_sample_ratio:float=0.25
    sample_file_num:int=1
    second_sample_point_num:int=100000
    random_seed:int=0
    abnormal_threshold:float=-1000
    log_value_epsilon:float=1e-6
    log_distance_epsilon:float=1e-12
    adjust_max_nfev:int=100

    point_num:int=field(init=False,default=0)
    coord_array:np.ndarray=field(init=False,default_factory=lambda:np.empty((0,0),dtype=np.float32))
    min_coord:np.ndarray=field(init=False,default_factory=lambda:np.empty(0,dtype=np.float32))
    max_coord:np.ndarray=field(init=False,default_factory=lambda:np.empty(0,dtype=np.float32))
    first_sample_ids:np.ndarray=field(init=False,default_factory=lambda:np.empty(0,dtype=np.int64))
    coord_transform_parameter:np.ndarray=field(init=False,default_factory=lambda:np.empty(0,dtype=np.float64))
    coord_transform_center:np.ndarray=field(init=False,default_factory=lambda:np.empty(0,dtype=np.float64))
    coord_transform_scale:np.ndarray=field(init=False,default_factory=lambda:np.empty(0,dtype=np.float64))

    init=init
    adjust_coord=adjust_coord
