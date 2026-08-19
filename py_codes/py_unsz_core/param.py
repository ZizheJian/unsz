import numpy as np


class Param:
    def __init__(self):
        self.coord_path=""
        self.original_data_path_vec=[]
        self.compressed_data_path_vec=[]
        self.decompressed_data_path_vec=[]

        self.dim=3
        self.coord_dtype=np.dtype(np.float32)
        self.data_dtype=np.dtype(np.float32)
        self.rel_eb=None
        self.abs_eb=None

        self.cross_time_compress=False
        self.if_adjust_coord=False
        self.actual_ref_num=5

        self.point_num=0
        self.coord_array=np.empty((0,0),dtype=np.float32)
        self.min_coord=np.empty(0,dtype=np.float32)
        self.max_coord=np.empty(0,dtype=np.float32)
        self.coord_transform_parameter=np.empty(0,dtype=np.float64)
        self.coord_transform_matrix=np.empty((0,0),dtype=np.float64)
