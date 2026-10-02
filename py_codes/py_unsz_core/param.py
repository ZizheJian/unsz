import numpy as np


class Param:
    def __init__(self):
        self.coord_path=""
        self.original_data_path_vec=[]

        self.dim=3
        self.coord_dtype=np.dtype(np.float32)
        self.data_dtype=np.dtype(np.float32)
        self.rel_eb=None
        self.abs_eb=None

        self.prediction_max_ref_num=4

        self.point_num=0
        self.coord_array=np.empty((0,0),dtype=np.float32)
        self.min_coord=np.empty(0,dtype=np.float32)
        self.max_coord=np.empty(0,dtype=np.float32)
        self.coord_transform_matrix=np.empty((0,0),dtype=np.float64)
