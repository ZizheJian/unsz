# 检验异常点在同一数据集的不同文件上是否相同

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from t4_adjust_coord import adjust_coord

file_name_list=[f"/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/attr0.dat.{i}" for i in range(0,162)]
abnormal_count=None
for file_name in file_name_list:
    data=np.fromfile(file_name,dtype=np.float32)
    if abnormal_count is None:
        abnormal_count=np.zeros(data.shape[0],dtype=np.int32)
    abnormal_count+=np.where(data<-1000,1,0)
unique,counts=np.unique(abnormal_count,return_counts=True)
for u,c in zip(unique,counts):
    print(f"abnormal_count={u}, count={c}")