from py_codes.py_unsz_core.param import Param

data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina"
param=Param()
param.coord_path=f"{data_dir}/coords.dat"
param.original_data_path_vec=[f"{data_dir}/attr0.dat.0"]
param.dim=3
param.rel_eb=1e-2

param.cross_time_compress=False
param.if_adjust_coord=True

param.init()
if param.if_adjust_coord:
    param.adjust_coord()
