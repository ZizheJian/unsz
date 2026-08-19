import sys
import pyvista as pv
import numpy as np

if_adjust_coord=True
if len(sys.argv)>1:
    if_adjust_coord=sys.argv[1].lower() in ["1","true","yes"]

data=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/attr0.dat.80",dtype=np.float32)
conn=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/conn.dat",dtype=np.int32).reshape(-1,4)[:,1:]
coord=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/coords.dat",dtype=np.float32).reshape(-1,3)
transform_matrix=np.array([
    [1.12670515e+00,0,0],
    [2.70622568e-01,1.28736120e+00,0],
    [-1.67488548e-04,-1.83679099e-04,3.29459469e-06],
],dtype=np.float64)

data=np.where(data<-1000,np.nan,data)
plot_coord=coord@transform_matrix if if_adjust_coord else coord.copy()
if not if_adjust_coord:
    plot_coord[:,2]*=0.001
print(f"if_adjust_coord={if_adjust_coord}")
print(f"data shape={data.shape}, min={np.nanmin(data)}, max={np.nanmax(data)}")
print(f"data valid percentage={np.sum(np.isfinite(data))/data.size}")
print(f"conn shape={conn.shape}, min={conn.min()}, max={conn.max()}")
print(f"plot coord range={np.column_stack((plot_coord.min(axis=0),plot_coord.max(axis=0)))}")

cells=np.hstack([np.full((conn.shape[0],1),3),conn]).ravel()
celltypes=np.full(conn.shape[0],pv.CellType.TRIANGLE,dtype=np.uint8)
grid=pv.UnstructuredGrid(cells,celltypes,plot_coord)
grid.point_data["value"]=data
clim=np.nanquantile(data,[0.05,0.95])
p=pv.Plotter(off_screen=True,window_size=(1200,900))
p.set_background("white")
p.add_mesh(grid,scalars="value",cmap="turbo",clim=clim,nan_color="lightgray")
p.view_isometric()
p.camera.zoom(1.5)
output_path="Katrina_80_adjusted.png" if if_adjust_coord else "Katrina_80.png"
p.show(screenshot=output_path)
print(f"生成图像：{output_path}")
