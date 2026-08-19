import sys
import pyvista as pv
import numpy as np

if_adjust_coord=True
if len(sys.argv)>1:
    if_adjust_coord=sys.argv[1].lower() in ["1","true","yes"]

data=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean/temperature.dat.15",dtype=np.float32)
conn=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean/conn.dat",dtype=np.int32).reshape(-1,2)
coord=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean/coords.dat",dtype=np.float32).reshape(-1,3)
transform_matrix=np.array([
    [0.57863758,0,0],
    [0.27448921,0.38810419,0],
    [-0.16613945,0.13620828,1.54695002],
],dtype=np.float64)

plot_coord=coord@transform_matrix if if_adjust_coord else coord.copy()
print(f"if_adjust_coord={if_adjust_coord}")
print(f"data shape={data.shape}, min={np.nanmin(data)}, max={np.nanmax(data)}")
print(f"data valid percentage={np.sum(np.isfinite(data))/data.size}")
print(f"conn shape={conn.shape}, min={conn.min()}, max={conn.max()}")
print(f"plot coord range={np.column_stack((plot_coord.min(axis=0),plot_coord.max(axis=0)))}")

cells=np.hstack([np.full((conn.shape[0],1),2),conn]).ravel()
grid=pv.PolyData()
grid.points=plot_coord
grid.lines=cells
grid.point_data["value"]=data
clim=np.nanquantile(data,[0.05,0.95])
p=pv.Plotter(off_screen=True,window_size=(1200,900))
p.set_background("white")
p.add_mesh(grid,scalars="value",cmap="turbo",clim=clim,nan_color="lightgray")
p.view_xy()
p.camera.zoom(1.5)
output_path="Ocean_temperature_15_adjusted.png" if if_adjust_coord else "Ocean_temperature_15.png"
p.show(screenshot=output_path)
print(f"生成图像：{output_path}")
