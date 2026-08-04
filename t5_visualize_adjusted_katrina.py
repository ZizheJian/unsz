import pyvista as pv
import numpy as np
from t4_adjust_coord import adjust_coord

conn=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/conn.dat",dtype=np.int32).reshape(-1,4)[:,1:]
coord=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/coords.dat",dtype=np.float32).reshape(-1,3)
data=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/attr0.dat.80",dtype=np.float32)

data=np.where(data<-1000,np.nan,data)
coord_transform_mat=adjust_coord(coord[np.isfinite(data)],data[np.isfinite(data)])
print("coord_transform_mat:",coord_transform_mat)
exit()
coord=coord@coord_transform_mat
print(f"data shape= {data.shape}, min= {np.nanmin(data)}, max= {np.nanmax(data)}")
print(f"data valid percentage: {np.sum(np.isfinite(data))/data.size}")
print(f"conn shape= {conn.shape}, min= {conn.min()}, max= {conn.max()}")
print(f"coord shape= {coord.shape}, min= {coord.min(axis=0)}, max= {coord.max(axis=0)}")

cells=np.hstack([np.full((conn.shape[0],1),3),conn]).ravel()
celltypes=np.full(conn.shape[0],pv.CellType.TRIANGLE,dtype=np.uint8)
plot_coord=coord.copy()
grid=pv.UnstructuredGrid(cells,celltypes,plot_coord)
grid.point_data["value"]=data

clim=np.nanquantile(data,[0.05,0.95])
print("clim:",clim)
p=pv.Plotter(off_screen=True,window_size=(1200,900))
p.set_background("white")
p.add_mesh(grid,scalars="value",cmap="turbo",clim=clim,nan_color="lightgray")
p.view_isometric()
p.camera.zoom(1.5)
p.show(screenshot="Katrina_80_adjusted.png")