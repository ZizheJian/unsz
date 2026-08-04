import pyvista as pv
import numpy as np

data=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean/layerThickness.dat.15",dtype=np.float32)
conn=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean/conn.dat",dtype=np.int32).reshape(-1,2)
coord=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean/coords.dat",dtype=np.float32).reshape(-1,3)

# r=np.linalg.norm(coord,axis=1)
# lon=np.arctan2(coord[:,1],coord[:,0])
# lat=np.arcsin(coord[:,2]/r)
# coord[:,0]=lon
# coord[:,1]=lat
# coord[:,2]=r

# p0=coord[conn[:,0]]
# p1=coord[conn[:,1]]
# p2=coord[conn[:,2]]
# l01=np.linalg.norm(p0-p1,axis=1)
# l12=np.linalg.norm(p1-p2,axis=1)
# l20=np.linalg.norm(p2-p0,axis=1)
# max_l=np.maximum.reduce([l01,l12,l20],axis=0)
# threshold=np.quantile(max_l,0.999)
# conn=conn[max_l<threshold,:]

print(f"data shape= {data.shape}, min= {np.nanmin(data)}, max= {np.nanmax(data)}")
print(f"data valid percentage: {np.sum(np.isfinite(data))/data.size}")
print(f"conn shape= {conn.shape}, min= {conn.min()}, max= {conn.max()}")
print(f"coord shape= {coord.shape}, min= {coord.min(axis=0)}, max= {coord.max(axis=0)}")

cells=np.hstack([np.full((conn.shape[0],1),2),conn]).ravel()
grid=pv.PolyData()
grid.points=coord
grid.lines=cells
grid.point_data["value"]=data

clim=np.nanquantile(data,[0.05,0.95])
print("clim:",clim)
p=pv.Plotter(off_screen=True,window_size=(1200,900))
p.set_background("white")
p.add_mesh(grid,scalars="value",cmap="turbo",clim=clim,nan_color="lightgray")
# p.view_isometric()
p.view_xy()
p.camera.zoom(1.5)
p.show(screenshot="Ocean_15.png")