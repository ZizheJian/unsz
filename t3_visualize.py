import argparse
from pathlib import Path
import numpy as np
import pyvista as pv

data_root_default="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
dataset_config={
    "katrina":{
        "directory":"Katrina",
        "variable":"attr0",
        "time_step":80,
        "cell_point_num":3,
        "view":"isometric",
        "zoom":0.8,
        # 仅为避免Katrina原坐标图被严重拉长而缩放z轴；不是PyVista自动变换，也不参与压缩或adjust coord训练
        "original_plot_scale":np.array([1,1,0.001],dtype=np.float64),
        "abnormal_upper_bound":-1000,
        "output_prefix":"Katrina_80",
        # second_power、least_squares、第一次采样0.25、第二次采样0.1、新矩阵权重0.5、训练4轮
        "transform_matrix":np.array([[1.3207279451534779,0,0],[-0.050022502016393564,1.119176596028389,0],[-0.0022639755300411725,0.00041386801901295816,0.02477688868808782]],dtype=np.float64),
    },
    "ocean-temperature":{
        "directory":"Ocean",
        "variable":"temperature",
        "time_step":15,
        "cell_point_num":2,
        "view":"xy",
        "zoom":1.5,
        "original_plot_scale":np.ones(3,dtype=np.float64),
        "abnormal_upper_bound":None,
        "output_prefix":"Ocean_temperature_15",
        # second_power、least_squares、第一次采样0.25、第二次采样0.1、新矩阵权重0.5、训练4轮
        "transform_matrix":np.array([[0.6613823200270561,0,0],[0.1410372745584055,0.5997719271039184,0],[0.04748698529576451,0.0520621994620818,1.475801498513099]],dtype=np.float64),
    },
}

def make_grid(coord,conn,cell_point_num):
    cells=np.column_stack((np.full(conn.shape[0],cell_point_num,dtype=np.int32),conn)).ravel()
    if cell_point_num==3:
        celltypes=np.full(conn.shape[0],pv.CellType.TRIANGLE,dtype=np.uint8)
        return pv.UnstructuredGrid(cells,celltypes,coord)
    grid=pv.PolyData()
    grid.points=coord
    grid.lines=cells
    return grid

def render(config,output_dir,if_adjust_coord,data,conn,coord,clim):
    plot_coord=coord@config["transform_matrix"] if if_adjust_coord else coord*config["original_plot_scale"]
    grid=make_grid(plot_coord,conn,config["cell_point_num"])
    grid.point_data["value"]=data
    plotter=pv.Plotter(off_screen=True,window_size=(1200,900))
    plotter.set_background("white")
    plotter.add_mesh(grid,scalars="value",cmap="turbo",clim=clim,nan_color="lightgray")
    if config["view"]=="isometric":
        plotter.view_isometric()
    else:
        plotter.view_xy()
    plotter.reset_camera()
    plotter.camera.zoom(config["zoom"])
    suffix="_adjusted" if if_adjust_coord else ""
    output_path=output_dir/f"{config['output_prefix']}{suffix}.png"
    plotter.show(screenshot=str(output_path))
    print(f"生成图像：{output_path}")
    print(f"绘图坐标范围：{np.column_stack((plot_coord.min(axis=0),plot_coord.max(axis=0))).tolist()}")

def visualize(dataset_name,data_root,output_dir):
    config=dataset_config[dataset_name]
    data_dir=data_root/config["directory"]
    data=np.fromfile(data_dir/f"{config['variable']}.dat.{config['time_step']}",dtype=np.float32)
    conn=np.fromfile(data_dir/"conn.dat",dtype=np.int32).reshape(-1,config["cell_point_num"]+1)[:,1:] if config["cell_point_num"]==3 else np.fromfile(data_dir/"conn.dat",dtype=np.int32).reshape(-1,2)
    coord=np.fromfile(data_dir/"coords.dat",dtype=np.float32).reshape(-1,3).astype(np.float64)
    if config["abnormal_upper_bound"] is not None:
        data=np.where(data<config["abnormal_upper_bound"],np.nan,data)
    if data.size!=coord.shape[0]:
        raise ValueError(f"{dataset_name}的数据点数{data.size}与坐标点数{coord.shape[0]}不一致")
    clim=np.nanquantile(data,[0.05,0.95])
    print(f"数据集：{dataset_name}")
    print(f"数据点数：{data.size}，有效点比例：{np.count_nonzero(np.isfinite(data))/data.size}")
    print(f"连接数：{conn.shape[0]}，原始坐标范围：{np.column_stack((coord.min(axis=0),coord.max(axis=0))).tolist()}")
    print(f"坐标变换矩阵：\n{config['transform_matrix']}")
    render(config,output_dir,False,data,conn,coord,clim)
    render(config,output_dir,True,data,conn,coord,clim)

parser=argparse.ArgumentParser(description="生成Katrina和Ocean坐标变换前后的field图像")
parser.add_argument("-dataset",choices=["all",*dataset_config],default="all")
parser.add_argument("-data-root",default=data_root_default)
parser.add_argument("-output-dir",default=".")
args=parser.parse_args()
data_root=Path(args.data_root)
output_dir=Path(args.output_dir)
output_dir.mkdir(parents=True,exist_ok=True)
selected_datasets=dataset_config if args.dataset=="all" else [args.dataset]
for dataset_name in selected_datasets:
    visualize(dataset_name,data_root,output_dir)
