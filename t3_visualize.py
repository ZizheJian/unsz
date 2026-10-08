import argparse
from pathlib import Path
import numpy as np
import pyvista as pv

data_root_default="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
dataset_config={
    "katrina":{
        "directory":"Katrina",
        "field":"attr0",
        "time_step":80,
        "cell_point_num":3,
        "view":"isometric",
        "zoom":0.8,
        # 仅为避免Katrina原坐标图被严重拉长而缩放z轴；不参与压缩或坐标变换训练
        "original_plot_scale":np.array([1,1,0.001],dtype=np.float64),
        "abnormal_upper_bound":-1000,
        "output_prefix":"Katrina_80",
        "coordinate_dim":3,
        "coordinate_transform_file":"attr0.unsz.coord_transform.txt",
    },
}

def read_coordinate_transform(path,coordinate_dim):
    token=path.read_text().split()
    if len(token)!=7+coordinate_dim or token[:2]!=["UNSZ_COORDINATE_TRANSFORM","3"]:
        raise ValueError(f"坐标变换文件格式不正确：{path}")
    if token[2]!="dimension" or int(token[3])!=coordinate_dim:
        raise ValueError(f"坐标变换文件维数不正确：{path}")
    if token[4]!="enabled" or token[5] not in ("0","1"):
        raise ValueError(f"坐标变换文件enabled字段不正确：{path}")
    if token[6]!="diagonal":
        raise ValueError(f"坐标变换文件字段不正确：{path}")
    diagonal=np.asarray(token[7:],dtype=np.float64)
    if not np.isfinite(diagonal).all():
        raise ValueError(f"坐标变换文件包含非有限数：{path}")
    return bool(int(token[5])),diagonal

def make_grid(coord,conn,cell_point_num):
    cells=np.column_stack((np.full(conn.shape[0],cell_point_num,dtype=np.int32),conn)).ravel()
    if cell_point_num==3:
        celltypes=np.full(conn.shape[0],pv.CellType.TRIANGLE,dtype=np.uint8)
        return pv.UnstructuredGrid(cells,celltypes,coord)
    grid=pv.PolyData()
    grid.points=coord
    grid.lines=cells
    return grid

def save_grid(config,output_dir,grid,clim,suffix=""):
    plotter=pv.Plotter(off_screen=True,window_size=(1200,900))
    plotter.set_background("white")
    plotter.add_mesh(grid,scalars="value",cmap="turbo",clim=clim,nan_color="lightgray",show_scalar_bar=False)
    plotter.add_scalar_bar(title="value")
    if config["view"]=="isometric":
        plotter.view_isometric()
    else:
        plotter.view_xy()
    plotter.reset_camera()
    plotter.camera.zoom(config["zoom"])
    output_path=output_dir/f"{config['output_prefix']}{suffix}.png"
    plotter.show(screenshot=str(output_path))
    print(f"生成图像：{output_path}")

def render(config,output_dir,use_coordinate_transform,data,conn,coord,clim):
    plot_coord=coord@config["transform_matrix"] if use_coordinate_transform else coord*config["original_plot_scale"]
    grid=make_grid(plot_coord,conn,config["cell_point_num"])
    grid.point_data["value"]=data
    save_grid(config,output_dir,grid,clim,"_adjusted" if use_coordinate_transform else "")
    print(f"绘图坐标范围：{np.column_stack((plot_coord.min(axis=0),plot_coord.max(axis=0))).tolist()}")

def visualize(dataset_name,data_root,output_dir):
    config=dict(dataset_config[dataset_name])
    data_dir=data_root/config["directory"]
    if "coordinate_transform_file" in config:
        enabled,diagonal=read_coordinate_transform(data_dir/config["coordinate_transform_file"],config["coordinate_dim"])
        config["coordinate_transform_enabled"]=enabled
        config["transform_matrix"]=np.diag(diagonal)
        print(f"坐标变换：enabled={int(enabled)}，diagonal={diagonal.tolist()}")
    data=np.fromfile(data_dir/f"{config['field']}.dat.{config['time_step']}",dtype=np.float32)
    if config["abnormal_upper_bound"] is not None:
        data=np.where(data<config["abnormal_upper_bound"],np.nan,data)
    clim=np.nanquantile(data,[0.05,0.95])
    print(f"数据集：{dataset_name}")
    print(f"数据点数：{data.size}，有效点比例：{np.count_nonzero(np.isfinite(data))/data.size}")
    if config["cell_point_num"]==3:
        conn=np.fromfile(data_dir/"conn.dat",dtype=np.int32).reshape(-1,config["cell_point_num"]+1)[:,1:]
    else:
        conn=np.fromfile(data_dir/"conn.dat",dtype=np.int32).reshape(-1,2)
    coord=np.fromfile(data_dir/"coords.dat",dtype=np.float32).reshape(-1,3).astype(np.float64)
    if data.size!=coord.shape[0]:
        raise ValueError(f"{dataset_name}的数据点数{data.size}与坐标点数{coord.shape[0]}不一致")
    print(f"连接数：{conn.shape[0]}，原始坐标范围：{np.column_stack((coord.min(axis=0),coord.max(axis=0))).tolist()}")
    print(f"坐标变换矩阵：\n{config['transform_matrix']}")
    render(config,output_dir,False,data,conn,coord,clim)
    if config["coordinate_transform_enabled"]:
        render(config,output_dir,True,data,conn,coord,clim)
    else:
        print("C++自动判据未启用坐标变换，不生成_adjusted图像")

def main():
    parser=argparse.ArgumentParser(description="读取C++生成的坐标变换结果，生成Katrina变换前后的field图像")
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

if __name__=="__main__":
    main()
