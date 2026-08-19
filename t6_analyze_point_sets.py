import os
import numpy as np
from scipy.spatial import cKDTree

data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
katrina_matrix=np.array([
    [1.12670515e0,0,0],
    [2.70622568e-1,1.28736120e0,0],
    [-1.67488548e-4,-1.83679099e-4,3.29459469e-6]
],dtype=np.float64)
dataset_list=[
    {"name":"Katrina-attr0","folder":"Katrina","variable":"attr0","time_steps":[0,40,80,120,161],"matrix":katrina_matrix},
    {"name":"Ocean-temperature","folder":"Ocean","variable":"temperature","time_steps":[0,7,15,23,30],"matrix":None},
]

def is_abnormal(data):
    return data<-1000

def read_coord(folder):
    coord=np.fromfile(os.path.join(data_root,folder,"coords.dat"),dtype=np.float32)
    if coord.size%3!=0:
        raise RuntimeError(f"{folder}坐标数量不能被3整除")
    return coord.reshape(-1,3).astype(np.float64)

def read_mask(dataset,time_step,point_num):
    path=os.path.join(data_root,dataset["folder"],f"{dataset['variable']}.dat.{time_step}")
    data=np.fromfile(path,dtype=np.float32)
    if data.size!=point_num:
        raise RuntimeError(f"{path}的点数为{data.size}，坐标点数为{point_num}")
    return is_abnormal(data)

def analyze_neighbors(coord,mask):
    point_num=coord.shape[0]
    abnormal_ratio=np.mean(mask)
    tree=cKDTree(coord)
    _,neighbor_ids=tree.query(coord,k=10,workers=1)
    neighbor_mask=mask[neighbor_ids[:,1:]]
    same_ratio=np.mean(neighbor_mask==mask[:,None])
    nearest_same_ratio=np.mean(neighbor_mask[:,0]==mask)
    predicted_mask=np.sum(neighbor_mask,axis=1)>=5
    vote_accuracy=np.mean(predicted_mask==mask)
    random_same_ratio=abnormal_ratio**2+(1-abnormal_ratio)**2
    normal=~mask
    result={
        "same_ratio":same_ratio,
        "nearest_same_ratio":nearest_same_ratio,
        "vote_accuracy":vote_accuracy,
        "random_same_ratio":random_same_ratio,
        "normal_nearest_same_ratio":np.mean(~neighbor_mask[normal,0]) if np.any(normal) else np.nan,
        "abnormal_nearest_same_ratio":np.mean(neighbor_mask[mask,0]) if np.any(mask) else np.nan,
    }
    if np.sum(normal)>1 and np.sum(mask)>1:
        normal_tree=cKDTree(coord[normal])
        abnormal_tree=cKDTree(coord[mask])
        normal_same_distance=normal_tree.query(coord[normal],k=2,workers=1)[0][:,1]
        normal_other_distance=abnormal_tree.query(coord[normal],k=1,workers=1)[0]
        abnormal_same_distance=abnormal_tree.query(coord[mask],k=2,workers=1)[0][:,1]
        abnormal_other_distance=normal_tree.query(coord[mask],k=1,workers=1)[0]
        result["normal_separation_ratio"]=np.median(normal_other_distance/normal_same_distance)
        result["abnormal_separation_ratio"]=np.median(abnormal_other_distance/abnormal_same_distance)
    return result

def print_neighbor_result(coord_name,result):
    print(f"  {coord_name}：")
    print(f"    最近邻同类比例：{result['nearest_same_ratio']:.6f}（正常点{result['normal_nearest_same_ratio']:.6f}，异常点{result['abnormal_nearest_same_ratio']:.6f}）")
    print(f"    9近邻中的同类边比例：{result['same_ratio']:.6f}，仅按类别比例的随机基线：{result['random_same_ratio']:.6f}")
    print(f"    9近邻多数投票准确率：{result['vote_accuracy']:.6f}")
    if "normal_separation_ratio" in result:
        print(f"    最近异类/最近同类距离中位数：正常点{result['normal_separation_ratio']:.3f}，异常点{result['abnormal_separation_ratio']:.3f}")

def main():
    for dataset in dataset_list:
        print(f"\n{dataset['name']}")
        coord=read_coord(dataset["folder"])
        print(f"  点数：{coord.shape[0]}，采样时间步：{dataset['time_steps']}")
        for time_step in dataset["time_steps"]:
            mask=read_mask(dataset,time_step,coord.shape[0])
            print(f"  时间步{time_step}：异常点比例{np.mean(mask):.6f}")
            print_neighbor_result("原始坐标",analyze_neighbors(coord,mask))
            if dataset["matrix"] is not None:
                print_neighbor_result("变换后坐标",analyze_neighbors(coord@dataset["matrix"],mask))

if __name__=="__main__":
    main()
