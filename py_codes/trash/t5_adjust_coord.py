# 我们想要知道邻近点的距离和可预测性之间的关系，但是先要对距离进行缩放，因为不同方向上的单位距离可能对应不同的可预测性，另外距离的缩放也会改变邻近点的选择。
# 关于可预测性，暂时定义为一阶导数
# 可以对数据进行分块，然后选择一些比较近的点，拟合(v1-v2)^2=(c1-c2)^T M (c1-c2)，其中v是数据，c是坐标，M是需要拟合的对称矩阵。
# 得到一个M以后，可以修正坐标并重新选择比较近的点，然后用这些点的原始坐标再次拟合，得到新的M。

import numpy as np
from scipy.spatial import cKDTree

def log_m(m):
    eigval,eigvec=np.linalg.eigh(m)
    return eigvec@np.diag(np.log(eigval))@eigvec.T

def exp_m(m):
    eigval,eigvec=np.linalg.eigh(m)
    return eigvec@np.diag(np.exp(eigval))@eigvec.T

def damp_m(m,m_history,m_weight=0.3):
    if m_history is None:
        return m
    else:
        return exp_m(log_m(m)*m_weight+log_m(m_history)*(1-m_weight))

def adjust_coord(coord,data):
    new_coord=coord
    m_history=None
    last_coord_range=np.zeros(3)
    stable_count=0
    for it in range(100):
        print(f"Iteration {it}:")
        ######## 根据修正后坐标选择邻近点 ########
        tree=cKDTree(new_coord)
        _,nearest_idx=tree.query(new_coord,k=5)
        nearest_idx=nearest_idx[:,1:]
        delta_v2=((data[:,None]-data[nearest_idx])**2).reshape(-1)
        delta_c=(coord[:,None,:]-coord[nearest_idx]).reshape(-1,3)
        delta_c2=np.column_stack([delta_c[:,0]**2,delta_c[:,1]**2,delta_c[:,2]**2,
                                2*delta_c[:,0]*delta_c[:,1],2*delta_c[:,0]*delta_c[:,2],2*delta_c[:,1]*delta_c[:,2]])
        ######## 求出m矩阵 ########
        m=np.linalg.lstsq(delta_c2,delta_v2,rcond=None)[0]
        m=np.array([[m[0],m[3],m[4]],
                    [m[3],m[1],m[5]],
                    [m[4],m[5],m[2]]])
        ######## 保证m是正定的 ########
        eigval,eigvec=np.linalg.eigh(m)
        print("eigval:\n",eigval)
        print("eigvec:\n",eigvec)
        eigval[eigval<eigval.max()*1e-10]=eigval.max()*1e-10
        m=eigvec@np.diag(eigval)@eigvec.T
        ######## 归一化m矩阵 ########
        m=m*3/np.trace(m)
        ######## 阻尼 ########
        m_history=m=damp_m(m,m_history)
        ######## 修正坐标 ########
        B=np.linalg.cholesky(m)
        new_coord=coord@B
        print(f"New coord[0] range: [[{new_coord[:,0].min()}, {new_coord[:,0].max()}] ({np.ptp(new_coord[:,0])})")
        print(f"New coord[1] range: [[{new_coord[:,1].min()}, {new_coord[:,1].max()}] ({np.ptp(new_coord[:,1])})")
        print(f"New coord[2] range: [[{new_coord[:,2].min()}, {new_coord[:,2].max()}] ({np.ptp(new_coord[:,2])})")
        if (np.abs(last_coord_range-np.ptp(new_coord,axis=0))<0.1*last_coord_range).all():
            stable_count+=1
        else:
            stable_count=0
        last_coord_range=np.ptp(new_coord,axis=0)
        if stable_count>=10:
            print("Converged.")
            break
    B=np.linalg.cholesky(m)
    print(B)
    print(f"{np.array([1,0,0])} -> {np.array([1,0,0])@B}")
    print(f"{np.array([0,1,0])} -> {np.array([0,1,0])@B}")
    print(f"{np.array([0,0,1])} -> {np.array([0,0,1])@B}")
    return B

if __name__=="__main__":
    data=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/attr0.dat.160",dtype=np.float32)
    coord=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/coords.dat",dtype=np.float32).reshape(-1,3)

    data,coord=data[data>-1000],coord[data>-1000]
    print(f"Valid data num: {coord.shape[0]}")
    print(f"Coord[0] range: [[{coord[:,0].min()}, {coord[:,0].max()}] ({np.ptp(coord[:,0])})")
    print(f"Coord[1] range: [[{coord[:,1].min()}, {coord[:,1].max()}] ({np.ptp(coord[:,1])})")
    print(f"Coord[2] range: [[{coord[:,2].min()}, {coord[:,2].max()}] ({np.ptp(coord[:,2])})")
    adjust_coord(coord,data)