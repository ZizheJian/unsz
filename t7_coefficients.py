# 使用1/d^k作为参考点的权重时，发现k越大越好，推测是因为较远的参考点本该使用负权重。
# 尝试使用a[i]/(d^k+b[i])做为权重，其中i是参考点的距离的顺序。

# 采样时不能只使用最近点，而是应该在一定范围内随机选点，有两个原因：1. 最近点可能没有包围目标点，2. 最近点可能是异常值或者不可用
# 当参考点数量不同时，可能需要使用不同的a、b、k。因此对不同参考点数量分别训练（min_ref_num=max_ref_num，但小于farthest_ref_idx）
# 1 RMSE= 0.026238097851083785, MAE= 0.004446944009042472
# 2 RMSE= 0.019392245697962048, MAE= 0.003338303250546527
# 3 RMSE= 0.016467703953698697, MAE= 0.0027605326222431776
# 4 RMSE= 0.014772417210545194, MAE= 0.002436365960102876
# 5 RMSE= 0.011946267236944917, MAE= 0.0020675575884701205
# 6 RMSE= 0.011479255497462391, MAE= 0.001898396494241238
# 7 RMSE= 0.012006514215890171, MAE= 0.0017749959739079724
# 8 RMSE= 0.010675992101714073, MAE= 0.001596664782722301 （这里开始出现负权重）
# 9 RMSE= 0.009953208122034325, MAE= 0.0015105804788447057
# 10 RMSE= 0.009257973740541572, MAE= 0.0014550226348736847

# 所以基本上使用4~5个参考点就可以了，但可能不能使用MSE来优化系数，而是应该使用固定系数。
# 四个参考点时，
# a= [0.97449082 0.87237605 1.36994827 1.86538999]
# b= [2.02260094e-11 2.22014987e-10 5.36458713e-09 6.50066916e-09]
# k= 3.6070108277674082
# 五个参考点时，
# a= [0.99045959 0.84959153 0.83315349 0.88359481 2.27752616]
# b= [8.14986914e-10 1.91159290e-09 1.92227743e-09 1.98419896e-09 1.56547749e-06]
# k= 3.6215397584806674
# 看起来没有任何规律，可能还是应该使用固定的a=1，b=0，k=3.61。


import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from t4_adjust_coord import adjust_coord

coord=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/coords.dat",dtype=np.float32).reshape(-1,3)
data=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/attr0.dat.80",dtype=np.float32)
min_ref_num=5 # 最小的参考点数量
max_ref_num=5 # 最大的参考点数量
farthest_ref_idx=10 # 选择参考点时可以用的最远参考点
if farthest_ref_idx<max_ref_num:
    raise ValueError("farthest_ref_idx should be greater than or equal to max_ref_num")

######## 修正坐标系 ########
data=np.where(data<-1000,np.nan,data)
coord_transform_mat=adjust_coord(coord[np.isfinite(data)],data[np.isfinite(data)])
coord=coord@coord_transform_mat
data,coord=data[np.isfinite(data)],coord[np.isfinite(data)]

tree=cKDTree(coord)
nearest_dist,nearest_idx=tree.query(coord,k=farthest_ref_idx+1)
nearest_dist=nearest_dist[:,1:]
nearest_idx=nearest_idx[:,1:]

record_num=200000
ref_array=np.zeros((record_num,max_ref_num))
dist_array=np.zeros((record_num,max_ref_num))
valid_array=np.zeros((record_num,max_ref_num))
tgt_array=np.zeros(record_num)
for r_id in range(record_num):
    tgt=np.random.randint(0,coord.shape[0])
    tgt_array[r_id]=data[tgt]
    ref_num=np.random.randint(min_ref_num,max_ref_num+1)
    ref_seq=np.sort(np.random.choice(np.arange(0,farthest_ref_idx),size=ref_num,replace=False))
    ref=nearest_idx[tgt][ref_seq]
    ref_array[r_id][:ref_num]=data[ref]
    dist_array[r_id][:ref_num]=nearest_dist[tgt][ref_seq]
    valid_array[r_id][:ref_num]=1

eps=1e-1
iter=0

def predict(p):
    a=p[:max_ref_num]
    b=np.cumsum(p[max_ref_num:2*max_ref_num])
    k=p[-1]
    temp_dist=dist_array+(1-valid_array)
    weight=a[None,:]/(temp_dist**k+b[None,:])*valid_array
    weight/=np.maximum(weight.sum(axis=1,keepdims=True),eps)
    pred=(weight*ref_array).sum(axis=1)
    return pred,weight

def residual(p):
    global iter
    pred,weight=predict(p)
    r=pred-tgt_array
    if iter%1000==0:
        print(f"iter= {iter}, RMSE= {np.sqrt(np.mean(r**2))}, MAE= {np.mean(np.abs(r))}")
        print(f"a= {p[:max_ref_num]}, b= {np.cumsum(p[max_ref_num:2*max_ref_num])}, k= {p[-1]}")
    iter+=1
    return r

p0=np.zeros(2*max_ref_num+1)
p0[:max_ref_num]=1
p0[max_ref_num:2*max_ref_num]=0
p0[-1]=2
lower=np.full(2*max_ref_num+1,-np.inf)
upper=np.full(2*max_ref_num+1,np.inf)
lower[0]=0.9
upper[0]=1.1
lower[max_ref_num:2*max_ref_num]=0
res=least_squares(residual,p0,bounds=(lower,upper),max_nfev=600,verbose=2,ftol=None)
p=res.x

for k in np.arange(0,6,1):
    p0[-1]=k
    pred,weight=predict(p0)
    rmse=np.sqrt(np.mean((pred-tgt_array)**2))
    mae=np.mean(np.abs(pred-tgt_array))
    print(f"k= {k}, RMSE= {rmse}, MAE= {mae}")

print(f"a= {p[:max_ref_num]}")
print(f"b= {np.cumsum(p[max_ref_num:2*max_ref_num])}")
print(f"k= {p[-1]}")
pred,weight=predict(p)
rmse=np.sqrt(np.mean((pred-tgt_array)**2))
mae=np.mean(np.abs(pred-tgt_array))
print(f"Final: RMSE= {rmse}, MAE= {mae}")

for i in range(record_num):
    if any(np.abs(weight[i,1:])>1):
        print(f"Record {i}")
        print(f"    dist= {dist_array[i]}")
        print(f"    weight= {weight[i]}")

for i in range(max_ref_num):
    m=valid_array[:,i]>0
    wi=weight[m,i]
    print(f"Ref {i}, count= {wi.size}, mean= {wi.mean()}, min= {wi.min()}, p1= {np.percentile(wi,1)}, p50= {np.percentile(wi,50)}, p99= {np.percentile(wi,99)}, max= {wi.max()}")