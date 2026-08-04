# 选择每个点的最近10个点，然后从中随机选择几个点，做预测，统计预测误差和一些其他量
# 1. 使用不同的预测公式，可以得到不同的预测误差
# 2. 参考点数量
# 3. 实际使用的第k近参考点的距离/平均距离
# 4. 参考点的分布
#       理论上来说，如果预测点在参考点组成的凸包内部，是比较好的
#       p=Σ(λ[i]*x[i])（p是预测点坐标，x[i]λ[i]是未知量）
#       Σ(λ[i])=k
#       最大化min(λ[i])
#       min(λ[i])越小，预测点就越偏。当min(λ[i])<0时，就是外插了

# 关于用凸包来判定参考点的分布，如果参考点和预测目标几乎共面或者共线，判定结果会有问题。
# 有两种解决方法：
# 1. 使用SVD检测局部的维度，然后看情况使用2D或者1D凸包
# 2. 计算预测目标到参考点的距离和参考点两两之间的距离（之后要统计一下点之间的间距分布作为检验）
#       预测目标到参考点的距离：
#           假设数据是三维均匀泊松分布，点密度是ρ，那么对于体积为v的区域，其中的点的数量n服从Poisson(ρ*v,n)=exp(-ρ*v)*(ρ*v)^n/n!。
#           距离待预测点第i近的参考点的距离是r[i]，定义y[i]=4πr[i]^3ρ/3，也就是半径为r[i]的球内的期望点数（注意y[i]是和实际点的位置无关的期望值，并且由于r[i]未知，所以y[i]也未知）。
#           因此，对于任意一个给定的y[i]，内部的点的数量n服从Poisson(y[i],n)=exp(-y[i])*y[i]^n/n!。
#           这一公式可以理解为，对于一个泊松过程，第i个事件发生的时间的期望为y[i]，而n是泊松过程在时间y[i]内实际发生的事件数量。
#           第一个时间发生的事件y[1]晚于y的概率是p(y[1]>y)=Poisson(y,0)=exp(-y)，说明y[1]服从指数分布Exp(1)。
#           由于是泊松过程，因此每个事件相对于前一个事件的延迟相互独立（注意不是事件发生时间相互独立，而是时间延迟相互独立），也就是说y[2]-y[1]~Exp(1)，y[3]-y[2]~Exp(1)...
#           y[i]是i个相互独立的Exp(1)变量的和，服从Gamma(i,1)分布。
#           E(y[i])=i*1=i，E(y[i]^(1/3))=Γ(i+1/3)/Γ(i)，E(r[i])=(4πρ/3)^(-1/3)*Γ(i+1/3)/Γ(i)。
#           E(y[i]^(2/3))=Γ(i+2/3)/Γ(i)，E(r[i]^2)=(4πρ/3)^(-2/3)*Γ(i+2/3)/Γ(i)。
#       参考点两两之间的距离：
#           总共k个参考点中，第k个位于半径为r[k]的球面上，其他的位于内部。我们这里假设所有k个点独立均匀分布在半径为r[k+1]的球内。
#           此时，两两之间的距离期望值是36r[k+1]/35，距离的平方的期望值是6r[k+1]^2/5。
#       如果用距离的一次项计算：
#           Σ(E(r[i]))=3/4*(4πρ/3)^(-1/3)*Γ(k+4/3)/Γ(k)
#           Σ(E(r[i]))/k=3/4*(4πρ/3)^(-1/3)*Γ(k+4/3)/Γ(k+1)=3/4*r[k+1]
#           比例=3/4/(36/35)=35/48
#       如果用距离的二次项计算：
#           Σ(E(r[i]))=3/5*(4πρ/3)^(-2/3)*Γ(k+5/3)/Γ(k)
#           Σ(E(r[i]))/k=3/5*(4πρ/3)^(-2/3)*Γ(k+5/3)/Γ(k+1)=3/5*r[k+1]
#           比例=3/5/(6/5)=1/2

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestRegressor
from t4_adjust_coord import adjust_coord
np.random.seed(0)

class record:
    def __init__(self):
        self.tgt=None
        self.ref_num=0
        self.ref=[]
        self.ref_dist=[]
        self.ref_dist_nan=[]
        self.pred_err=[]#反比于距离的0、0.5、1、1.5、2、2.5次幂
        self.avg_tgt_ref_dist=0
        self.ms_tgt_ref_dist=0
        self.interpolation_score=[]

def max_min_lambda_score(p,X):
    k,d=X.shape

    # 令t<=min(lambda[i])。优化目标是max(t)，也就是min(0*lambda[0]+...+0*lambda[k-1]+(-1)*t)
    c=np.zeros(k+1)
    c[-1]=-1

    # 添加不等式约束。-lambda[i]+t<=0
    A_ub=np.zeros((k,k+1))
    A_ub[:,:k]=-np.eye(k)
    A_ub[:,-1]=1
    b_ub=np.zeros(k)

    # 添加等式约束。前d行对应坐标约束，最后一行对应lambda的和为k
    A_eq=np.zeros((d+1,k+1))
    A_eq[:d,:k]=(X-p).T
    A_eq[d,:k]=1
    b_eq=np.r_[np.zeros(d),k]

    bounds=[(None,None)]*(k+1)

    res=linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds,method="highs")
    if not res.success:
        return np.nan,None
    return res.x[-1],res.x[:k]

coord=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/coords.dat",dtype=np.float32).reshape(-1,3)
data=np.fromfile("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/attr0.dat.80",dtype=np.float32)
min_ref_num=1
max_ref_num=10

######## 修正坐标系 ########
data=np.where(data<-1000,np.nan,data)
coord_transform_mat=adjust_coord(coord[np.isfinite(data)],data[np.isfinite(data)])
coord=coord@coord_transform_mat
data,coord=data[np.isfinite(data)],coord[np.isfinite(data)]

tree=cKDTree(coord)
nearest_dist,nearest_idx=tree.query(coord,k=max_ref_num+1)
nearest_dist=nearest_dist[:,1:]
nearest_idx=nearest_idx[:,1:]

records=[]
for r_id in range(10000):
    rec=record()
    # 预测目标
    rec.tgt=np.random.randint(0,coord.shape[0])
    # 参考点数量
    rec.ref_num=np.random.randint(min_ref_num,max_ref_num+1)
    # 参考点
    ref_seq=np.sort(np.random.choice(np.arange(0,max_ref_num),size=rec.ref_num,replace=False))
    rec.ref=nearest_idx[rec.tgt][ref_seq]
    # 参考点距离
    rec.ref_dist=nearest_dist[rec.tgt][ref_seq]
    rec.ref_dist_nan=[rec.ref_dist[i] if i<rec.ref_num else np.nan for i in range(max_ref_num)]
    # 预测误差
    for i in np.arange(0,5,0.5):
        rec.pred_err.append(np.abs(data[rec.tgt]-np.sum(data[rec.ref]/(rec.ref_dist**i))/np.sum(1/(rec.ref_dist**i))))
    # 平均参考点距离
    rec.avg_tgt_ref_dist=np.mean(rec.ref_dist)
    rec.ms_tgt_ref_dist=np.mean(rec.ref_dist**2)
    # 凸包溢出量
    if rec.ref_num>=4:
        max_min_lambda,_=max_min_lambda_score(coord[rec.tgt],coord[rec.ref])
        rec.interpolation_score.append(max_min_lambda)
    else:
        rec.interpolation_score.append(np.nan)
    # 距离比值
    if rec.ref_num>=2:
        avg_ref_ref_dist=0
        ms_ref_ref_dist=0
        for i in range(rec.ref_num):
            for j in range(i+1,rec.ref_num):
                avg_ref_ref_dist+=np.linalg.norm(coord[rec.ref[i]]-coord[rec.ref[j]])
                ms_ref_ref_dist+=np.linalg.norm(coord[rec.ref[i]]-coord[rec.ref[j]])**2
        avg_ref_ref_dist=2*avg_ref_ref_dist/(rec.ref_num*(rec.ref_num-1))
        ms_ref_ref_dist=2*ms_ref_ref_dist/(rec.ref_num*(rec.ref_num-1))
        rec.interpolation_score.append(rec.avg_tgt_ref_dist/avg_ref_ref_dist/(35/48))
        rec.interpolation_score.append(rec.ms_tgt_ref_dist/ms_ref_ref_dist/(1/2))
    else:
        rec.interpolation_score.append(np.nan)
        rec.interpolation_score.append(np.nan)
    records.append(rec)

mae=np.mean([rec.pred_err for rec in records],axis=0)
print("平均绝对预测误差：",mae)
mse=np.sqrt(np.mean([np.array(rec.pred_err)**2 for rec in records],axis=0))
print("平均预测误差：",mse)

rf=RandomForestRegressor(n_estimators=500,random_state=0,n_jobs=-1)
X=np.array([[rec.ref_num]+list(rec.ref_dist_nan)+[rec.avg_tgt_ref_dist,rec.ms_tgt_ref_dist]+list(rec.interpolation_score) for rec in records])
y=np.array([rec.pred_err[-1] for rec in records])
rf.fit(X,y)
print(f"最后一个预测器特征重要性：{rf.feature_importances_}")

# 有两个问题：
# 1. 假设参考点权重反比于距离的i次幂，那么i越大，预测越准。这可能是因为较远的参考点本该使用负权重
# 2. 用svd判断局部数据的维度，对与两种参考点分布的评分方式写出二维和一维的版本
# 3. alpha-beta-gamma滤波