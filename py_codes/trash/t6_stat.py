# 从目标点最近的10个正常点中随机选择1~10个参考点，分析几何和参考值指标对反比例预测误差的解释能力。

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split

data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina"
record_num=10000
min_ref_num=1
max_ref_num=10
inverse_distance_power=3.61
random_seed=0
coord_transform_matrix=np.array([[1.12670515e0,0,0],[2.70622568e-1,1.28736120e0,0],[-1.67488548e-4,-1.83679099e-4,3.29459469e-6]],dtype=np.float64)

def max_min_lambda_score(target_coord,ref_coord):
    ref_num,dim=ref_coord.shape
    objective=np.zeros(ref_num+1)
    objective[-1]=-1
    inequality_matrix=np.zeros((ref_num,ref_num+1))
    inequality_matrix[:,:ref_num]=-np.eye(ref_num)
    inequality_matrix[:,-1]=1
    equality_matrix=np.zeros((dim+1,ref_num+1))
    equality_matrix[:dim,:ref_num]=(ref_coord-target_coord).T
    equality_matrix[dim,:ref_num]=1
    result=linprog(objective,A_ub=inequality_matrix,b_ub=np.zeros(ref_num),A_eq=equality_matrix,b_eq=np.r_[np.zeros(dim),ref_num],bounds=[(None,None)]*(ref_num+1),method="highs")
    return result.x[-1] if result.success else np.nan

def mean_pairwise_squared_distance_linear(coord):
    point_num=coord.shape[0]
    if point_num<2:
        return np.nan
    relative_coord=coord-coord[0]
    pairwise_squared_distance_sum=point_num*np.sum(relative_coord**2)-np.sum(np.sum(relative_coord,axis=0)**2)
    return pairwise_squared_distance_sum/(point_num*(point_num-1)/2)

def predict(ref_value,ref_distance,power):
    weight=ref_distance**(-power)
    weight/=np.sum(weight)
    return np.sum(weight*ref_value),weight

def evaluate_feature_group(name,feature_array,target_log_error,feature_ids):
    train_ids,test_ids=train_test_split(np.arange(target_log_error.size),test_size=0.3,random_state=random_seed)
    model=RandomForestRegressor(n_estimators=300,min_samples_leaf=8,max_features=0.8,random_state=random_seed,n_jobs=1)
    model.fit(feature_array[train_ids][:,feature_ids],target_log_error[train_ids])
    prediction=model.predict(feature_array[test_ids][:,feature_ids])
    mae=mean_absolute_error(target_log_error[test_ids],prediction)
    r2=r2_score(target_log_error[test_ids],prediction)
    rank_correlation=spearmanr(prediction,target_log_error[test_ids]).statistic
    print(f"{name}：test log-MAE={mae:.6f},R2={r2:.6f},预测值与实际log(error)的Spearman相关={rank_correlation:.6f}")
    importance=permutation_importance(model,feature_array[test_ids][:,feature_ids],target_log_error[test_ids],scoring="neg_mean_absolute_error",n_repeats=5,random_state=random_seed,n_jobs=1)
    order=np.argsort(importance.importances_mean)[::-1]
    for local_id in order:
        feature_id=feature_ids[local_id]
        print(f"  {feature_name_vec[feature_id]}：置乱后log-MAE增加{importance.importances_mean[local_id]:.6f}（基线log-MAE的{importance.importances_mean[local_id]/mae*100:.2f}%）")

coord=np.fromfile(f"{data_dir}/coords.dat",dtype=np.float32).reshape(-1,3).astype(np.float64)
data=np.fromfile(f"{data_dir}/attr0.dat.80",dtype=np.float32).astype(np.float64)
normal=np.isfinite(data)&(data>=-1000)
coord=coord[normal]@coord_transform_matrix
data=data[normal]

tree=cKDTree(coord)
nearest_distance,nearest_id=tree.query(coord,k=max_ref_num+1,workers=1)
nearest_distance=nearest_distance[:,1:]
nearest_id=nearest_id[:,1:]
rng=np.random.default_rng(random_seed)

feature_name_vec=["ref_num","nearest_ref_dist","farthest_ref_dist","mean_ref_dist","rms_ref_dist","ref_dist_cv","centroid_offset_ratio","direction_resultant","direction_cov_min_ratio","distribution_s2","convex_score","weighted_ref_std","ref_value_range","gradient_proxy"]
feature_vec=[]
prediction_error_vec=[]
power_error_vec=[[] for _ in np.arange(0,5,0.5)]
for record_id in range(record_num):
    target_id=rng.integers(coord.shape[0])
    ref_num=rng.integers(min_ref_num,max_ref_num+1)
    ref_seq=np.sort(rng.choice(max_ref_num,size=ref_num,replace=False))
    ref_id=nearest_id[target_id,ref_seq]
    ref_distance=nearest_distance[target_id,ref_seq]
    ref_coord=coord[ref_id]
    ref_value=data[ref_id]
    target_coord=coord[target_id]
    predicted_value,weight=predict(ref_value,ref_distance,inverse_distance_power)
    prediction_error_vec.append(abs(data[target_id]-predicted_value))
    for power_id,power in enumerate(np.arange(0,5,0.5)):
        power_prediction,_=predict(ref_value,ref_distance,power)
        power_error_vec[power_id].append(abs(data[target_id]-power_prediction))
    mean_ref_dist=np.mean(ref_distance)
    rms_ref_dist=np.sqrt(np.mean(ref_distance**2))
    ref_dist_cv=np.std(ref_distance)/mean_ref_dist if mean_ref_dist>0 else 0
    delta_coord=ref_coord-target_coord
    centroid_offset_ratio=np.linalg.norm(np.mean(delta_coord,axis=0))/rms_ref_dist if rms_ref_dist>0 else 0
    unit_direction=delta_coord/ref_distance[:,None]
    direction_resultant=np.linalg.norm(np.mean(unit_direction,axis=0))
    direction_covariance=unit_direction.T@unit_direction/ref_num
    direction_cov_min_ratio=np.linalg.eigvalsh(direction_covariance)[0]/np.trace(direction_covariance)
    mean_target_ref_squared_dist=np.mean(ref_distance**2)
    mean_ref_ref_squared_dist=mean_pairwise_squared_distance_linear(ref_coord)
    distribution_s2=2*mean_target_ref_squared_dist/mean_ref_ref_squared_dist if np.isfinite(mean_ref_ref_squared_dist) and mean_ref_ref_squared_dist>0 else np.nan
    convex_score=max_min_lambda_score(target_coord,ref_coord) if ref_num>=4 else np.nan
    weighted_ref_std=np.sqrt(np.sum(weight*(ref_value-predicted_value)**2))
    ref_value_range=np.ptp(ref_value)
    gradient_proxy=weighted_ref_std/rms_ref_dist if rms_ref_dist>0 else 0
    feature_vec.append([ref_num,ref_distance[0],ref_distance[-1],mean_ref_dist,rms_ref_dist,ref_dist_cv,centroid_offset_ratio,direction_resultant,direction_cov_min_ratio,distribution_s2,convex_score,weighted_ref_std,ref_value_range,gradient_proxy])

feature_array=np.asarray(feature_vec)
prediction_error=np.asarray(prediction_error_vec)
print(f"记录数量：{record_num},正常点数量：{coord.shape[0]},目标预测误差MAE={np.mean(prediction_error):.9g},RMSE={np.sqrt(np.mean(prediction_error**2)):.9g}")
print("不同距离指数的误差：")
for power,error_vec in zip(np.arange(0,5,0.5),power_error_vec):
    error=np.asarray(error_vec)
    print(f"  power={power:.1f}：MAE={np.mean(error):.9g},RMSE={np.sqrt(np.mean(error**2)):.9g}")

log_error=np.log(prediction_error+np.finfo(np.float64).tiny)
print("各指标与log(error)的Spearman相关系数：")
for feature_id,feature_name in enumerate(feature_name_vec):
    finite=np.isfinite(feature_array[:,feature_id])
    correlation=spearmanr(feature_array[finite,feature_id],log_error[finite]).statistic
    print(f"  {feature_name}：{correlation:.6f}")
distance_feature_ids=[1,2,3,4]
distance_correlation=spearmanr(feature_array[:,distance_feature_ids]).statistic
print("距离指标之间的Spearman相关系数：")
for row_id,feature_id in enumerate(distance_feature_ids):
    print(f"  {feature_name_vec[feature_id]}："+",".join(f"{distance_correlation[row_id,column_id]:.6f}" for column_id in range(len(distance_feature_ids))))

geometry_feature_ids=list(range(11))
geometry_without_mean_feature_ids=[feature_id for feature_id in geometry_feature_ids if feature_id!=3]
nearest_and_distribution_feature_ids=[0,1,6,7,8,9,10]
value_feature_ids=list(range(11,14))
median_log_error=np.median(log_error)
_,test_ids=train_test_split(np.arange(prediction_error.size),test_size=0.3,random_state=random_seed)
constant_mae=mean_absolute_error(log_error[test_ids],np.full(test_ids.size,median_log_error))
print(f"常数中位数基线：test log-MAE={constant_mae:.6f}")
evaluate_feature_group("仅几何指标",feature_array,log_error,geometry_feature_ids)
evaluate_feature_group("几何指标去掉平均参考点距离",feature_array,log_error,geometry_without_mean_feature_ids)
evaluate_feature_group("只保留最近距离和非距离几何指标",feature_array,log_error,nearest_and_distribution_feature_ids)
evaluate_feature_group("仅参考值指标",feature_array,log_error,value_feature_ids)
evaluate_feature_group("全部指标",feature_array,log_error,geometry_feature_ids+value_feature_ids)
