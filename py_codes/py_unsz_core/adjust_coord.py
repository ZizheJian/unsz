import numpy as np
import scipy
import time
from py_codes.py_unsz_core.is_abnormal import is_abnormal

def select_file_ids(file_num,sample_file_num):
    sample_file_num=min(sample_file_num,file_num)
    file_position_float=(np.arange(sample_file_num)+0.5)*(file_num-1)/sample_file_num
    return np.round(file_position_float).astype(np.int64)

def initial_affine_parameter(dim):
    coefficient=np.zeros((dim*(dim+1)//2,dim+1),dtype=np.float64)
    for d in range(dim):
        coefficient[d*(d+3)//2,0]=1
    return coefficient.ravel()

def affine_lower_triangular_matrix(coord,parameter): # 根据坐标生成下三角变换矩阵，每个非零元素都是坐标的一次函数
    dim=coord.shape[-1]
    coefficient=parameter.reshape(dim*(dim+1)//2,dim+1)
    value=coefficient[:,0]+np.einsum("nd,kd->nk",coord,coefficient[:,1:])
    matrix=np.zeros((coord.shape[0],dim,dim),dtype=np.float64)
    for row in range(dim):
        for col in range(row+1):
            matrix[:,row,col]=value[:,row*(row+1)//2+col]
    return matrix

def transform_coord(coord,parameter,matrix_function=affine_lower_triangular_matrix):
    matrix=matrix_function(coord,parameter)
    return np.einsum("ni,nij->nj",coord,matrix)

def log_loss(value_diff,predicted_distance,epsilon=0): 
    # 返回对数距离拟合所使用的残差
    return np.log(np.abs(value_diff)+epsilon)-np.log(predicted_distance+epsilon)

def initial_constant_parameter(dim):
    parameter=np.zeros(dim*(dim+1)//2,dtype=np.float64)
    for d in range(dim):
        parameter[d*(d+3)//2]=1
    return parameter

def constant_lower_triangular_matrix(parameter,dim):
    matrix=np.zeros((dim,dim),dtype=np.float64)
    for row in range(dim):
        for col in range(row+1):
            matrix[row,col]=parameter[row*(row+1)//2+col]
    return matrix

def squared_distance_loss(value_diff,coord_diff,parameter,dim):
    transform_matrix=constant_lower_triangular_matrix(parameter,dim)
    transformed_coord_diff=coord_diff@transform_matrix
    predicted_squared_distance=np.sum(transformed_coord_diff**2,axis=1)
    return value_diff**2-predicted_squared_distance

def normalize_constant_matrix(transform_matrix):
    squared_frobenius_norm=np.sum(transform_matrix**2)
    if squared_frobenius_norm==0:
        raise RuntimeError("坐标变换矩阵不能是零矩阵")
    return transform_matrix*np.sqrt(transform_matrix.shape[0]/squared_frobenius_norm)

def positive_normalized_cholesky(transform_matrix,min_eigenvalue_ratio):
    metric_matrix=transform_matrix@transform_matrix.T
    eigenvalue,eigenvector=np.linalg.eigh(metric_matrix)
    if eigenvalue[-1]<=0:
        raise RuntimeError("坐标变换矩阵不能是零矩阵")
    eigenvalue=np.maximum(eigenvalue,eigenvalue[-1]*min_eigenvalue_ratio)
    metric_matrix=eigenvector@np.diag(eigenvalue)@eigenvector.T
    metric_matrix*=transform_matrix.shape[0]/np.trace(metric_matrix)
    metric_matrix=(metric_matrix+metric_matrix.T)/2
    return np.linalg.cholesky(metric_matrix)

def adjust_coord_cross_time(param):
    raise NotImplementedError("暂未实现跨时间步的坐标调整")

def adjust_coord_no_cross_time(param): # 访问多个文件来决定公共变换矩阵，但这个矩阵只会应用到每个文件的坐标上，而不会跨时间步使用
    first_sample_ratio=0.25 # 读取数据后，只保留这些点用于后续操作（包括建树）
    sample_file_num=3
    second_sample_ratio=0.25 # 选择点对的第一个点的时候，选择这么多点
    random_seed=0
    adjust_max_nfev=100
    iteration_num=100
    new_matrix_weight=0.3
    min_eigenvalue_ratio=1e-16
    time_dict={"adjust_coord总耗时":time.perf_counter()}

    rng=np.random.default_rng(random_seed)
    time_dict["读取sampled files"]=time.perf_counter()
    file_ids=select_file_ids(len(param.original_data_path_vec),sample_file_num)
    full_sampled_data=np.asarray([np.fromfile(param.original_data_path_vec[file_id],dtype=param.data_dtype) for file_id in file_ids])
    time_dict["读取sampled files"]=time.perf_counter()-time_dict["读取sampled files"]
    time_dict["第一次采样、应用已知变换"]=0
    time_dict["有效点检查和第二次采样"]=0
    time_dict["建树"]=0
    time_dict["最近邻查询"]=0
    time_dict["整理点对"]=0
    time_dict["合并点对数组"]=0
    time_dict["优化变换矩阵"]=0
    time_dict["正定、归一化和缓冲"]=0
    time_dict["定期变换全部坐标"]=0
    transform_matrix=np.eye(param.dim,dtype=np.float64)
    first_sample_num=int(np.ceil(param.point_num*first_sample_ratio))
    
    for iteration in range(iteration_num):
        print(f"Iteration {iteration+1}/{iteration_num}",flush=True)
        stage_start_time=time.perf_counter()
        param.first_sample_ids=np.sort(rng.choice(param.point_num,size=first_sample_num,replace=False))
        original_sampled_coord=param.coord_array[param.first_sample_ids]
        sampled_coord=original_sampled_coord@transform_matrix
        sampled_data=full_sampled_data[:,param.first_sample_ids]
        time_dict["第一次采样、应用已知变换"]+=time.perf_counter()-stage_start_time
    
        pair_coord1=[]
        pair_coord2=[]
        pair_value_diff=[]
        remaining_sample_num=int(np.ceil(sampled_data.size*second_sample_ratio))
        for sampled_file_id,data in enumerate(sampled_data):
            stage_start_time=time.perf_counter()
            valid_ids=np.flatnonzero(~is_abnormal(data))
            if valid_ids.size<=param.actual_ref_num:
                raise ValueError(f"第{file_ids[sampled_file_id]}个数据文件在第一次采样后只有{valid_ids.size}个有效点，无法寻找{param.actual_ref_num}个参考点")
            current_sample_num=min(valid_ids.size,int(np.ceil(remaining_sample_num/(len(sampled_data)-sampled_file_id))))
            target_ids=np.sort(rng.choice(valid_ids,size=current_sample_num,replace=False))
            remaining_sample_num-=current_sample_num
            time_dict["有效点检查和第二次采样"]+=time.perf_counter()-stage_start_time
    
            stage_start_time=time.perf_counter()
            tree=scipy.spatial.cKDTree(sampled_coord[valid_ids])
            time_dict["建树"]+=time.perf_counter()-stage_start_time
    
            stage_start_time=time.perf_counter()
            query_num=param.actual_ref_num+1
            _,neighbor_tree_ids=tree.query(sampled_coord[target_ids],k=query_num)
            neighbor_tree_ids=np.asarray(neighbor_tree_ids).reshape(-1,query_num)
            time_dict["最近邻查询"]+=time.perf_counter()-stage_start_time
    
            stage_start_time=time.perf_counter()
            for target_id,current_neighbor_tree_ids in zip(target_ids,neighbor_tree_ids):
                neighbor_ids=valid_ids[current_neighbor_tree_ids]
                neighbor_ids=neighbor_ids[neighbor_ids!=target_id][:param.actual_ref_num]
                pair_coord1.append(np.repeat(original_sampled_coord[target_id][None,:],neighbor_ids.size,axis=0))
                pair_coord2.append(original_sampled_coord[neighbor_ids])
                pair_value_diff.append(data[target_id]-data[neighbor_ids])
            time_dict["整理点对"]+=time.perf_counter()-stage_start_time
    
        stage_start_time=time.perf_counter()
        pair_coord1=np.concatenate(pair_coord1)
        pair_coord2=np.concatenate(pair_coord2)
        pair_value_diff=np.concatenate(pair_value_diff)
        pair_coord_diff=pair_coord1-pair_coord2
        time_dict["合并点对数组"]+=time.perf_counter()-stage_start_time
    
        def residual(parameter):
            return squared_distance_loss(pair_value_diff,pair_coord_diff,parameter,param.dim)
    
        stage_start_time=time.perf_counter()
        initial_parameter=transform_matrix[np.tril_indices(param.dim)]
        result=scipy.optimize.least_squares(residual,initial_parameter,max_nfev=adjust_max_nfev,verbose=0)
        time_dict["优化变换矩阵"]+=time.perf_counter()-stage_start_time
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
            raise RuntimeError("坐标变换优化失败：参数或残差中出现了非有限值")

        stage_start_time=time.perf_counter()
        fitted_transform_matrix=constant_lower_triangular_matrix(result.x,param.dim)
        fitted_transform_matrix=positive_normalized_cholesky(fitted_transform_matrix,min_eigenvalue_ratio)
        if iteration==0:
            transform_matrix=fitted_transform_matrix
        else:
            transform_matrix=(1-new_matrix_weight)*transform_matrix+new_matrix_weight*fitted_transform_matrix
            transform_matrix=normalize_constant_matrix(transform_matrix)
        time_dict["正定、归一化和缓冲"]+=time.perf_counter()-stage_start_time

        normalized_loss=np.mean(squared_distance_loss(pair_value_diff,pair_coord_diff,transform_matrix[np.tril_indices(param.dim)],param.dim)**2)
        if (iteration+1)%10==0 and iteration+1<iteration_num:
            stage_start_time=time.perf_counter()
            iteration_transformed_coord=param.coord_array@transform_matrix
            time_dict["定期变换全部坐标"]+=time.perf_counter()-stage_start_time
            print(f"Iteration {iteration+1}: loss={normalized_loss}")
            print(f"transformed coord range: [{', '.join(f'[{minimum},{maximum}]' for minimum,maximum in zip(iteration_transformed_coord.min(axis=0),iteration_transformed_coord.max(axis=0)))}]")
            print(f"transform matrix:\n{transform_matrix}")
    
    time_dict["最终变换全部坐标"]=time.perf_counter()
    param.coord_transform_parameter=transform_matrix[np.tril_indices(param.dim)]
    param.coord_transform_matrix=transform_matrix
    transformed_coord=param.coord_array@transform_matrix
    time_dict["最终变换全部坐标"]=time.perf_counter()-time_dict["最终变换全部坐标"]
    time_dict["adjust_coord总耗时"]=time.perf_counter()-time_dict["adjust_coord总耗时"]
    
    print(f"Iteration {iteration_num}: loss={normalized_loss}")
    print(f"original coord range: [{', '.join(f'[{minimum},{maximum}]' for minimum,maximum in zip(param.coord_array.min(axis=0),param.coord_array.max(axis=0)))}]")
    print(f"transformed coord range: [{', '.join(f'[{minimum},{maximum}]' for minimum,maximum in zip(transformed_coord.min(axis=0),transformed_coord.max(axis=0)))}]")
    print(f"transform matrix at original coord center {(param.min_coord+param.max_coord)/2}:\n{transform_matrix}")
    print(f"metric matrix eigenvalues: {np.linalg.eigvalsh(transform_matrix@transform_matrix.T)}")
    print("累计分段耗时：")
    for name,value in time_dict.items():
        if name!="adjust_coord总耗时":
            print(f"  {name}：{value:.6f} s")
    print(f"  adjust_coord总耗时：{time_dict['adjust_coord总耗时']:.6f} s")

def adjust_coord(param)->None:
    if param.cross_time_compress:
        adjust_coord_cross_time(param)
    else:
        adjust_coord_no_cross_time(param)

    
