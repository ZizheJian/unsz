import numpy as np
import scipy
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

def adjust_coord(param)->None:
    first_sample_ratio=0.25 # 读取数据后，只保留这些点用于后续操作（包括建树）
    sample_file_num=3
    second_sample_ratio=0.25 # 选择点对的第一个点的时候，选择这么多点
    random_seed=0
    adjust_max_nfev=100

    if param.cross_time_compress:
        raise NotImplementedError("暂未实现跨时间步的坐标调整")
    else:
        rng=np.random.default_rng(random_seed)
        first_sample_num=int(np.ceil(param.point_num*first_sample_ratio))
        param.first_sample_ids=np.sort(rng.choice(param.point_num,size=first_sample_num,replace=False))
        sampled_coord=param.coord_array[param.first_sample_ids]

        file_ids=select_file_ids(len(param.original_data_path_vec),sample_file_num)
        sampled_data=[]
        for file_id in file_ids:
            data=np.fromfile(param.original_data_path_vec[file_id],dtype=param.data_dtype)
            sampled_data.append(data[param.first_sample_ids])
        sampled_data=np.asarray(sampled_data)

        pair_coord1=[]
        pair_coord2=[]
        pair_value_diff=[]
        pair_abs_eb=[]
        remaining_sample_num=int(np.ceil(sampled_data.size*second_sample_ratio))
        for sampled_file_id,data in enumerate(sampled_data):
            valid_mask=~is_abnormal(data)
            valid_ids=np.flatnonzero(valid_mask)

            current_sample_num=min(valid_ids.size,int(np.ceil(remaining_sample_num/(len(sampled_data)-sampled_file_id))))
            target_ids=np.sort(rng.choice(valid_ids,size=current_sample_num,replace=False))
            remaining_sample_num-=current_sample_num

            tree=scipy.spatial.cKDTree(sampled_coord[valid_ids])
            query_num=min(valid_ids.size,param.actual_ref_num+1)
            _,neighbor_tree_ids=tree.query(sampled_coord[target_ids],k=query_num)
            neighbor_tree_ids=np.asarray(neighbor_tree_ids).reshape(-1,query_num)

            for target_id,current_neighbor_tree_ids in zip(target_ids,neighbor_tree_ids):
                neighbor_ids=valid_ids[current_neighbor_tree_ids]
                neighbor_ids=neighbor_ids[neighbor_ids!=target_id]
                pair_coord1.append(np.repeat(sampled_coord[target_id][None,:],neighbor_ids.size,axis=0))
                pair_coord2.append(sampled_coord[neighbor_ids])
                pair_value_diff.append(data[target_id]-data[neighbor_ids])
                pair_abs_eb.append(np.full(neighbor_ids.size,param.abs_eb[file_ids[sampled_file_id]]))

        pair_coord1=np.concatenate(pair_coord1)
        pair_coord2=np.concatenate(pair_coord2)
        pair_value_diff=np.concatenate(pair_value_diff)
        pair_abs_eb=np.concatenate(pair_abs_eb)

        def residual(parameter):
            transformed_coord1=transform_coord(pair_coord1,parameter)
            transformed_coord2=transform_coord(pair_coord2,parameter)
            predicted_distance=np.linalg.norm(transformed_coord1-transformed_coord2,axis=1)
            return log_loss(pair_value_diff,predicted_distance,pair_abs_eb)

        initial_parameter=initial_affine_parameter(param.dim)
        result=scipy.optimize.least_squares(residual,initial_parameter,max_nfev=adjust_max_nfev,verbose=2)
        if not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
            raise RuntimeError("坐标变换优化失败：参数或残差中出现了非有限值")
        if not result.success:
            print(f"坐标变换优化提前停止，将使用当前最优结果：{result.message}")

        param.coord_transform_parameter=result.x
        transformed_coord=transform_coord(param.coord_array,result.x)
        center_transform_matrix=affine_lower_triangular_matrix(((param.min_coord+param.max_coord)/2)[None,:],result.x)[0]
        print(f"original coord range: [{', '.join(f'[{minimum},{maximum}]' 
            for minimum,maximum in zip(param.coord_array.min(axis=0),param.coord_array.max(axis=0)))}]")
        print(f"transformed coord range: [{', '.join(f'[{minimum},{maximum}]'
            for minimum,maximum in zip(transformed_coord.min(axis=0),transformed_coord.max(axis=0)))}]")
        print(f"transform matrix at original coord center:\n{center_transform_matrix}")
