import numpy as np


def log_loss(value_diff,predicted_distance,value_epsilon,distance_epsilon):
    """返回对数距离拟合所使用的残差，之后可替换为其他损失。"""
    return np.log(np.abs(value_diff)+value_epsilon)-np.log(predicted_distance+distance_epsilon)


def affine_lower_triangular_matrix(coord,parameter,coord_center,coord_scale):
    """根据坐标生成下三角变换矩阵；每个非零元素都是坐标的一次函数。"""
    dim=coord.shape[-1]
    normalized_coord=(coord-coord_center)/coord_scale
    coefficient=parameter.reshape(dim*(dim+1)//2,dim+1)
    value=coefficient[:,0]+normalized_coord@coefficient[:,1:].T
    matrix=np.zeros((*coord.shape[:-1],dim,dim),dtype=np.float64)
    lower_row,lower_col=np.tril_indices(dim)
    matrix[...,lower_row,lower_col]=value
    return matrix


def transform_coord(coord,parameter,coord_center,coord_scale,matrix_function=affine_lower_triangular_matrix):
    matrix=matrix_function(coord,parameter,coord_center,coord_scale)
    return np.einsum("...i,...ij->...j",coord,matrix)


def initial_affine_parameter(dim):
    coefficient=np.zeros((dim*(dim+1)//2,dim+1),dtype=np.float64)
    lower_row,lower_col=np.tril_indices(dim)
    coefficient[lower_row==lower_col,0]=1
    return coefficient.ravel()


def select_file_ids(file_num,sample_file_num):
    if sample_file_num<=0:
        raise ValueError(f"sample_file_num必须是正数，当前值为{sample_file_num}")
    if sample_file_num>file_num:
        raise ValueError(f"sample_file_num不能超过数据文件数，当前分别为{sample_file_num}和{file_num}")
    return ((np.arange(sample_file_num)+0.5)*file_num/sample_file_num).astype(np.int64)


def adjust_coord(self)->None:
    try:
        from scipy.optimize import least_squares
        from scipy.spatial import cKDTree
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("adjust_coord需要SciPy，请先安装scipy") from error

    if self.cross_time_compress:
        raise NotImplementedError("暂未实现跨时间步的坐标调整")
    if not 0<self.first_sample_ratio<=1:
        raise ValueError(f"first_sample_ratio必须在(0,1]内，当前值为{self.first_sample_ratio}")
    if self.actual_ref_num<=0:
        raise ValueError(f"actual_ref_num必须是正数，当前值为{self.actual_ref_num}")
    if self.second_sample_point_num<=0:
        raise ValueError(f"second_sample_point_num必须是正数，当前值为{self.second_sample_point_num}")

    rng=np.random.default_rng(self.random_seed)
    original_point_num=self.point_num
    first_sample_num=max(1,int(np.ceil(original_point_num*self.first_sample_ratio)))
    self.first_sample_ids=np.sort(rng.choice(original_point_num,size=first_sample_num,replace=False))
    sampled_coord=self.coord_array[self.first_sample_ids].astype(np.float64,copy=True)

    file_ids=select_file_ids(len(self.original_data_path_vec),self.sample_file_num)
    sampled_data=[]
    for file_id in file_ids:
        data=np.fromfile(self.original_data_path_vec[file_id],dtype=self.data_dtype)
        sampled_data.append(data[self.first_sample_ids])
    sampled_data=np.asarray(sampled_data)

    pair_coord1=[]
    pair_coord2=[]
    pair_value_diff=[]
    remaining_sample_num=self.second_sample_point_num
    for sampled_file_id,data in enumerate(sampled_data):
        valid_mask=np.isfinite(data)&(data>=self.abnormal_threshold)
        valid_ids=np.flatnonzero(valid_mask)
        if valid_ids.size<=self.actual_ref_num:
            raise ValueError(f"第{file_ids[sampled_file_id]}个数据文件在第一次采样后只有{valid_ids.size}个有效点，无法寻找{self.actual_ref_num}个参考点")

        current_sample_num=min(valid_ids.size,int(np.ceil(remaining_sample_num/(len(sampled_data)-sampled_file_id))))
        target_ids=np.sort(rng.choice(valid_ids,size=current_sample_num,replace=False))
        remaining_sample_num-=current_sample_num

        tree=cKDTree(sampled_coord[valid_ids])
        query_num=min(valid_ids.size,self.actual_ref_num+1)
        _,neighbor_tree_ids=tree.query(sampled_coord[target_ids],k=query_num)
        neighbor_tree_ids=np.asarray(neighbor_tree_ids)
        if neighbor_tree_ids.ndim==1:
            neighbor_tree_ids=neighbor_tree_ids[:,None]

        for target_id,current_neighbor_tree_ids in zip(target_ids,neighbor_tree_ids):
            neighbor_ids=valid_ids[current_neighbor_tree_ids]
            neighbor_ids=neighbor_ids[neighbor_ids!=target_id][:self.actual_ref_num]
            if neighbor_ids.size!=self.actual_ref_num:
                raise RuntimeError(f"点{self.first_sample_ids[target_id]}没有找到{self.actual_ref_num}个不同于自身的有效参考点")
            pair_coord1.append(np.repeat(sampled_coord[target_id][None,:],self.actual_ref_num,axis=0))
            pair_coord2.append(sampled_coord[neighbor_ids])
            pair_value_diff.append(data[target_id]-data[neighbor_ids])

    pair_coord1=np.concatenate(pair_coord1)
    pair_coord2=np.concatenate(pair_coord2)
    pair_value_diff=np.concatenate(pair_value_diff).astype(np.float64,copy=False)
    coord_center=sampled_coord.mean(axis=0)
    coord_scale=sampled_coord.std(axis=0)
    coord_scale[coord_scale==0]=1

    def residual(parameter):
        transformed_coord1=transform_coord(pair_coord1,parameter,coord_center,coord_scale)
        transformed_coord2=transform_coord(pair_coord2,parameter,coord_center,coord_scale)
        predicted_distance=np.linalg.norm(transformed_coord1-transformed_coord2,axis=1)
        return log_loss(pair_value_diff,predicted_distance,self.log_value_epsilon,self.log_distance_epsilon)

    initial_parameter=initial_affine_parameter(self.dim)
    result=least_squares(residual,initial_parameter,max_nfev=self.adjust_max_nfev,verbose=2)
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun).all():
        raise RuntimeError("坐标变换优化失败：参数或残差中出现了非有限值")
    if not result.success:
        print(f"坐标变换优化提前停止，将使用当前最优结果：{result.message}")

    self.coord_transform_parameter=result.x
    self.coord_transform_center=coord_center
    self.coord_transform_scale=coord_scale
    transformed_coord=transform_coord(sampled_coord,result.x,coord_center,coord_scale)
    self.coord_array=transformed_coord.astype(self.coord_dtype,copy=False)
    self.point_num=first_sample_num
    self.min_coord=self.coord_array.min(axis=0)
    self.max_coord=self.coord_array.max(axis=0)

    print(f"第一次采样点数：{first_sample_num}/{original_point_num}")
    print(f"点对数量：{pair_value_diff.size}")
    print(f"优化后的损失：{np.mean(result.fun**2)}")
    print(f"坐标变换参数：\n{result.x.reshape(-1,self.dim+1)}")
