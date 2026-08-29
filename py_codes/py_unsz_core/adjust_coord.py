import numpy as np
import scipy
import time
from pathlib import Path
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

def fourth_power_loss(value_diff,coord_diff,parameter,dim):
    transform_matrix=constant_lower_triangular_matrix(parameter,dim)
    transformed_coord_diff=coord_diff@transform_matrix
    predicted_squared_distance=np.sum(transformed_coord_diff**2,axis=1)
    return value_diff**2-predicted_squared_distance

def fourth_power_loss_jacobian(coord_diff,parameter,dim):
    transform_matrix=constant_lower_triangular_matrix(parameter,dim)
    transformed_coord_diff=coord_diff@transform_matrix
    jacobian=np.empty((coord_diff.shape[0],parameter.size),dtype=np.float64)
    parameter_id=0
    for row in range(dim):
        for col in range(row+1):
            jacobian[:,parameter_id]=-2*coord_diff[:,row]*transformed_coord_diff[:,col]
            parameter_id+=1
    return jacobian

def second_power_loss(value_diff,coord_diff,parameter,dim):
    transform_matrix=constant_lower_triangular_matrix(parameter,dim)
    transformed_coord_diff=coord_diff@transform_matrix
    predicted_distance=np.sqrt(np.sum(transformed_coord_diff**2,axis=1))
    return np.abs(value_diff)-predicted_distance

def second_power_loss_jacobian(coord_diff,parameter,dim):
    transform_matrix=constant_lower_triangular_matrix(parameter,dim)
    transformed_coord_diff=coord_diff@transform_matrix
    predicted_distance=np.sqrt(np.sum(transformed_coord_diff**2,axis=1))
    inverse_distance=np.divide(1,predicted_distance,out=np.zeros_like(predicted_distance),where=predicted_distance!=0)
    jacobian=np.empty((coord_diff.shape[0],parameter.size),dtype=np.float64)
    parameter_id=0
    for row in range(dim):
        for col in range(row+1):
            jacobian[:,parameter_id]=-coord_diff[:,row]*transformed_coord_diff[:,col]*inverse_distance
            parameter_id+=1
    return jacobian

def second_power_normal_equation(value_diff,coord_diff,parameter,dim,chunk_size=65536):
    transform_matrix=constant_lower_triangular_matrix(parameter,dim)
    normal_matrix=np.zeros((parameter.size,parameter.size),dtype=np.float64)
    normal_gradient=np.zeros(parameter.size,dtype=np.float64)
    squared_loss=0
    for begin in range(0,value_diff.size,chunk_size):
        end=min(begin+chunk_size,value_diff.size)
        current_coord_diff=coord_diff[begin:end]
        transformed_coord_diff=current_coord_diff@transform_matrix
        predicted_distance=np.sqrt(np.sum(transformed_coord_diff**2,axis=1))
        residual=np.abs(value_diff[begin:end])-predicted_distance
        inverse_distance=np.divide(1,predicted_distance,out=np.zeros_like(predicted_distance),where=predicted_distance!=0)
        jacobian=np.empty((end-begin,parameter.size),dtype=np.float64)
        parameter_id=0
        for row in range(dim):
            for col in range(row+1):
                jacobian[:,parameter_id]=-current_coord_diff[:,row]*transformed_coord_diff[:,col]*inverse_distance
                parameter_id+=1
        normal_matrix+=jacobian.T@jacobian
        normal_gradient+=jacobian.T@residual
        squared_loss+=residual@residual
    return squared_loss,normal_matrix,normal_gradient

def optimize_second_power_gauss_newton(value_diff,coord_diff,initial_parameter,dim,max_nfev):
    parameter=initial_parameter.copy()
    nfev=0
    njev=0
    optimality=np.inf
    for _ in range(max_nfev):
        squared_loss,normal_matrix,normal_gradient=second_power_normal_equation(value_diff,coord_diff,parameter,dim)
        nfev+=1
        njev+=1
        optimality=np.max(np.abs(normal_gradient))
        step=np.linalg.lstsq(normal_matrix,-normal_gradient,rcond=None)[0]
        if np.linalg.norm(step)<=1e-10*(np.linalg.norm(parameter)+1e-10):
            break
        accepted=False
        step_ratio=1
        while nfev<max_nfev and step_ratio>=1/256:
            candidate_parameter=parameter+step_ratio*step
            candidate_residual=second_power_loss(value_diff,coord_diff,candidate_parameter,dim)
            candidate_squared_loss=candidate_residual@candidate_residual
            nfev+=1
            if candidate_squared_loss<squared_loss:
                relative_reduction=(squared_loss-candidate_squared_loss)/max(squared_loss,np.finfo(np.float64).tiny)
                parameter=candidate_parameter
                accepted=True
                break
            step_ratio/=2
        if not accepted or relative_reduction<=1e-10:
            break
    residual=second_power_loss(value_diff,coord_diff,parameter,dim)
    return scipy.optimize.OptimizeResult(x=parameter,fun=residual,nfev=nfev,njev=njev,optimality=optimality)

def second_power_value_gradient(value_diff,coord_diff,parameter,dim,chunk_size=65536):
    transform_matrix=constant_lower_triangular_matrix(parameter,dim)
    squared_loss=0
    gradient=np.zeros(parameter.size,dtype=np.float64)
    for begin in range(0,value_diff.size,chunk_size):
        end=min(begin+chunk_size,value_diff.size)
        current_coord_diff=coord_diff[begin:end]
        transformed_coord_diff=current_coord_diff@transform_matrix
        predicted_distance=np.sqrt(np.sum(transformed_coord_diff**2,axis=1))
        residual=np.abs(value_diff[begin:end])-predicted_distance
        inverse_distance=np.divide(1,predicted_distance,out=np.zeros_like(predicted_distance),where=predicted_distance!=0)
        parameter_id=0
        for row in range(dim):
            for col in range(row+1):
                jacobian_column=-current_coord_diff[:,row]*transformed_coord_diff[:,col]*inverse_distance
                gradient[parameter_id]+=jacobian_column@residual
                parameter_id+=1
        squared_loss+=residual@residual
    inverse_num=1/value_diff.size
    return squared_loss*inverse_num,2*gradient*inverse_num

def optimize_second_power_lbfgs(value_diff,coord_diff,initial_parameter,dim,max_iteration):
    result=scipy.optimize.minimize(lambda parameter:second_power_value_gradient(value_diff,coord_diff,parameter,dim),initial_parameter,jac=True,method="L-BFGS-B",options={"maxiter":max_iteration,"ftol":1e-12,"gtol":1e-8,"maxls":20})
    residual=second_power_loss(value_diff,coord_diff,result.x,dim)
    return scipy.optimize.OptimizeResult(x=result.x,fun=residual,nfev=result.nfev,njev=result.njev,optimality=np.max(np.abs(result.jac)))

def log_distance_loss(value_diff,coord_diff,abs_eb,parameter,dim):
    transform_matrix=constant_lower_triangular_matrix(parameter,dim)
    transformed_coord_diff=coord_diff@transform_matrix
    predicted_distance=np.sqrt(np.sum(transformed_coord_diff**2,axis=1))
    epsilon=np.maximum(abs_eb,np.finfo(np.float64).tiny)
    return np.log(np.abs(value_diff)+epsilon)-np.log(predicted_distance+epsilon)

def log_distance_loss_jacobian(coord_diff,abs_eb,parameter,dim):
    transform_matrix=constant_lower_triangular_matrix(parameter,dim)
    transformed_coord_diff=coord_diff@transform_matrix
    predicted_distance=np.sqrt(np.sum(transformed_coord_diff**2,axis=1))
    epsilon=np.maximum(abs_eb,np.finfo(np.float64).tiny)
    inverse_denominator=np.divide(1,predicted_distance*(predicted_distance+epsilon),out=np.zeros_like(predicted_distance),where=predicted_distance!=0)
    jacobian=np.empty((coord_diff.shape[0],parameter.size),dtype=np.float64)
    parameter_id=0
    for row in range(dim):
        for col in range(row+1):
            jacobian[:,parameter_id]=-coord_diff[:,row]*transformed_coord_diff[:,col]*inverse_denominator
            parameter_id+=1
    return jacobian

def inverse_distance_prediction_error(target_value,ref_value,coord_diff,parameter,dim,inverse_distance_power):
    transform_matrix=constant_lower_triangular_matrix(parameter,dim)
    transformed_coord_diff=np.einsum("nkd,df->nkf",coord_diff,transform_matrix)
    squared_distance=np.einsum("nkd,nkd->nk",transformed_coord_diff,transformed_coord_diff)
    zero_distance=squared_distance==0
    safe_squared_distance=np.where(zero_distance,1,squared_distance)
    weight=safe_squared_distance**(-inverse_distance_power/2)
    predicted_value=np.sum(weight*ref_value,axis=1)/np.sum(weight,axis=1)
    has_zero_distance=np.any(zero_distance,axis=1)
    if np.any(has_zero_distance):
        row_id=np.flatnonzero(has_zero_distance)
        predicted_value[row_id]=ref_value[row_id,np.argmax(zero_distance[row_id],axis=1)]
    return target_value-predicted_value

def generalized_gaussian_bit_residual(normalized_error,file_group,shape):
    if not shape>0:
        raise ValueError("广义高斯分布的形状参数必须大于0")
    residual=np.empty(normalized_error.size,dtype=np.float64)
    for group_id in np.unique(file_group):
        group_mask=file_group==group_id
        group_error=normalized_error[group_mask]
        scale=(shape*np.mean(np.abs(group_error)**shape))**(1/shape)
        scale=max(scale,np.finfo(np.float64).eps)
        probability=scipy.stats.gennorm.cdf(group_error+1,shape,scale=scale)-scipy.stats.gennorm.cdf(group_error-1,shape,scale=scale)
        estimated_bit=-np.log2(np.clip(probability,np.finfo(np.float64).tiny,1))
        residual[group_mask]=np.sqrt(estimated_bit)
    return residual

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

def adjust_coord_no_cross_time(param,loss_function_name="fourth_power",iteration_num=100,matrix_history_path=None,first_sample_ratio=0.25,second_sample_ratio=0.25,new_matrix_weight=0.3,adjust_max_nfev=100,sample_file_num=3,matrix_save_interval=5,optimizer_name="least_squares"): # 访问多个文件来决定公共变换矩阵，但这个矩阵只会应用到每个文件的坐标上，而不会跨时间步使用
    generalized_gaussian_shape=0.5
    inverse_distance_power=3.61
    random_seed=0
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
    matrix_history=[]
    iteration_history=[]
    loss_history=[]
    nfev_history=[]
    njev_history=[]
    optimality_history=[]
    elapsed_time_history=[]
    pair_num_history=[]
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
    
        prediction_coord_diff=[]
        prediction_target_value=[]
        prediction_ref_value=[]
        prediction_abs_eb=[]
        prediction_file_group=[]
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
            neighbor_ids=valid_ids[neighbor_tree_ids]
            nonself_mask=neighbor_ids!=target_ids[:,None]
            selected_mask=nonself_mask & (np.cumsum(nonself_mask,axis=1)<=param.actual_ref_num)
            selected_num=np.sum(selected_mask,axis=1)
            if np.any(selected_num!=param.actual_ref_num):
                failed_target_id=target_ids[np.flatnonzero(selected_num!=param.actual_ref_num)[0]]
                failed_selected_num=selected_num[np.flatnonzero(selected_num!=param.actual_ref_num)[0]]
                raise RuntimeError(f"第{file_ids[sampled_file_id]}个数据文件的目标点{failed_target_id}只找到了{failed_selected_num}个非自身参考点")
            selected_neighbor_ids=neighbor_ids[selected_mask].reshape(-1,param.actual_ref_num)
            prediction_coord_diff.append(original_sampled_coord[target_ids,None,:]-original_sampled_coord[selected_neighbor_ids])
            prediction_target_value.append(data[target_ids])
            prediction_ref_value.append(data[selected_neighbor_ids])
            prediction_abs_eb.append(np.full(target_ids.size,param.abs_eb[file_ids[sampled_file_id]]))
            prediction_file_group.append(np.full(target_ids.size,sampled_file_id))
            time_dict["整理点对"]+=time.perf_counter()-stage_start_time
    
        stage_start_time=time.perf_counter()
        prediction_coord_diff=np.concatenate(prediction_coord_diff).astype(np.float64,copy=False)
        prediction_target_value=np.concatenate(prediction_target_value).astype(np.float64,copy=False)
        prediction_ref_value=np.concatenate(prediction_ref_value).astype(np.float64,copy=False)
        prediction_abs_eb=np.concatenate(prediction_abs_eb).astype(np.float64,copy=False)
        prediction_file_group=np.concatenate(prediction_file_group).astype(np.int64,copy=False)
        pair_coord_diff=prediction_coord_diff.reshape(-1,param.dim)
        pair_value_diff=(prediction_target_value[:,None]-prediction_ref_value).ravel()
        pair_abs_eb=np.repeat(prediction_abs_eb,param.actual_ref_num)
        time_dict["合并点对数组"]+=time.perf_counter()-stage_start_time
    
        def residual(parameter):
            if loss_function_name=="fourth_power":
                return fourth_power_loss(pair_value_diff,pair_coord_diff,parameter,param.dim)
            if loss_function_name=="second_power":
                return second_power_loss(pair_value_diff,pair_coord_diff,parameter,param.dim)
            if loss_function_name=="log_distance":
                return log_distance_loss(pair_value_diff,pair_coord_diff,pair_abs_eb,parameter,param.dim)
            if loss_function_name=="prediction_ggd_bit":
                prediction_error=inverse_distance_prediction_error(prediction_target_value,prediction_ref_value,prediction_coord_diff,parameter,param.dim,inverse_distance_power)
                return generalized_gaussian_bit_residual(prediction_error/prediction_abs_eb,prediction_file_group,generalized_gaussian_shape)
            raise ValueError(f"未知的坐标调整损失函数：{loss_function_name}")
    
        stage_start_time=time.perf_counter()
        initial_parameter=transform_matrix[np.tril_indices(param.dim)]
        if optimizer_name=="gauss_newton":
            if loss_function_name!="second_power":
                raise ValueError("gauss_newton优化器目前只支持second_power损失")
            result=optimize_second_power_gauss_newton(pair_value_diff,pair_coord_diff,initial_parameter,param.dim,adjust_max_nfev)
        elif optimizer_name=="lbfgs":
            if loss_function_name!="second_power":
                raise ValueError("lbfgs优化器目前只支持second_power损失")
            result=optimize_second_power_lbfgs(pair_value_diff,pair_coord_diff,initial_parameter,param.dim,adjust_max_nfev)
        else:
            if optimizer_name!="least_squares":
                raise ValueError(f"未知的坐标调整优化器：{optimizer_name}")
            if loss_function_name=="fourth_power":
                jacobian=lambda parameter:fourth_power_loss_jacobian(pair_coord_diff,parameter,param.dim)
            elif loss_function_name=="second_power":
                jacobian=lambda parameter:second_power_loss_jacobian(pair_coord_diff,parameter,param.dim)
            elif loss_function_name=="log_distance":
                jacobian=lambda parameter:log_distance_loss_jacobian(pair_coord_diff,pair_abs_eb,parameter,param.dim)
            else:
                jacobian="2-point"
            result=scipy.optimize.least_squares(residual,initial_parameter,jac=jacobian,max_nfev=adjust_max_nfev,verbose=0)
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

        normalized_loss=np.mean(residual(transform_matrix[np.tril_indices(param.dim)])**2)
        if (iteration+1)%matrix_save_interval==0 or iteration+1==iteration_num:
            matrix_history.append(transform_matrix.copy())
            iteration_history.append(iteration+1)
            loss_history.append(normalized_loss)
            nfev_history.append(result.nfev)
            njev_history.append(np.nan if result.njev is None else result.njev)
            optimality_history.append(result.optimality)
            elapsed_time_history.append(time.perf_counter()-time_dict["adjust_coord总耗时"])
            pair_num_history.append(pair_coord_diff.shape[0])
        if matrix_history_path and ((iteration+1)%matrix_save_interval==0 or iteration+1==iteration_num):
            history_path=Path(matrix_history_path)
            history_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez(history_path,loss_function_name=loss_function_name,optimizer_name=optimizer_name,iteration=np.asarray(iteration_history),matrix=np.asarray(matrix_history),loss=np.asarray(loss_history),nfev=np.asarray(nfev_history),njev=np.asarray(njev_history),optimality=np.asarray(optimality_history),elapsed_time=np.asarray(elapsed_time_history),pair_num=np.asarray(pair_num_history),first_sample_ratio=first_sample_ratio,second_sample_ratio=second_sample_ratio,new_matrix_weight=new_matrix_weight,adjust_max_nfev=adjust_max_nfev,sample_file_num=sample_file_num,actual_ref_num=param.actual_ref_num,inverse_distance_power=inverse_distance_power,generalized_gaussian_shape=generalized_gaussian_shape,matrix_save_interval=matrix_save_interval)
        if (iteration+1)%10==0 and iteration+1<iteration_num:
            stage_start_time=time.perf_counter()
            iteration_transformed_coord=param.coord_array@transform_matrix
            time_dict["定期变换全部坐标"]+=time.perf_counter()-stage_start_time
            print(f"Iteration {iteration+1}: loss={normalized_loss},nfev={result.nfev},njev={result.njev},optimality={result.optimality}")
            print(f"transformed coord range: [{', '.join(f'[{minimum},{maximum}]' for minimum,maximum in zip(iteration_transformed_coord.min(axis=0),iteration_transformed_coord.max(axis=0)))}]")
            print(f"transform matrix:\n{transform_matrix}")
    
    time_dict["最终变换全部坐标"]=time.perf_counter()
    param.coord_transform_parameter=transform_matrix[np.tril_indices(param.dim)]
    param.coord_transform_matrix=transform_matrix
    transformed_coord=param.coord_array@transform_matrix
    time_dict["最终变换全部坐标"]=time.perf_counter()-time_dict["最终变换全部坐标"]
    time_dict["adjust_coord总耗时"]=time.perf_counter()-time_dict["adjust_coord总耗时"]
    
    print(f"Iteration {iteration_num}: loss={normalized_loss},nfev={result.nfev},njev={result.njev},optimality={result.optimality}")
    print(f"original coord range: [{', '.join(f'[{minimum},{maximum}]' for minimum,maximum in zip(param.coord_array.min(axis=0),param.coord_array.max(axis=0)))}]")
    print(f"transformed coord range: [{', '.join(f'[{minimum},{maximum}]' for minimum,maximum in zip(transformed_coord.min(axis=0),transformed_coord.max(axis=0)))}]")
    print(f"transform matrix at original coord center {(param.min_coord+param.max_coord)/2}:\n{transform_matrix}")
    print(f"metric matrix eigenvalues: {np.linalg.eigvalsh(transform_matrix@transform_matrix.T)}")
    print("累计分段耗时：")
    for name,value in time_dict.items():
        if name!="adjust_coord总耗时":
            print(f"  {name}：{value:.6f} s")
    print(f"  adjust_coord总耗时：{time_dict['adjust_coord总耗时']:.6f} s")

def adjust_coord(param,loss_function_name="fourth_power",iteration_num=100,matrix_history_path=None,first_sample_ratio=0.25,second_sample_ratio=0.25,new_matrix_weight=0.3,adjust_max_nfev=100,sample_file_num=3,matrix_save_interval=5,optimizer_name="least_squares")->None:
    if param.cross_time_compress:
        adjust_coord_cross_time(param)
    else:
        adjust_coord_no_cross_time(param,loss_function_name,iteration_num,matrix_history_path,first_sample_ratio,second_sample_ratio,new_matrix_weight,adjust_max_nfev,sample_file_num,matrix_save_interval,optimizer_name)

    
