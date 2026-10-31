#pragma once

namespace unsz
{
    // 为每个预测目标计算rbf_reference_num个参考点的RBF插值权重，预测值是参考点重建值的加权和。
    // 记参考点为x_1..x_n，目标为x，h为目标到各参考点平方距离的平均值，核函数φ(d²)=exp(-rbf_shape*sqrt(d²/h))。
    // 除以h使核函数与坐标的绝对尺度无关。
    // 权重w满足带常数项的RBF插值方程：A w+μ1=k，1ᵀw=1，其中A_ij=φ(|x_i-x_j|²)+rbf_nugget·δ_ij，k_i=φ(|x-x_i|²)。
    // nugget给对角线加一个小量，防止参考点很近时A接近奇异。1ᵀw=1保证常数场被准确预测。
    // 解法：先解A u=k和A v=1，再取μ=(1ᵀu-1)/(1ᵀv)，w=u-μv。A是对称正定矩阵，用LDLᵀ分解求解。
    // 退化情况都退回最近参考点（权重全给第0个参考点）并置rbf_fallback：参考点不足n个、分解主元过小、结果出现非有限值。
    // 目标与某个参考点重合时直接复制该参考点，不置rbf_fallback。
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_batched_rbf_weights(const vector<int>&target_ids,const vector<int>&ref_ids,const vector<uint64_t>&ref_begin,vector<T_calc>&weights,vector<uint8_t>&rbf_fallback)
    {
        if(ref_begin.size()!=target_ids.size()+1||ref_begin.back()!=ref_ids.size())
            throw runtime_error("批量RBF预测结构不完整");
        auto start_time=chrono::steady_clock::now();
        weights.assign(ref_ids.size(),0);
        rbf_fallback.assign(target_ids.size(),0);
        vector<uint32_t> batched_target_ids;
        batched_target_ids.reserve(target_ids.size());
        uint64_t boundary_fallback_target_num=0,rbf_target_num=0,nearest_fallback_target_num=0;
        // 参考点不足rbf_reference_num个的目标（预测顺序中最前面的点）直接用最近参考点，其余目标进入批量求解。
        for(uint32_t prediction_index=0;prediction_index<target_ids.size();prediction_index++)
        {
            uint64_t begin=ref_begin[prediction_index];
            uint32_t ref_num=(uint32_t)(ref_begin[prediction_index+1]-begin);
            if(ref_num==rbf_reference_num)
            {
                batched_target_ids.push_back(prediction_index);
                continue;
            }
            boundary_fallback_target_num++;
            rbf_fallback[prediction_index]=1;
            if(ref_num>0)
                weights[begin]=1;
            nearest_fallback_target_num++;
        }
        if(weight_mode=="idw")
        {
            // 消融实验：与UMC相同的反距离权重w_i∝1/d_i，参考点与RBF相同。目标与某个参考点重合时直接复制该参考点
            for(uint32_t prediction_index:batched_target_ids)
            {
                uint64_t begin=ref_begin[prediction_index];
                int target_id=target_ids[prediction_index];
                T_calc inverse_distance_sum=0;
                int zero_distance_ref=-1;
                for(uint32_t ref_index=0;ref_index<rbf_reference_num&&zero_distance_ref<0;ref_index++)
                {
                    T_calc squared_distance=0;
                    for(uint32_t dimension=0;dimension<dim;dimension++)
                    {
                        T_calc difference=(T_calc)coord_array[ref_ids[begin+ref_index]][dimension]-(T_calc)coord_array[target_id][dimension];
                        squared_distance+=difference*difference;
                    }
                    if(squared_distance==0)
                        zero_distance_ref=(int)ref_index;
                    else
                    {
                        weights[begin+ref_index]=1/sqrt(squared_distance);
                        inverse_distance_sum+=weights[begin+ref_index];
                    }
                }
                if(zero_distance_ref>=0)
                {
                    fill(weights.begin()+begin,weights.begin()+begin+rbf_reference_num,0);
                    weights[begin+zero_distance_ref]=1;
                }
                else
                    for(uint32_t ref_index=0;ref_index<rbf_reference_num;ref_index++)
                        weights[begin+ref_index]/=inverse_distance_sum;
            }
            printf("IDW weights: targets=%llu,boundary_fallback_targets=%llu,time=%f s\n",(unsigned long long)batched_target_ids.size(),(unsigned long long)boundary_fallback_target_num,seconds_since(start_time));
            return;
        }

        // 每批同时求解batch_capacity个目标，每个目标占一个lane。矩阵和向量按[行][列][lane]存放，最内层循环遍历lane，便于编译器向量化。
        // valid[lane]=0的lane仍按单位矩阵参与计算，使所有lane执行相同的指令，最后再退回最近参考点。
        constexpr uint32_t batch_capacity=32;
        uint32_t ref_num=rbf_reference_num;
        vector<T_calc> matrix((uint64_t)ref_num*ref_num*batch_capacity),target_squared_distance((uint64_t)ref_num*batch_capacity),target_kernel((uint64_t)ref_num*batch_capacity),target_solution((uint64_t)ref_num*batch_capacity),constant_solution((uint64_t)ref_num*batch_capacity),distance_normalizer(batch_capacity),work_value(batch_capacity);
        vector<uint8_t> valid(batch_capacity),direct_solution(batch_capacity);
        vector<uint64_t> reference_begin(batch_capacity);
        const T_calc shape=(T_calc)rbf_shape;
        const T_calc squared_shape=shape*shape;
        auto evaluate_kernel=[&](T_calc normalized_squared_distance)
        {
            T_calc scaled_squared_distance=squared_shape*normalized_squared_distance;
            T_calc scaled_distance=sqrt(scaled_squared_distance);
            return (T_calc)exp(-scaled_distance);
        };
        auto matrix_value=[&](uint32_t row,uint32_t column,uint32_t lane)->T_calc&
        {
            return matrix[((uint64_t)row*ref_num+column)*batch_capacity+lane];
        };
        auto vector_value=[&](vector<T_calc> &values,uint32_t row,uint32_t lane)->T_calc&
        {
            return values[(uint64_t)row*batch_capacity+lane];
        };

        for(uint64_t batch_begin=0;batch_begin<batched_target_ids.size();batch_begin+=batch_capacity)
        {
            uint32_t lane_num=(uint32_t)min<uint64_t>(batch_capacity,batched_target_ids.size()-batch_begin);
            // 第1步：计算目标到各参考点的平方距离和归一化系数h。
            for(uint32_t lane=0;lane<lane_num;lane++)
            {
                uint32_t prediction_index=batched_target_ids[batch_begin+lane];
                uint64_t begin=ref_begin[prediction_index];
                reference_begin[lane]=begin;
                valid[lane]=0;
                direct_solution[lane]=0;
                T_calc distance_sum=0;
                bool has_zero_distance=false;
                uint32_t zero_distance_ref=0;
                int target_id=target_ids[prediction_index];
                for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                {
                    int ref_id=ref_ids[begin+ref_index];
                    T_calc squared_distance=0;
                    for(uint32_t dimension=0;dimension<dim;dimension++)
                    {
                        T_calc difference=(T_calc)coord_array[ref_id][dimension]-(T_calc)coord_array[target_id][dimension];
                        squared_distance+=difference*difference;
                    }
                    vector_value(target_squared_distance,ref_index,lane)=squared_distance;
                    distance_sum+=squared_distance;
                    if(squared_distance==0)
                    {
                        has_zero_distance=true;
                        zero_distance_ref=ref_index;
                        break;
                    }
                }
                if(has_zero_distance)
                {
                    weights[begin+zero_distance_ref]=1;
                    direct_solution[lane]=1;
                    rbf_target_num++;
                    continue;
                }
                distance_normalizer[lane]=distance_sum/ref_num;
                valid[lane]=distance_normalizer[lane]>0&&isfinite(distance_normalizer[lane]);
            }

            // 第2步：组装对称矩阵A（参考点两两之间的核函数值，对角线加nugget）和右端向量k（目标到各参考点的核函数值）。
            for(uint32_t row=0;row<ref_num;row++)
            {
                for(uint32_t column=row;column<ref_num;column++)
                {
                    fill(work_value.begin(),work_value.begin()+lane_num,0);
                    for(uint32_t dimension=0;dimension<dim;dimension++)
                    {
                        for(uint32_t lane=0;lane<lane_num;lane++)
                        {
                            uint64_t begin=reference_begin[lane];
                            int row_ref_id=ref_ids[begin+row],column_ref_id=ref_ids[begin+column];
                            T_calc difference=(T_calc)coord_array[row_ref_id][dimension]-(T_calc)coord_array[column_ref_id][dimension];
                            work_value[lane]+=difference*difference;
                        }
                    }
                    for(uint32_t lane=0;lane<lane_num;lane++)
                    {
                        T_calc kernel_value=valid[lane]?evaluate_kernel(work_value[lane]/distance_normalizer[lane]):(row==column?1:0);
                        matrix_value(row,column,lane)=kernel_value;
                        matrix_value(column,row,lane)=kernel_value;
                    }
                }
                for(uint32_t lane=0;lane<lane_num;lane++)
                {
                    matrix_value(row,row,lane)+=(T_calc)rbf_nugget;
                    vector_value(target_kernel,row,lane)=valid[lane]?evaluate_kernel(vector_value(target_squared_distance,row,lane)/distance_normalizer[lane]):0;
                }
            }

            // 第3步：原地做LDLᵀ分解，A=L·D·Lᵀ。对角线保存D，下三角保存L（L的对角线为1，不存）。
            // D_r=A_rr-Σ_{c<r}L_rc²·D_c，L_nr=(A_nr-Σ_{c<r}L_nc·L_rc·D_c)/D_r。D_r不大于1e-12时视为退化。
            for(uint32_t row=0;row<ref_num;row++)
            {
                #pragma omp simd
                for(uint32_t lane=0;lane<lane_num;lane++)
                    work_value[lane]=matrix_value(row,row,lane);
                for(uint32_t column=0;column<row;column++)
                {
                    #pragma omp simd
                    for(uint32_t lane=0;lane<lane_num;lane++)
                        work_value[lane]-=matrix_value(row,column,lane)*matrix_value(row,column,lane)*matrix_value(column,column,lane);
                }
                #pragma omp simd
                for(uint32_t lane=0;lane<lane_num;lane++)
                {
                    if(!(work_value[lane]>(T_calc)1e-12)||!isfinite(work_value[lane]))
                        valid[lane]=0;
                    matrix_value(row,row,lane)=valid[lane]?work_value[lane]:1;
                }
                for(uint32_t next_row=row+1;next_row<ref_num;next_row++)
                {
                    #pragma omp simd
                    for(uint32_t lane=0;lane<lane_num;lane++)
                        work_value[lane]=matrix_value(next_row,row,lane);
                    for(uint32_t column=0;column<row;column++)
                    {
                        #pragma omp simd
                        for(uint32_t lane=0;lane<lane_num;lane++)
                            work_value[lane]-=matrix_value(next_row,column,lane)*matrix_value(row,column,lane)*matrix_value(column,column,lane);
                    }
                    #pragma omp simd
                    for(uint32_t lane=0;lane<lane_num;lane++)
                        matrix_value(next_row,row,lane)=valid[lane]?work_value[lane]/matrix_value(row,row,lane):0;
                }
            }

            // 第4步：同时解A u=k（target_solution）和A v=1（constant_solution），依次做前代L y=b、除以D、回代Lᵀx=y。
            for(uint32_t row=0;row<ref_num;row++)
            {
                #pragma omp simd
                for(uint32_t lane=0;lane<lane_num;lane++)
                {
                    vector_value(target_solution,row,lane)=vector_value(target_kernel,row,lane);
                    vector_value(constant_solution,row,lane)=1;
                }
                for(uint32_t column=0;column<row;column++)
                {
                    #pragma omp simd
                    for(uint32_t lane=0;lane<lane_num;lane++)
                    {
                        vector_value(target_solution,row,lane)-=matrix_value(row,column,lane)*vector_value(target_solution,column,lane);
                        vector_value(constant_solution,row,lane)-=matrix_value(row,column,lane)*vector_value(constant_solution,column,lane);
                    }
                }
            }
            for(uint32_t row=0;row<ref_num;row++)
            {
                #pragma omp simd
                for(uint32_t lane=0;lane<lane_num;lane++)
                {
                    vector_value(target_solution,row,lane)/=matrix_value(row,row,lane);
                    vector_value(constant_solution,row,lane)/=matrix_value(row,row,lane);
                }
            }
            for(uint32_t reverse_row=ref_num;reverse_row>0;reverse_row--)
            {
                uint32_t row=reverse_row-1;
                for(uint32_t next_row=row+1;next_row<ref_num;next_row++)
                {
                    #pragma omp simd
                    for(uint32_t lane=0;lane<lane_num;lane++)
                    {
                        vector_value(target_solution,row,lane)-=matrix_value(next_row,row,lane)*vector_value(target_solution,next_row,lane);
                        vector_value(constant_solution,row,lane)-=matrix_value(next_row,row,lane)*vector_value(constant_solution,next_row,lane);
                    }
                }
            }
            // 第5步：用拉格朗日乘子μ=(1ᵀu-1)/(1ᵀv)施加权重和为1的约束，w=u-μv。退化的目标退回最近参考点。
            for(uint32_t lane=0;lane<lane_num;lane++)
            {
                if(direct_solution[lane])
                    continue;
                uint32_t prediction_index=batched_target_ids[batch_begin+lane];
                uint64_t begin=ref_begin[prediction_index];
                T_calc target_sum=0,constant_sum=0;
                for(uint32_t row=0;row<ref_num;row++)
                {
                    target_sum+=vector_value(target_solution,row,lane);
                    constant_sum+=vector_value(constant_solution,row,lane);
                }
                if(abs(constant_sum)<(T_calc)1e-12||!isfinite(constant_sum))
                    valid[lane]=0;
                T_calc lagrange_multiplier=valid[lane]?(target_sum-1)/constant_sum:0;
                for(uint32_t row=0;row<ref_num;row++)
                {
                    T_calc weight=vector_value(target_solution,row,lane)-vector_value(constant_solution,row,lane)*lagrange_multiplier;
                    if(!isfinite(weight))
                        valid[lane]=0;
                    weights[begin+row]=weight;
                }
                if(valid[lane])
                    rbf_target_num++;
                else
                {
                    rbf_fallback[prediction_index]=1;
                    fill(weights.begin()+begin,weights.begin()+begin+ref_num,0);
                    weights[begin]=1;
                    nearest_fallback_target_num++;
                }
            }
        }
        printf("Batched RBF weights: vectorized_targets=%llu,boundary_fallback_targets=%llu,rbf_targets=%llu,nearest_fallback_targets=%llu,time=%f s\n",(unsigned long long)batched_target_ids.size(),(unsigned long long)boundary_fallback_target_num,(unsigned long long)rbf_target_num,(unsigned long long)nearest_fallback_target_num,seconds_since(start_time));
    }

}
