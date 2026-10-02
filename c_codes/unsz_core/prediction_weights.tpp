#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::write_normalized_idw_prediction_weights(int pred_id,array_view_c<int> ref_ids,T_calc *weights,array_view_c<T_coord> target_ref_squared_distances)
    {
        if(!target_ref_squared_distances.empty()&&target_ref_squared_distances.size()!=ref_ids.size())
            throw runtime_error("预计算距离数量与IDW参考点数量不一致");
        if(ref_ids.empty())
            return;
        fill(weights,weights+ref_ids.size(),0);
        T_calc weight_sum=0;
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            T_calc squared_distance;
            if(target_ref_squared_distances.empty())
                squared_distance=(T_calc)get_dist_to_coord(coord_array[pred_id],coord_array[ref_ids[ref_index]],dim);
            else
                squared_distance=(T_calc)target_ref_squared_distances[ref_index];
            if(squared_distance==0)
            {
                weights[ref_index]=1;
                return;
            }
            T_calc weight=pow(squared_distance,(T_calc)(-idw_power/2));
            if(!isfinite(weight))
            {
                weights[ref_index]=1;
                return;
            }
            weights[ref_index]=weight;
            weight_sum+=weight;
        }
        if(!(weight_sum>0)||!isfinite(weight_sum))
        {
            fill(weights,weights+ref_ids.size(),0);
            weights[0]=1;
            return;
        }
        for(uint32_t weight_index=0;weight_index<ref_ids.size();weight_index++)
            weights[weight_index]/=weight_sum;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::append_normalized_idw_prediction_weights(int pred_id,array_view_c<int> ref_ids,vector<T_calc> &weights,array_view_c<T_coord> target_ref_squared_distances)
    {
        uint64_t begin=weights.size();
        weights.resize(begin+ref_ids.size());
        write_normalized_idw_prediction_weights(pred_id,ref_ids,ref_ids.empty()?nullptr:weights.data()+begin,target_ref_squared_distances);
    }


    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::solve_scalar_rbf_weights(int pred_id,array_view_c<int> ref_ids,vector<T_calc> &weights,array_view_c<T_coord> precomputed_target_ref_squared_distances)
    {
        // 仅处理参考点数未达到固定批宽、但仍满足RBF下限的少量边界目标。
        uint32_t reference_num=(uint32_t)ref_ids.size();
        if(reference_num<2)
            return false;
        if(!precomputed_target_ref_squared_distances.empty()&&precomputed_target_ref_squared_distances.size()!=reference_num)
            throw runtime_error("预计算距离数量与RBF参考点数量不一致");
        weights.assign(reference_num,0);

        // 用目标到参考点的平均平方距离归一化本次局部几何。
        thread_local vector<T_calc> target_reference_squared_distances;
        target_reference_squared_distances.assign(reference_num,0);
        T_calc distance_normalizer=0;
        for(uint32_t reference_index=0;reference_index<reference_num;reference_index++)
        {
            if(!precomputed_target_ref_squared_distances.empty())
                target_reference_squared_distances[reference_index]=(T_calc)precomputed_target_ref_squared_distances[reference_index];
            else
            {
                int reference_id=ref_ids[reference_index];
                for(uint32_t dimension=0;dimension<dim;dimension++)
                {
                    T_calc coordinate_difference=(T_calc)coord_array[reference_id][dimension]-(T_calc)coord_array[pred_id][dimension];
                    target_reference_squared_distances[reference_index]+=coordinate_difference*coordinate_difference;
                }
            }
            if(target_reference_squared_distances[reference_index]==0)
            {
                weights[reference_index]=1;
                return true;
            }
            distance_normalizer+=target_reference_squared_distances[reference_index];
        }
        distance_normalizer/=reference_num;
        if(!(distance_normalizer>0)||!isfinite(distance_normalizer))
            return false;

        uint8_t kernel_id;
        if(rbf_kernel=="gaussian")
            kernel_id=0;
        else if(rbf_kernel=="inverse_multiquadric")
            kernel_id=1;
        else if(rbf_kernel=="exponential")
            kernel_id=2;
        else if(rbf_kernel=="matern32")
            kernel_id=3;
        else if(rbf_kernel=="matern52")
            kernel_id=4;
        else
            throw runtime_error("未知的RBF核函数："+rbf_kernel);

        const T_calc shape=(T_calc)rbf_shape;
        const T_calc squared_shape=shape*shape;
        auto evaluate_kernel=[&](T_calc normalized_squared_distance)
        {
            T_calc scaled_squared_distance=squared_shape*normalized_squared_distance;
            if(kernel_id==0)
                return (T_calc)exp(-scaled_squared_distance);
            if(kernel_id==1)
                return (T_calc)1/(T_calc)sqrt((T_calc)1+scaled_squared_distance);
            T_calc scaled_distance=sqrt(scaled_squared_distance);
            if(kernel_id==2)
                return (T_calc)exp(-scaled_distance);
            if(kernel_id==3)
            {
                T_calc value=sqrt((T_calc)3)*scaled_distance;
                return ((T_calc)1+value)*(T_calc)exp(-value);
            }
            T_calc value=sqrt((T_calc)5)*scaled_distance;
            return ((T_calc)1+value+value*value/(T_calc)3)*(T_calc)exp(-value);
        };

        uint32_t matrix_size=reference_num;
        thread_local vector<T_calc> system_matrix;
        system_matrix.assign((uint64_t)matrix_size*matrix_size,0);
        auto matrix_value=[&](uint32_t row,uint32_t column)->T_calc&
        {
            return system_matrix[(uint64_t)row*matrix_size+column];
        };
        thread_local vector<T_calc> target_kernel,solution;
        target_kernel.assign(reference_num,0);
        for(uint32_t row=0;row<reference_num;row++)
        {
            int row_reference_id=ref_ids[row];
            for(uint32_t column=row;column<reference_num;column++)
            {
                int column_reference_id=ref_ids[column];
                T_calc reference_squared_distance=0;
                for(uint32_t dimension=0;dimension<dim;dimension++)
                {
                    T_calc coordinate_difference=(T_calc)coord_array[row_reference_id][dimension]-(T_calc)coord_array[column_reference_id][dimension];
                    reference_squared_distance+=coordinate_difference*coordinate_difference;
                }
                T_calc kernel_value=evaluate_kernel(reference_squared_distance/distance_normalizer);
                matrix_value(row,column)=kernel_value;
                matrix_value(column,row)=kernel_value;
            }
            matrix_value(row,row)+=(T_calc)rbf_nugget;
            target_kernel[row]=evaluate_kernel(target_reference_squared_distances[row]/distance_normalizer);
        }

        // 先对正定核矩阵K做LDLT，再通过Schur补处理权重和为1的约束。
        bool factorization_succeeded=true;
        for(uint32_t row=0;row<reference_num;row++)
        {
            T_calc diagonal=matrix_value(row,row);
            for(uint32_t column=0;column<row;column++)
                diagonal-=matrix_value(row,column)*matrix_value(row,column)*matrix_value(column,column);
            if(!(diagonal>(T_calc)1e-12)||!isfinite(diagonal))
            {
                factorization_succeeded=false;
                break;
            }
            matrix_value(row,row)=diagonal;
            for(uint32_t next_row=row+1;next_row<reference_num;next_row++)
            {
                T_calc value=matrix_value(next_row,row);
                for(uint32_t column=0;column<row;column++)
                    value-=matrix_value(next_row,column)*matrix_value(row,column)*matrix_value(column,column);
                matrix_value(next_row,row)=value/diagonal;
            }
        }
        bool solved=false;
        if(factorization_succeeded)
        {
            thread_local vector<T_calc> target_kernel_solution,constant_solution;
            target_kernel_solution.assign(reference_num,0);
            constant_solution.assign(reference_num,0);
            for(uint32_t row=0;row<reference_num;row++)
            {
                T_calc target_kernel_value=target_kernel[row],constant_value=1;
                for(uint32_t column=0;column<row;column++)
                {
                    target_kernel_value-=matrix_value(row,column)*target_kernel_solution[column];
                    constant_value-=matrix_value(row,column)*constant_solution[column];
                }
                target_kernel_solution[row]=target_kernel_value;
                constant_solution[row]=constant_value;
            }
            for(uint32_t row=0;row<reference_num;row++)
            {
                target_kernel_solution[row]/=matrix_value(row,row);
                constant_solution[row]/=matrix_value(row,row);
            }
            for(uint32_t reverse_row=reference_num;reverse_row>0;reverse_row--)
            {
                uint32_t row=reverse_row-1;
                for(uint32_t next_row=row+1;next_row<reference_num;next_row++)
                {
                    target_kernel_solution[row]-=matrix_value(next_row,row)*target_kernel_solution[next_row];
                    constant_solution[row]-=matrix_value(next_row,row)*constant_solution[next_row];
                }
            }
            T_calc target_kernel_sum=accumulate(target_kernel_solution.begin(),target_kernel_solution.end(),(T_calc)0);
            T_calc constant_sum=accumulate(constant_solution.begin(),constant_solution.end(),(T_calc)0);
            if(abs(constant_sum)>=(T_calc)1e-12&&isfinite(constant_sum))
            {
                T_calc lagrange_multiplier=(target_kernel_sum-1)/constant_sum;
                solution.assign(reference_num,0);
                for(uint32_t reference_index=0;reference_index<reference_num;reference_index++)
                    solution[reference_index]=target_kernel_solution[reference_index]-constant_solution[reference_index]*lagrange_multiplier;
                solved=true;
            }
        }
        if(!solved)
            return false;
        for(uint32_t reference_index=0;reference_index<reference_num;reference_index++)
        {
            if(!isfinite(solution[reference_index]))
                return false;
            weights[reference_index]=solution[reference_index];
        }
        return true;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_batched_rbf_weights(const vector<T_coord>&reference_coords,const vector<T_coord>&target_ref_squared_distances,const vector<int>&target_ids,const vector<int>&ref_ids,const vector<uint64_t>&ref_begin,vector<T_calc>&weights,vector<uint8_t>&rbf_fallback)
    {
        if(ref_begin.size()!=target_ids.size()+1||ref_begin.back()!=ref_ids.size())
            throw runtime_error("批量RBF预测结构不完整");
        if(reference_coords.size()!=(uint64_t)ref_ids.size()*dim)
            throw runtime_error("批量RBF参考坐标数量与DAG不一致");
        if(target_ref_squared_distances.size()!=ref_ids.size())
            throw runtime_error("批量RBF目标距离数量与DAG不一致");
        auto start_time=chrono::steady_clock::now();
        weights.assign(ref_ids.size(),0);
        rbf_fallback.assign(target_ids.size(),0);
        vector<uint32_t> batched_target_ids;
        batched_target_ids.reserve(target_ids.size());
        vector<T_calc> scalar_weights;
        uint64_t scalar_target_num=0,rbf_target_num=0,nearest_fallback_target_num=0;
        for(uint32_t prediction_index=0;prediction_index<target_ids.size();prediction_index++)
        {
            uint64_t begin=ref_begin[prediction_index];
            uint32_t ref_num=(uint32_t)(ref_begin[prediction_index+1]-begin);
            if(ref_num==prediction_max_ref_num&&ref_num>=prediction_min_ref_num)
            {
                batched_target_ids.push_back(prediction_index);
                continue;
            }
            scalar_target_num++;
            bool solved=false;
            if(ref_num>=prediction_min_ref_num)
                solved=solve_scalar_rbf_weights(target_ids[prediction_index],array_view_c<int>(ref_ids.data()+begin,ref_num),scalar_weights,array_view_c<T_coord>(target_ref_squared_distances.data()+begin,ref_num));
            if(solved)
            {
                copy(scalar_weights.begin(),scalar_weights.end(),weights.begin()+begin);
                rbf_target_num++;
            }
            else
            {
                rbf_fallback[prediction_index]=1;
                if(ref_num>0)
                    weights[begin]=1;
                nearest_fallback_target_num++;
            }
        }

        constexpr uint32_t batch_capacity=32;
        uint32_t ref_num=prediction_max_ref_num;
        vector<T_calc> matrix((uint64_t)ref_num*ref_num*batch_capacity),target_kernel((uint64_t)ref_num*batch_capacity),target_solution((uint64_t)ref_num*batch_capacity),constant_solution((uint64_t)ref_num*batch_capacity),distance_normalizer(batch_capacity),work_value(batch_capacity);
        vector<uint8_t> valid(batch_capacity),direct_solution(batch_capacity);
        vector<uint64_t> reference_begin(batch_capacity);
        uint8_t kernel_id;
        if(rbf_kernel=="gaussian")
            kernel_id=0;
        else if(rbf_kernel=="inverse_multiquadric")
            kernel_id=1;
        else if(rbf_kernel=="exponential")
            kernel_id=2;
        else if(rbf_kernel=="matern32")
            kernel_id=3;
        else if(rbf_kernel=="matern52")
            kernel_id=4;
        else
            throw runtime_error("未知的RBF核函数："+rbf_kernel);
        const T_calc shape=(T_calc)rbf_shape;
        const T_calc squared_shape=shape*shape;
        auto evaluate_kernel=[&](T_calc normalized_squared_distance)
        {
            T_calc scaled_squared_distance=squared_shape*normalized_squared_distance;
            if(kernel_id==0)
                return (T_calc)exp(-scaled_squared_distance);
            if(kernel_id==1)
                return (T_calc)1/(T_calc)sqrt((T_calc)1+scaled_squared_distance);
            T_calc scaled_distance=sqrt(scaled_squared_distance);
            if(kernel_id==2)
                return (T_calc)exp(-scaled_distance);
            if(kernel_id==3)
            {
                T_calc value=sqrt((T_calc)3)*scaled_distance;
                return ((T_calc)1+value)*(T_calc)exp(-value);
            }
            T_calc value=sqrt((T_calc)5)*scaled_distance;
            return ((T_calc)1+value+value*value/(T_calc)3)*(T_calc)exp(-value);
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
                for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                {
                    T_calc squared_distance=(T_calc)target_ref_squared_distances[begin+ref_index];
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
                            T_calc difference=(T_calc)reference_coords[((begin+row)*dim)+dimension]-(T_calc)reference_coords[((begin+column)*dim)+dimension];
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
                    uint64_t begin=reference_begin[lane];
                    matrix_value(row,row,lane)+=(T_calc)rbf_nugget;
                    vector_value(target_kernel,row,lane)=valid[lane]?evaluate_kernel((T_calc)target_ref_squared_distances[begin+row]/distance_normalizer[lane]):0;
                }
            }

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
        printf("Batched RBF weights: vectorized_targets=%llu,scalar_targets=%llu,rbf_targets=%llu,nearest_fallback_targets=%llu,time=%f s\n",(unsigned long long)batched_target_ids.size(),(unsigned long long)scalar_target_num,(unsigned long long)rbf_target_num,(unsigned long long)nearest_fallback_target_num,chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_rbf_weights_from_loaded_dag()
    {
        auto prepare=[&](prediction_structure_c<T_calc>&structure)
        {
            if(structure.prediction_weights.size()==structure.reference_ids.size()&&structure.rbf_fallback_flags.size()==structure.target_ids.size())
                return;
            auto start_time=chrono::steady_clock::now();
            vector<T_coord> reference_coords((uint64_t)structure.reference_ids.size()*dim),target_ref_squared_distances(structure.reference_ids.size());
            for(uint32_t prediction_index=0;prediction_index<structure.target_ids.size();prediction_index++)
            {
                uint64_t begin=structure.reference_offsets[prediction_index],end=structure.reference_offsets[prediction_index+1];
                int pred_id=structure.target_ids[prediction_index];
                for(uint64_t ref_offset=begin;ref_offset<end;ref_offset++)
                {
                    int ref_id=structure.reference_ids[ref_offset];
                    for(uint32_t dimension=0;dimension<dim;dimension++)
                        reference_coords[ref_offset*dim+dimension]=coord_array[ref_id][dimension];
                    target_ref_squared_distances[ref_offset]=get_dist_to_coord(coord_array[pred_id],coord_array[ref_id],dim);
                }
            }
            printf("Reconstructed temporary RBF geometry from DAG: coordinates=%llu,distances=%llu,time=%f s\n",(unsigned long long)reference_coords.size(),(unsigned long long)target_ref_squared_distances.size(),chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
            generate_batched_rbf_weights(reference_coords,target_ref_squared_distances,structure.target_ids,structure.reference_ids,structure.reference_offsets,structure.prediction_weights,structure.rbf_fallback_flags);
        };
        exception_ptr weight_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(prediction_structure_vec.size()>1&&thread_num>1)
        for(int64_t structure_index=0;structure_index<(int64_t)prediction_structure_vec.size();structure_index++)
        {
            try
            {
                prepare(prediction_structure_vec[structure_index]);
            }
            catch(...)
            {
#pragma omp critical
                if(!weight_exception)
                    weight_exception=current_exception();
            }
        }
        if(weight_exception)
            rethrow_exception(weight_exception);
    }
}
