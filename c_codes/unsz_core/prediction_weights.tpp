#pragma once

namespace unsz
{
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
        printf("Batched RBF weights: vectorized_targets=%llu,boundary_fallback_targets=%llu,rbf_targets=%llu,nearest_fallback_targets=%llu,time=%f s\n",(unsigned long long)batched_target_ids.size(),(unsigned long long)boundary_fallback_target_num,(unsigned long long)rbf_target_num,(unsigned long long)nearest_fallback_target_num,chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
    }

}
