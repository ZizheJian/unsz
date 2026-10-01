#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    array_view_c<int> param_c<T_coord,T_data>::get_pred_ref_ids(uint32_t pred_ref_id) const
    {
        uint64_t begin=pred_ref_begin[pred_ref_id];
        uint32_t reference_num=(uint32_t)(pred_ref_begin[pred_ref_id+1]-begin);
        if(reference_num==0)
            return {};
        return {pred_ref_ids.data()+begin,reference_num};
    }

    template<typename T_coord,typename T_data>
    array_view_c<typename param_c<T_coord,T_data>::T_calc> param_c<T_coord,T_data>::get_pred_ref_weights(uint32_t pred_ref_id) const
    {
        if(pred_ref_prediction_weights.empty())
            return {};
        if(pred_ref_prediction_weights.size()!=pred_ref_ids.size())
            throw runtime_error("预测参考权重数量与参考点数量不一致");
        uint64_t begin=pred_ref_begin[pred_ref_id];
        uint32_t reference_num=(uint32_t)(pred_ref_begin[pred_ref_id+1]-begin);
        if(reference_num==0)
            return {};
        return {pred_ref_prediction_weights.data()+begin,reference_num};
    }

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
    void param_c<T_coord,T_data>::load_or_generate_pred_refs_structure()
    {
        if (dag_cache_path.empty())
        {
            generate_pred_refs_structure();
            return;
        }
        if(filesystem::exists(dag_cache_path))
        {
            if(partition_size_experimental>0||partition_num_experimental>0)
                build_partitions();
            load_pred_refs_structure();
        }
        else
            generate_pred_refs_structure();
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::save_pred_refs_structure()
    {
        static constexpr uint64_t magic=0x4452505a534e55;//"UNSZPRD"
        static constexpr uint32_t version=1;
        vector<unsigned char> bytes;
        uint64_t ref_num_sum=pred_ref_ids.size();
        bytes.reserve(32+(uint64_t)prediction_round_num*4+(uint64_t)point_num*8+ref_num_sum*4);
        append_binary(bytes,magic);
        append_binary(bytes,version);
        append_binary(bytes,point_num);
        append_binary(bytes,dim);
        append_binary(bytes,prediction_round_num);
        for (vector<int> &target_ids:prediction_target_id_groups)
            append_binary(bytes,(uint32_t)target_ids.size());
        uint32_t prediction_id=0;
        for (vector<int> &target_ids:prediction_target_id_groups)
            for (int target_id:target_ids)
            {
                if (prediction_id>=pred_target_ids.size() || pred_target_ids[prediction_id]!=target_id)
                    throw runtime_error("预测目标顺序与参考结构不一致，无法保存");
                append_binary(bytes,(uint32_t)target_id);
                auto ref_ids=get_pred_ref_ids(prediction_id);
                append_binary(bytes,ref_ids.size());
                for (int ref_id:ref_ids)
                    append_binary(bytes,(uint32_t)ref_id);
                prediction_id++;
            }
        if (prediction_id!=point_num)
            throw runtime_error("保存的预测目标数量与点数不一致");
        filesystem::path cache_path(dag_cache_path);
        if(cache_path.has_parent_path())
            filesystem::create_directories(cache_path.parent_path());
        FILE *file=fopen(dag_cache_path.c_str(),"wb");
        if (!file)
            throw runtime_error("无法创建DAG缓存文件："+dag_cache_path);
        if (fwrite(bytes.data(),1,bytes.size(),file)!=bytes.size())
        {
            fclose(file);
            throw runtime_error("写入DAG缓存文件失败："+dag_cache_path);
        }
        if (fclose(file)!=0)
            throw runtime_error("关闭DAG缓存文件失败："+dag_cache_path);
        printf("Saved DAG cache: %s, bytes=%llu\n",dag_cache_path.c_str(),(unsigned long long)bytes.size());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::load_pred_refs_structure()
    {
        static constexpr uint64_t expected_magic=0x4452505a534e55;
        static constexpr uint32_t expected_version=1;
        auto start_time=chrono::steady_clock::now();
        FILE *file=fopen(dag_cache_path.c_str(),"rb");
        if (!file)
            throw runtime_error("找不到DAG缓存文件："+dag_cache_path);
        if (fseek(file,0,SEEK_END)!=0)
        {
            fclose(file);
            throw runtime_error("无法定位DAG缓存文件末尾："+dag_cache_path);
        }
        long byte_num_signed=ftell(file);
        if (byte_num_signed<0 || fseek(file,0,SEEK_SET)!=0)
        {
            fclose(file);
            throw runtime_error("无法读取DAG缓存文件大小："+dag_cache_path);
        }
        vector<unsigned char> bytes((uint64_t)byte_num_signed);
        if (fread(bytes.data(),1,bytes.size(),file)!=bytes.size())
        {
            fclose(file);
            throw runtime_error("读取DAG缓存文件失败："+dag_cache_path);
        }
        if (fclose(file)!=0)
            throw runtime_error("关闭DAG缓存文件失败："+dag_cache_path);
        const unsigned char *cursor=bytes.data(),*end=bytes.data()+bytes.size();
        uint64_t magic=read_binary<uint64_t>(cursor,end);
        uint32_t version=read_binary<uint32_t>(cursor,end);
        uint32_t file_point_num=read_binary<uint32_t>(cursor,end);
        uint32_t file_dim=read_binary<uint32_t>(cursor,end);
        uint32_t file_round_num=read_binary<uint32_t>(cursor,end);
        if (magic!=expected_magic || version!=expected_version)
            throw runtime_error("DAG缓存文件格式或版本不正确："+dag_cache_path);
        if (file_point_num!=point_num || file_dim!=dim || file_round_num!=prediction_round_num)
            throw runtime_error("DAG缓存文件的点数、维数或轮数与当前参数不一致");
        vector<uint32_t> round_target_num_vec(prediction_round_num);
        uint64_t target_num_sum=0;
        for (uint32_t round_id=0;round_id<prediction_round_num;round_id++)
        {
            round_target_num_vec[round_id]=read_binary<uint32_t>(cursor,end);
            target_num_sum+=round_target_num_vec[round_id];
        }
        if (target_num_sum!=point_num)
            throw runtime_error("DAG缓存文件中的目标数量与点数不一致");
        prediction_target_id_groups.clear();
        prediction_target_id_groups.resize(prediction_round_num);
        pred_target_ids.clear();
        pred_target_ids.reserve(point_num);
        pred_ref_ids.clear();
        pred_ref_ids.reserve((uint64_t)point_num*prediction_max_ref_num);
        pred_ref_begin.clear();
        pred_ref_begin.reserve((uint64_t)point_num+1);
        pred_ref_begin.push_back(0);
        pred_ref_prediction_weights.clear();
        pred_ref_rbf_fallback_vec.clear();
        bool generate_idw_weights=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::idw;});
        if(generate_idw_weights)
            pred_ref_prediction_weights.reserve((uint64_t)point_num*prediction_max_ref_num);
        vector<uint8_t> available(point_num,0);
        for (uint32_t round_id=0;round_id<prediction_round_num;round_id++)
        {
            prediction_target_id_groups[round_id].reserve(round_target_num_vec[round_id]);
            for (uint32_t round_target_id=0;round_target_id<round_target_num_vec[round_id];round_target_id++)
            {
                uint32_t target_id=read_binary<uint32_t>(cursor,end);
                uint32_t ref_num=read_binary<uint32_t>(cursor,end);
                if (target_id>=point_num || available[target_id])
                    throw runtime_error("DAG缓存文件包含越界或重复的目标编号");
                if(ref_num>prediction_max_ref_num)
                    throw runtime_error("DAG缓存文件的单点参考点数量超过当前prediction_max_ref_num");
                prediction_target_id_groups[round_id].push_back(target_id);
                pred_target_ids.push_back((int)target_id);
                for (uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                {
                    uint32_t ref_id=read_binary<uint32_t>(cursor,end);
                    if (ref_id>=point_num || !available[ref_id])
                        throw runtime_error("DAG缓存文件包含越界或尚不可用的参考点编号");
                    pred_ref_ids.push_back(ref_id);
                }
                pred_ref_begin.push_back(pred_ref_ids.size());
                if(generate_idw_weights)
                    append_normalized_idw_prediction_weights(target_id,get_pred_ref_ids((uint32_t)pred_target_ids.size()-1),pred_ref_prediction_weights);
                available[target_id]=1;
            }
        }
        if (cursor!=end)
            throw runtime_error("DAG缓存文件末尾存在未解析数据");
        prepare_pred_refs_structure();
        printf("Loaded DAG cache: %s, bytes=%llu, time=%f s\n",dag_cache_path.c_str(),(unsigned long long)bytes.size(),chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::validate_pred_refs_structure() const
    {
        bool has_idw=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::idw;});
        bool has_rbf=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::rbf;});
        if(pred_target_ids.size()!=point_num||pred_ref_begin.size()!=(uint64_t)point_num+1)
            throw runtime_error("预测参考结构没有覆盖所有点");
        if(pred_ref_begin.back()!=pred_ref_ids.size())
            throw runtime_error("预测参考点偏移与参考点数组大小不一致");
        if(has_idw&&pred_ref_prediction_weights.size()!=pred_ref_ids.size())
            throw runtime_error("归一化IDW权重没有覆盖所有参考点");
        if(has_rbf&&!pred_ref_prediction_weights.empty()&&pred_ref_prediction_weights.size()!=pred_ref_ids.size())
            throw runtime_error("RBF权重没有覆盖所有参考点");
        if (partition_size_experimental>0 || partition_num_experimental>0)
            for (uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
                for (int ref_id:get_pred_ref_ids(pred_ref_id))
                    if (point_partition_id_vec[pred_target_ids[pred_ref_id]]!=point_partition_id_vec[ref_id])
                        throw runtime_error("partition预测DAG包含跨partition参考点");
        if(!pred_ref_rbf_fallback_vec.empty()&&pred_ref_rbf_fallback_vec.size()!=point_num)
            throw runtime_error("RBF回退标记数量与点数不一致");
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_pred_refs_structure()
    {
        partition_pred_ref_ids_vec.clear();
        if(!partition_point_ids_vec.empty())
        {
            partition_pred_ref_ids_vec.resize(partition_point_ids_vec.size());
            for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
            {
                uint32_t pred_id=(uint32_t)pred_target_ids[pred_ref_id];
                uint32_t partition_id=point_partition_id_vec[pred_id];
                if(partition_id>=partition_pred_ref_ids_vec.size())
                    throw runtime_error("预测目标的partition编号越界");
                partition_pred_ref_ids_vec[partition_id].push_back(pred_ref_id);
            }
        }
        prepare_multihead_execution_order();
        printf("Prepared continuous reference arrays: targets=%u, references=%llu\n",point_num,(unsigned long long)pred_ref_ids.size());
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::calculate_scalar_rbf_prediction_weights(int pred_id,array_view_c<int> ref_ids,vector<T_calc> &weights,array_view_c<T_coord> precomputed_target_ref_squared_distances)
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
    void param_c<T_coord,T_data>::calculate_batched_rbf_prediction_weights(const vector<T_coord> &reference_coords,const vector<T_coord> &target_ref_squared_distances)
    {
        if(reference_coords.size()!=(uint64_t)pred_ref_ids.size()*dim)
            throw runtime_error("批量RBF参考坐标数量与DAG不一致");
        if(target_ref_squared_distances.size()!=pred_ref_ids.size())
            throw runtime_error("批量RBF目标距离数量与DAG不一致");
        auto start_time=chrono::steady_clock::now();
        pred_ref_prediction_weights.assign(pred_ref_ids.size(),0);
        pred_ref_rbf_fallback_vec.assign(point_num,0);
        vector<uint32_t> batched_target_ids;
        batched_target_ids.reserve(point_num);
        vector<T_calc> scalar_weights;
        uint64_t scalar_target_num=0,rbf_target_num=0,nearest_fallback_target_num=0;
        for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
        {
            uint64_t begin=pred_ref_begin[pred_ref_id];
            uint32_t ref_num=(uint32_t)(pred_ref_begin[pred_ref_id+1]-begin);
            if(ref_num==prediction_max_ref_num&&ref_num>=prediction_min_ref_num)
            {
                batched_target_ids.push_back(pred_ref_id);
                continue;
            }
            scalar_target_num++;
            bool solved=false;
            if(ref_num>=prediction_min_ref_num)
                solved=calculate_scalar_rbf_prediction_weights(pred_target_ids[pred_ref_id],array_view_c<int>(pred_ref_ids.data()+begin,ref_num),scalar_weights,array_view_c<T_coord>(target_ref_squared_distances.data()+begin,ref_num));
            if(solved)
            {
                copy(scalar_weights.begin(),scalar_weights.end(),pred_ref_prediction_weights.begin()+begin);
                rbf_target_num++;
            }
            else
            {
                pred_ref_rbf_fallback_vec[pred_ref_id]=1;
                if(ref_num>0)
                    pred_ref_prediction_weights[begin]=1;
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
                uint32_t pred_ref_id=batched_target_ids[batch_begin+lane];
                uint64_t begin=pred_ref_begin[pred_ref_id];
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
                    pred_ref_prediction_weights[begin+zero_distance_ref]=1;
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
                uint32_t pred_ref_id=batched_target_ids[batch_begin+lane];
                uint64_t begin=pred_ref_begin[pred_ref_id];
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
                    pred_ref_prediction_weights[begin+row]=weight;
                }
                if(valid[lane])
                    rbf_target_num++;
                else
                {
                    pred_ref_rbf_fallback_vec[pred_ref_id]=1;
                    fill(pred_ref_prediction_weights.begin()+begin,pred_ref_prediction_weights.begin()+begin+ref_num,0);
                    pred_ref_prediction_weights[begin]=1;
                    nearest_fallback_target_num++;
                }
            }
        }
        printf("Batched RBF weights: vectorized_targets=%llu,scalar_targets=%llu,rbf_targets=%llu,nearest_fallback_targets=%llu,time=%f s\n",(unsigned long long)batched_target_ids.size(),(unsigned long long)scalar_target_num,(unsigned long long)rbf_target_num,(unsigned long long)nearest_fallback_target_num,chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_rbf_prediction_weights()
    {
        if(pred_ref_prediction_weights.size()==pred_ref_ids.size()&&pred_ref_rbf_fallback_vec.size()==point_num)
            return;
        auto start_time=chrono::steady_clock::now();
        vector<T_coord> reference_coords((uint64_t)pred_ref_ids.size()*dim),target_ref_squared_distances(pred_ref_ids.size());
        for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
        {
            uint64_t begin=pred_ref_begin[pred_ref_id];
            uint64_t end=pred_ref_begin[pred_ref_id+1];
            int pred_id=pred_target_ids[pred_ref_id];
            for(uint64_t ref_offset=begin;ref_offset<end;ref_offset++)
            {
                int ref_id=pred_ref_ids[ref_offset];
                for(uint32_t dimension=0;dimension<dim;dimension++)
                    reference_coords[ref_offset*dim+dimension]=coord_array[ref_id][dimension];
                target_ref_squared_distances[ref_offset]=get_dist_to_coord(coord_array[pred_id],coord_array[ref_id],dim);
            }
        }
        printf("Reconstructed temporary RBF geometry from DAG: coordinates=%llu,distances=%llu,time=%f s\n",(unsigned long long)reference_coords.size(),(unsigned long long)target_ref_squared_distances.size(),chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
        calculate_batched_rbf_prediction_weights(reference_coords,target_ref_squared_distances);
    }
}
