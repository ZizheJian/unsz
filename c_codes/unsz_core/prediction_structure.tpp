#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    array_view_c<int> param_c<T_coord,T_data>::get_pred_ref_ids(uint32_t pred_ref_id) const
    {
        uint64_t begin=pred_ref_begin[pred_ref_id],end=pred_ref_begin[pred_ref_id+1];
        return array_view_c<int>(begin==end?nullptr:pred_ref_ids.data()+begin,(uint32_t)(end-begin));
    }

    template<typename T_coord,typename T_data>
    array_view_c<double> param_c<T_coord,T_data>::get_pred_ref_weights(uint32_t pred_ref_id,bool label_normalized,bool value_normalized) const
    {
        uint64_t begin=pred_ref_begin[pred_ref_id],end=pred_ref_begin[pred_ref_id+1];
        const vector<double> &weights=label_normalized?pred_ref_label_normalized_weights:(value_normalized?pred_ref_value_normalized_weights:pred_ref_weights);
        return array_view_c<double>(begin==end?nullptr:weights.data()+begin,(uint32_t)(end-begin));
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::load_or_generate_pred_refs_structure()
    {
        if (prediction_structure_path.empty())
        {
            generate_pred_refs_structure();
            return;
        }
        FILE *file=fopen(prediction_structure_path.c_str(),"rb");
        if (file)
        {
            if (fclose(file)!=0)
                throw runtime_error("关闭预测结构文件失败："+prediction_structure_path);
            if (partition_size_experimental>0 || partition_num_experimental>0)
                build_partitions();
            load_pred_refs_structure();
        }
        else
            generate_pred_refs_structure();
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::print_prediction_geometry_statistics_result()
    {
        if (!print_prediction_geometry_statistics)
            return;
        double full_diagonal_squared=0;
        for (uint32_t d=0;d<dim;d++)
        {
            double span=(double)max_coord[d]-min_coord[d];
            full_diagonal_squared+=span*span;
        }
        double full_diagonal=sqrt(full_diagonal_squared);
        uint32_t prediction_id=0;
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            vector<double> round_min(dim,numeric_limits<double>::infinity()),round_max(dim,-numeric_limits<double>::infinity());
            double reference_distance_sum=0,nearest_reference_distance_sum=0;
            uint64_t reference_num=0,target_with_reference_num=0;
            for (int target_id:round_target_ids[round_id])
            {
                if (prediction_id>=pred_target_ids.size() || pred_target_ids[prediction_id]!=target_id)
                    throw runtime_error("几何统计发现预测目标顺序不一致");
                for (uint32_t d=0;d<dim;d++)
                {
                    round_min[d]=min(round_min[d],(double)coord_array[target_id][d]);
                    round_max[d]=max(round_max[d],(double)coord_array[target_id][d]);
                }
                uint64_t ref_begin=pred_ref_begin[prediction_id],ref_end=pred_ref_begin[prediction_id+1];
                if (ref_begin<ref_end)
                    target_with_reference_num++;
                for (uint64_t ref_index=ref_begin;ref_index<ref_end;ref_index++)
                {
                    int ref_id=pred_ref_ids[ref_index];
                    double squared_distance=0;
                    for (uint32_t d=0;d<dim;d++)
                    {
                        double difference=(double)coord_array[target_id][d]-coord_array[ref_id][d];
                        squared_distance+=difference*difference;
                    }
                    double distance=sqrt(squared_distance);
                    reference_distance_sum+=distance;
                    if (ref_index==ref_begin)
                        nearest_reference_distance_sum+=distance;
                    reference_num++;
                }
                prediction_id++;
            }
            printf("Prediction geometry: round=%u,target_num=%llu,span_ratio=[",round_id,(unsigned long long)round_target_ids[round_id].size());
            for (uint32_t d=0;d<dim;d++)
            {
                double full_span=(double)max_coord[d]-min_coord[d];
                double ratio=round_target_ids[round_id].empty() || full_span==0?0:(round_max[d]-round_min[d])/full_span;
                printf("%s%.9f",d==0?"":",",ratio);
            }
            double mean_reference_distance=reference_num==0?0:reference_distance_sum/reference_num;
            double mean_nearest_reference_distance=target_with_reference_num==0?0:nearest_reference_distance_sum/target_with_reference_num;
            printf("],ref_num=%llu,mean_ref_distance=%.9g,mean_ref_distance_ratio=%.9g,mean_nearest_ref_distance=%.9g,mean_nearest_ref_distance_ratio=%.9g\n",(unsigned long long)reference_num,mean_reference_distance,full_diagonal==0?0:mean_reference_distance/full_diagonal,mean_nearest_reference_distance,full_diagonal==0?0:mean_nearest_reference_distance/full_diagonal);
        }
        if (prediction_id!=point_num)
            throw runtime_error("几何统计没有覆盖全部预测目标");
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::save_pred_refs_structure()
    {
        static constexpr uint64_t magic=0x4452505a534e55;//"UNSZPRD"
        static constexpr uint32_t version=1;
        vector<unsigned char> bytes;
        uint64_t ref_num_sum=pred_ref_ids.size();
        bytes.reserve(32+(uint64_t)predict_round_num*4+(uint64_t)point_num*8+ref_num_sum*4);
        append_binary(bytes,magic);
        append_binary(bytes,version);
        append_binary(bytes,point_num);
        append_binary(bytes,dim);
        append_binary(bytes,predict_round_num);
        for (vector<int> &target_ids:round_target_ids)
            append_binary(bytes,(uint32_t)target_ids.size());
        uint32_t prediction_id=0;
        for (vector<int> &target_ids:round_target_ids)
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
        FILE *file=fopen(prediction_structure_path.c_str(),"wb");
        if (!file)
            throw runtime_error("无法创建预测结构文件："+prediction_structure_path);
        if (fwrite(bytes.data(),1,bytes.size(),file)!=bytes.size())
        {
            fclose(file);
            throw runtime_error("写入预测结构文件失败："+prediction_structure_path);
        }
        if (fclose(file)!=0)
            throw runtime_error("关闭预测结构文件失败："+prediction_structure_path);
        printf("Saved prediction structure: %s, bytes=%llu\n",prediction_structure_path.c_str(),(unsigned long long)bytes.size());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::load_pred_refs_structure()
    {
        static constexpr uint64_t expected_magic=0x4452505a534e55;
        static constexpr uint32_t expected_version=1;
        auto start_time=chrono::steady_clock::now();
        FILE *file=fopen(prediction_structure_path.c_str(),"rb");
        if (!file)
            throw runtime_error("找不到预测结构文件："+prediction_structure_path);
        if (fseek(file,0,SEEK_END)!=0)
        {
            fclose(file);
            throw runtime_error("无法定位预测结构文件末尾："+prediction_structure_path);
        }
        long byte_num_signed=ftell(file);
        if (byte_num_signed<0 || fseek(file,0,SEEK_SET)!=0)
        {
            fclose(file);
            throw runtime_error("无法读取预测结构文件大小："+prediction_structure_path);
        }
        vector<unsigned char> bytes((uint64_t)byte_num_signed);
        if (fread(bytes.data(),1,bytes.size(),file)!=bytes.size())
        {
            fclose(file);
            throw runtime_error("读取预测结构文件失败："+prediction_structure_path);
        }
        if (fclose(file)!=0)
            throw runtime_error("关闭预测结构文件失败："+prediction_structure_path);
        const unsigned char *cursor=bytes.data(),*end=bytes.data()+bytes.size();
        uint64_t magic=read_binary<uint64_t>(cursor,end);
        uint32_t version=read_binary<uint32_t>(cursor,end);
        uint32_t file_point_num=read_binary<uint32_t>(cursor,end);
        uint32_t file_dim=read_binary<uint32_t>(cursor,end);
        uint32_t file_round_num=read_binary<uint32_t>(cursor,end);
        if (magic!=expected_magic || version!=expected_version)
            throw runtime_error("预测结构文件格式或版本不正确："+prediction_structure_path);
        if (file_point_num!=point_num || file_dim!=dim || file_round_num!=predict_round_num)
            throw runtime_error("预测结构文件的点数、维数或轮数与当前参数不一致");
        vector<uint32_t> round_target_num_vec(predict_round_num);
        uint64_t target_num_sum=0;
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            round_target_num_vec[round_id]=read_binary<uint32_t>(cursor,end);
            target_num_sum+=round_target_num_vec[round_id];
        }
        if (target_num_sum!=point_num)
            throw runtime_error("预测结构文件中的目标数量与点数不一致");
        round_target_ids.clear();
        round_target_ids.resize(predict_round_num);
        pred_target_ids.clear();
        pred_target_ids.reserve(point_num);
        pred_ref_ids.clear();
        pred_ref_begin.clear();
        pred_ref_begin.reserve((uint64_t)point_num+1);
        pred_ref_begin.push_back(0);
        pred_ref_weights.clear();
        vector<uint8_t> available(point_num,0);
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            round_target_ids[round_id].reserve(round_target_num_vec[round_id]);
            for (uint32_t round_target_id=0;round_target_id<round_target_num_vec[round_id];round_target_id++)
            {
                uint32_t target_id=read_binary<uint32_t>(cursor,end);
                uint32_t ref_num=read_binary<uint32_t>(cursor,end);
                if (target_id>=point_num || available[target_id])
                    throw runtime_error("预测结构文件包含越界或重复的目标编号");
                round_target_ids[round_id].push_back(target_id);
                pred_target_ids.push_back((int)target_id);
                for (uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                {
                    uint32_t ref_id=read_binary<uint32_t>(cursor,end);
                    if (ref_id>=point_num || !available[ref_id])
                        throw runtime_error("预测结构文件包含越界或尚不可用的参考点编号");
                    pred_ref_ids.push_back(ref_id);
                    double dist=(double)get_dist_to_coord(coord_array[target_id],coord_array[ref_id],dim);
                    pred_ref_weights.push_back(dist==0?numeric_limits<double>::infinity():pow(dist,-inverse_distance_power/2));
                }
                pred_ref_begin.push_back(pred_ref_ids.size());
                available[target_id]=1;
            }
        }
        if (cursor!=end)
            throw runtime_error("预测结构文件末尾存在未解析数据");
        print_prediction_geometry_statistics_result();
        prepare_pred_refs_structure();
        printf("Loaded prediction structure: %s, bytes=%llu, time=%f s\n",prediction_structure_path.c_str(),(unsigned long long)bytes.size(),chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_pred_refs_structure()
    {
        if (pred_target_ids.size()!=point_num || pred_ref_begin.size()!=(uint64_t)point_num+1 || pred_ref_ids.size()!=pred_ref_weights.size())
            throw runtime_error("预测参考结构没有覆盖所有点");
        if (partition_size_experimental>0 || partition_num_experimental>0)
            for (uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
                for (int ref_id:get_pred_ref_ids(pred_ref_id))
                    if (point_partition_id_vec[pred_target_ids[pred_ref_id]]!=point_partition_id_vec[ref_id])
                        throw runtime_error("partition预测结构包含跨partition参考点");
        pred_ref_label_normalized_weights.clear();
        pred_ref_value_normalized_weights.clear();
        if (use_normalized_weight_experimental)
        {
            bool normalize_label_weights=point_set_predictor=="continuous_actual" || point_set_predictor=="continuous_reflect";
            auto normalize_weights=[&](vector<double> &normalized_weights,uint32_t ref_num_limit)
            {
                normalized_weights.assign(pred_ref_weights.size(),0);
                for (uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
                {
                    uint64_t begin=pred_ref_begin[pred_ref_id],end=pred_ref_begin[pred_ref_id+1];
                    uint64_t use_end=min<uint64_t>(end,begin+ref_num_limit);
                    bool has_infinite=false;
                    for (uint64_t i=begin;i<use_end;i++)
                        if (isinf(pred_ref_weights[i]))
                        {
                            normalized_weights[i]=1;
                            has_infinite=true;
                            break;
                        }
                    if (has_infinite)
                        continue;
                    double weight_sum=0;
                    for (uint64_t i=begin;i<use_end;i++)
                        weight_sum+=pred_ref_weights[i];
                    if (weight_sum>0)
                        for (uint64_t i=begin;i<use_end;i++)
                            normalized_weights[i]=pred_ref_weights[i]/weight_sum;
                }
            };
            if (normalize_label_weights)
                normalize_weights(pred_ref_label_normalized_weights,actual_ref_num);
            normalize_weights(pred_ref_value_normalized_weights,actual_ref_num);
            printf(normalize_label_weights?"Pre-normalized label and single-point-set value weights\n":"Pre-normalized single-point-set value weights\n");
        }
        prediction_id_to_original_id.clear();
        original_id_to_prediction_id.clear();
        if (use_prediction_order_layout_experimental)
        {
            prediction_id_to_original_id.resize(point_num);
            original_id_to_prediction_id.assign(point_num,numeric_limits<uint32_t>::max());
            for (uint32_t prediction_id=0;prediction_id<point_num;prediction_id++)
            {
                int original_id=pred_target_ids[prediction_id];
                if (original_id<0 || (uint32_t)original_id>=point_num || original_id_to_prediction_id[original_id]!=numeric_limits<uint32_t>::max())
                    throw runtime_error("预测顺序包含无效或重复的原始点编号");
                prediction_id_to_original_id[prediction_id]=original_id;
                original_id_to_prediction_id[original_id]=prediction_id;
            }
            Array2D<T_coord> prediction_order_coord_array(point_num,dim);
            for (uint32_t prediction_id=0;prediction_id<point_num;prediction_id++)
                memcpy(prediction_order_coord_array[prediction_id],coord_array[prediction_id_to_original_id[prediction_id]],dim*sizeof(T_coord));
            coord_array=move(prediction_order_coord_array);
            for (vector<int> &target_ids:round_target_ids)
                for (int &target_id:target_ids)
                    target_id=original_id_to_prediction_id[target_id];
            for (uint32_t prediction_id=0;prediction_id<point_num;prediction_id++)
            {
                pred_target_ids[prediction_id]=prediction_id;
                for (uint64_t ref_index=pred_ref_begin[prediction_id];ref_index<pred_ref_begin[prediction_id+1];ref_index++)
                {
                    int &ref_id=pred_ref_ids[ref_index];
                    ref_id=original_id_to_prediction_id[ref_id];
                    if ((uint32_t)ref_id>=prediction_id)
                        throw runtime_error("预测参考点没有位于目标点之前");
                }
            }
            printf("Reordered internal coordinates and reference ids into prediction order\n");
        }
        bool expect_normalized_label_weights=use_normalized_weight_experimental && (point_set_predictor=="continuous_actual" || point_set_predictor=="continuous_reflect");
        if (use_normalized_weight_experimental && pred_ref_value_normalized_weights.size()!=pred_ref_ids.size())
            throw runtime_error("归一化数值预测权重没有覆盖所有点");
        if (expect_normalized_label_weights && pred_ref_label_normalized_weights.size()!=pred_ref_ids.size())
            throw runtime_error("归一化预测权重没有覆盖所有点");
        partition_pred_ref_ids_vec.clear();
        if(!partition_point_ids_vec.empty())
        {
            partition_pred_ref_ids_vec.resize(partition_point_ids_vec.size());
            for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
            {
                uint32_t pred_id=(uint32_t)pred_target_ids[pred_ref_id];
                uint32_t original_id=use_prediction_order_layout_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                uint32_t partition_id=point_partition_id_vec[original_id];
                if(partition_id>=partition_pred_ref_ids_vec.size())
                    throw runtime_error("预测目标的partition编号越界");
                partition_pred_ref_ids_vec[partition_id].push_back(pred_ref_id);
            }
        }
        printf("Prepared continuous reference arrays: targets=%u, references=%llu\n",point_num,(unsigned long long)pred_ref_ids.size());
    }
}
