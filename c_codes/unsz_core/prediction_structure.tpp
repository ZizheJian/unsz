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
        if (dag_cache_path.empty())
        {
            generate_pred_refs_structure();
            return;
        }
        FILE *file=fopen(dag_cache_path.c_str(),"rb");
        if (file)
        {
            if (fclose(file)!=0)
                throw runtime_error("关闭DAG缓存文件失败："+dag_cache_path);
            if (partition_size_experimental>0 || partition_num_experimental>0)
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
        round_target_ids.clear();
        round_target_ids.resize(prediction_round_num);
        pred_target_ids.clear();
        pred_target_ids.reserve(point_num);
        pred_ref_ids.clear();
        pred_ref_begin.clear();
        pred_ref_begin.reserve((uint64_t)point_num+1);
        pred_ref_begin.push_back(0);
        pred_ref_weights.clear();
        vector<uint8_t> available(point_num,0);
        for (uint32_t round_id=0;round_id<prediction_round_num;round_id++)
        {
            round_target_ids[round_id].reserve(round_target_num_vec[round_id]);
            for (uint32_t round_target_id=0;round_target_id<round_target_num_vec[round_id];round_target_id++)
            {
                uint32_t target_id=read_binary<uint32_t>(cursor,end);
                uint32_t ref_num=read_binary<uint32_t>(cursor,end);
                if (target_id>=point_num || available[target_id])
                    throw runtime_error("DAG缓存文件包含越界或重复的目标编号");
                round_target_ids[round_id].push_back(target_id);
                pred_target_ids.push_back((int)target_id);
                for (uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                {
                    uint32_t ref_id=read_binary<uint32_t>(cursor,end);
                    if (ref_id>=point_num || !available[ref_id])
                        throw runtime_error("DAG缓存文件包含越界或尚不可用的参考点编号");
                    pred_ref_ids.push_back(ref_id);
                    double dist=(double)get_dist_to_coord(coord_array[target_id],coord_array[ref_id],dim);
                    pred_ref_weights.push_back(dist==0?numeric_limits<double>::infinity():pow(dist,-idw_power/2));
                }
                pred_ref_begin.push_back(pred_ref_ids.size());
                available[target_id]=1;
            }
        }
        if (cursor!=end)
            throw runtime_error("DAG缓存文件末尾存在未解析数据");
        prepare_pred_refs_structure();
        printf("Loaded DAG cache: %s, bytes=%llu, time=%f s\n",dag_cache_path.c_str(),(unsigned long long)bytes.size(),chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
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
                        throw runtime_error("partition预测DAG包含跨partition参考点");
        pred_ref_label_normalized_weights.clear();
        pred_ref_value_normalized_weights.clear();
        pred_ref_rbf_weights.clear();
        single_point_set_rbf10_fast_begin=0;
        bool has_idw=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::idw;});
        bool use_normalized_idw_weights=has_idw&&(point_set_config_vec.size()==1||label_value_execution=="fused");
        if(use_normalized_idw_weights)
        {
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
            normalize_weights(pred_ref_value_normalized_weights,prediction_max_ref_num);
            printf("Pre-normalized IDW weights\n");
        }
        if(use_normalized_idw_weights&&pred_ref_value_normalized_weights.size()!=pred_ref_ids.size())
            throw runtime_error("归一化数值预测权重没有覆盖所有点");
        partition_pred_ref_ids_vec.clear();
        if(!partition_point_ids_vec.empty())
        {
            partition_pred_ref_ids_vec.resize(partition_point_ids_vec.size());
            for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
            {
                uint32_t pred_id=(uint32_t)pred_target_ids[pred_ref_id];
                uint32_t original_id=pred_id;
                uint32_t partition_id=point_partition_id_vec[original_id];
                if(partition_id>=partition_pred_ref_ids_vec.size())
                    throw runtime_error("预测目标的partition编号越界");
                partition_pred_ref_ids_vec[partition_id].push_back(pred_ref_id);
            }
        }
        prepare_multihead_execution_order();
        printf("Prepared continuous reference arrays: targets=%u, references=%llu\n",point_num,(unsigned long long)pred_ref_ids.size());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_rbf_weights()
    {
        if(pred_ref_rbf_weights.size()==pred_ref_ids.size())
            return;
        auto start_time=chrono::steady_clock::now();
        pred_ref_rbf_weights.assign(pred_ref_ids.size(),numeric_limits<double>::quiet_NaN());
        uint64_t prepared_target_num=0;
        vector<double> weights;
        for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
        {
            auto ref_ids=get_pred_ref_ids(pred_ref_id);
            uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,ref_ids.size());
            if(ref_num<prediction_min_ref_num)
                continue;
            array_view_c<int> selected_ref_ids(ref_ids.begin(),ref_num);
            if(!calculate_rbf_weights(pred_target_ids[pred_ref_id],selected_ref_ids,weights))
                continue;
            uint64_t begin=pred_ref_begin[pred_ref_id];
            for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                pred_ref_rbf_weights[begin+ref_index]=weights[ref_index];
            prepared_target_num++;
        }
        single_point_set_rbf10_fast_begin=point_num;
        if(prediction_max_ref_num==10)
            for(uint32_t reverse_pred_ref_id=point_num;reverse_pred_ref_id>0;reverse_pred_ref_id--)
            {
                uint32_t pred_ref_id=reverse_pred_ref_id-1;
                uint64_t begin=pred_ref_begin[pred_ref_id];
                if(pred_ref_begin[pred_ref_id+1]-begin!=10||!isfinite(pred_ref_rbf_weights[begin]))
                    break;
                single_point_set_rbf10_fast_begin=pred_ref_id;
            }
        printf("Precomputed RBF weights: targets=%llu,time=%f s\n",(unsigned long long)prepared_target_num,chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
    }
}
