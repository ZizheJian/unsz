#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::load_or_generate_prediction_structures()
    {
        if((partition_size>0||partition_num>0)&&!dag_cache_path.empty())
            throw runtime_error("独立partition预测DAG暂不支持缓存");
        if (dag_cache_path.empty())
        {
            generate_prediction_structures_and_weights();
            return;
        }
        if(filesystem::exists(dag_cache_path))
        {
            load_prediction_structure_cache();
        }
        else
            generate_prediction_structures_and_weights();
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::save_prediction_structure_cache()
    {
        if(prediction_structure_vec.size()!=1)
            throw runtime_error("DAG缓存只支持单个预测结构");
        const prediction_structure_c<T_calc>&structure=prediction_structure_vec.front();
        static constexpr uint64_t magic=0x4452505a534e55;//"UNSZPRD"
        static constexpr uint32_t version=1;
        vector<unsigned char> bytes;
        uint64_t ref_num_sum=structure.reference_ids.size();
        bytes.reserve(32+(uint64_t)prediction_round_num*4+(uint64_t)point_num*8+ref_num_sum*4);
        append_binary(bytes,magic);
        append_binary(bytes,version);
        append_binary(bytes,point_num);
        append_binary(bytes,dim);
        append_binary(bytes,prediction_round_num);
        for(const vector<int>&target_ids:structure.target_id_groups)
            append_binary(bytes,(uint32_t)target_ids.size());
        uint32_t prediction_id=0;
        for(const vector<int>&target_ids:structure.target_id_groups)
            for (int target_id:target_ids)
            {
                if(prediction_id>=structure.target_ids.size()||structure.target_ids[prediction_id]!=target_id)
                    throw runtime_error("预测目标顺序与参考结构不一致，无法保存");
                append_binary(bytes,(uint32_t)target_id);
                auto ref_ids=structure.get_reference_ids(prediction_id);
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
    void param_c<T_coord,T_data>::load_prediction_structure_cache()
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
        prediction_structure_vec.clear();
        prediction_structure_vec.resize(1);
        prediction_structure_c<T_calc>&structure=prediction_structure_vec.front();
        structure.target_id_groups.resize(prediction_round_num);
        structure.target_ids.reserve(point_num);
        structure.reference_ids.reserve((uint64_t)point_num*prediction_max_ref_num);
        structure.reference_offsets.reserve((uint64_t)point_num+1);
        structure.reference_offsets.push_back(0);
        bool generate_idw_weights=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::idw;});
        if(generate_idw_weights)
            structure.prediction_weights.reserve((uint64_t)point_num*prediction_max_ref_num);
        vector<uint8_t> available(point_num,0);
        for (uint32_t round_id=0;round_id<prediction_round_num;round_id++)
        {
            structure.target_id_groups[round_id].reserve(round_target_num_vec[round_id]);
            for (uint32_t round_target_id=0;round_target_id<round_target_num_vec[round_id];round_target_id++)
            {
                uint32_t target_id=read_binary<uint32_t>(cursor,end);
                uint32_t ref_num=read_binary<uint32_t>(cursor,end);
                if (target_id>=point_num || available[target_id])
                    throw runtime_error("DAG缓存文件包含越界或重复的目标编号");
                if(ref_num>prediction_max_ref_num)
                    throw runtime_error("DAG缓存文件的单点参考点数量超过当前prediction_max_ref_num");
                structure.target_id_groups[round_id].push_back(target_id);
                structure.target_ids.push_back((int)target_id);
                for (uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                {
                    uint32_t ref_id=read_binary<uint32_t>(cursor,end);
                    if (ref_id>=point_num || !available[ref_id])
                        throw runtime_error("DAG缓存文件包含越界或尚不可用的参考点编号");
                    structure.reference_ids.push_back(ref_id);
                }
                structure.reference_offsets.push_back(structure.reference_ids.size());
                if(generate_idw_weights)
                    append_normalized_idw_prediction_weights(target_id,structure.get_reference_ids((uint32_t)structure.target_ids.size()-1),structure.prediction_weights);
                available[target_id]=1;
            }
        }
        if (cursor!=end)
            throw runtime_error("DAG缓存文件末尾存在未解析数据");
        printf("Loaded DAG cache: %s, bytes=%llu, time=%f s\n",dag_cache_path.c_str(),(unsigned long long)bytes.size(),chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::validate_prediction_structures() const
    {
        bool has_idw=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::idw;});
        bool has_rbf=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::rbf;});
        if(prediction_structure_vec.empty())
            throw runtime_error("预测结构不存在");
        uint64_t target_num=0;
        vector<uint8_t> target_seen(point_num,0),available(point_num,0);
        for(uint32_t structure_id=0;structure_id<prediction_structure_vec.size();structure_id++)
        {
            const prediction_structure_c<T_calc>&structure=prediction_structure_vec[structure_id];
            fill(available.begin(),available.end(),0);
            if(structure.reference_offsets.size()!=structure.target_ids.size()+1||structure.reference_offsets.empty()||structure.reference_offsets.front()!=0||structure.reference_offsets.back()!=structure.reference_ids.size())
                throw runtime_error("预测参考结构不完整");
            if(structure.eb_multiplier.size()!=structure.target_ids.size())
                throw runtime_error("EB倍率没有覆盖所有目标");
            if((has_idw||has_rbf)&&structure.prediction_weights.size()!=structure.reference_ids.size())
                throw runtime_error("预测权重没有覆盖所有参考点");
            if(has_rbf&&structure.rbf_fallback_flags.size()!=structure.target_ids.size())
                throw runtime_error("RBF回退标记没有覆盖所有目标");
            for(uint32_t prediction_id=0;prediction_id<structure.target_ids.size();prediction_id++)
            {
                int pred_id=structure.target_ids[prediction_id];
                if(pred_id<0||(uint32_t)pred_id>=point_num||target_seen[pred_id])
                    throw runtime_error("预测结构包含越界或重复的目标点");
                uint64_t begin=structure.reference_offsets[prediction_id],end=structure.reference_offsets[prediction_id+1];
                for(uint64_t ref_offset=begin;ref_offset<end;ref_offset++)
                {
                    int ref_id=structure.reference_ids[ref_offset];
                    if(ref_id<0||(uint32_t)ref_id>=point_num||!available[ref_id])
                        throw runtime_error("预测结构包含尚不可用的参考点");
                }
                available[pred_id]=1;
                target_seen[pred_id]=1;
            }
            target_num+=structure.target_ids.size();
        }
        if(target_num!=point_num)
            throw runtime_error("预测结构没有覆盖所有点");
    }

}
