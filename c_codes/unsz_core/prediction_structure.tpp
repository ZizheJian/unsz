#pragma once

namespace unsz
{
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
            if (independent_block_size_experimental>0 || independent_block_num_experimental>0)
                build_independent_blocks();
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
        uint64_t ref_num_sum=0;
        for (auto &entry:pred_refs_vec)
            ref_num_sum+=entry.second.size();
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
                if (prediction_id>=pred_refs_vec.size() || pred_refs_vec[prediction_id].first!=target_id)
                    throw runtime_error("预测目标顺序与参考结构不一致，无法保存");
                append_binary(bytes,(uint32_t)target_id);
                append_binary(bytes,(uint32_t)pred_refs_vec[prediction_id].second.size());
                for (int ref_id:pred_refs_vec[prediction_id].second)
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
        pred_refs_vec.clear();
        pred_refs_vec.reserve(point_num);
        pred_ref_weights_vec.clear();
        pred_ref_weights_vec.reserve(point_num);
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
                pred_refs_vec.push_back({(int)target_id,vector<int>()});
                pred_ref_weights_vec.push_back(vector<double>());
                vector<int> &ref_ids=pred_refs_vec.back().second;
                vector<double> &ref_weights=pred_ref_weights_vec.back();
                ref_ids.reserve(ref_num);
                ref_weights.reserve(ref_num);
                for (uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                {
                    uint32_t ref_id=read_binary<uint32_t>(cursor,end);
                    if (ref_id>=point_num || !available[ref_id])
                        throw runtime_error("预测结构文件包含越界或尚不可用的参考点编号");
                    ref_ids.push_back(ref_id);
                    double dist=(double)get_dist_to_coord(coord_array[target_id],coord_array[ref_id],dim);
                    ref_weights.push_back(dist==0?numeric_limits<double>::infinity():pow(dist,-inverse_distance_power/2));
                }
                available[target_id]=1;
            }
        }
        if (cursor!=end)
            throw runtime_error("预测结构文件末尾存在未解析数据");
        prepare_pred_refs_structure();
        printf("Loaded prediction structure: %s, bytes=%llu, time=%f s\n",prediction_structure_path.c_str(),(unsigned long long)bytes.size(),chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_pred_refs_structure()
    {
        if (pred_refs_vec.size()!=point_num || pred_ref_weights_vec.size()!=point_num)
            throw runtime_error("预测参考结构没有覆盖所有点");
        if (independent_block_size_experimental>0 || independent_block_num_experimental>0)
            for (auto &entry:pred_refs_vec)
                for (int ref_id:entry.second)
                    if (point_independent_block_id_vec[entry.first]!=point_independent_block_id_vec[ref_id])
                        throw runtime_error("独立分块预测结构包含跨块参考点");
        pred_ref_label_normalized_weights_vec.clear();
        pred_ref_value_normalized_weights_vec.clear();
        if (use_normalized_weight_experimental)
        {
            bool normalize_label_weights=point_set_predictor=="continuous_actual" || point_set_predictor=="continuous_reflect";
            auto normalize_weights=[](vector<double> &weights,uint32_t ref_num_limit)
            {
                vector<double> normalized_weights(weights.size(),0);
                uint32_t ref_num=min<uint32_t>(ref_num_limit,weights.size());
                for (uint32_t i=0;i<ref_num;i++)
                    if (isinf(weights[i]))
                    {
                        normalized_weights[i]=1;
                        return normalized_weights;
                    }
                double weight_sum=0;
                for (uint32_t i=0;i<ref_num;i++)
                    weight_sum+=weights[i];
                if (weight_sum>0)
                    for (uint32_t i=0;i<ref_num;i++)
                        normalized_weights[i]=weights[i]/weight_sum;
                return normalized_weights;
            };
            if (normalize_label_weights)
                pred_ref_label_normalized_weights_vec.reserve(point_num);
            pred_ref_value_normalized_weights_vec.reserve(point_num);
            for (vector<double> &weights:pred_ref_weights_vec)
            {
                if (normalize_label_weights)
                    pred_ref_label_normalized_weights_vec.push_back(normalize_weights(weights,point_set_ref_num));
                pred_ref_value_normalized_weights_vec.push_back(normalize_weights(weights,actual_ref_num));
            }
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
                int original_id=pred_refs_vec[prediction_id].first;
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
                pred_refs_vec[prediction_id].first=prediction_id;
                for (int &ref_id:pred_refs_vec[prediction_id].second)
                {
                    ref_id=original_id_to_prediction_id[ref_id];
                    if ((uint32_t)ref_id>=prediction_id)
                        throw runtime_error("预测参考点没有位于目标点之前");
                }
            }
            printf("Reordered internal coordinates and reference ids into prediction order\n");
        }
        bool expect_normalized_label_weights=use_normalized_weight_experimental && (point_set_predictor=="continuous_actual" || point_set_predictor=="continuous_reflect");
        if (use_normalized_weight_experimental && pred_ref_value_normalized_weights_vec.size()!=point_num)
            throw runtime_error("归一化数值预测权重没有覆盖所有点");
        if (expect_normalized_label_weights && pred_ref_label_normalized_weights_vec.size()!=point_num)
            throw runtime_error("归一化预测权重没有覆盖所有点");
        continuous_pred_target_ids.clear();
        continuous_pred_ref_ids.clear();
        continuous_pred_ref_begin.clear();
        continuous_pred_ref_weights.clear();
        continuous_pred_ref_label_normalized_weights.clear();
        continuous_pred_ref_value_normalized_weights.clear();
        if (use_continuous_reference_array_experimental)
        {
            uint64_t ref_num_sum=0;
            for (auto &entry:pred_refs_vec)
                ref_num_sum+=entry.second.size();
            continuous_pred_target_ids.resize(point_num);
            continuous_pred_ref_begin.resize((uint64_t)point_num+1);
            continuous_pred_ref_ids.reserve(ref_num_sum);
            continuous_pred_ref_weights.reserve(ref_num_sum);
            if (expect_normalized_label_weights)
                continuous_pred_ref_label_normalized_weights.reserve(ref_num_sum);
            if (use_normalized_weight_experimental)
                continuous_pred_ref_value_normalized_weights.reserve(ref_num_sum);
            for (uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
            {
                continuous_pred_target_ids[pred_ref_id]=pred_refs_vec[pred_ref_id].first;
                continuous_pred_ref_begin[pred_ref_id]=continuous_pred_ref_ids.size();
                continuous_pred_ref_ids.insert(continuous_pred_ref_ids.end(),pred_refs_vec[pred_ref_id].second.begin(),pred_refs_vec[pred_ref_id].second.end());
                continuous_pred_ref_weights.insert(continuous_pred_ref_weights.end(),pred_ref_weights_vec[pred_ref_id].begin(),pred_ref_weights_vec[pred_ref_id].end());
                if (expect_normalized_label_weights)
                    continuous_pred_ref_label_normalized_weights.insert(continuous_pred_ref_label_normalized_weights.end(),pred_ref_label_normalized_weights_vec[pred_ref_id].begin(),pred_ref_label_normalized_weights_vec[pred_ref_id].end());
                if (use_normalized_weight_experimental)
                    continuous_pred_ref_value_normalized_weights.insert(continuous_pred_ref_value_normalized_weights.end(),pred_ref_value_normalized_weights_vec[pred_ref_id].begin(),pred_ref_value_normalized_weights_vec[pred_ref_id].end());
            }
            continuous_pred_ref_begin[point_num]=continuous_pred_ref_ids.size();
            if (continuous_pred_ref_ids.size()!=continuous_pred_ref_weights.size() || (expect_normalized_label_weights && continuous_pred_ref_ids.size()!=continuous_pred_ref_label_normalized_weights.size()) || (use_normalized_weight_experimental && continuous_pred_ref_ids.size()!=continuous_pred_ref_value_normalized_weights.size()))
                throw runtime_error("连续参考数组的编号和权重数量不一致");
            printf("Prepared continuous reference arrays: targets=%u, references=%llu\n",point_num,(unsigned long long)continuous_pred_ref_ids.size());
        }
    }
}
