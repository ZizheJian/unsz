#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::decompress()
    {
        if (point_set_config_vec.size()!=point_set_classifier_vec.size()+1)
            throw runtime_error("点集配置数量必须等于分类函数数量加1");
        bool has_point_set_error_bound=false;
        for (const point_set_config_c &config:point_set_config_vec)
            has_point_set_error_bound|=config.error_mode!=point_set_error_mode_e::use_global;
        if (!got_rel_eb && !got_abs_eb && !has_point_set_error_bound)
            throw runtime_error("解压时没有指定误差上限");
        if (eb_schedule_experimental!="none" && eb_schedule_experimental!="round" && eb_schedule_experimental!="point" && eb_schedule_experimental!="round_point" && eb_schedule_experimental!="round_ratio" && eb_schedule_experimental!="round_point_ratio")
            throw runtime_error("未知的实验性EB序列模式："+eb_schedule_experimental);
        if (!(eb_schedule_start_ratio>0) || eb_schedule_start_ratio>1)
            throw runtime_error("实验性EB序列起始倍率必须在(0,1]内");
        vector<unsigned char> &container=current_zstd_bs_vec;
        const unsigned char *cursor=container.data(),*end=container.data()+container.size();
        const unsigned char magic_v7[8]={'U','N','S','Z','P','S','0','7'};
        const unsigned char magic_v8[8]={'U','N','S','Z','P','S','0','8'};
        if ((uint64_t)(end-cursor)<8)
            throw runtime_error("不是受支持的UNSZ点集文件");
        bool use_legacy_point_set_format=memcmp(cursor,magic_v7,8)==0;
        if (!use_legacy_point_set_format && memcmp(cursor,magic_v8,8)!=0)
            throw runtime_error("不是受支持的UNSZ点集文件");
        cursor+=8;
        uint32_t version=read_binary<uint32_t>(cursor,end);
        if ((use_legacy_point_set_format && version!=7) || (!use_legacy_point_set_format && version!=8))
            throw runtime_error("不支持的UNSZ文件版本："+to_string(version));
        uint32_t file_point_num=read_binary<uint32_t>(cursor,end);
        uint32_t data_type_bytes=read_binary<uint32_t>(cursor,end);
        uint32_t point_set_num=read_binary<uint32_t>(cursor,end);
        if (file_point_num!=point_num)
            throw runtime_error("压缩文件点数与坐标点数不一致");
        if (data_type_bytes!=sizeof(T_data))
            throw runtime_error("压缩文件的数据类型大小与解压程序不一致");
        if (point_set_num==0 || point_set_num>point_num)
            throw runtime_error("压缩文件中的点集数量无效");
        if (use_legacy_point_set_format && point_set_num>2)
            throw runtime_error("UNSZ v7文件只支持最多两个点集");
        if (point_set_num>point_set_config_vec.size())
            throw runtime_error("压缩文件的点集数量超过当前用户配置");
        if (use_partition_adaptive_predictor_experimental)
        {
            uint32_t block_config_num=read_binary<uint32_t>(cursor,end);
            if (block_config_num!=partition_point_ids_vec.size() || (uint64_t)(end-cursor)<block_config_num)
                throw runtime_error("partition预测配置数量与partition数量不一致");
            partition_prediction_config_id_vec.assign(cursor,cursor+block_config_num);
            cursor+=block_config_num;
            uint32_t candidate_num=get_partition_prediction_config_vec().size();
            for (uint8_t config_id:partition_prediction_config_id_vec)
                if (config_id>=candidate_num)
                    throw runtime_error("压缩文件中的partition预测配置编号越界");
            prepare_partition_adaptive_sample_mask();
        }

        uint32_t label_channel_num=point_set_num-1;
        vector<vector<uint32_t>> label_symbol_vec_vec(label_channel_num);
        vector<uint64_t> label_channel_target_num_vec(label_channel_num,0);
        if (use_legacy_point_set_format)
        {
            uint64_t label_zstd_byte_num=read_binary<uint64_t>(cursor,end);
            if ((uint64_t)(end-cursor)<label_zstd_byte_num)
                throw runtime_error("点集标签流不完整");
            if (point_set_num==1)
            {
                if (label_zstd_byte_num!=0)
                    throw runtime_error("单点集文件不应包含点集标签流");
            }
            else
            {
                vector<unsigned char> label_zstd_bytes(cursor,cursor+label_zstd_byte_num);
                vector<unsigned char> label_position_delta_bytes=zstd_decompress(label_zstd_bytes);
                label_symbol_vec_vec[0]=decode_binary_one_position_delta_vec(label_position_delta_bytes,point_num);
                label_channel_target_num_vec[0]=point_num;
            }
            cursor+=label_zstd_byte_num;
        }
        else
        {
            uint32_t file_label_channel_num=read_binary<uint32_t>(cursor,end);
            if (file_label_channel_num!=label_channel_num)
                throw runtime_error("点集条件标签通道数量与点集数量不一致");
            for (uint32_t label_channel_id=0;label_channel_id<label_channel_num;label_channel_id++)
            {
                uint64_t target_num=read_binary<uint64_t>(cursor,end);
                uint64_t label_zstd_byte_num=read_binary<uint64_t>(cursor,end);
                if (target_num>point_num || (uint64_t)(end-cursor)<label_zstd_byte_num)
                    throw runtime_error("点集条件标签通道"+to_string(label_channel_id)+"不完整");
                vector<unsigned char> label_zstd_bytes(cursor,cursor+label_zstd_byte_num);
                vector<unsigned char> label_position_delta_bytes=label_zstd_byte_num==0?vector<unsigned char>():zstd_decompress(label_zstd_bytes);
                label_symbol_vec_vec[label_channel_id]=decode_binary_one_position_delta_vec(label_position_delta_bytes,target_num);
                label_channel_target_num_vec[label_channel_id]=target_num;
                cursor+=label_zstd_byte_num;
            }
        }

        point_set_point_num_vec.resize(point_set_num);
        point_set_min_data_vec.resize(point_set_num);
        point_set_max_data_vec.resize(point_set_num);
        point_set_abs_eb_vec.resize(point_set_num);
        point_set_codec_vec.resize(point_set_num);
        uint32_t multi_huffman_group_num=get_multi_huffman_group_num();
        vector<vector<vector<int32_t>>> point_set_group_qb_vec(point_set_num,vector<vector<int32_t>>(multi_huffman_group_num));
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            uint8_t codec=read_binary<uint8_t>(cursor,end);
            if (codec>(uint8_t)point_set_codec_e::hybrid_loo)
                throw runtime_error("压缩文件包含未知的点集编码方式");
            point_set_codec_vec[point_set_id]=(point_set_codec_e)codec;
            point_set_point_num_vec[point_set_id]=read_binary<uint32_t>(cursor,end);
            point_set_min_data_vec[point_set_id]=read_binary<T_data>(cursor,end);
            point_set_max_data_vec[point_set_id]=read_binary<T_data>(cursor,end);
            uint64_t qb_num=read_binary<uint64_t>(cursor,end);
            uint64_t qb_zstd_byte_num=read_binary<uint64_t>(cursor,end);
            if ((uint64_t)(end-cursor)<qb_zstd_byte_num)
                throw runtime_error("点集"+to_string(point_set_id)+"的数值流不完整");
            if (qb_num==0)
            {
                if (qb_zstd_byte_num!=0)
                    throw runtime_error("空数值流具有非空Zstd字节流");
            }
            else
            {
                vector<unsigned char> qb_zstd_bytes(cursor,cursor+qb_zstd_byte_num);
                vector<unsigned char> qb_huffman_bytes=zstd_decompress(qb_zstd_bytes);
                if (multi_huffman_mode_experimental=="none")
                    point_set_group_qb_vec[point_set_id][0]=huffman_decode<int32_t>(qb_huffman_bytes.data(),qb_huffman_bytes.size(),qb_num);
                else
                    point_set_group_qb_vec[point_set_id]=decode_multi_huffman_groups(qb_huffman_bytes,multi_huffman_group_num,qb_num);
            }
            cursor+=qb_zstd_byte_num;
            if (point_set_codec_vec[point_set_id]==point_set_codec_e::constant && qb_num!=0)
                throw runtime_error("常量点集不应包含量化索引");
            const point_set_config_c &config=point_set_config_vec[point_set_id];
            if (point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                point_set_abs_eb_vec[point_set_id]=0;
            else if (config.error_mode==point_set_error_mode_e::absolute)
                point_set_abs_eb_vec[point_set_id]=(T_data)config.error_bound;
            else if (config.error_mode==point_set_error_mode_e::relative)
                point_set_abs_eb_vec[point_set_id]=(T_data)(config.error_bound*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]));
            else
            {
                if (!got_rel_eb && !got_abs_eb)
                    throw runtime_error("点集"+to_string(point_set_id)+"没有指定误差上限");
                point_set_abs_eb_vec[point_set_id]=got_abs_eb?abs_eb:rel_eb*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]);
            }
            if (point_set_codec_vec[point_set_id]!=point_set_codec_e::constant && !(point_set_abs_eb_vec[point_set_id]>0))
                throw runtime_error("点集"+to_string(point_set_id)+"的绝对误差上限必须大于0");
        }
        if (cursor!=end)
            throw runtime_error("压缩文件末尾存在未解析数据");

        point_set_id_vec.assign(point_num,point_set_num-1);
        if (label_channel_num>0)
        {
            if (point_set_predictor!="inverse_proportion" && point_set_predictor!="majority" && point_set_predictor!="continuous_actual" && point_set_predictor!="continuous_reflect")
                throw runtime_error("未知的点集标签预测器："+point_set_predictor);
            bool continuous=point_set_predictor=="continuous_actual" || point_set_predictor=="continuous_reflect";
            bool use_majority=point_set_predictor=="majority";
            bool use_normalized_label_weights=use_normalized_weight_experimental && continuous;
            for (uint32_t label_channel_id=0;label_channel_id<label_channel_num;label_channel_id++)
            {
                vector<uint32_t> label_symbol_internal_vec;
                uint64_t label_symbol_id=0;
                if (!use_prediction_order_label_experimental)
                {
                    label_symbol_internal_vec.assign(point_num,0);
                    for (uint32_t original_id=0;original_id<point_num;original_id++)
                    {
                        uint32_t internal_id=use_prediction_order_layout_experimental?original_id_to_prediction_id[original_id]:original_id;
                        if (point_set_id_vec[internal_id]<label_channel_id)
                            continue;
                        if (label_symbol_id>=label_symbol_vec_vec[label_channel_id].size())
                            throw runtime_error("点集条件标签通道"+to_string(label_channel_id)+"的修正值不足");
                        label_symbol_internal_vec[internal_id]=label_symbol_vec_vec[label_channel_id][label_symbol_id++];
                    }
                }
                vector<uint8_t> binary_label_vec(point_num,1);
                vector<double> binary_label_value_vec(continuous?point_num:0,0);
                if (continuous)
                    for (uint32_t point_id=0;point_id<point_num;point_id++)
                        if (point_set_id_vec[point_id]<label_channel_id)
                            binary_label_value_vec[point_id]=1;
                label_symbol_id=use_prediction_order_label_experimental?0:label_symbol_id;
                uint64_t decoded_target_num=0;
                if(thread_num>1)
                {
                    if(partition_pred_ref_ids_vec.empty())
                        throw runtime_error("多线程标签解码需要独立partition预测结构");
                    vector<uint32_t> prediction_order_symbol_vec;
                    if(use_prediction_order_label_experimental)
                    {
                        prediction_order_symbol_vec.assign(point_num,0);
                        for(uint32_t pred_ref_id=0;pred_ref_id<pred_target_ids.size();pred_ref_id++)
                        {
                            int pred_id=pred_target_ids[pred_ref_id];
                            if(point_set_id_vec[pred_id]<label_channel_id)
                                continue;
                            if(label_symbol_id>=label_symbol_vec_vec[label_channel_id].size())
                                throw runtime_error("点集条件标签通道"+to_string(label_channel_id)+"的修正值不足");
                            prediction_order_symbol_vec[pred_ref_id]=label_symbol_vec_vec[label_channel_id][label_symbol_id++];
                        }
                    }
                    vector<uint64_t> partition_decoded_target_num_vec(partition_pred_ref_ids_vec.size(),0);
                    exception_ptr label_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
                    for(int64_t partition_index=0;partition_index<(int64_t)partition_pred_ref_ids_vec.size();partition_index++)
                    {
                        try
                        {
                            uint32_t partition_id=(uint32_t)partition_index;
                            uint64_t local_decoded_target_num=0;
                            for(uint32_t pred_ref_id:partition_pred_ref_ids_vec[partition_id])
                            {
                                int pred_id=pred_target_ids[pred_ref_id];
                                if(point_set_id_vec[pred_id]<label_channel_id)
                                    continue;
                                auto ref_ids=get_pred_ref_ids(pred_ref_id);
                                auto ref_weights=get_pred_ref_weights(pred_ref_id,use_normalized_label_weights,false);
                                double predicted_value=continuous?(use_normalized_label_weights?predict_binary_label_value_normalized(ref_ids,ref_weights,binary_label_value_vec):predict_binary_label_value(ref_ids,ref_weights,binary_label_value_vec)):0;
                                uint32_t predicted_label=continuous?(predicted_value>=0.5):predict_binary_label(ref_ids,ref_weights,binary_label_vec,use_majority);
                                uint32_t symbol=use_prediction_order_label_experimental?prediction_order_symbol_vec[pred_ref_id]:label_symbol_internal_vec[pred_id];
                                if(symbol>1)
                                    throw runtime_error("点集标签修正值不是0或1");
                                uint32_t actual_label=symbol==0?predicted_label:1-predicted_label;
                                binary_label_vec[pred_id]=actual_label;
                                if(continuous)
                                {
                                    if(symbol==0)
                                        binary_label_value_vec[pred_id]=predicted_value;
                                    else if(point_set_predictor=="continuous_actual" || predicted_value==0.5)
                                        binary_label_value_vec[pred_id]=actual_label;
                                    else
                                        binary_label_value_vec[pred_id]=1-predicted_value;
                                    actual_label=binary_label_value_vec[pred_id]>=0.5;
                                }
                                if(actual_label==0)
                                    point_set_id_vec[pred_id]=label_channel_id;
                                local_decoded_target_num++;
                            }
                            partition_decoded_target_num_vec[partition_id]=local_decoded_target_num;
                        }
                        catch(...)
                        {
#pragma omp critical
                            if(!label_exception)
                                label_exception=current_exception();
                        }
                    }
                    if(label_exception)
                        rethrow_exception(label_exception);
                    decoded_target_num=accumulate(partition_decoded_target_num_vec.begin(),partition_decoded_target_num_vec.end(),(uint64_t)0);
                }
                else
                {
                    for(uint32_t pred_ref_id=0;pred_ref_id<pred_target_ids.size();pred_ref_id++)
                    {
                        int pred_id=pred_target_ids[pred_ref_id];
                        if(point_set_id_vec[pred_id]<label_channel_id)
                            continue;
                        auto ref_ids=get_pred_ref_ids(pred_ref_id);
                        auto ref_weights=get_pred_ref_weights(pred_ref_id,use_normalized_label_weights,false);
                        double predicted_value=continuous?(use_normalized_label_weights?predict_binary_label_value_normalized(ref_ids,ref_weights,binary_label_value_vec):predict_binary_label_value(ref_ids,ref_weights,binary_label_value_vec)):0;
                        uint32_t predicted_label=continuous?(predicted_value>=0.5):predict_binary_label(ref_ids,ref_weights,binary_label_vec,use_majority);
                        if(use_prediction_order_label_experimental && label_symbol_id>=label_symbol_vec_vec[label_channel_id].size())
                            throw runtime_error("点集条件标签通道"+to_string(label_channel_id)+"的修正值不足");
                        uint32_t symbol=use_prediction_order_label_experimental?label_symbol_vec_vec[label_channel_id][label_symbol_id++]:label_symbol_internal_vec[pred_id];
                        if(symbol>1)
                            throw runtime_error("点集标签修正值不是0或1");
                        uint32_t actual_label=symbol==0?predicted_label:1-predicted_label;
                        binary_label_vec[pred_id]=actual_label;
                        if(continuous)
                        {
                            if(symbol==0)
                                binary_label_value_vec[pred_id]=predicted_value;
                            else if(point_set_predictor=="continuous_actual" || predicted_value==0.5)
                                binary_label_value_vec[pred_id]=actual_label;
                            else
                                binary_label_value_vec[pred_id]=1-predicted_value;
                            actual_label=binary_label_value_vec[pred_id]>=0.5;
                        }
                        if(actual_label==0)
                            point_set_id_vec[pred_id]=label_channel_id;
                        decoded_target_num++;
                    }
                }
                if (decoded_target_num!=label_channel_target_num_vec[label_channel_id] || label_symbol_id!=label_symbol_vec_vec[label_channel_id].size())
                    throw runtime_error("点集条件标签通道"+to_string(label_channel_id)+"的修正值数量不一致");
            }
        }
        vector<uint32_t> decoded_point_set_num(point_set_num,0);
        for (uint32_t point_set_id:point_set_id_vec)
            decoded_point_set_num[point_set_id]++;
        if (decoded_point_set_num!=point_set_point_num_vec)
            throw runtime_error("解码后的点集数量与文件头不一致");

        vector<double> round_eb_multiplier_vec(predict_round_num,1);
        double schedule_common_ratio=1,point_eb_multiplier=1;
        vector<double> round_point_common_ratio_vec(predict_round_num,1);
        bool use_round_schedule=eb_schedule_experimental=="round" || eb_schedule_experimental=="round_point" || eb_schedule_experimental=="round_ratio" || eb_schedule_experimental=="round_point_ratio";
        bool use_round_point_schedule=eb_schedule_experimental=="round_point" || eb_schedule_experimental=="round_point_ratio";
        bool parameter_is_round_ratio=eb_schedule_experimental=="round_ratio" || eb_schedule_experimental=="round_point_ratio";
        if (use_round_schedule && predict_round_num>1)
        {
            schedule_common_ratio=parameter_is_round_ratio?1/eb_schedule_start_ratio:pow(1/eb_schedule_start_ratio,1.0/(predict_round_num-1));
            for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
                round_eb_multiplier_vec[round_id]=parameter_is_round_ratio?pow(eb_schedule_start_ratio,predict_round_num-1-round_id):pow(eb_schedule_start_ratio,(double)(predict_round_num-1-round_id)/(predict_round_num-1));
            if (use_round_point_schedule)
                for (uint32_t round_id=0;round_id+1<predict_round_num;round_id++)
                    if (round_target_ids[round_id].size()>1)
                        round_point_common_ratio_vec[round_id]=pow(round_eb_multiplier_vec[round_id+1]/round_eb_multiplier_vec[round_id],1.0/(round_target_ids[round_id].size()-1));
        }
        else if (eb_schedule_experimental=="point")
        {
            point_eb_multiplier=eb_schedule_start_ratio;
            if (point_num>1)
                schedule_common_ratio=pow(1/eb_schedule_start_ratio,1.0/(point_num-1));
        }

        current_decompressed_data.assign(point_num,0);
        vector<vector<uint64_t>> qb_index(point_set_num,vector<uint64_t>(multi_huffman_group_num,0));
        bool use_normalized_value_weights=use_normalized_weight_experimental && point_set_num==1 && point_set_codec_vec[0]==point_set_codec_e::inverse_proportion;
        bool has_previous_time=cross_time_predictor_experimental=="previous" && previous_time_decompressed_data.size()==point_num && previous_time_point_set_id_vec.size()==point_num;
        prepare_time_interpolation();
        uint32_t pred_ref_index=0;
        if(thread_num>1)
        {
            if(partition_pred_ref_ids_vec.empty())
                throw runtime_error("多线程数值解码需要独立partition预测结构");
            vector<double> pred_eb_multiplier_vec(point_num,1);
            vector<int32_t> parallel_qb_vec(point_num);
            vector<uint8_t> parallel_has_qb_vec(point_num,0);
            double schedule_point_eb_multiplier=point_eb_multiplier;
            for(uint32_t round_id=0;round_id<predict_round_num;round_id++)
            {
                double round_point_eb_multiplier=round_eb_multiplier_vec[round_id];
                for(uint32_t i=0;i<round_target_ids[round_id].size();i++)
                {
                    uint32_t current_pred_ref_id=pred_ref_index++;
                    int pred_id=pred_target_ids[current_pred_ref_id];
                    uint32_t point_set_id=point_set_id_vec[pred_id];
                    pred_eb_multiplier_vec[current_pred_ref_id]=(eb_schedule_experimental=="round" || eb_schedule_experimental=="round_ratio")?round_eb_multiplier_vec[round_id]:(use_round_point_schedule?round_point_eb_multiplier:schedule_point_eb_multiplier);
                    if(eb_schedule_experimental=="point")
                        schedule_point_eb_multiplier=min(1.0,schedule_point_eb_multiplier*schedule_common_ratio);
                    else if(use_round_point_schedule)
                        round_point_eb_multiplier=min(round_eb_multiplier_vec[min(round_id+1,predict_round_num-1)],round_point_eb_multiplier*round_point_common_ratio_vec[round_id]);
                    if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                        continue;
                    uint32_t huffman_group_id=get_multi_huffman_group_id(round_id,pred_id);
                    if(qb_index[point_set_id][huffman_group_id]>=point_set_group_qb_vec[point_set_id][huffman_group_id].size())
                        throw runtime_error("点集"+to_string(point_set_id)+"的量化索引不足");
                    parallel_qb_vec[current_pred_ref_id]=point_set_group_qb_vec[point_set_id][huffman_group_id][qb_index[point_set_id][huffman_group_id]++];
                    parallel_has_qb_vec[current_pred_ref_id]=1;
                }
            }
            if(pred_ref_index!=point_num)
                throw runtime_error("多线程数值解码没有映射全部目标");
            exception_ptr prediction_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
            for(int64_t partition_index=0;partition_index<(int64_t)partition_pred_ref_ids_vec.size();partition_index++)
            {
                try
                {
                    uint32_t partition_id=(uint32_t)partition_index;
                    for(uint32_t current_pred_ref_id:partition_pred_ref_ids_vec[partition_id])
                    {
                        int pred_id=pred_target_ids[current_pred_ref_id];
                        uint32_t point_set_id=point_set_id_vec[pred_id];
                        if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                        {
                            current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                            continue;
                        }
                        if(!parallel_has_qb_vec[current_pred_ref_id])
                            throw runtime_error("多线程数值解码缺少量化索引");
                        auto ref_ids=get_pred_ref_ids(current_pred_ref_id);
                        auto ref_weights=get_pred_ref_weights(current_pred_ref_id,false,use_normalized_value_weights);
                        bool use_partition_config=use_partition_adaptive_predictor_experimental && !partition_adaptive_sample_mask[pred_id];
                        uint32_t original_id=use_prediction_order_layout_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                        T_data pred=0;
                        bool use_cross_time_prediction=false;
                        if(has_previous_time && previous_time_point_set_id_vec[original_id]==point_set_id)
                        {
                            pred=previous_time_decompressed_data[original_id];
                            use_cross_time_prediction=true;
                        }
                        else if(cross_time_predictor_experimental=="interpolation")
                            use_cross_time_prediction=predict_time_interpolation(original_id,point_set_id,pred);
                        if(!use_cross_time_prediction)
                            pred=use_partition_config?predict_partition_adaptive_value(pred_id,ref_ids,current_decompressed_data,partition_prediction_config_id_vec[partition_id]):(use_normalized_value_weights?predict_value_normalized(ref_ids,ref_weights,current_decompressed_data):predict_value(pred_id,ref_ids,ref_weights,point_set_id,current_decompressed_data));
                        double point_abs_eb=point_set_abs_eb_vec[point_set_id]*pred_eb_multiplier_vec[current_pred_ref_id];
                        current_decompressed_data[pred_id]=pred+parallel_qb_vec[current_pred_ref_id]*2*point_abs_eb;
                    }
                }
                catch(...)
                {
#pragma omp critical
                    if(!prediction_exception)
                        prediction_exception=current_exception();
                }
            }
            if(prediction_exception)
                rethrow_exception(prediction_exception);
        }
        else
        {
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            double round_point_eb_multiplier=round_eb_multiplier_vec[round_id];
            for (uint32_t i=0;i<round_target_ids[round_id].size();i++)
            {
                uint32_t current_pred_ref_id=pred_ref_index++;
                auto ref_ids=get_pred_ref_ids(current_pred_ref_id);
                auto ref_weights=get_pred_ref_weights(current_pred_ref_id,false,use_normalized_value_weights);
                int pred_id=pred_target_ids[current_pred_ref_id];
                uint32_t point_set_id=point_set_id_vec[pred_id];
                double eb_multiplier=(eb_schedule_experimental=="round" || eb_schedule_experimental=="round_ratio")?round_eb_multiplier_vec[round_id]:(use_round_point_schedule?round_point_eb_multiplier:point_eb_multiplier);
                if (eb_schedule_experimental=="point")
                    point_eb_multiplier=min(1.0,point_eb_multiplier*schedule_common_ratio);
                else if (use_round_point_schedule)
                    round_point_eb_multiplier=min(round_eb_multiplier_vec[min(round_id+1,predict_round_num-1)],round_point_eb_multiplier*round_point_common_ratio_vec[round_id]);
                if (point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                    current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                else
                {
                    uint32_t huffman_group_id=get_multi_huffman_group_id(round_id,pred_id);
                    if (qb_index[point_set_id][huffman_group_id]>=point_set_group_qb_vec[point_set_id][huffman_group_id].size())
                        throw runtime_error("点集"+to_string(point_set_id)+"的量化索引不足");
                    bool use_partition_config=use_partition_adaptive_predictor_experimental && !partition_adaptive_sample_mask[pred_id];
                    uint32_t original_id=use_prediction_order_layout_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                    T_data pred=0;
                    bool use_cross_time_prediction=false;
                    if (has_previous_time && previous_time_point_set_id_vec[original_id]==point_set_id)
                    {
                        pred=previous_time_decompressed_data[original_id];
                        use_cross_time_prediction=true;
                    }
                    else if (cross_time_predictor_experimental=="interpolation")
                        use_cross_time_prediction=predict_time_interpolation(original_id,point_set_id,pred);
                    if (!use_cross_time_prediction)
                        pred=use_partition_config?predict_partition_adaptive_value(pred_id,ref_ids,current_decompressed_data,partition_prediction_config_id_vec[point_partition_id_vec[pred_id]]):(use_normalized_value_weights?predict_value_normalized(ref_ids,ref_weights,current_decompressed_data):predict_value(pred_id,ref_ids,ref_weights,point_set_id,current_decompressed_data));
                    int32_t qb=point_set_group_qb_vec[point_set_id][huffman_group_id][qb_index[point_set_id][huffman_group_id]++];
                    double point_abs_eb=point_set_abs_eb_vec[point_set_id]*eb_multiplier;
                    current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                }
            }
        }
        }
        if (pred_ref_index!=pred_target_ids.size())
            throw runtime_error("逐轮预测目标数量与预测参考结构不一致");
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
            for (uint32_t group_id=0;group_id<multi_huffman_group_num;group_id++)
                if (qb_index[point_set_id][group_id]!=point_set_group_qb_vec[point_set_id][group_id].size())
                    throw runtime_error("点集"+to_string(point_set_id)+"的Huffman分组"+to_string(group_id)+"存在未使用的量化索引");
        uint32_t primary_point_set_id=0;
        while (point_set_point_num_vec[primary_point_set_id]==0)
            primary_point_set_id++;
        current_min_data=point_set_min_data_vec[primary_point_set_id];
        current_max_data=point_set_max_data_vec[primary_point_set_id];
        current_abs_eb=point_set_abs_eb_vec[primary_point_set_id];
        if (use_prediction_order_layout_experimental)
        {
            vector<T_data> prediction_order_data=move(current_decompressed_data);
            vector<uint32_t> prediction_order_point_set_id_vec=move(point_set_id_vec);
            current_decompressed_data.resize(point_num);
            point_set_id_vec.resize(point_num);
            for (uint32_t prediction_id=0;prediction_id<point_num;prediction_id++)
            {
                uint32_t original_id=prediction_id_to_original_id[prediction_id];
                current_decompressed_data[original_id]=prediction_order_data[prediction_id];
                point_set_id_vec[original_id]=prediction_order_point_set_id_vec[prediction_id];
            }
        }
    }
}
