#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::decompress()
    {
        if (!got_rel_eb && !got_abs_eb && point_set_rel_eb_config_vec.empty() && point_set_abs_eb_config_vec.empty())
            throw runtime_error("解压时没有指定误差上限");
        if (eb_schedule_experimental!="none" && eb_schedule_experimental!="round" && eb_schedule_experimental!="point" && eb_schedule_experimental!="round_point" && eb_schedule_experimental!="round_ratio" && eb_schedule_experimental!="round_point_ratio")
            throw runtime_error("未知的实验性EB序列模式："+eb_schedule_experimental);
        if (!(eb_schedule_start_ratio>0) || eb_schedule_start_ratio>1)
            throw runtime_error("实验性EB序列起始倍率必须在(0,1]内");
        vector<unsigned char> &container=current_zstd_bs_vec;
        const unsigned char *cursor=container.data(),*end=container.data()+container.size();
        const unsigned char magic[8]={'U','N','S','Z','P','S','0','7'};
        if ((uint64_t)(end-cursor)<8 || memcmp(cursor,magic,8)!=0)
            throw runtime_error("不是受支持的UNSZ点集文件");
        cursor+=8;
        uint32_t version=read_binary<uint32_t>(cursor,end);
        if (version!=7)
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
        if (point_set_num>2)
            throw runtime_error("当前0/1标签修正流只支持最多两个点集");
        if (use_block_adaptive_predictor_experimental)
        {
            uint32_t block_config_num=read_binary<uint32_t>(cursor,end);
            if (block_config_num!=independent_block_point_ids_vec.size() || (uint64_t)(end-cursor)<block_config_num)
                throw runtime_error("block预测配置数量与独立空间块不一致");
            block_prediction_config_id_vec.assign(cursor,cursor+block_config_num);
            cursor+=block_config_num;
            uint32_t candidate_num=get_block_prediction_config_vec().size();
            for (uint8_t config_id:block_prediction_config_id_vec)
                if (config_id>=candidate_num)
                    throw runtime_error("压缩文件中的block预测配置编号越界");
            prepare_block_adaptive_sample_mask();
        }

        uint64_t label_zstd_byte_num=read_binary<uint64_t>(cursor,end);
        if ((uint64_t)(end-cursor)<label_zstd_byte_num)
            throw runtime_error("点集标签流不完整");
        vector<uint32_t> label_symbol_vec;
        if (point_set_num==1)
        {
            if (label_zstd_byte_num!=0)
                throw runtime_error("单点集文件不应包含点集标签流");
        }
        else
        {
            vector<unsigned char> label_zstd_bytes(cursor,cursor+label_zstd_byte_num);
            vector<unsigned char> label_position_delta_bytes=zstd_decompress(label_zstd_bytes);
            label_symbol_vec=decode_binary_one_position_delta_vec(label_position_delta_bytes,point_num);
        }
        cursor+=label_zstd_byte_num;

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
            if (point_set_id<point_set_abs_eb_config_vec.size())
                point_set_abs_eb_vec[point_set_id]=point_set_abs_eb_config_vec[point_set_id];
            else if (point_set_id<point_set_rel_eb_config_vec.size())
                point_set_abs_eb_vec[point_set_id]=point_set_rel_eb_config_vec[point_set_id]*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]);
            else
                point_set_abs_eb_vec[point_set_id]=got_abs_eb?abs_eb:rel_eb*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]);
            if (point_set_codec_vec[point_set_id]!=point_set_codec_e::constant && !(point_set_abs_eb_vec[point_set_id]>0))
                throw runtime_error("点集"+to_string(point_set_id)+"的绝对误差上限必须大于0");
        }
        if (cursor!=end)
            throw runtime_error("压缩文件末尾存在未解析数据");

        if (point_set_num>1 && use_prediction_order_layout_experimental && !use_prediction_order_label_experimental)
        {
            vector<uint32_t> label_symbol_file_order_vec=move(label_symbol_vec);
            label_symbol_vec.resize(point_num);
            for (uint32_t prediction_id=0;prediction_id<point_num;prediction_id++)
                label_symbol_vec[prediction_id]=label_symbol_file_order_vec[prediction_id_to_original_id[prediction_id]];
        }

        point_set_id_vec.assign(point_num,0);
        if (point_set_num>1)
        {
            if (point_set_predictor!="inverse_proportion" && point_set_predictor!="majority" && point_set_predictor!="continuous_actual" && point_set_predictor!="continuous_reflect")
                throw runtime_error("未知的点集标签预测器："+point_set_predictor);
            bool continuous=point_set_predictor=="continuous_actual" || point_set_predictor=="continuous_reflect";
            bool use_majority=point_set_predictor=="majority";
            bool use_normalized_label_weights=use_normalized_weight_experimental && continuous;
            point_set_prediction_value_vec.assign(point_num,0);
            for (uint32_t pred_ref_id=0;pred_ref_id<pred_refs_vec.size();pred_ref_id++)
            {
                auto &pred_refs=pred_refs_vec[pred_ref_id];
                auto &ref_weights=use_normalized_label_weights?pred_ref_label_normalized_weights_vec[pred_ref_id]:pred_ref_weights_vec[pred_ref_id];
                int pred_id=pred_refs.first;
                double predicted_value=continuous?(use_normalized_label_weights?predict_point_set_value_normalized(pred_refs.second,ref_weights):predict_point_set_value(pred_refs.second,ref_weights)):0;
                uint32_t predicted_id=continuous?(predicted_value>=0.5):predict_point_set_id(pred_refs.second,ref_weights,use_majority);
                uint32_t symbol=label_symbol_vec[use_prediction_order_label_experimental?pred_ref_id:pred_id];
                if (symbol>1)
                    throw runtime_error("点集标签修正值不是0或1");
                uint32_t actual_id=symbol==0?predicted_id:1-predicted_id;
                if (actual_id>=point_set_num)
                    throw runtime_error("点集标签解码结果越界");
                if (continuous)
                {
                    if (symbol==0)
                        point_set_prediction_value_vec[pred_id]=predicted_value;
                    else if (point_set_predictor=="continuous_actual" || predicted_value==0.5)
                        point_set_prediction_value_vec[pred_id]=actual_id;
                    else
                        point_set_prediction_value_vec[pred_id]=1-predicted_value;
                    actual_id=point_set_prediction_value_vec[pred_id]>=0.5;
                }
                point_set_id_vec[pred_id]=actual_id;
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
        uint64_t pred_ref_index=0;
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            double round_point_eb_multiplier=round_eb_multiplier_vec[round_id];
            for (uint32_t i=0;i<round_target_ids[round_id].size();i++)
            {
                auto &pred_refs=pred_refs_vec[pred_ref_index];
                auto &ref_weights=use_normalized_value_weights?pred_ref_value_normalized_weights_vec[pred_ref_index]:pred_ref_weights_vec[pred_ref_index];
                pred_ref_index++;
                int pred_id=pred_refs.first;
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
                    bool use_block_config=use_block_adaptive_predictor_experimental && !block_adaptive_sample_mask[pred_id];
                    uint32_t previous_time_id=use_prediction_order_layout_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                    bool use_previous_time=has_previous_time && previous_time_point_set_id_vec[previous_time_id]==point_set_id;
                    T_data pred=use_previous_time?previous_time_decompressed_data[previous_time_id]:(use_block_config?predict_block_adaptive_value(pred_id,pred_refs.second,current_decompressed_data,block_prediction_config_id_vec[point_independent_block_id_vec[pred_id]]):(use_normalized_value_weights?predict_value_normalized(pred_refs.second,ref_weights,current_decompressed_data):predict_value(pred_id,pred_refs.second,ref_weights,point_set_id,current_decompressed_data)));
                    int32_t qb=point_set_group_qb_vec[point_set_id][huffman_group_id][qb_index[point_set_id][huffman_group_id]++];
                    double point_abs_eb=point_set_abs_eb_vec[point_set_id]*eb_multiplier;
                    current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                }
            }
        }
        if (pred_ref_index!=pred_refs_vec.size())
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
            if (point_set_prediction_value_vec.size()==point_num)
            {
                vector<double> prediction_order_value_vec=move(point_set_prediction_value_vec);
                point_set_prediction_value_vec.resize(point_num);
                for (uint32_t prediction_id=0;prediction_id<point_num;prediction_id++)
                    point_set_prediction_value_vec[prediction_id_to_original_id[prediction_id]]=prediction_order_value_vec[prediction_id];
            }
        }
    }
}
