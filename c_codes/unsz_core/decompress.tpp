#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::decompress_single_point_set()
    {
        if(point_set_config_vec.size()!=1)
            throw runtime_error("单点集解压要求field恰好配置一个点集");
        resolve_single_point_set_time_mode();
        if(value_flow=="joint_spatiotemporal")
            throw runtime_error("联合时空解压必须通过联合时空入口处理");
        point_set_codec_e codec=point_set_config_vec[0].codec;
        auto run_fixed=[&](auto mode_constant,auto predictor_constant)
        {
            constexpr time_dimension_mode_e mode=decltype(mode_constant)::value;
            constexpr point_set_codec_e predictor=decltype(predictor_constant)::value;
            decompress_point_sets<false,mode,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,predictor>();
        };
        auto dispatch_predictor=[&](auto mode_constant)
        {
            if(codec==point_set_codec_e::idw)
                run_fixed(mode_constant,integral_constant<point_set_codec_e,point_set_codec_e::idw>());
            else if(codec==point_set_codec_e::rbf)
                run_fixed(mode_constant,integral_constant<point_set_codec_e,point_set_codec_e::rbf>());
            else
                throw runtime_error("单点集解压只支持IDW或RBF空间预测器");
        };
        prediction_framework_c framework=get_prediction_framework(1);
        if(framework.value_mode==time_dimension_mode_e::spatial)
            dispatch_predictor(integral_constant<time_dimension_mode_e,time_dimension_mode_e::spatial>());
        else if(framework.value_mode==time_dimension_mode_e::constant_extrapolation)
            dispatch_predictor(integral_constant<time_dimension_mode_e,time_dimension_mode_e::constant_extrapolation>());
        else if(framework.value_mode==time_dimension_mode_e::linear_interpolation)
            dispatch_predictor(integral_constant<time_dimension_mode_e,time_dimension_mode_e::linear_interpolation>());
        else if(framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
            dispatch_predictor(integral_constant<time_dimension_mode_e,time_dimension_mode_e::constant_extrapolation_spatial_residual>());
        else if(framework.value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
            dispatch_predictor(integral_constant<time_dimension_mode_e,time_dimension_mode_e::linear_interpolation_spatial_residual>());
        else
            throw runtime_error("未知的单点集数值预测流程："+value_flow);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::decompress_two_point_sets()
    {
        if(point_set_config_vec.size()!=2)
            throw runtime_error("两点集解压要求field恰好配置两个点集");
        prediction_framework_c framework=get_prediction_framework(2);
        if(framework.execution==label_value_execution_e::staged)
        {
            auto dispatch_value=[&](auto label_mode_constant)
            {
                constexpr time_dimension_mode_e label_mode=decltype(label_mode_constant)::value;
                switch(framework.value_mode)
                {
                    case time_dimension_mode_e::spatial:return decompress_two_point_sets_fixed<label_mode,time_dimension_mode_e::spatial,label_value_execution_e::staged>();
                    case time_dimension_mode_e::constant_extrapolation:return decompress_two_point_sets_fixed<label_mode,time_dimension_mode_e::constant_extrapolation,label_value_execution_e::staged>();
                    case time_dimension_mode_e::linear_interpolation:return decompress_two_point_sets_fixed<label_mode,time_dimension_mode_e::linear_interpolation,label_value_execution_e::staged>();
                    case time_dimension_mode_e::constant_extrapolation_spatial_residual:return decompress_two_point_sets_fixed<label_mode,time_dimension_mode_e::constant_extrapolation_spatial_residual,label_value_execution_e::staged>();
                    case time_dimension_mode_e::linear_interpolation_spatial_residual:return decompress_two_point_sets_fixed<label_mode,time_dimension_mode_e::linear_interpolation_spatial_residual,label_value_execution_e::staged>();
                    default:throw runtime_error("分阶段两点集解压不支持当前数值流程");
                }
            };
            if(framework.label_mode==time_dimension_mode_e::spatial)
                return dispatch_value(integral_constant<time_dimension_mode_e,time_dimension_mode_e::spatial>());
            if(framework.label_mode==time_dimension_mode_e::constant_extrapolation)
                return dispatch_value(integral_constant<time_dimension_mode_e,time_dimension_mode_e::constant_extrapolation>());
            if(framework.label_mode==time_dimension_mode_e::endpoint_agreement)
                return dispatch_value(integral_constant<time_dimension_mode_e,time_dimension_mode_e::endpoint_agreement>());
        }
        else if(framework.execution==label_value_execution_e::fused)
        {
            if(framework.label_mode==time_dimension_mode_e::spatial&&framework.value_mode==time_dimension_mode_e::spatial)
                return decompress_two_point_sets_fixed<time_dimension_mode_e::spatial,time_dimension_mode_e::spatial,label_value_execution_e::fused>();
            if(framework.label_mode==time_dimension_mode_e::constant_extrapolation&&framework.value_mode==time_dimension_mode_e::spatial)
                return decompress_two_point_sets_fixed<time_dimension_mode_e::constant_extrapolation,time_dimension_mode_e::spatial,label_value_execution_e::fused>();
            if(framework.label_mode==time_dimension_mode_e::constant_extrapolation&&framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
                return decompress_two_point_sets_fixed<time_dimension_mode_e::constant_extrapolation,time_dimension_mode_e::constant_extrapolation_spatial_residual,label_value_execution_e::fused>();
            if(framework.label_mode==time_dimension_mode_e::endpoint_agreement&&framework.value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                return decompress_two_point_sets_fixed<time_dimension_mode_e::endpoint_agreement,time_dimension_mode_e::linear_interpolation_spatial_residual,label_value_execution_e::fused>();
        }
        throw runtime_error("两点集解压尚未实现当前标签、数值和执行方式的组合");
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e label_mode,time_dimension_mode_e value_mode,label_value_execution_e execution>
    void param_c<T_coord,T_data>::decompress_two_point_sets_fixed()
    {
        point_set_codec_e predictor=point_set_codec_e::constant;
        for(const point_set_config_c &config:point_set_config_vec)
            if(config.codec!=point_set_codec_e::constant)
            {
                if(predictor!=point_set_codec_e::constant&&predictor!=config.codec)
                    throw runtime_error("同一次两点集解压中的非常量点集必须使用同一种空间预测器");
                predictor=config.codec;
            }
        if(predictor==point_set_codec_e::idw)
            return decompress_point_sets<true,value_mode,label_mode,execution,point_set_codec_e::idw>();
        if(predictor==point_set_codec_e::rbf)
            return decompress_point_sets<true,value_mode,label_mode,execution,point_set_codec_e::rbf>();
        throw runtime_error("两点集解压至少需要一个使用IDW或RBF的非常量点集");
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::decompress_point_sets()
    {
        constexpr bool use_normalized_idw_weights=fixed_predictor==point_set_codec_e::idw;
        if constexpr(two_point_sets)
            point_set_predictor=fixed_predictor==point_set_codec_e::rbf?"rbf":"idw";
        auto decompression_stage_start_time=chrono::steady_clock::now();
        double stream_decode_time=0,label_decode_time=0,value_decode_time=0;
        if(point_set_config_vec.empty())
            throw runtime_error("尚未配置点集");
        if(eb_schedule!="none" && eb_schedule!="round" && eb_schedule!="point" && eb_schedule!="inverse_rank_log2" && eb_schedule!="round_point" && eb_schedule!="round_ratio" && eb_schedule!="round_point_ratio" && eb_schedule!="round_point_ratio_dag" && eb_schedule!="round_point_ratio_dag_direct" && eb_schedule!="round_point_ratio_dag_variance_direct")
            throw runtime_error("未知的EB序列模式："+eb_schedule);
        if (!(eb_schedule_start_ratio>0) || eb_schedule_start_ratio>1)
            throw runtime_error("EB序列起始倍率必须在(0,1]内");
        vector<unsigned char> &container=current_zstd_bs_vec;
        const unsigned char *cursor=container.data(),*end=container.data()+container.size();
        const unsigned char magic_v7[8]={'U','N','S','Z','P','S','0','7'};
        const unsigned char magic_v8[8]={'U','N','S','Z','P','S','0','8'};
        const unsigned char magic_v9[8]={'U','N','S','Z','P','S','0','9'};
        if ((uint64_t)(end-cursor)<8)
            throw runtime_error("不是受支持的UNSZ点集文件");
        bool use_legacy_point_set_format=memcmp(cursor,magic_v7,8)==0;
        bool use_partition_stream_format=memcmp(cursor,magic_v9,8)==0;
        if (!use_legacy_point_set_format && !use_partition_stream_format && memcmp(cursor,magic_v8,8)!=0)
            throw runtime_error("不是受支持的UNSZ点集文件");
        cursor+=8;
        uint32_t version=read_binary<uint32_t>(cursor,end);
        if ((use_legacy_point_set_format && version!=7) || (use_partition_stream_format && version!=9) || (!use_legacy_point_set_format && !use_partition_stream_format && version!=8))
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
        if(!two_point_sets&&point_set_num!=1)
            throw runtime_error("多点集码流不能使用单点集field配置解压");
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
        vector<vector<uint8_t>> label_symbol_vec_vec(label_channel_num);
        vector<uint64_t> label_channel_target_num_vec(label_channel_num,0);
        point_set_point_num_vec.resize(point_set_num);
        point_set_min_data_vec.resize(point_set_num);
        point_set_max_data_vec.resize(point_set_num);
        point_set_abs_eb_vec.resize(point_set_num);
        point_set_codec_vec.resize(point_set_num);
        vector<vector<int32_t>> point_set_qb_vec(point_set_num);
        vector<vector<vector<uint8_t>>> partition_label_symbol_vec_vec;
        vector<vector<vector<int32_t>>> partition_point_set_qb_vec;
        if(use_partition_stream_format)
        {
            if(partition_pred_ref_ids_vec.empty())
                throw runtime_error("UNSZ v9 partition码流需要使用相同的partition参数");
            for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
            {
                uint8_t codec=read_binary<uint8_t>(cursor,end);
                if(codec!=(uint8_t)point_set_codec_e::idw&&codec!=(uint8_t)point_set_codec_e::constant&&codec!=(uint8_t)point_set_codec_e::rbf)
                    throw runtime_error("压缩文件包含未知的点集编码方式");
                point_set_codec_vec[point_set_id]=(point_set_codec_e)codec;
                point_set_point_num_vec[point_set_id]=read_binary<uint32_t>(cursor,end);
                point_set_min_data_vec[point_set_id]=read_binary<T_data>(cursor,end);
                point_set_max_data_vec[point_set_id]=read_binary<T_data>(cursor,end);
            }
            uint32_t file_partition_num=read_binary<uint32_t>(cursor,end);
            if(file_partition_num!=partition_pred_ref_ids_vec.size())
                throw runtime_error("压缩文件的partition数量与当前预测DAG不一致");
            vector<uint64_t> partition_payload_begin_vec(file_partition_num+1);
            for(uint32_t partition_id=0;partition_id<=file_partition_num;partition_id++)
                partition_payload_begin_vec[partition_id]=read_binary<uint64_t>(cursor,end);
            if(partition_payload_begin_vec[0]!=0 || partition_payload_begin_vec.back()!=(uint64_t)(end-cursor))
                throw runtime_error("partition码流偏移表与文件长度不一致");
            for(uint32_t partition_id=0;partition_id<file_partition_num;partition_id++)
                if(partition_payload_begin_vec[partition_id]>partition_payload_begin_vec[partition_id+1])
                    throw runtime_error("partition码流偏移表不是递增的");
            partition_label_symbol_vec_vec.resize(file_partition_num,vector<vector<uint8_t>>(label_channel_num));
            partition_point_set_qb_vec.resize(file_partition_num,vector<vector<int32_t>>(point_set_num));
            const unsigned char *partition_payload_base=cursor;
            exception_ptr lossless_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
            for(int64_t partition_index=0;partition_index<(int64_t)file_partition_num;partition_index++)
            {
                try
                {
                    uint32_t partition_id=(uint32_t)partition_index;
                    const unsigned char *partition_cursor=partition_payload_base+partition_payload_begin_vec[partition_id];
                    const unsigned char *partition_end=partition_payload_base+partition_payload_begin_vec[partition_id+1];
                    for(uint32_t label_channel_id=0;label_channel_id<label_channel_num;label_channel_id++)
                    {
                        uint64_t target_num=read_binary<uint64_t>(partition_cursor,partition_end);
                        uint64_t zstd_byte_num=read_binary<uint64_t>(partition_cursor,partition_end);
                        if(target_num>partition_pred_ref_ids_vec[partition_id].size() || (uint64_t)(partition_end-partition_cursor)<zstd_byte_num)
                            throw runtime_error("partition标签流不完整");
                        if(target_num==0)
                        {
                            if(zstd_byte_num!=0)
                                throw runtime_error("空partition标签流具有非空Zstd字节流");
                        }
                        else
                        {
                            vector<unsigned char> delta_bytes=zstd_decompress(partition_cursor,zstd_byte_num);
                            partition_label_symbol_vec_vec[partition_id][label_channel_id]=decode_binary_one_position_delta_vec(delta_bytes,target_num);
                        }
                        partition_cursor+=zstd_byte_num;
                    }
                    for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                    {
                        uint64_t qb_num=read_binary<uint64_t>(partition_cursor,partition_end);
                        uint64_t zstd_byte_num=read_binary<uint64_t>(partition_cursor,partition_end);
                        if(qb_num>partition_pred_ref_ids_vec[partition_id].size() || (uint64_t)(partition_end-partition_cursor)<zstd_byte_num)
                            throw runtime_error("partition数值流不完整");
                        if(qb_num==0)
                        {
                            if(zstd_byte_num!=0)
                                throw runtime_error("空partition数值流具有非空Zstd字节流");
                        }
                        else
                        {
                            vector<unsigned char> huffman_bytes=zstd_decompress(partition_cursor,zstd_byte_num);
                            partition_point_set_qb_vec[partition_id][point_set_id]=huffman_decode<int32_t>(huffman_bytes.data(),huffman_bytes.size(),qb_num);
                        }
                        partition_cursor+=zstd_byte_num;
                    }
                    if(partition_cursor!=partition_end)
                        throw runtime_error("partition码流末尾存在未解析数据");
                }
                catch(...)
                {
#pragma omp critical
                    if(!lossless_exception)
                        lossless_exception=current_exception();
                }
            }
            if(lossless_exception)
                rethrow_exception(lossless_exception);
            for(uint32_t partition_id=0;partition_id<file_partition_num;partition_id++)
                for(uint32_t label_channel_id=0;label_channel_id<label_channel_num;label_channel_id++)
                    label_channel_target_num_vec[label_channel_id]+=partition_label_symbol_vec_vec[partition_id][label_channel_id].size();
            cursor=end;
        }
        else if (use_legacy_point_set_format)
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
                vector<unsigned char> label_position_delta_bytes=zstd_decompress(cursor,label_zstd_byte_num);
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
                vector<unsigned char> label_position_delta_bytes;
                if(label_zstd_byte_num>0)
                    label_position_delta_bytes=zstd_decompress(cursor,label_zstd_byte_num);
                label_symbol_vec_vec[label_channel_id]=decode_binary_one_position_delta_vec(label_position_delta_bytes,target_num);
                label_channel_target_num_vec[label_channel_id]=target_num;
                cursor+=label_zstd_byte_num;
            }
        }

        if(!use_partition_stream_format)
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            uint8_t codec=read_binary<uint8_t>(cursor,end);
            if(codec!=(uint8_t)point_set_codec_e::idw&&codec!=(uint8_t)point_set_codec_e::constant&&codec!=(uint8_t)point_set_codec_e::rbf)
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
                vector<unsigned char> qb_huffman_bytes=zstd_decompress(cursor,qb_zstd_byte_num);
                point_set_qb_vec[point_set_id]=huffman_decode<int32_t>(qb_huffman_bytes.data(),qb_huffman_bytes.size(),qb_num);
            }
            cursor+=qb_zstd_byte_num;
            if (point_set_codec_vec[point_set_id]==point_set_codec_e::constant && qb_num!=0)
                throw runtime_error("常量点集不应包含量化索引");
            const point_set_config_c &config=point_set_config_vec[point_set_id];
            if (point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                point_set_abs_eb_vec[point_set_id]=0;
            else if (config.error_mode==point_set_error_mode_e::absolute)
                point_set_abs_eb_vec[point_set_id]=(T_data)config.eb;
            else
                point_set_abs_eb_vec[point_set_id]=(T_data)(config.eb*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]));
            if (point_set_codec_vec[point_set_id]!=point_set_codec_e::constant && !(point_set_abs_eb_vec[point_set_id]>0))
                throw runtime_error("点集"+to_string(point_set_id)+"的绝对误差上限必须大于0");
        }
        if(use_partition_stream_format)
            for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
            {
                const point_set_config_c &config=point_set_config_vec[point_set_id];
                if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                    point_set_abs_eb_vec[point_set_id]=0;
                else if(config.error_mode==point_set_error_mode_e::absolute)
                    point_set_abs_eb_vec[point_set_id]=(T_data)config.eb;
                else
                    point_set_abs_eb_vec[point_set_id]=(T_data)(config.eb*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]));
                if(point_set_codec_vec[point_set_id]!=point_set_codec_e::constant && !(point_set_abs_eb_vec[point_set_id]>0))
                    throw runtime_error("点集"+to_string(point_set_id)+"的绝对误差上限必须大于0");
            }
        if (cursor!=end)
            throw runtime_error("压缩文件末尾存在未解析数据");
        stream_decode_time=chrono::duration<double>(chrono::steady_clock::now()-decompression_stage_start_time).count();
        decompression_stage_start_time=chrono::steady_clock::now();

        prediction_framework_c prediction_framework;
        prediction_framework.has_label=two_point_sets;
        prediction_framework.value_mode=fixed_value_mode;
        prediction_framework.label_mode=fixed_label_mode;
        prediction_framework.execution=two_point_sets?fixed_execution:label_value_execution_e::no_label;
        current_prediction_framework=prediction_framework;
        constexpr time_dimension_mode_e value_mode=fixed_value_mode;
        auto predict_configured_value=[&](uint32_t pred_ref_id,int pred_id,array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,uint32_t point_set_id,vector<T_data> &data)
        {
            static_assert(fixed_predictor==point_set_codec_e::idw||fixed_predictor==point_set_codec_e::rbf,"固定数值预测器必须是IDW或RBF");
            if constexpr(fixed_predictor==point_set_codec_e::idw)
            {
                if constexpr(two_point_sets)
                    return predict_value_two_point_sets_idw(ref_ids,ref_weights,data,point_set_id);
                else
                    return predict_value_one_point_set(pred_ref_id,pred_ref_prediction_weights,data);
            }
            else
            {
                if constexpr(two_point_sets)
                    return predict_value_two_point_sets_rbf(pred_ref_id,ref_ids,ref_weights,data,point_set_id);
                else
                    return predict_value_one_point_set(pred_ref_id,pred_ref_prediction_weights,data);
            }
        };
        auto predict_configured_label=[&](uint32_t pred_ref_id,const vector<uint8_t> &label_vec)
        {
            static_assert(fixed_predictor==point_set_codec_e::idw||fixed_predictor==point_set_codec_e::rbf,"固定标签预测器必须是IDW或RBF");
            if(pred_ref_id>=prediction_10_fast_begin)
                return predict_label_10(pred_ref_begin[pred_ref_id],pred_ref_prediction_weights,label_vec);
            return predict_label(pred_ref_id,pred_ref_prediction_weights,label_vec);
        };
        constexpr bool use_fused_two_point_set_prediction=two_point_sets&&fixed_execution==label_value_execution_e::fused;
        constexpr bool use_selected_fused_temporal_flow=use_fused_two_point_set_prediction&&((fixed_label_mode==time_dimension_mode_e::constant_extrapolation&&fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)||(fixed_label_mode==time_dimension_mode_e::endpoint_agreement&&fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual));
        bool use_linear_previous_label=use_linear_previous_label_experimental&&use_fused_two_point_set_prediction&&fixed_label_mode==time_dimension_mode_e::constant_extrapolation&&previous_time_point_set_id_vec.size()==point_num;
        bool use_original_order_constant_label_phase=use_selected_fused_temporal_flow&&fixed_label_mode==time_dimension_mode_e::constant_extrapolation&&value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual&&previous_time_point_set_id_vec.size()==point_num;
        if(use_fused_two_point_set_prediction)
        {
            if(point_set_num!=2)
                throw runtime_error("融合两点集预测要求压缩文件恰好包含两个点集");
            if(use_partition_stream_format || thread_num>1)
                throw runtime_error("融合两点集预测目前只支持单线程非partition路径");
            if(point_set_predictor!="idw"&&point_set_predictor!="rbf")
                throw runtime_error("融合两点集解压当前流程不支持所选标签预测器");
            if(use_partition_adaptive_predictor_experimental)
                throw runtime_error("融合两点集预测目前不能和partition自适应预测器同时使用");
            if(use_selected_fused_temporal_flow&&use_linear_previous_label_experimental)
                throw runtime_error("选定的融合时间流程不使用线性上一时刻标签修正流");
            if(!use_selected_fused_temporal_flow&&value_mode!=time_dimension_mode_e::spatial)
                throw runtime_error("当前融合两点集流程不支持所选数值预测方式");
            for(point_set_codec_e codec:point_set_codec_vec)
                if(codec!=point_set_codec_e::constant&&codec!=point_set_codec_e::idw&&codec!=point_set_codec_e::rbf)
                    throw runtime_error("融合两点集预测只支持constant、IDW或RBF预测器");
        }
        bool data_dependent_eb_schedule=eb_schedule=="round_point_ratio_dag" || eb_schedule=="round_point_ratio_dag_direct" || eb_schedule=="round_point_ratio_dag_variance_direct";
        if(data_dependent_eb_schedule || cached_pred_eb_multiplier_vec.size()!=point_num)
            cached_pred_eb_multiplier_vec=generate_pred_eb_multiplier_vec();
        const vector<double> &pred_eb_multiplier_vec=cached_pred_eb_multiplier_vec;
        vector<uint64_t> qb_index(point_set_num,0);
        if(use_fused_two_point_set_prediction)
            current_decompressed_data.assign(point_num,0);

        if(use_staged_point_set_ids)
        {
            auto point_set_id_iterator=staged_point_set_id_map.find(current_time_step);
            if(point_set_id_iterator==staged_point_set_id_map.end())
                throw runtime_error("staged解压缺少time step "+to_string(current_time_step)+"的标签");
            point_set_id_vec=point_set_id_iterator->second;
            if(point_set_id_vec.size()!=point_num)
                throw runtime_error("staged解压的标签点数不正确");
        }
        else
        {
            point_set_id_vec.assign(point_num,point_set_num-1);
            if(label_channel_num>0)
            {
                if(point_set_predictor!="idw"&&point_set_predictor!="rbf")
                    throw runtime_error("未知的点集标签预测器："+point_set_predictor);
                prepare_linear_interpolation();
                bool use_previous_time_label=fixed_label_mode==time_dimension_mode_e::constant_extrapolation&&previous_time_point_set_id_vec.size()==point_num;
                bool use_causal_time_label=fixed_label_mode==time_dimension_mode_e::constant_extrapolation&&previous_time_point_set_id_vec.size()==point_num;
                constexpr bool try_time_label=fixed_label_mode==time_dimension_mode_e::constant_extrapolation||fixed_label_mode==time_dimension_mode_e::endpoint_agreement;
                bool use_direct_fused_label_stream=use_fused_two_point_set_prediction;
                for (uint32_t label_channel_id=0;label_channel_id<label_channel_num;label_channel_id++)
                {
                vector<uint8_t> label_symbol_internal_vec;
                uint64_t label_symbol_id=0;
                if(use_original_order_constant_label_phase||use_linear_previous_label)
                {
                    if(label_channel_id!=0 || label_channel_target_num_vec[0]!=point_num || label_symbol_vec_vec[0].size()!=point_num)
                        throw runtime_error("原始顺序上一时刻标签修正流的符号数量不正确");
                    for(uint32_t original_id=0;original_id<point_num;original_id++)
                    {
                        uint32_t internal_id=original_id;
                        uint32_t predicted_label=previous_time_point_set_id_vec[original_id];
                        point_set_id_vec[internal_id]=label_symbol_vec_vec[0][original_id]?1-predicted_label:predicted_label;
                    }
                    continue;
                }
                if(use_causal_time_label&&!use_fused_two_point_set_prediction)
                {
                    uint64_t decoded_target_num=0;
                    if(use_partition_stream_format)
                    {
                        for(uint32_t partition_id=0;partition_id<partition_pred_ref_ids_vec.size();partition_id++)
                        {
                            uint64_t local_symbol_id=0;
                            for(uint32_t pred_ref_id:partition_pred_ref_ids_vec[partition_id])
                            {
                                int pred_id=pred_target_ids[pred_ref_id];
                                if(point_set_id_vec[pred_id]<label_channel_id)
                                    continue;
                                if(local_symbol_id>=partition_label_symbol_vec_vec[partition_id][label_channel_id].size())
                                    throw runtime_error("partition跨时间标签流的修正值不足");
                                uint32_t symbol=partition_label_symbol_vec_vec[partition_id][label_channel_id][local_symbol_id++];
                                if(symbol>1)
                                    throw runtime_error("点集标签修正值不是0或1");
                                uint32_t original_id=pred_id;
                                uint32_t predicted_label;
                                if(!predict_time_label_fixed<fixed_label_mode>(original_id,label_channel_id,predicted_label))
                                    throw runtime_error("因果时间标签预测缺少历史标签");
                                uint32_t actual_label=symbol==0?predicted_label:1-predicted_label;
                                if(actual_label==0)
                                    point_set_id_vec[pred_id]=label_channel_id;
                                decoded_target_num++;
                            }
                            if(local_symbol_id!=partition_label_symbol_vec_vec[partition_id][label_channel_id].size())
                                throw runtime_error("partition跨时间标签流存在未使用的修正值");
                        }
                    }
                    else
                    {
                        for(uint32_t original_id=0;original_id<point_num;original_id++)
                        {
                            uint32_t internal_id=original_id;
                            if(point_set_id_vec[internal_id]<label_channel_id)
                                continue;
                            if(label_symbol_id>=label_symbol_vec_vec[label_channel_id].size())
                                throw runtime_error("跨时间点集标签流的修正值不足");
                            uint32_t symbol=label_symbol_vec_vec[label_channel_id][label_symbol_id++];
                            if(symbol>1)
                                throw runtime_error("点集标签修正值不是0或1");
                            uint32_t predicted_label;
                            if(!predict_time_label_fixed<fixed_label_mode>(original_id,label_channel_id,predicted_label))
                                throw runtime_error("因果时间标签预测缺少历史标签");
                            uint32_t actual_label=symbol==0?predicted_label:1-predicted_label;
                            if(actual_label==0)
                                point_set_id_vec[internal_id]=label_channel_id;
                            decoded_target_num++;
                        }
                    }
                    if(decoded_target_num!=label_channel_target_num_vec[label_channel_id]||(!use_partition_stream_format&&label_symbol_id!=label_symbol_vec_vec[label_channel_id].size()))
                        throw runtime_error("跨时间点集标签通道"+to_string(label_channel_id)+"的修正值数量不一致");
                    continue;
                }
                if(!use_partition_stream_format)
                {
                    if(use_direct_fused_label_stream)
                        label_symbol_id=label_symbol_vec_vec[label_channel_id].size();
                    else
                    {
                        label_symbol_internal_vec.assign(point_num,0);
                        for (uint32_t original_id=0;original_id<point_num;original_id++)
                        {
                            uint32_t internal_id=original_id;
                            if (point_set_id_vec[internal_id]<label_channel_id)
                                continue;
                            if (label_symbol_id>=label_symbol_vec_vec[label_channel_id].size())
                                throw runtime_error("点集条件标签通道"+to_string(label_channel_id)+"的修正值不足");
                            label_symbol_internal_vec[internal_id]=label_symbol_vec_vec[label_channel_id][label_symbol_id++];
                        }
                    }
                }
                vector<uint8_t> binary_label_vec;
                if(!(use_fused_two_point_set_prediction&&!use_selected_fused_temporal_flow&&use_normalized_idw_weights))
                    binary_label_vec.assign(point_num,1);
                uint64_t decoded_target_num=0;
                if(use_partition_stream_format || thread_num>1)
                {
                    if(partition_pred_ref_ids_vec.empty())
                        throw runtime_error("多线程标签解码需要独立partition预测DAG");
                    vector<uint64_t> partition_decoded_target_num_vec(partition_pred_ref_ids_vec.size(),0);
                    exception_ptr label_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
                    for(int64_t partition_index=0;partition_index<(int64_t)partition_pred_ref_ids_vec.size();partition_index++)
                    {
                        try
                        {
                            uint32_t partition_id=(uint32_t)partition_index;
                            uint64_t local_decoded_target_num=0;
                            uint64_t local_symbol_id=0;
                            for(uint32_t pred_ref_id:partition_pred_ref_ids_vec[partition_id])
                            {
                                int pred_id=pred_target_ids[pred_ref_id];
                                if(point_set_id_vec[pred_id]<label_channel_id)
                                    continue;
                                uint32_t original_id=pred_id;
                                uint32_t predicted_label;
                                bool used_time_label=try_time_label&&predict_time_label_fixed<fixed_label_mode>(original_id,label_channel_id,predicted_label);
                                if(!used_time_label)
                                    predicted_label=predict_configured_label(pred_ref_id,binary_label_vec);
                                if(use_partition_stream_format && local_symbol_id>=partition_label_symbol_vec_vec[partition_id][label_channel_id].size())
                                    throw runtime_error("partition标签流的修正值不足");
                                uint32_t symbol;
                                if(use_partition_stream_format)
                                    symbol=partition_label_symbol_vec_vec[partition_id][label_channel_id][local_symbol_id++];
                                else
                                    symbol=label_symbol_internal_vec[pred_id];
                                if(symbol>1)
                                    throw runtime_error("点集标签修正值不是0或1");
                                uint32_t actual_label=symbol==0?predicted_label:1-predicted_label;
                                binary_label_vec[pred_id]=actual_label;
                                if(actual_label==0)
                                    point_set_id_vec[pred_id]=label_channel_id;
                                local_decoded_target_num++;
                            }
                            if(use_partition_stream_format && local_symbol_id!=partition_label_symbol_vec_vec[partition_id][label_channel_id].size())
                                throw runtime_error("partition标签流存在未使用的修正值");
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
                else if(use_selected_fused_temporal_flow)
                {
                    if(label_channel_id!=0)
                        throw runtime_error("融合两点集时间预测只能处理一个标签通道");
                    uint32_t pred_ref_id=0;
                    uint64_t ref_begin=pred_ref_begin[0];
                    for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
                        for(uint32_t round_target_id=0;round_target_id<round_target_ids[round_id].size();round_target_id++,pred_ref_id++)
                        {
                            int pred_id=pred_target_ids[pred_ref_id];
                            uint32_t original_id=pred_id;
                            uint64_t ref_end=pred_ref_begin[pred_ref_id+1];
                            uint32_t ref_num=(uint32_t)(ref_end-ref_begin);
                            array_view_c<int> ref_ids(ref_num?pred_ref_ids.data()+ref_begin:nullptr,ref_num);
                            auto ref_weights=get_pred_ref_weights(pred_ref_id);
                            ref_begin=ref_end;
                            uint32_t predicted_label;
                            bool used_time_label=try_time_label&&predict_time_label_fixed<fixed_label_mode>(original_id,0,predicted_label);
                            if(!used_time_label)
                                predicted_label=predict_configured_label(pred_ref_id,binary_label_vec);
                            uint32_t symbol;
                            if(use_direct_fused_label_stream)
                                symbol=label_symbol_vec_vec[0][pred_id];
                            else
                                symbol=label_symbol_internal_vec[pred_id];
                            if(symbol>1)
                                throw runtime_error("点集标签修正值不是0或1");
                            uint32_t point_set_id=symbol==0?predicted_label:1-predicted_label;
                            binary_label_vec[pred_id]=(uint8_t)point_set_id;
                            point_set_id_vec[pred_id]=(uint8_t)point_set_id;
                            if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                                current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                            else
                            {
                                if(qb_index[point_set_id]>=point_set_qb_vec[point_set_id].size())
                                    throw runtime_error("点集"+to_string(point_set_id)+"的量化索引不足");
                                T_data pred=0;
                                bool use_time_prediction=false;
                                if(value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                                {
                                    use_time_prediction=predict_time_spatial_residual_baseline_fixed<fixed_value_mode>(original_id,point_set_id,pred);
                                    if(use_time_prediction)
                                        pred+=predict_configured_value(pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_time_spatial_residual_data);
                                }
                                if(!use_time_prediction)
                                    pred=predict_configured_value(pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_decompressed_data);
                                int32_t qb=point_set_qb_vec[point_set_id][qb_index[point_set_id]++];
                                double point_abs_eb=point_set_abs_eb_vec[point_set_id]*pred_eb_multiplier_vec[pred_ref_id];
                                current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                            }
                            update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,point_set_id);
                        }
                    if(pred_ref_id!=pred_target_ids.size())
                        throw runtime_error("融合两点集时间预测没有处理全部目标");
                    decoded_target_num=pred_ref_id;
                }
                else if(use_fused_two_point_set_prediction)
                {
                    if(label_channel_id!=0)
                        throw runtime_error("融合两点集预测只能处理一个标签通道");
                    uint32_t pred_ref_id=0;
                    uint64_t ref_begin=pred_ref_begin[0];
                    for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
                        for(uint32_t round_target_id=0;round_target_id<round_target_ids[round_id].size();round_target_id++,pred_ref_id++)
                    {
                        int pred_id=pred_target_ids[pred_ref_id];
                        uint64_t ref_end=pred_ref_begin[pred_ref_id+1];
                        uint32_t ref_num=(uint32_t)(ref_end-ref_begin);
                        array_view_c<int> ref_ids(ref_num?pred_ref_ids.data()+ref_begin:nullptr,ref_num);
                        auto ref_weights=get_pred_ref_weights(pred_ref_id);
                        ref_begin=ref_end;
                        uint32_t predicted_label=0,uniform_ref_point_set_id=0;
                        bool uniform_ref_point_set=false;
                        if(use_previous_time_label)
                        {
                            uint32_t original_id=pred_id;
                            predicted_label=previous_time_point_set_id_vec[original_id];
                        }
                        else if(use_normalized_idw_weights)
                        {
                            uint32_t label_ref_num=min<uint32_t>(prediction_max_ref_num,ref_ids.size());
                            if(label_ref_num==4)
                            {
                                uint32_t point_set_id0=point_set_id_vec[ref_ids[0]],point_set_id1=point_set_id_vec[ref_ids[1]],point_set_id2=point_set_id_vec[ref_ids[2]],point_set_id3=point_set_id_vec[ref_ids[3]];
                                T_calc weight0=ref_weights[0],weight1=ref_weights[1],weight2=ref_weights[2],weight3=ref_weights[3];
                                T_calc label0_weight=0,label1_weight=0;
                                if(point_set_id0)
                                    label1_weight+=weight0;
                                else
                                    label0_weight+=weight0;
                                if(point_set_id1)
                                    label1_weight+=weight1;
                                else
                                    label0_weight+=weight1;
                                if(point_set_id2)
                                    label1_weight+=weight2;
                                else
                                    label0_weight+=weight2;
                                if(point_set_id3)
                                    label1_weight+=weight3;
                                else
                                    label0_weight+=weight3;
                                predicted_label=label1_weight>label0_weight;
                                uniform_ref_point_set=point_set_id0==point_set_id1 && point_set_id0==point_set_id2 && point_set_id0==point_set_id3;
                                uniform_ref_point_set_id=point_set_id0;
                            }
                            else
                            {
                                T_calc label_weight_sum[2]={0,0};
                                uniform_ref_point_set=label_ref_num>0;
                                if(uniform_ref_point_set)
                                    uniform_ref_point_set_id=point_set_id_vec[ref_ids[0]];
                                for(uint32_t ref_index=0;ref_index<label_ref_num;ref_index++)
                                {
                                    uint32_t ref_point_set_id=point_set_id_vec[ref_ids[ref_index]];
                                    label_weight_sum[ref_point_set_id]+=ref_weights[ref_index];
                                    uniform_ref_point_set&=ref_point_set_id==uniform_ref_point_set_id;
                                }
                                predicted_label=label_weight_sum[1]>label_weight_sum[0];
                            }
                        }
                        else
                            predicted_label=predict_configured_label(pred_ref_id,binary_label_vec);
                        uint32_t symbol;
                        if(use_direct_fused_label_stream)
                            symbol=label_symbol_vec_vec[0][pred_id];
                        else
                            symbol=label_symbol_internal_vec[pred_id];
                        uint32_t actual_label=symbol==0?predicted_label:1-predicted_label;
                        if(!use_normalized_idw_weights)
                            binary_label_vec[pred_id]=actual_label;
                        point_set_id_vec[pred_id]=actual_label;
                        uint32_t point_set_id=actual_label;
                        if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                            current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                        else
                        {
                            if(qb_index[point_set_id]>=point_set_qb_vec[point_set_id].size())
                                throw runtime_error("点集"+to_string(point_set_id)+"的量化索引不足");
                            int32_t qb=point_set_qb_vec[point_set_id][qb_index[point_set_id]++];
                            double point_abs_eb=point_set_abs_eb_vec[point_set_id]*pred_eb_multiplier_vec[pred_ref_id];
                            T_data predicted_value;
                            if(use_previous_time_label)
                            {
                                if constexpr(fixed_predictor==point_set_codec_e::rbf)
                                    predicted_value=predict_configured_value(pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_decompressed_data);
                                else
                                    predicted_value=predict_value_two_point_sets_idw(ref_ids,ref_weights,current_decompressed_data,point_set_id);
                            }
                            else
                            {
                                bool use_normalized_value=use_normalized_idw_weights&&uniform_ref_point_set&&uniform_ref_point_set_id==point_set_id;
                                if(use_normalized_value)
                                    predicted_value=predict_value_one_point_set(pred_ref_id,pred_ref_prediction_weights,current_decompressed_data);
                                else
                                    predicted_value=predict_configured_value(pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_decompressed_data);
                            }
                            current_decompressed_data[pred_id]=predicted_value+qb*2*point_abs_eb;
                        }
                    }
                    if(pred_ref_id!=pred_target_ids.size())
                        throw runtime_error("融合两点集预测没有处理全部目标");
                    decoded_target_num=pred_ref_id;
                }
                else
                {
                    for(uint32_t pred_ref_id=0;pred_ref_id<pred_target_ids.size();pred_ref_id++)
                    {
                        int pred_id=pred_target_ids[pred_ref_id];
                        if(point_set_id_vec[pred_id]<label_channel_id)
                            continue;
                        uint32_t original_id=pred_id;
                        uint32_t predicted_label;
                        bool used_time_label=try_time_label&&predict_time_label_fixed<fixed_label_mode>(original_id,label_channel_id,predicted_label);
                        if(!used_time_label)
                            predicted_label=predict_configured_label(pred_ref_id,binary_label_vec);
                        uint32_t symbol=label_symbol_internal_vec[pred_id];
                        if(symbol>1)
                            throw runtime_error("点集标签修正值不是0或1");
                        uint32_t actual_label=symbol==0?predicted_label:1-predicted_label;
                        binary_label_vec[pred_id]=actual_label;
                        if(actual_label==0)
                            point_set_id_vec[pred_id]=label_channel_id;
                        decoded_target_num++;
                    }
                }
                if (decoded_target_num!=label_channel_target_num_vec[label_channel_id] || label_symbol_id!=label_symbol_vec_vec[label_channel_id].size())
                    throw runtime_error("点集条件标签通道"+to_string(label_channel_id)+"的修正值数量不一致");
                }
            }
        }
        vector<uint32_t> decoded_point_set_num(point_set_num,0);
        for(uint32_t point_set_id:point_set_id_vec)
            decoded_point_set_num[point_set_id]++;
        if (decoded_point_set_num!=point_set_point_num_vec)
            throw runtime_error("解码后的点集数量与文件头不一致");
        label_decode_time=chrono::duration<double>(chrono::steady_clock::now()-decompression_stage_start_time).count();
        if(decode_labels_only)
            return;
        decompression_stage_start_time=chrono::steady_clock::now();

        bool use_single_point_set_rbf_fast_path=!two_point_sets&&point_set_codec_vec[0]==point_set_codec_e::rbf&&value_mode==time_dimension_mode_e::spatial&&!use_partition_stream_format&&thread_num==1&&!use_partition_adaptive_predictor_experimental&&!use_linear_previous_label&&pred_ref_prediction_weights.size()==pred_ref_ids.size();
        constexpr bool supports_two_point_set_constant_residual_rbf_fast_path=two_point_sets&&fixed_execution==label_value_execution_e::fused&&fixed_label_mode==time_dimension_mode_e::constant_extrapolation&&fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual&&fixed_predictor==point_set_codec_e::rbf;
        bool use_two_point_set_constant_residual_rbf_fast_path=supports_two_point_set_constant_residual_rbf_fast_path&&use_original_order_constant_label_phase&&point_set_codec_vec.size()==2&&point_set_codec_vec[0]==point_set_codec_e::rbf&&point_set_codec_vec[1]==point_set_codec_e::constant&&previous_time_decompressed_data.size()==point_num&&current_time_spatial_residual_data.size()==point_num&&!use_partition_stream_format&&thread_num==1&&!use_partition_adaptive_predictor_experimental&&!use_linear_previous_label&&pred_ref_prediction_weights.size()==pred_ref_ids.size();
        bool use_multihead_execution=!multihead_execution_pred_ref_ids.empty();
        if(use_multihead_execution&&!use_single_point_set_rbf_fast_path)
            throw runtime_error("多头执行要求单点集纯空间RBF快速路径");
        if(!use_fused_two_point_set_prediction||use_original_order_constant_label_phase||use_linear_previous_label)
        {
            if(use_single_point_set_rbf_fast_path)
                current_decompressed_data.resize(point_num);
            else
                current_decompressed_data.assign(point_num,0);
        }
        constexpr bool use_normalized_value_weights=!two_point_sets&&use_normalized_idw_weights;
        bool has_previous_time=value_mode==time_dimension_mode_e::constant_extrapolation&&previous_time_decompressed_data.size()==point_num&&previous_time_point_set_id_vec.size()==point_num;
        prepare_linear_interpolation();
        uint32_t pred_ref_index=use_fused_two_point_set_prediction&&!use_original_order_constant_label_phase&&!use_linear_previous_label?point_num:0;
        if(!use_fused_two_point_set_prediction||use_original_order_constant_label_phase||use_linear_previous_label)
        {
        if(use_two_point_set_constant_residual_rbf_fast_path)
        {
            const vector<int32_t> &qb_vec=point_set_qb_vec[0];
            uint64_t ref_begin=pred_ref_begin[0],qb_id=0;
            for(uint32_t current_pred_ref_id=0;current_pred_ref_id<point_num;current_pred_ref_id++)
            {
                int pred_id=pred_target_ids[current_pred_ref_id];
                uint64_t ref_end=pred_ref_begin[current_pred_ref_id+1];
                if(point_set_id_vec[pred_id]==1)
                    current_decompressed_data[pred_id]=point_set_min_data_vec[1];
                else
                {
                    if(qb_id>=qb_vec.size())
                        throw runtime_error("点集0的量化索引不足");
                    uint32_t ref_num=(uint32_t)(ref_end-ref_begin);
                    array_view_c<int> ref_ids(ref_num?pred_ref_ids.data()+ref_begin:nullptr,ref_num);
                    auto ref_weights=get_pred_ref_weights(current_pred_ref_id);
                    bool use_time_prediction=previous_time_point_set_id_vec[pred_id]==0;
                    T_data time_baseline=use_time_prediction?previous_time_decompressed_data[pred_id]:0;
                    vector<T_data> &prediction_data=use_time_prediction?current_time_spatial_residual_data:current_decompressed_data;
                    T_data pred=time_baseline+predict_configured_value(current_pred_ref_id,pred_id,ref_ids,ref_weights,0,prediction_data);
                    double point_abs_eb=point_set_abs_eb_vec[0]*pred_eb_multiplier_vec[current_pred_ref_id];
                    current_decompressed_data[pred_id]=pred+qb_vec[qb_id++]*2*point_abs_eb;
                    if(use_time_prediction)
                        current_time_spatial_residual_data[pred_id]=current_decompressed_data[pred_id]-time_baseline;
                }
                ref_begin=ref_end;
            }
            qb_index[0]=qb_id;
            pred_ref_index=point_num;
        }
        else if(use_single_point_set_rbf_fast_path)
        {
            const vector<int32_t> &qb_vec=point_set_qb_vec[0];
            if(qb_vec.size()!=point_num)
                throw runtime_error("单点集RBF快速路径的量化索引数量不等于点数");
            double base_abs_eb=point_set_abs_eb_vec[0];
            auto decode_prediction=[&](uint32_t pred_ref_id,T_data pred)
            {
                uint32_t pred_id=(uint32_t)pred_target_ids[pred_ref_id];
                uint32_t qb_id=use_multihead_execution?multihead_qb_index_by_pred_ref_id[pred_ref_id]:pred_ref_id;
                double point_abs_eb=base_abs_eb*pred_eb_multiplier_vec[pred_ref_id];
                current_decompressed_data[pred_id]=pred+qb_vec[qb_id]*2*point_abs_eb;
            };
            if(use_multihead_execution)
            {
                for(uint32_t pred_ref_id:multihead_execution_pred_ref_ids)
                {
                    if(pred_ref_id<prediction_10_fast_begin)
                        decode_prediction(pred_ref_id,predict_value_one_point_set(pred_ref_id,pred_ref_prediction_weights,current_decompressed_data));
                    else
                        decode_prediction(pred_ref_id,predict_value_one_point_set_10(pred_ref_begin[pred_ref_id],pred_ref_prediction_weights,current_decompressed_data));
                }
            }
            else
            {
                for(uint32_t pred_ref_id=0;pred_ref_id<prediction_10_fast_begin;pred_ref_id++)
                    decode_prediction(pred_ref_id,predict_value_one_point_set(pred_ref_id,pred_ref_prediction_weights,current_decompressed_data));
                uint64_t ref_begin=pred_ref_begin[prediction_10_fast_begin];
                for(uint32_t pred_ref_id=prediction_10_fast_begin;pred_ref_id<point_num;pred_ref_id++,ref_begin+=10)
                    decode_prediction(pred_ref_id,predict_value_one_point_set_10(ref_begin,pred_ref_prediction_weights,current_decompressed_data));
            }
            qb_index[0]=point_num;
            pred_ref_index=point_num;
        }
        else if(use_partition_stream_format||thread_num>1)
        {
            if(partition_pred_ref_ids_vec.empty())
                throw runtime_error("多线程数值解码需要独立partition预测DAG");
            vector<int32_t> parallel_qb_vec(use_partition_stream_format?0:point_num);
            vector<uint8_t> parallel_has_qb_vec(use_partition_stream_format?0:point_num,0);
            for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
            {
                for(uint32_t i=0;i<round_target_ids[round_id].size();i++)
                {
                    uint32_t current_pred_ref_id=pred_ref_index++;
                    int pred_id=pred_target_ids[current_pred_ref_id];
                    uint32_t point_set_id=two_point_sets?point_set_id_vec[pred_id]:0;
                    if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                        continue;
                    if(use_partition_stream_format)
                        continue;
                    if(qb_index[point_set_id]>=point_set_qb_vec[point_set_id].size())
                        throw runtime_error("点集"+to_string(point_set_id)+"的量化索引不足");
                    parallel_qb_vec[current_pred_ref_id]=point_set_qb_vec[point_set_id][qb_index[point_set_id]++];
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
                    vector<uint64_t> local_qb_index_vec(point_set_num,0);
                    for(uint32_t current_pred_ref_id:partition_pred_ref_ids_vec[partition_id])
                    {
                        int pred_id=pred_target_ids[current_pred_ref_id];
                        uint32_t point_set_id=two_point_sets?point_set_id_vec[pred_id]:0;
                        if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                        {
                            current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                            uint32_t original_id=pred_id;
                            update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,point_set_id);
                            continue;
                        }
                        if(!use_partition_stream_format && !parallel_has_qb_vec[current_pred_ref_id])
                            throw runtime_error("多线程数值解码缺少量化索引");
                        auto ref_ids=get_pred_ref_ids(current_pred_ref_id);
                        auto ref_weights=get_pred_ref_weights(current_pred_ref_id);
                        bool use_partition_config=use_partition_adaptive_predictor_experimental && !partition_adaptive_sample_mask[pred_id];
                        uint32_t original_id=pred_id;
                        T_data pred=0;
                        bool use_cross_time_prediction=false;
                        if(has_previous_time && previous_time_point_set_id_vec[original_id]==point_set_id)
                        {
                            pred=previous_time_decompressed_data[original_id];
                            use_cross_time_prediction=true;
                        }
                        else if(value_mode==time_dimension_mode_e::linear_interpolation)
                            use_cross_time_prediction=predict_linear_interpolation(original_id,point_set_id,pred);
                        else if(value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual||value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
                        {
                            use_cross_time_prediction=predict_time_spatial_residual_baseline_fixed<fixed_value_mode>(original_id,point_set_id,pred);
                            if(use_cross_time_prediction)
                            {
                                if(use_partition_config)
                                    pred+=predict_partition_adaptive_value(pred_id,ref_ids,current_time_spatial_residual_data,partition_prediction_config_id_vec[partition_id]);
                                else if(use_normalized_value_weights)
                                    pred+=predict_value_one_point_set(current_pred_ref_id,pred_ref_prediction_weights,current_time_spatial_residual_data);
                                else
                                    pred+=predict_configured_value(current_pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_time_spatial_residual_data);
                            }
                        }
                        if(!use_cross_time_prediction)
                        {
                            if(use_partition_config)
                                pred=predict_partition_adaptive_value(pred_id,ref_ids,current_decompressed_data,partition_prediction_config_id_vec[partition_id]);
                            else if(use_normalized_value_weights)
                                pred=predict_value_one_point_set(current_pred_ref_id,pred_ref_prediction_weights,current_decompressed_data);
                            else
                                pred=predict_configured_value(current_pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_decompressed_data);
                        }
                        double point_abs_eb=point_set_abs_eb_vec[point_set_id]*pred_eb_multiplier_vec[current_pred_ref_id];
                        int32_t qb=0;
                        if(use_partition_stream_format)
                        {
                            if(local_qb_index_vec[point_set_id]>=partition_point_set_qb_vec[partition_id][point_set_id].size())
                                throw runtime_error("partition数值流的量化索引不足");
                            qb=partition_point_set_qb_vec[partition_id][point_set_id][local_qb_index_vec[point_set_id]++];
                        }
                        else
                            qb=parallel_qb_vec[current_pred_ref_id];
                        current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                        update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,point_set_id);
                    }
                    if(use_partition_stream_format)
                        for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                            if(local_qb_index_vec[point_set_id]!=partition_point_set_qb_vec[partition_id][point_set_id].size())
                                throw runtime_error("partition数值流存在未使用的量化索引");
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
        for (uint32_t round_id=0;round_id<prediction_round_num;round_id++)
        {
            for (uint32_t i=0;i<round_target_ids[round_id].size();i++)
            {
                uint32_t current_pred_ref_id=pred_ref_index++;
                auto ref_ids=get_pred_ref_ids(current_pred_ref_id);
                auto ref_weights=get_pred_ref_weights(current_pred_ref_id);
                int pred_id=pred_target_ids[current_pred_ref_id];
                uint32_t point_set_id=two_point_sets?point_set_id_vec[pred_id]:0;
                double eb_multiplier=pred_eb_multiplier_vec[current_pred_ref_id];
                if (point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                {
                    current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                    uint32_t original_id=pred_id;
                    update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,point_set_id);
                }
                else
                {
                    if(qb_index[point_set_id]>=point_set_qb_vec[point_set_id].size())
                        throw runtime_error("点集"+to_string(point_set_id)+"的量化索引不足");
                    bool use_partition_config=use_partition_adaptive_predictor_experimental && !partition_adaptive_sample_mask[pred_id];
                    uint32_t original_id=pred_id;
                    T_data pred=0;
                    bool use_cross_time_prediction=false;
                    if (has_previous_time && previous_time_point_set_id_vec[original_id]==point_set_id)
                    {
                        pred=previous_time_decompressed_data[original_id];
                        use_cross_time_prediction=true;
                    }
                    else if (value_mode==time_dimension_mode_e::linear_interpolation)
                        use_cross_time_prediction=predict_linear_interpolation(original_id,point_set_id,pred);
                    else if(value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual||value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
                    {
                        use_cross_time_prediction=predict_time_spatial_residual_baseline_fixed<fixed_value_mode>(original_id,point_set_id,pred);
                        if(use_cross_time_prediction)
                        {
                            if(use_partition_config)
                                pred+=predict_partition_adaptive_value(pred_id,ref_ids,current_time_spatial_residual_data,partition_prediction_config_id_vec[point_partition_id_vec[pred_id]]);
                            else if(use_normalized_value_weights)
                                pred+=predict_value_one_point_set(current_pred_ref_id,pred_ref_prediction_weights,current_time_spatial_residual_data);
                            else
                                pred+=predict_configured_value(current_pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_time_spatial_residual_data);
                        }
                    }
                    if (!use_cross_time_prediction)
                    {
                        if(use_linear_previous_label)
                        {
                            if(point_set_codec_vec[point_set_id]==point_set_codec_e::rbf)
                                pred=predict_configured_value(current_pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_decompressed_data);
                            else
                                pred=predict_value_two_point_sets_idw(ref_ids,ref_weights,current_decompressed_data,point_set_id);
                        }
                        else
                        {
                            if(use_partition_config)
                                pred=predict_partition_adaptive_value(pred_id,ref_ids,current_decompressed_data,partition_prediction_config_id_vec[point_partition_id_vec[pred_id]]);
                            else if(use_normalized_value_weights)
                                pred=predict_value_one_point_set(current_pred_ref_id,pred_ref_prediction_weights,current_decompressed_data);
                            else
                                pred=predict_configured_value(current_pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_decompressed_data);
                        }
                    }
                    int32_t qb=point_set_qb_vec[point_set_id][qb_index[point_set_id]++];
                    double point_abs_eb=point_set_abs_eb_vec[point_set_id]*eb_multiplier;
                    current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                    update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,point_set_id);
                }
            }
        }
        }
        }
        value_decode_time=chrono::duration<double>(chrono::steady_clock::now()-decompression_stage_start_time).count();
        if (pred_ref_index!=pred_target_ids.size())
            throw runtime_error("逐轮预测目标数量与预测参考结构不一致");
        if(!use_partition_stream_format)
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
            if(qb_index[point_set_id]!=point_set_qb_vec[point_set_id].size())
                throw runtime_error("点集"+to_string(point_set_id)+"存在未使用的量化索引");
        uint32_t primary_point_set_id=0;
        while (point_set_point_num_vec[primary_point_set_id]==0)
            primary_point_set_id++;
        current_min_data=point_set_min_data_vec[primary_point_set_id];
        current_max_data=point_set_max_data_vec[primary_point_set_id];
        current_abs_eb=point_set_abs_eb_vec[primary_point_set_id];
        if(print_compression_stage_time)
            printf("decompression stage time: stream_decode=%f s, label_decode=%f s, value_decode=%f s\n",stream_decode_time,label_decode_time,value_decode_time);
    }
}
