#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::dispatch_single_point_set_decompression()
    {
        one_point_set::decompress(*this);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::dispatch_two_point_sets_decompression_by_mode()
    {
        two_point_sets::fused::decompress(*this);
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::decompress_current_file()
    {
        decompression_file_state_c<T_data> state;
        auto decompression_stage_start_time=chrono::steady_clock::now();
        if(point_set_config_vec.empty())
            throw runtime_error("尚未配置点集");
        if(eb_schedule!="none"&&eb_schedule!="round"&&eb_schedule!="point"&&eb_schedule!="inverse_rank_log2"&&eb_schedule!="round_point"&&eb_schedule!="round_ratio"&&eb_schedule!="round_point_ratio")
            throw runtime_error("未知的EB序列模式："+eb_schedule);
        if(!(eb_schedule_start_ratio>0)||eb_schedule_start_ratio>1)
            throw runtime_error("EB序列起始倍率必须在(0,1]内");
        decode_current_compression_streams<two_point_sets>(state);
        uint32_t point_set_num=state.point_set_num;
        for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            const point_set_config_c &config=point_set_config_vec[point_set_id];
            if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                point_set_abs_eb_vec[point_set_id]=0;
            else if(config.error_mode==point_set_error_mode_e::absolute)
                point_set_abs_eb_vec[point_set_id]=(T_data)config.eb;
            else
                point_set_abs_eb_vec[point_set_id]=(T_data)(config.eb*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]));
            if(point_set_codec_vec[point_set_id]!=point_set_codec_e::constant&&!(point_set_abs_eb_vec[point_set_id]>0))
                throw runtime_error("点集"+to_string(point_set_id)+"的绝对误差上限必须大于0");
        }
        state.stream_decode_time=chrono::duration<double>(chrono::steady_clock::now()-decompression_stage_start_time).count();

        prediction_framework_c prediction_framework;
        prediction_framework.has_label=two_point_sets;
        prediction_framework.value_mode=fixed_value_mode;
        prediction_framework.label_mode=fixed_label_mode;
        prediction_framework.execution=two_point_sets?fixed_execution:label_value_execution_e::no_label;
        current_prediction_framework=prediction_framework;
        for(const prediction_structure_c<T_calc>&structure:prediction_structure_vec)
            if(structure.eb_multiplier.size()!=structure.target_ids.size())
                throw runtime_error("预处理没有生成完整的EB倍率");

        prepare_temporal_prediction();
        if constexpr(two_point_sets)
            point_set_id_vec.assign(point_num,point_set_num-1);
        else
        {
            point_set_id_vec.clear();
        }
        if constexpr(two_point_sets&&fixed_execution==label_value_execution_e::fused)
        {
            decompress_labels_and_values<two_point_sets,fixed_value_mode,fixed_label_mode,fixed_execution,fixed_predictor>(state);
            vector<uint32_t> decoded_point_set_num(point_set_num,0);
            for(uint32_t point_set_id:point_set_id_vec)
                decoded_point_set_num[point_set_id]++;
            if(decoded_point_set_num!=point_set_point_num_vec)
                throw runtime_error("解码后的点集数量与文件头不一致");
            if(print_compression_stage_time)
                printf("decompression stage time: stream_decode=%f s, label_decode=%f s, value_decode=%f s\n",state.stream_decode_time,state.label_decode_time,state.value_decode_time);
            return;
        }
        else
        {
            decompress_values<two_point_sets,fixed_value_mode,fixed_label_mode,fixed_execution,fixed_predictor>(state);
            vector<uint32_t> decoded_point_set_num(point_set_num,0);
            if constexpr(two_point_sets)
                for(uint32_t point_set_id:point_set_id_vec)
                    decoded_point_set_num[point_set_id]++;
            else
                decoded_point_set_num[0]=point_num;
            if(decoded_point_set_num!=point_set_point_num_vec)
                throw runtime_error("解码后的点集数量与文件头不一致");
            if(print_compression_stage_time)
                printf("decompression stage time: stream_decode=%f s, label_decode=%f s, value_decode=%f s\n",state.stream_decode_time,state.label_decode_time,state.value_decode_time);
            return;
        }
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::decompress_labels_and_values(decompression_file_state_c<T_data>&state)
    {
        if(state.point_set_num!=2||!state.has_label)
            throw runtime_error("融合解压要求恰好两个点集和一个标签通道");
        constexpr bool use_temporal_flow=fixed_label_mode!=time_dimension_mode_e::spatial||fixed_value_mode!=time_dimension_mode_e::spatial;
        constexpr bool try_time_label=fixed_label_mode==time_dimension_mode_e::constant_extrapolation||fixed_label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual;
        vector<uint8_t> binary_label_vec(point_num,1);
        current_decompressed_data.assign(point_num,0);
        auto stage_start_time=chrono::steady_clock::now();
        auto decompress_targets=[&](const prediction_structure_c<T_calc>&prediction_structure,const vector<uint8_t>&label_symbols,const vector<vector<int32_t>>&point_set_qbs,vector<uint64_t>&qb_indices,const unsigned char*&unpredictable_value_cursor,const unsigned char*unpredictable_value_end)
        {
            const vector<int>&target_ids=prediction_structure.target_ids;
            const vector<int>&ref_ids_vec=prediction_structure.reference_ids;
            const vector<uint64_t>&ref_begin_vec=prediction_structure.reference_offsets;
            const vector<T_calc>&weights=prediction_structure.prediction_weights;
            const vector<uint8_t>&rbf_fallback=prediction_structure.rbf_fallback_flags;
            const vector<double>&eb_multipliers=prediction_structure.eb_multiplier;
            if(ref_begin_vec.size()!=target_ids.size()+1||weights.size()!=ref_ids_vec.size()||eb_multipliers.size()!=target_ids.size())
                throw runtime_error("融合解压收到不完整的预测结构");
            if(label_symbols.size()!=target_ids.size())
                throw runtime_error("融合标签流的符号数量与目标数量不一致");
            if constexpr(fixed_predictor==point_set_codec_e::rbf)
                if(rbf_fallback.size()!=target_ids.size())
                    throw runtime_error("融合RBF解压收到不完整的fallback标记");
            for(uint32_t prediction_index=0;prediction_index<target_ids.size();prediction_index++)
            {
                int pred_id=target_ids[prediction_index];
                uint64_t ref_begin=ref_begin_vec[prediction_index],ref_end=ref_begin_vec[prediction_index+1];
                uint32_t ref_num=(uint32_t)(ref_end-ref_begin);
                array_view_c<int> ref_ids(ref_num?ref_ids_vec.data()+ref_begin:nullptr,ref_num);
                array_view_c<T_calc> ref_weights(ref_num?weights.data()+ref_begin:nullptr,ref_num);
                bool has_rbf_weights=true;
                if constexpr(fixed_predictor==point_set_codec_e::rbf)
                    has_rbf_weights=!rbf_fallback[prediction_index];
                auto predict_configured_label=[&]()
                {
                    if(ref_num==10)
                        return predict_label_10(ref_begin,ref_ids_vec,weights,binary_label_vec);
                    return predict_label(prediction_index,ref_ids_vec,ref_begin_vec,weights,binary_label_vec);
                };
                uint32_t predicted_label;
                if constexpr(use_temporal_flow)
                {
                    bool used_time_label=false;
                    if constexpr(try_time_label)
                        used_time_label=predict_temporal_label<fixed_label_mode>(pred_id,0,predicted_label);
                    if(!used_time_label)
                        predicted_label=predict_configured_label();
                }
                else if constexpr(fixed_predictor==point_set_codec_e::rbf)
                    predicted_label=predict_label_and_value_two_point_sets_rbf(has_rbf_weights,ref_ids,ref_weights,binary_label_vec,current_decompressed_data,0,false).first;
                else
                    predicted_label=predict_label_and_value_two_point_sets_idw(ref_ids,ref_weights,current_decompressed_data,0,false).first;
                uint32_t symbol=label_symbols[prediction_index];
                if(symbol>1)
                    throw runtime_error("点集标签修正值不是0或1");
                uint32_t point_set_id=symbol==0?predicted_label:1-predicted_label;
                binary_label_vec[pred_id]=(uint8_t)point_set_id;
                point_set_id_vec[pred_id]=(uint8_t)point_set_id;
                if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                {
                    current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                    update_temporal_residual<fixed_value_mode>(pred_id,pred_id,point_set_id);
                    continue;
                }
                if(qb_indices[point_set_id]>=point_set_qbs[point_set_id].size())
                    throw runtime_error("点集"+to_string(point_set_id)+"的量化索引不足");
                auto predict_configured_value=[&](vector<T_data>&data,const vector<uint8_t>*valid_ref_mask=nullptr)
                {
                    if constexpr(fixed_predictor==point_set_codec_e::idw)
                        return predict_value_two_point_sets_idw(ref_ids,ref_weights,data,point_set_id,valid_ref_mask);
                    return predict_value_two_point_sets_rbf(has_rbf_weights,ref_ids,ref_weights,data,point_set_id,valid_ref_mask);
                };
                T_data predicted_value=0;
                bool use_time_prediction=false;
                if constexpr(use_temporal_flow)
                {
                    if constexpr(fixed_value_mode==time_dimension_mode_e::constant_extrapolation)
                    {
                        use_time_prediction=previous_time_decompressed_data.size()==point_num&&previous_time_point_set_id_vec.size()==point_num&&previous_time_point_set_id_vec[pred_id]==point_set_id;
                        if(use_time_prediction)
                            predicted_value=previous_time_decompressed_data[pred_id];
                    }
                    else if constexpr(fixed_value_mode==time_dimension_mode_e::linear_interpolation)
                        use_time_prediction=predict_linear_interpolation(pred_id,point_set_id,predicted_value);
                    else if constexpr(fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                    {
                        use_time_prediction=predict_temporal_value_baseline<fixed_value_mode>(pred_id,point_set_id,predicted_value);
                        if(use_time_prediction)
                            predicted_value+=predict_configured_value(current_time_spatial_residual_data,&current_time_spatial_residual_valid);
                    }
                    if(!use_time_prediction)
                        predicted_value=predict_configured_value(current_decompressed_data);
                }
                else if constexpr(fixed_predictor==point_set_codec_e::rbf)
                    predicted_value=predict_label_and_value_two_point_sets_rbf(has_rbf_weights,ref_ids,ref_weights,binary_label_vec,current_decompressed_data,point_set_id,true).second;
                else
                    predicted_value=predict_label_and_value_two_point_sets_idw(ref_ids,ref_weights,current_decompressed_data,point_set_id,true).second;
                int32_t qb=point_set_qbs[point_set_id][qb_indices[point_set_id]++];
                double point_abs_eb=point_set_abs_eb_vec[point_set_id]*eb_multipliers[prediction_index];
                if(qb==numeric_limits<int32_t>::min())
                    current_decompressed_data[pred_id]=read_binary<T_data>(unpredictable_value_cursor,unpredictable_value_end);
                else
                    current_decompressed_data[pred_id]=predicted_value+qb*2*point_abs_eb;
                update_temporal_residual<fixed_value_mode>(pred_id,pred_id,point_set_id);
            }
            for(uint32_t point_set_id=0;point_set_id<state.point_set_num;point_set_id++)
                if(qb_indices[point_set_id]!=point_set_qbs[point_set_id].size())
                    throw runtime_error("点集"+to_string(point_set_id)+"存在未使用的量化索引");
            if(unpredictable_value_cursor!=unpredictable_value_end)
                throw runtime_error("融合数值流存在未使用的不可预测值");
        };
        vector<vector<uint8_t>> structure_label_symbols(prediction_structure_vec.size());
        vector<vector<vector<int32_t>>> structure_point_set_qbs(prediction_structure_vec.size());
        for(uint32_t structure_id=0;structure_id<prediction_structure_vec.size();structure_id++)
        {
            structure_label_symbols[structure_id]=move(state.structure_label_symbols[structure_id]);
            structure_point_set_qbs[structure_id]=move(state.structure_point_set_qb_vec[structure_id]);
        }
        vector<vector<uint64_t>> structure_qb_indices(prediction_structure_vec.size(),vector<uint64_t>(state.point_set_num,0));
        vector<const unsigned char*> structure_unpredictable_value_cursors=state.structure_unpredictable_value_begin_vec;
        exception_ptr decompression_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(prediction_structure_vec.size()>1&&thread_num>1)
        for(int64_t structure_index=0;structure_index<(int64_t)prediction_structure_vec.size();structure_index++)
        {
            try
            {
                decompress_targets(prediction_structure_vec[structure_index],structure_label_symbols[structure_index],structure_point_set_qbs[structure_index],structure_qb_indices[structure_index],structure_unpredictable_value_cursors[structure_index],state.structure_unpredictable_value_end_vec[structure_index]);
            }
            catch(...)
            {
#pragma omp critical
                if(!decompression_exception)
                    decompression_exception=current_exception();
            }
        }
        if(decompression_exception)
            rethrow_exception(decompression_exception);
        state.label_decode_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        state.value_decode_time=0;
    }


    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::decompress_values(decompression_file_state_c<T_data>&state)
    {
        uint32_t point_set_num=state.point_set_num;
        vector<vector<vector<int32_t>>> &structure_point_set_qb_vec=state.structure_point_set_qb_vec;
        vector<const unsigned char*>&structure_unpredictable_value_begin_vec=state.structure_unpredictable_value_begin_vec,&structure_unpredictable_value_end_vec=state.structure_unpredictable_value_end_vec;
        constexpr bool supports_single_point_set_rbf_fast_path=!two_point_sets&&fixed_predictor==point_set_codec_e::rbf&&fixed_value_mode==time_dimension_mode_e::spatial;
        bool use_single_point_set_rbf_fast_path=supports_single_point_set_rbf_fast_path&&prediction_structure_vec.size()==1&&thread_num==1&&!use_partition_adaptive_predictor_experimental&&prediction_structure_vec.front().prediction_weights.size()==prediction_structure_vec.front().reference_ids.size();
        if(use_single_point_set_rbf_fast_path)
            current_decompressed_data.resize(point_num);
        else
            current_decompressed_data.assign(point_num,0);
        auto stage_start_time=chrono::steady_clock::now();
        auto decompress_targets=[&](const prediction_structure_c<T_calc>&prediction_structure,auto get_qb,auto get_unpredictable_value)
        {
            const vector<int>&target_ids=prediction_structure.target_ids;
            const vector<int>&reference_ids=prediction_structure.reference_ids;
            const vector<uint64_t>&reference_begin=prediction_structure.reference_offsets;
            const vector<T_calc>&prediction_weights=prediction_structure.prediction_weights;
            const vector<double>&pred_eb_multiplier_vec=prediction_structure.eb_multiplier;
            if(reference_begin.size()!=target_ids.size()+1||pred_eb_multiplier_vec.size()!=target_ids.size())
                throw runtime_error("数值解压收到不完整的预测结构");
            for(uint32_t prediction_index=0;prediction_index<target_ids.size();prediction_index++)
            {
                int pred_id=target_ids[prediction_index];
                uint32_t point_set_id=two_point_sets?point_set_id_vec[pred_id]:0;
                if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                {
                    current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                    update_temporal_residual<fixed_value_mode>(pred_id,pred_id,point_set_id);
                    continue;
                }
                uint64_t ref_begin=reference_begin[prediction_index];
                uint32_t ref_num=(uint32_t)(reference_begin[prediction_index+1]-ref_begin);
                array_view_c<int> ref_ids(ref_num?reference_ids.data()+ref_begin:nullptr,ref_num);
                array_view_c<T_calc> ref_weights(ref_num?prediction_weights.data()+ref_begin:nullptr,ref_num);
                bool use_adaptive_config=should_use_selected_adaptive_prediction(pred_id);
                auto predict_spatial_value=[&](vector<T_data>&data,const vector<uint8_t>*valid_ref_mask=nullptr)
                {
                    if(use_adaptive_config)
                        return predict_selected_adaptive_value(pred_id,ref_ids,data);
                    return predict_value_one_point_set(prediction_index,reference_ids,reference_begin,prediction_weights,data);
                };
                T_data pred=0;
                bool use_cross_time_prediction=false;
                if constexpr(fixed_value_mode==time_dimension_mode_e::constant_extrapolation)
                {
                    bool has_previous_time_data=previous_time_decompressed_data.size()==point_num;
                    if constexpr(two_point_sets)
                        has_previous_time_data=has_previous_time_data&&previous_time_point_set_id_vec.size()==point_num&&previous_time_point_set_id_vec[pred_id]==point_set_id;
                    if(has_previous_time_data)
                    {
                        pred=previous_time_decompressed_data[pred_id];
                        use_cross_time_prediction=true;
                    }
                }
                else if constexpr(fixed_value_mode==time_dimension_mode_e::linear_interpolation)
                    use_cross_time_prediction=predict_linear_interpolation(pred_id,point_set_id,pred);
                else if constexpr(fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
                {
                    use_cross_time_prediction=predict_temporal_value_baseline<fixed_value_mode>(pred_id,point_set_id,pred);
                    if(use_cross_time_prediction)
                        pred+=predict_spatial_value(current_time_spatial_residual_data,&current_time_spatial_residual_valid);
                }
                if(!use_cross_time_prediction)
                    pred=predict_spatial_value(current_decompressed_data);
                int32_t qb=get_qb(prediction_index,point_set_id);
                double point_abs_eb=point_set_abs_eb_vec[point_set_id]*pred_eb_multiplier_vec[prediction_index];
                if(qb==numeric_limits<int32_t>::min())
                {
                    T_data unpredictable_value=get_unpredictable_value(point_set_id);
                    current_decompressed_data[pred_id]=unpredictable_value;
                }
                else
                    current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                update_temporal_residual<fixed_value_mode>(pred_id,pred_id,point_set_id);
            }
        };
        if(use_single_point_set_rbf_fast_path)
        {
            const prediction_structure_c<T_calc>&structure=prediction_structure_vec.front();
            const vector<int32_t> &qb_vec=structure_point_set_qb_vec[0][0];
            if(qb_vec.size()!=point_num)
                throw runtime_error("单点集RBF快速路径的量化索引数量不等于点数");
            double base_abs_eb=point_set_abs_eb_vec[0];
            const unsigned char *unpredictable_value_cursor=structure_unpredictable_value_begin_vec[0],*unpredictable_value_end=structure_unpredictable_value_end_vec[0];
            auto decode_prediction=[&](uint32_t prediction_index,T_data pred)
            {
                uint32_t pred_id=(uint32_t)structure.target_ids[prediction_index];
                double point_abs_eb=base_abs_eb*structure.eb_multiplier[prediction_index];
                int32_t qb=qb_vec[prediction_index];
                if(qb==numeric_limits<int32_t>::min())
                    current_decompressed_data[pred_id]=read_binary<T_data>(unpredictable_value_cursor,unpredictable_value_end);
                else
                    current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
            };
            for(uint32_t prediction_index=0;prediction_index<point_num;prediction_index++)
                decode_prediction(prediction_index,predict_value_one_point_set(prediction_index,structure.reference_ids,structure.reference_offsets,structure.prediction_weights,current_decompressed_data));
            if(unpredictable_value_cursor!=unpredictable_value_end)
                throw runtime_error("单点集RBF快速路径存在未使用的不可预测值");
        }
        else
        {
            exception_ptr prediction_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(prediction_structure_vec.size()>1&&thread_num>1)
            for(int64_t structure_index=0;structure_index<(int64_t)prediction_structure_vec.size();structure_index++)
            {
                try
                {
                    vector<uint64_t> local_qb_index_vec(point_set_num,0);
                    const unsigned char *unpredictable_value_cursor=structure_unpredictable_value_begin_vec[structure_index],*unpredictable_value_end=structure_unpredictable_value_end_vec[structure_index];
                    auto get_qb=[&](uint32_t,uint32_t point_set_id)
                    {
                        const vector<int32_t>&qb_vec=structure_point_set_qb_vec[structure_index][point_set_id];
                        if(local_qb_index_vec[point_set_id]>=qb_vec.size())
                            throw runtime_error("数值流的量化索引不足");
                        return qb_vec[local_qb_index_vec[point_set_id]++];
                    };
                    auto get_unpredictable_value=[&](uint32_t)
                    {
                        return read_binary<T_data>(unpredictable_value_cursor,unpredictable_value_end);
                    };
                    decompress_targets(prediction_structure_vec[structure_index],get_qb,get_unpredictable_value);
                    for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                    {
                        const vector<int32_t>&qb_vec=structure_point_set_qb_vec[structure_index][point_set_id];
                        if(local_qb_index_vec[point_set_id]!=qb_vec.size())
                            throw runtime_error("数值流存在未使用的量化索引");
                    }
                    if(unpredictable_value_cursor!=unpredictable_value_end)
                        throw runtime_error("数值流存在未使用的不可预测值");
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
        state.value_decode_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
    }
}
