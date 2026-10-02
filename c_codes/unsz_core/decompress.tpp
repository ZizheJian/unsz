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
    void param_c<T_coord,T_data>::validate_decoded_point_set_counts() const
    {
        if(point_set_point_num_vec.size()==1)
        {
            if(point_set_point_num_vec[0]!=point_num)
                throw runtime_error("解码后的点集数量与文件头不一致");
            return;
        }
        if(point_set_point_num_vec.size()!=2||point_set_id_vec.size()!=point_num)
            throw runtime_error("解码后的点集数量与文件头不一致");
        array<uint32_t,2> decoded_point_set_num{{0,0}};
        for(uint32_t point_set_id:point_set_id_vec)
            decoded_point_set_num[point_set_id]++;
        if(decoded_point_set_num[0]!=point_set_point_num_vec[0]||decoded_point_set_num[1]!=point_set_point_num_vec[1])
            throw runtime_error("解码后的点集数量与文件头不一致");
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::decompress_current_file()
    {
        decompression_file_state_c<T_data> state;
        auto decompression_stage_start_time=chrono::steady_clock::now();
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
            if(print_compression_stage_time)
                printf("decompression stage time: stream_decode=%f s, label_decode=%f s, value_decode=%f s\n",state.stream_decode_time,state.label_decode_time,state.value_decode_time);
            return;
        }
        else
        {
            decompress_values<two_point_sets,fixed_value_mode,fixed_label_mode,fixed_execution,fixed_predictor>(state);
            if(print_compression_stage_time)
                printf("decompression stage time: stream_decode=%f s, label_decode=%f s, value_decode=%f s\n",state.stream_decode_time,state.label_decode_time,state.value_decode_time);
            return;
        }
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::decompress_labels_and_values(decompression_file_state_c<T_data>&state)
    {
        constexpr bool try_time_label=fixed_label_mode==time_dimension_mode_e::constant_extrapolation||fixed_label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual;
        constexpr bool try_time_value=fixed_value_mode==time_dimension_mode_e::constant_extrapolation||fixed_value_mode==time_dimension_mode_e::linear_interpolation||fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual;
        constexpr bool use_spatial_residual=fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual;
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
                uint32_t predicted_label=0;
                bool used_time_label=false;
                if constexpr(try_time_label)
                    used_time_label=predict_temporal_label<fixed_label_mode>(pred_id,0,predicted_label);
                array<T_data,2> temporal_value_predictions{{0,0}};
                array<uint8_t,2> used_time_value_predictions{{0,0}};
                array<spatial_value_source_c<T_data>,2> spatial_value_sources;
                for(uint32_t point_set_id=0;point_set_id<2;point_set_id++)
                {
                    if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                        continue;
                    if constexpr(try_time_value)
                        used_time_value_predictions[point_set_id]=predict_temporal_value<fixed_value_mode>(pred_id,point_set_id,temporal_value_predictions[point_set_id]);
                    if(used_time_value_predictions[point_set_id])
                    {
                        if constexpr(use_spatial_residual)
                            spatial_value_sources[point_set_id]={&current_time_spatial_residual_data,&current_time_spatial_residual_valid};
                    }
                    else
                        spatial_value_sources[point_set_id].data=&current_decompressed_data;
                }
                bool needs_spatial_value_prediction=spatial_value_sources[0].data||spatial_value_sources[1].data;
                fused_spatial_prediction_c<T_data> spatial_prediction;
                if(!used_time_label||needs_spatial_value_prediction)
                {
                    if constexpr(fixed_predictor==point_set_codec_e::rbf)
                        spatial_prediction=predict_fused_spatial_rbf(has_rbf_weights,ref_ids,ref_weights,!used_time_label,spatial_value_sources);
                    else
                        spatial_prediction=predict_fused_spatial_idw(ref_ids,ref_weights,!used_time_label,spatial_value_sources);
                    if(!used_time_label)
                        predicted_label=spatial_prediction.label;
                }
                uint32_t symbol=label_symbols[prediction_index];
                uint32_t point_set_id=symbol==0?predicted_label:1-predicted_label;
                point_set_id_vec[pred_id]=(uint8_t)point_set_id;
                if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                {
                    current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                    update_temporal_residual<fixed_value_mode>(pred_id,pred_id,point_set_id);
                    continue;
                }
                if(qb_indices[point_set_id]>=point_set_qbs[point_set_id].size())
                    throw runtime_error("点集"+to_string(point_set_id)+"的量化索引不足");
                T_data predicted_value=temporal_value_predictions[point_set_id];
                if(spatial_value_sources[point_set_id].data)
                {
                    if(used_time_value_predictions[point_set_id])
                        predicted_value+=spatial_prediction.values[point_set_id];
                    else
                        predicted_value=spatial_prediction.values[point_set_id];
                }
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
        exception_ptr decompression_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(prediction_structure_vec.size()>1&&thread_num>1)
        for(int64_t structure_index=0;structure_index<(int64_t)prediction_structure_vec.size();structure_index++)
        {
            try
            {
                vector<uint64_t> qb_indices(state.point_set_num,0);
                const unsigned char*unpredictable_value_cursor=state.structure_unpredictable_value_begin_vec[structure_index];
                decompress_targets(prediction_structure_vec[structure_index],state.structure_label_symbols[structure_index],state.structure_point_set_qb_vec[structure_index],qb_indices,unpredictable_value_cursor,state.structure_unpredictable_value_end_vec[structure_index]);
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
        constexpr bool supports_single_point_set_spatial_fast_path=!two_point_sets&&(fixed_predictor==point_set_codec_e::idw||fixed_predictor==point_set_codec_e::rbf)&&fixed_value_mode==time_dimension_mode_e::spatial;
        bool use_single_point_set_spatial_fast_path=supports_single_point_set_spatial_fast_path&&prediction_structure_vec.size()==1&&thread_num==1&&!use_partition_adaptive_predictor_experimental&&prediction_structure_vec.front().prediction_weights.size()==prediction_structure_vec.front().reference_ids.size();
        if(use_single_point_set_spatial_fast_path)
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
            const vector<uint8_t>&rbf_fallback=prediction_structure.rbf_fallback_flags;
            const vector<double>&pred_eb_multiplier_vec=prediction_structure.eb_multiplier;
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
                    if constexpr(!two_point_sets)
                        return predict_value_one_point_set(prediction_index,reference_ids,reference_begin,prediction_weights,data);
                    if constexpr(fixed_predictor==point_set_codec_e::idw)
                        return predict_value_two_point_sets_idw(ref_ids,ref_weights,data,point_set_id,valid_ref_mask);
                    return predict_value_two_point_sets_rbf(!rbf_fallback[prediction_index],ref_ids,ref_weights,data,point_set_id,valid_ref_mask);
                };
                T_data pred=0;
                bool use_cross_time_prediction=false;
                if constexpr(fixed_value_mode==time_dimension_mode_e::constant_extrapolation||fixed_value_mode==time_dimension_mode_e::linear_interpolation||fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                {
                    use_cross_time_prediction=predict_temporal_value<fixed_value_mode>(pred_id,point_set_id,pred);
                    if constexpr(fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
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
        if(use_single_point_set_spatial_fast_path)
        {
            const prediction_structure_c<T_calc>&structure=prediction_structure_vec.front();
            const vector<int32_t> &qb_vec=structure_point_set_qb_vec[0][0];
            if(qb_vec.size()!=point_num)
                throw runtime_error("单点集空间快速路径的量化索引数量不等于点数");
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
                throw runtime_error("单点集空间快速路径存在未使用的不可预测值");
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
