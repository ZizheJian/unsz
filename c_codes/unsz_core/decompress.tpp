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
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,point_set_codec_e fixed_predictor>
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
        current_prediction_framework=prediction_framework;
        prepare_temporal_prediction();
        if constexpr(two_point_sets)
        {
            point_set_id_vec.resize(point_num);
            if(prediction_data_uses_prediction_order)
                prediction_order_point_set_id_vec.resize(point_num);
        }
        else
        {
            point_set_id_vec.clear();
            prediction_order_point_set_id_vec.clear();
        }
        if constexpr(two_point_sets)
        {
            decompress_labels_and_values<two_point_sets,fixed_value_mode,fixed_label_mode,fixed_predictor>(state);
            if(print_compression_stage_time)
                printf("decompression stage time: stream_decode=%f s, label_decode=%f s, value_decode=%f s\n",state.stream_decode_time,state.label_decode_time,state.value_decode_time);
            return;
        }
        else
        {
            decompress_values<two_point_sets,fixed_value_mode,fixed_label_mode,fixed_predictor>(state);
            if(print_compression_stage_time)
                printf("decompression stage time: stream_decode=%f s, label_decode=%f s, value_decode=%f s\n",state.stream_decode_time,state.label_decode_time,state.value_decode_time);
            return;
        }
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::decompress_labels_and_values(decompression_file_state_c<T_data>&state)
    {
        constexpr bool try_time_label=fixed_label_mode==time_dimension_mode_e::constant_extrapolation||fixed_label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual;
        constexpr bool try_time_value=fixed_value_mode==time_dimension_mode_e::constant_extrapolation||fixed_value_mode==time_dimension_mode_e::linear_interpolation||fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual;
        constexpr bool use_spatial_residual=fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual;
        current_decompressed_data.resize(point_num);
        if(prediction_data_uses_prediction_order)
            prediction_order_decompressed_data.resize(point_num);
        vector<T_data>&reconstructed_data=prediction_data_uses_prediction_order?prediction_order_decompressed_data:current_decompressed_data;
        auto stage_start_time=chrono::steady_clock::now();
        auto decompress_targets=[&](const prediction_structure_c<T_calc>&prediction_structure,const vector<uint8_t>&label_symbols,const vector<vector<int32_t>>&point_set_qbs,vector<uint64_t>&qb_indices,const unsigned char*&unpredictable_value_cursor,const unsigned char*unpredictable_value_end)
        {
            const vector<int>&target_ids=prediction_structure.target_ids;
            const vector<int>&ref_ids_vec=prediction_structure.reference_ids;
            const vector<T_calc>&weights=prediction_structure.prediction_weights;
            const vector<uint8_t>&rbf_fallback=prediction_structure.rbf_fallback_flags;
            const vector<T_calc>&eb_multipliers=prediction_structure.eb_multiplier;
            bool standard_reference_layout=prediction_structure.uses_standard_reference_layout(prediction_max_ref_num);
            for(uint32_t prediction_index=0;prediction_index<target_ids.size();prediction_index++)
            {
                int pred_id=target_ids[prediction_index];
                uint32_t storage_id=prediction_data_uses_prediction_order?prediction_index:(uint32_t)pred_id;
                uint32_t predicted_label=0;
                bool used_time_label=false;
                if constexpr(try_time_label)
                    used_time_label=predict_temporal_label<fixed_label_mode>(storage_id,0,predicted_label);
                uint32_t symbol=label_symbols[prediction_structure_vec.size()==1?(uint32_t)pred_id:prediction_index];
                uint32_t point_set_id;
                T_data predicted_value=0,temporal_prediction=0;
                bool used_time_value=false;
                if(used_time_label)
                {
                    point_set_id=symbol==0?predicted_label:1-predicted_label;
                    point_set_id_vec[pred_id]=(uint8_t)point_set_id;
                    if(prediction_data_uses_prediction_order)
                        prediction_order_point_set_id_vec[storage_id]=(uint8_t)point_set_id;
                    if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                    {
                        reconstructed_data[storage_id]=point_set_min_data_vec[point_set_id];
                        continue;
                    }
                    bool has_rbf_weights=true;
                    if constexpr(fixed_predictor==point_set_codec_e::rbf)
                        has_rbf_weights=!rbf_fallback[prediction_index];
                    auto predict_spatial_value=[&](const vector<T_data>&data,const vector<uint8_t>*valid_ref_mask=nullptr)
                    {
                        typename prediction_structure_c<T_calc>::reference_range_c reference_range=prediction_structure.get_reference_range(prediction_index,prediction_max_ref_num,standard_reference_layout);
                        array_view_c<int> ref_ids(reference_range.num?ref_ids_vec.data()+reference_range.begin:nullptr,reference_range.num);
                        array_view_c<T_calc> ref_weights(reference_range.num?weights.data()+reference_range.begin:nullptr,reference_range.num);
                        return predict_value_two_point_sets_rbf(has_rbf_weights,ref_ids,ref_weights,data,point_set_id,valid_ref_mask);
                    };
                    if constexpr(try_time_value)
                        used_time_value=predict_temporal_value<fixed_value_mode>(storage_id,point_set_id,temporal_prediction);
                    predicted_value=temporal_prediction;
                    if(used_time_value)
                    {
                        if constexpr(use_spatial_residual)
                            predicted_value+=predict_spatial_value(current_time_spatial_residual_data,&current_time_spatial_residual_valid);
                    }
                    else
                        predicted_value=predict_spatial_value(reconstructed_data);
                }
                else
                {
                    typename prediction_structure_c<T_calc>::reference_range_c reference_range=prediction_structure.get_reference_range(prediction_index,prediction_max_ref_num,standard_reference_layout);
                    array_view_c<int> ref_ids(reference_range.num?ref_ids_vec.data()+reference_range.begin:nullptr,reference_range.num);
                    array_view_c<T_calc> ref_weights(reference_range.num?weights.data()+reference_range.begin:nullptr,reference_range.num);
                    bool has_rbf_weights=true;
                    if constexpr(fixed_predictor==point_set_codec_e::rbf)
                        has_rbf_weights=!rbf_fallback[prediction_index];
                    array<T_data,2> temporal_value_predictions{{0,0}};
                    array<uint8_t,2> used_time_value_predictions{{0,0}};
                    array<spatial_value_source_c<T_data>,2> spatial_value_sources;
                    for(uint32_t candidate_point_set_id=0;candidate_point_set_id<2;candidate_point_set_id++)
                    {
                        if(point_set_codec_vec[candidate_point_set_id]==point_set_codec_e::constant)
                            continue;
                        if constexpr(try_time_value)
                            used_time_value_predictions[candidate_point_set_id]=predict_temporal_value<fixed_value_mode>(storage_id,candidate_point_set_id,temporal_value_predictions[candidate_point_set_id]);
                        if(used_time_value_predictions[candidate_point_set_id])
                        {
                            if constexpr(use_spatial_residual)
                                spatial_value_sources[candidate_point_set_id]={&current_time_spatial_residual_data,&current_time_spatial_residual_valid};
                        }
                        else
                            spatial_value_sources[candidate_point_set_id].data=&reconstructed_data;
                    }
                    fused_spatial_prediction_c<T_data> spatial_prediction;
                    spatial_prediction=predict_fused_spatial_rbf(has_rbf_weights,ref_ids,ref_weights,true,spatial_value_sources);
                    predicted_label=spatial_prediction.label;
                    point_set_id=symbol==0?predicted_label:1-predicted_label;
                    point_set_id_vec[pred_id]=(uint8_t)point_set_id;
                    if(prediction_data_uses_prediction_order)
                        prediction_order_point_set_id_vec[storage_id]=(uint8_t)point_set_id;
                    used_time_value=used_time_value_predictions[point_set_id];
                    temporal_prediction=temporal_value_predictions[point_set_id];
                    predicted_value=temporal_prediction;
                    if(spatial_value_sources[point_set_id].data)
                    {
                        if(used_time_value_predictions[point_set_id])
                            predicted_value+=spatial_prediction.values[point_set_id];
                        else
                            predicted_value=spatial_prediction.values[point_set_id];
                    }
                }
                if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                {
                    reconstructed_data[storage_id]=point_set_min_data_vec[point_set_id];
                    continue;
                }
                if(qb_indices[point_set_id]>=point_set_qbs[point_set_id].size())
                    throw runtime_error("点集"+to_string(point_set_id)+"的量化索引不足");
                int32_t qb=point_set_qbs[point_set_id][qb_indices[point_set_id]++];
                T_data point_abs_eb=point_set_abs_eb_vec[point_set_id]*eb_multipliers[prediction_index];
                if(qb==numeric_limits<int32_t>::min())
                    reconstructed_data[storage_id]=read_binary<T_data>(unpredictable_value_cursor,unpredictable_value_end);
                else
                    reconstructed_data[storage_id]=predicted_value+(T_data)qb*((T_data)2*point_abs_eb);
                store_temporal_residual<fixed_value_mode>(storage_id,used_time_value,temporal_prediction);
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
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::decompress_values(decompression_file_state_c<T_data>&state)
    {
        uint32_t point_set_num=state.point_set_num;
        vector<vector<vector<int32_t>>> &structure_point_set_qb_vec=state.structure_point_set_qb_vec;
        vector<const unsigned char*>&structure_unpredictable_value_begin_vec=state.structure_unpredictable_value_begin_vec,&structure_unpredictable_value_end_vec=state.structure_unpredictable_value_end_vec;
        constexpr bool supports_single_point_set_fast_path=!two_point_sets&&fixed_predictor==point_set_codec_e::rbf&&(fixed_value_mode==time_dimension_mode_e::spatial||fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual);
        bool use_single_point_set_fast_path=supports_single_point_set_fast_path&&point_set_codec_vec[0]!=point_set_codec_e::constant;
        if constexpr(supports_single_point_set_fast_path)
            for(const prediction_structure_c<T_calc>&structure:prediction_structure_vec)
                use_single_point_set_fast_path&=structure.prediction_weights.size()==structure.reference_ids.size();
        if constexpr(fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
            use_single_point_set_fast_path&=previous_time_value_available;
        else if constexpr(fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
            use_single_point_set_fast_path&=temporal_value_interpolation_available;
        current_decompressed_data.resize(point_num);
        if(prediction_data_uses_prediction_order)
            prediction_order_decompressed_data.resize(point_num);
        auto stage_start_time=chrono::steady_clock::now();
        auto decompress_targets=[&](const prediction_structure_c<T_calc>&prediction_structure,auto get_qb,auto get_unpredictable_value)
        {
            const vector<int>&target_ids=prediction_structure.target_ids;
            const vector<int>&reference_ids=prediction_structure.reference_ids;
            const vector<T_calc>&prediction_weights=prediction_structure.prediction_weights;
            const vector<uint8_t>&rbf_fallback=prediction_structure.rbf_fallback_flags;
            const vector<T_calc>&pred_eb_multiplier_vec=prediction_structure.eb_multiplier;
            bool standard_reference_layout=prediction_structure.uses_standard_reference_layout(prediction_max_ref_num);
            vector<T_data>&reconstructed_data=prediction_data_uses_prediction_order?prediction_order_decompressed_data:current_decompressed_data;
            for(uint32_t prediction_index=0;prediction_index<target_ids.size();prediction_index++)
            {
                int pred_id=target_ids[prediction_index];
                uint32_t storage_id=prediction_data_uses_prediction_order?prediction_index:(uint32_t)pred_id;
                uint32_t point_set_id=two_point_sets?point_set_id_vec[pred_id]:0;
                if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                {
                    reconstructed_data[storage_id]=point_set_min_data_vec[point_set_id];
                    continue;
                }
                auto predict_spatial_value=[&](vector<T_data>&data,const vector<uint8_t>*valid_ref_mask=nullptr)
                {
                    typename prediction_structure_c<T_calc>::reference_range_c reference_range=prediction_structure.get_reference_range(prediction_index,prediction_max_ref_num,standard_reference_layout);
                    array_view_c<int> ref_ids(reference_range.num?reference_ids.data()+reference_range.begin:nullptr,reference_range.num);
                    if constexpr(!two_point_sets)
                        return predict_value_one_point_set(reference_range.begin,reference_range.num,reference_ids,prediction_weights,data);
                    array_view_c<T_calc> ref_weights(reference_range.num?prediction_weights.data()+reference_range.begin:nullptr,reference_range.num);
                    return predict_value_two_point_sets_rbf(!rbf_fallback[prediction_index],ref_ids,ref_weights,data,point_set_id,valid_ref_mask);
                };
                T_data pred=0,temporal_prediction=0;
                bool use_cross_time_prediction=false;
                if constexpr(fixed_value_mode==time_dimension_mode_e::constant_extrapolation||fixed_value_mode==time_dimension_mode_e::linear_interpolation||fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                {
                    use_cross_time_prediction=predict_temporal_value<fixed_value_mode>(storage_id,point_set_id,temporal_prediction);
                    pred=temporal_prediction;
                    if constexpr(fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                        if(use_cross_time_prediction)
                            pred+=predict_spatial_value(current_time_spatial_residual_data,&current_time_spatial_residual_valid);
                }
                if(!use_cross_time_prediction)
                    pred=predict_spatial_value(reconstructed_data);
                int32_t qb=get_qb(prediction_index,point_set_id);
                T_data point_abs_eb=point_set_abs_eb_vec[point_set_id]*pred_eb_multiplier_vec[prediction_index];
                if(qb==numeric_limits<int32_t>::min())
                {
                    T_data unpredictable_value=get_unpredictable_value(point_set_id);
                    reconstructed_data[storage_id]=unpredictable_value;
                }
                else
                    reconstructed_data[storage_id]=pred+(T_data)qb*((T_data)2*point_abs_eb);
                store_temporal_residual<fixed_value_mode>(storage_id,use_cross_time_prediction,temporal_prediction);
            }
        };
        if(use_single_point_set_fast_path)
        {
            auto decompress_structure=[&](uint32_t structure_index)
            {
                const prediction_structure_c<T_calc>&structure=prediction_structure_vec[structure_index];
                const vector<int32_t>&qb_vec=structure_point_set_qb_vec[structure_index][0];
                uint32_t target_num=(uint32_t)structure.target_ids.size();
                if(qb_vec.size()!=target_num)
                    throw runtime_error("单点集快速路径的量化索引数量与目标数量不一致");
                const unsigned char *unpredictable_value_cursor=structure_unpredictable_value_begin_vec[structure_index],*unpredictable_value_end=structure_unpredictable_value_end_vec[structure_index];
                vector<T_data>&reconstructed_data=prediction_data_uses_prediction_order?prediction_order_decompressed_data:current_decompressed_data;
                if constexpr(fixed_value_mode==time_dimension_mode_e::spatial)
                {
                    double base_abs_eb=point_set_abs_eb_vec[0];
                    auto reconstruct_prediction=[&](uint32_t prediction_index,T_data prediction)
                    {
                        uint32_t pred_id=(uint32_t)structure.target_ids[prediction_index];
                        uint32_t reconstructed_id=prediction_data_uses_prediction_order?prediction_index:pred_id;
                        T_data point_abs_eb=base_abs_eb*structure.eb_multiplier[prediction_index];
                        int32_t qb=qb_vec[prediction_index];
                        if(qb==numeric_limits<int32_t>::min())
                            reconstructed_data[reconstructed_id]=read_binary<T_data>(unpredictable_value_cursor,unpredictable_value_end);
                        else
                            reconstructed_data[reconstructed_id]=prediction+(T_data)qb*((T_data)2*point_abs_eb);
                    };
                    uint32_t general_target_num=min<uint32_t>(prediction_max_ref_num,target_num);
                    for(uint32_t prediction_index=0;prediction_index<general_target_num;prediction_index++)
                    {
                        typename prediction_structure_c<T_calc>::reference_range_c reference_range=structure.get_reference_range(prediction_index,prediction_max_ref_num,false);
                        reconstruct_prediction(prediction_index,predict_value_one_point_set(reference_range.begin,reference_range.num,structure.reference_ids,structure.prediction_weights,reconstructed_data));
                    }
                    bool standard_reference_layout=structure.uses_standard_reference_layout(prediction_max_ref_num);
                    uint64_t fixed_ref_begin=(uint64_t)prediction_max_ref_num*(prediction_max_ref_num-1)/2;
                    if(prediction_max_ref_num==4&&standard_reference_layout)
                    {
                        for(uint32_t prediction_index=general_target_num;prediction_index<target_num;prediction_index++,fixed_ref_begin+=4)
                            reconstruct_prediction(prediction_index,predict_value_one_point_set_4(fixed_ref_begin,structure.reference_ids,structure.prediction_weights,reconstructed_data));
                    }
                    else if(prediction_max_ref_num==10&&standard_reference_layout)
                    {
                        for(uint32_t prediction_index=general_target_num;prediction_index<target_num;prediction_index++,fixed_ref_begin+=10)
                            reconstruct_prediction(prediction_index,predict_value_one_point_set_10(fixed_ref_begin,structure.reference_ids,structure.prediction_weights,reconstructed_data));
                    }
                    else
                    {
                        for(uint32_t prediction_index=general_target_num;prediction_index<target_num;prediction_index++)
                        {
                            typename prediction_structure_c<T_calc>::reference_range_c reference_range=structure.get_reference_range(prediction_index,prediction_max_ref_num,false);
                            reconstruct_prediction(prediction_index,predict_value_one_point_set(reference_range.begin,reference_range.num,structure.reference_ids,structure.prediction_weights,reconstructed_data));
                        }
                    }
                }
                else
                {
                    const vector<T_data>*left_time_data=nullptr,*right_time_data=nullptr;
                    if constexpr(fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
                        left_time_data=previous_time_value_source;
                    else
                    {
                        const temporal_dependency_c&dependency=temporal_value_dependency_vec[current_time_step_index];
                        left_time_data=&reconstructed_time_data_vec[(uint32_t)dependency.source_indices[0]];
                        right_time_data=&reconstructed_time_data_vec[(uint32_t)dependency.source_indices[1]];
                    }
                    auto predict_temporal=[&](uint32_t storage_id)
                    {
                        if constexpr(fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
                            return (*left_time_data)[storage_id];
                        return (T_data)(temporal_weights[0]*(T_calc)(*left_time_data)[storage_id]+temporal_weights[1]*(T_calc)(*right_time_data)[storage_id]);
                    };
                    auto reconstruct_prediction=[&](uint32_t prediction_index,T_data prediction,T_data temporal_prediction)
                    {
                        uint32_t pred_id=(uint32_t)structure.target_ids[prediction_index];
                        uint32_t storage_id=prediction_data_uses_prediction_order?prediction_index:pred_id;
                        T_data point_abs_eb=point_set_abs_eb_vec[0]*structure.eb_multiplier[prediction_index];
                        int32_t qb=qb_vec[prediction_index];
                        if(qb==numeric_limits<int32_t>::min())
                            reconstructed_data[storage_id]=read_binary<T_data>(unpredictable_value_cursor,unpredictable_value_end);
                        else
                            reconstructed_data[storage_id]=prediction+(T_data)qb*((T_data)2*point_abs_eb);
                        current_time_spatial_residual_data[storage_id]=reconstructed_data[storage_id]-temporal_prediction;
                        current_time_spatial_residual_valid[storage_id]=1;
                    };
                    uint32_t general_target_num=min<uint32_t>(prediction_max_ref_num,target_num);
                    for(uint32_t prediction_index=0;prediction_index<general_target_num;prediction_index++)
                    {
                        uint32_t storage_id=prediction_data_uses_prediction_order?prediction_index:(uint32_t)structure.target_ids[prediction_index];
                        T_data temporal_prediction=predict_temporal(storage_id);
                        typename prediction_structure_c<T_calc>::reference_range_c reference_range=structure.get_reference_range(prediction_index,prediction_max_ref_num,false);
                        T_data spatial_prediction=predict_value_one_point_set(reference_range.begin,reference_range.num,structure.reference_ids,structure.prediction_weights,current_time_spatial_residual_data);
                        reconstruct_prediction(prediction_index,temporal_prediction+spatial_prediction,temporal_prediction);
                    }
                    bool standard_reference_layout=structure.uses_standard_reference_layout(prediction_max_ref_num);
                    uint64_t fixed_ref_begin=(uint64_t)prediction_max_ref_num*(prediction_max_ref_num-1)/2;
                    if(prediction_max_ref_num==10&&standard_reference_layout)
                    {
                        for(uint32_t prediction_index=general_target_num;prediction_index<target_num;prediction_index++,fixed_ref_begin+=10)
                        {
                            uint32_t storage_id=prediction_data_uses_prediction_order?prediction_index:(uint32_t)structure.target_ids[prediction_index];
                            T_data temporal_prediction=predict_temporal(storage_id);
                            T_data spatial_prediction=predict_value_one_point_set_10(fixed_ref_begin,structure.reference_ids,structure.prediction_weights,current_time_spatial_residual_data);
                            reconstruct_prediction(prediction_index,temporal_prediction+spatial_prediction,temporal_prediction);
                        }
                    }
                    else
                    {
                        for(uint32_t prediction_index=general_target_num;prediction_index<target_num;prediction_index++)
                        {
                            uint32_t storage_id=prediction_data_uses_prediction_order?prediction_index:(uint32_t)structure.target_ids[prediction_index];
                            T_data temporal_prediction=predict_temporal(storage_id);
                            typename prediction_structure_c<T_calc>::reference_range_c reference_range=structure.get_reference_range(prediction_index,prediction_max_ref_num,false);
                            T_data spatial_prediction=predict_value_one_point_set(reference_range.begin,reference_range.num,structure.reference_ids,structure.prediction_weights,current_time_spatial_residual_data);
                            reconstruct_prediction(prediction_index,temporal_prediction+spatial_prediction,temporal_prediction);
                        }
                    }
                }
                if(unpredictable_value_cursor!=unpredictable_value_end)
                    throw runtime_error("单点集快速路径存在未使用的不可预测值");
            };
            exception_ptr prediction_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(prediction_structure_vec.size()>1&&thread_num>1)
            for(int64_t structure_index=0;structure_index<(int64_t)prediction_structure_vec.size();structure_index++)
            {
                try
                {
                    decompress_structure((uint32_t)structure_index);
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
            exception_ptr prediction_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(prediction_structure_vec.size()>1&&thread_num>1)
            for(int64_t structure_index=0;structure_index<(int64_t)prediction_structure_vec.size();structure_index++)
            {
                try
                {
                    const unsigned char *unpredictable_value_cursor=structure_unpredictable_value_begin_vec[structure_index],*unpredictable_value_end=structure_unpredictable_value_end_vec[structure_index];
                    auto get_unpredictable_value=[&](uint32_t)
                    {
                        return read_binary<T_data>(unpredictable_value_cursor,unpredictable_value_end);
                    };
                    if constexpr(!two_point_sets)
                    {
                        const vector<int32_t>&qb_vec=structure_point_set_qb_vec[structure_index][0];
                        if(qb_vec.size()!=prediction_structure_vec[structure_index].target_ids.size())
                            throw runtime_error("单点集数值流的量化索引数量与目标数量不一致");
                        auto get_qb=[&](uint32_t prediction_index,uint32_t)
                        {
                            return qb_vec[prediction_index];
                        };
                        decompress_targets(prediction_structure_vec[structure_index],get_qb,get_unpredictable_value);
                    }
                    else
                    {
                        vector<uint64_t> local_qb_index_vec(point_set_num,0);
                        auto get_qb=[&](uint32_t,uint32_t point_set_id)
                        {
                            const vector<int32_t>&qb_vec=structure_point_set_qb_vec[structure_index][point_set_id];
                            if(local_qb_index_vec[point_set_id]>=qb_vec.size())
                                throw runtime_error("数值流的量化索引不足");
                            return qb_vec[local_qb_index_vec[point_set_id]++];
                        };
                        decompress_targets(prediction_structure_vec[structure_index],get_qb,get_unpredictable_value);
                        for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                        {
                            const vector<int32_t>&qb_vec=structure_point_set_qb_vec[structure_index][point_set_id];
                            if(local_qb_index_vec[point_set_id]!=qb_vec.size())
                                throw runtime_error("数值流存在未使用的量化索引");
                        }
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
