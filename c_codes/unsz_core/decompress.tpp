#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::decompress_current_file_by_mode()
    {
        uint32_t point_set_num=(uint32_t)point_set_config_vec.size();
        if(point_set_num!=1&&point_set_num!=2)
            throw runtime_error("解压只支持一个或两个点集");
        if(point_set_num==1)
            resolve_single_point_set_time_mode();
        check_label_value_time_step_order();
        for(const point_set_config_c&config:point_set_config_vec)
            if(config.codec!=point_set_codec_e::constant&&config.codec!=point_set_codec_e::rbf)
                throw runtime_error("解压只支持constant或RBF空间预测器");
        dispatch_prediction_framework(get_prediction_framework(point_set_num),[&](auto two_point_sets,auto value_mode,auto label_mode)
        {
            this->template decompress_current_file<decltype(two_point_sets)::value,decltype(value_mode)::value,decltype(label_mode)::value>();
        });
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::print_current_decompression_diagnostics() const
    {
        if(print_compression_stage_time)
            printf("decompression stage time: stream_decode=%f s, label_decode=%f s, value_decode=%f s\n",current_decompression_diagnostics.stream_decode_time,current_decompression_diagnostics.label_decode_time,current_decompression_diagnostics.value_decode_time);
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
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode>
    void param_c<T_coord,T_data>::decompress_current_file()
    {
        decompression_file_state_c<T_data> state;
        auto decompression_stage_start_time=chrono::steady_clock::now();
        decode_current_compression_streams<two_point_sets>(state);
        uint32_t point_set_num=state.point_set_num;
        for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            point_set_abs_eb_vec[point_set_id]=point_set_codec_vec[point_set_id]==point_set_codec_e::constant?0:point_set_abs_eb(point_set_id,true);
            if(point_set_codec_vec[point_set_id]!=point_set_codec_e::constant&&!(point_set_abs_eb_vec[point_set_id]>0))
                throw runtime_error("点集"+to_string(point_set_id)+"的绝对误差上限必须大于0");
        }
        state.stream_decode_time=seconds_since(decompression_stage_start_time);

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
            decompress_labels_and_values<fixed_value_mode,fixed_label_mode>(state);
        else
            decompress_values<fixed_value_mode>(state);
        current_decompression_diagnostics.stream_decode_time=state.stream_decode_time;
        current_decompression_diagnostics.label_decode_time=state.label_decode_time;
        current_decompression_diagnostics.value_decode_time=state.value_decode_time;
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode>
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
            bool standard_reference_layout=prediction_structure.uses_standard_reference_layout(rbf_reference_num);
            for(uint32_t prediction_index=0;prediction_index<target_ids.size();prediction_index++)
            {
                int pred_id=target_ids[prediction_index];
                uint32_t storage_id=prediction_structure.get_storage_id(prediction_index,pred_id,prediction_data_uses_prediction_order);
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
                    bool has_rbf_weights=!rbf_fallback[prediction_index];
                    auto predict_spatial_value=[&](const vector<T_data>&data,const vector<uint8_t>*valid_ref_mask=nullptr)
                    {
                        typename prediction_structure_c<T_calc>::reference_range_c reference_range=prediction_structure.get_reference_range(prediction_index,rbf_reference_num,standard_reference_layout);
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
                    typename prediction_structure_c<T_calc>::reference_range_c reference_range=prediction_structure.get_reference_range(prediction_index,rbf_reference_num,standard_reference_layout);
                    array_view_c<int> ref_ids(reference_range.num?ref_ids_vec.data()+reference_range.begin:nullptr,reference_range.num);
                    array_view_c<T_calc> ref_weights(reference_range.num?weights.data()+reference_range.begin:nullptr,reference_range.num);
                    bool has_rbf_weights=!rbf_fallback[prediction_index];
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
        parallel_for_tasks((uint32_t)prediction_structure_vec.size(),thread_num,[&](uint32_t structure_index)
        {
            vector<uint64_t> qb_indices(state.point_set_num,0);
            const unsigned char*unpredictable_value_cursor=state.structure_unpredictable_value_begin_vec[structure_index];
            decompress_targets(prediction_structure_vec[structure_index],state.structure_label_symbols[structure_index],state.structure_point_set_qb_vec[structure_index],qb_indices,unpredictable_value_cursor,state.structure_unpredictable_value_end_vec[structure_index]);
        });
        state.label_decode_time=seconds_since(stage_start_time);
        state.value_decode_time=0;
    }


    // 单点集时逐点解压，与compress_structure_values一一对应。
    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e value_mode>
    void param_c<T_coord,T_data>::decompress_values(decompression_file_state_c<T_data>&state)
    {
        constexpr bool use_spatial_residual=value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual;
        vector<vector<vector<int32_t>>> &structure_point_set_qb_vec=state.structure_point_set_qb_vec;
        vector<const unsigned char*>&structure_unpredictable_value_begin_vec=state.structure_unpredictable_value_begin_vec,&structure_unpredictable_value_end_vec=state.structure_unpredictable_value_end_vec;
        current_decompressed_data.resize(point_num);
        if(prediction_data_uses_prediction_order)
            prediction_order_decompressed_data.resize(point_num);
        vector<T_data>&reconstructed_data=prediction_data_uses_prediction_order?prediction_order_decompressed_data:current_decompressed_data;
        auto stage_start_time=chrono::steady_clock::now();
        if(point_set_codec_vec[0]==point_set_codec_e::constant)
        {
            for(uint32_t structure_id=0;structure_id<prediction_structure_vec.size();structure_id++)
            {
                if(!structure_point_set_qb_vec[structure_id][0].empty())
                    throw runtime_error("常量点集的数值流不能包含量化索引");
                if(structure_unpredictable_value_begin_vec[structure_id]!=structure_unpredictable_value_end_vec[structure_id])
                    throw runtime_error("常量点集的数值流不能包含不可预测值");
            }
            fill(reconstructed_data.begin(),reconstructed_data.end(),point_set_min_data_vec[0]);
            state.value_decode_time=seconds_since(stage_start_time);
            return;
        }
        T_data base_abs_eb=point_set_abs_eb_vec[0];
        auto cross_slice_predictor=get_single_point_set_cross_slice_predictor<value_mode>();
        bool use_cross_slice_prediction=cross_slice_predictor.first;
        const auto&predict_cross_slice=cross_slice_predictor.second;
        parallel_for_tasks((uint32_t)prediction_structure_vec.size(),thread_num,[&](uint32_t structure_index)
        {
            const prediction_structure_c<T_calc>&structure=prediction_structure_vec[structure_index];
            const vector<int>&target_ids=structure.target_ids;
            const vector<int32_t>&qb_vec=structure_point_set_qb_vec[structure_index][0];
            if(qb_vec.size()!=target_ids.size())
                throw runtime_error("单点集数值流的量化索引数量与目标数量不一致");
            const unsigned char *unpredictable_value_cursor=structure_unpredictable_value_begin_vec[structure_index],*unpredictable_value_end=structure_unpredictable_value_end_vec[structure_index];
            // 跨slice预测是否可用对整个slice相同，因此生成可用和不可用两个循环版本，循环内不再判断。
            auto decompress_targets=[&](auto use_cross_slice_constant)
            {
                constexpr bool use_cross_slice=decltype(use_cross_slice_constant)::value;
                auto decompress_target=[&](uint32_t prediction_index,uint64_t ref_begin,uint32_t ref_num) __attribute__((always_inline))
                {
                    uint32_t storage_id=structure.get_storage_id(prediction_index,target_ids[prediction_index],prediction_data_uses_prediction_order);
                    T_data cross_slice_prediction=0,prediction;
                    if constexpr(use_cross_slice)
                    {
                        cross_slice_prediction=predict_cross_slice(storage_id);
                        prediction=cross_slice_prediction;
                        if constexpr(use_spatial_residual)
                            prediction+=predict_value_one_point_set(ref_begin,ref_num,structure.reference_ids,structure.prediction_weights,current_time_spatial_residual_data);
                    }
                    else
                        prediction=predict_value_one_point_set(ref_begin,ref_num,structure.reference_ids,structure.prediction_weights,reconstructed_data);
                    int32_t qb=qb_vec[prediction_index];
                    T_data point_abs_eb=base_abs_eb*structure.eb_multiplier[prediction_index];
                    if(qb==numeric_limits<int32_t>::min())
                        reconstructed_data[storage_id]=read_binary<T_data>(unpredictable_value_cursor,unpredictable_value_end);
                    else
                        reconstructed_data[storage_id]=prediction+(T_data)qb*((T_data)2*point_abs_eb);
                    if constexpr(use_spatial_residual&&use_cross_slice)
                        current_time_spatial_residual_data[storage_id]=reconstructed_data[storage_id]-cross_slice_prediction;
                };
                // 与compress_structure_values相同，标准参考布局时第二个循环直接递推参考区间。
                uint32_t target_num=(uint32_t)target_ids.size(),general_target_num=structure.uses_standard_reference_layout(rbf_reference_num)?min<uint32_t>(rbf_reference_num,target_num):target_num;
                for(uint32_t prediction_index=0;prediction_index<general_target_num;prediction_index++)
                {
                    typename prediction_structure_c<T_calc>::reference_range_c reference_range=structure.get_reference_range(prediction_index,rbf_reference_num,false);
                    decompress_target(prediction_index,reference_range.begin,reference_range.num);
                }
                uint64_t ref_begin=(uint64_t)rbf_reference_num*(rbf_reference_num-1)/2;
                for(uint32_t prediction_index=general_target_num;prediction_index<target_num;prediction_index++,ref_begin+=rbf_reference_num)
                    decompress_target(prediction_index,ref_begin,rbf_reference_num);
            };
            if(use_cross_slice_prediction)
                decompress_targets(true_type());
            else
                decompress_targets(false_type());
            if(unpredictable_value_cursor!=unpredictable_value_end)
                throw runtime_error("单点集数值流存在未使用的不可预测值");
        });
        state.value_decode_time=seconds_since(stage_start_time);
    }
}
