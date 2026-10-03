#pragma once

namespace unsz
{
    template<typename T_data>
    __attribute__((always_inline)) inline int32_t quantize_or_store_unpredictable(T_data value,T_data prediction,double abs_eb,double inverse_double_abs_eb,vector<T_data>&unpredictable_values,T_data&reconstructed_value)
    {
        double scaled_error=((double)value-prediction)*inverse_double_abs_eb;
        constexpr double min_predictable_qb=(double)numeric_limits<int32_t>::min()+0.5,max_predictable_qb=(double)numeric_limits<int32_t>::max()+0.5;
        if(scaled_error>min_predictable_qb&&scaled_error<max_predictable_qb)
        {
            int32_t qb=(int32_t)llrint(scaled_error);
            reconstructed_value=prediction+qb*2*abs_eb;
            return qb;
        }
        unpredictable_values.push_back(value);
        reconstructed_value=value;
        return numeric_limits<int32_t>::min();
    }

    namespace compression_dispatch
    {
        template<typename Function>
        void label_mode(time_dimension_mode_e mode,Function function)
        {
            if(mode==time_dimension_mode_e::spatial)
                return function(integral_constant<uint32_t,0>());
            if(mode==time_dimension_mode_e::constant_extrapolation)
                return function(integral_constant<uint32_t,1>());
            if(mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual)
                return function(integral_constant<uint32_t,2>());
            throw runtime_error("两点集标签压缩不支持当前流程");
        }

        template<typename Function>
        void value_mode(time_dimension_mode_e mode,Function function)
        {
            if(mode==time_dimension_mode_e::spatial)
                return function(integral_constant<uint32_t,0>(),false_type());
            if(mode==time_dimension_mode_e::constant_extrapolation)
                return function(integral_constant<uint32_t,1>(),false_type());
            if(mode==time_dimension_mode_e::linear_interpolation)
                return function(integral_constant<uint32_t,2>(),false_type());
            if(mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
                return function(integral_constant<uint32_t,1>(),true_type());
            if(mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                return function(integral_constant<uint32_t,2>(),true_type());
            throw runtime_error("数值压缩不支持当前流程");
        }
    }

    template<typename T_coord,typename T_data>
    template<point_set_codec_e predictor>
    void param_c<T_coord,T_data>::dispatch_single_point_set_compression()
    {
        prediction_framework_c framework=get_prediction_framework(1);
        compression_dispatch::value_mode(framework.value_mode,[&](auto time_ref_num_constant,auto use_spatial_residual_constant)
        {
            constexpr uint32_t time_ref_num=decltype(time_ref_num_constant)::value;
            constexpr bool use_spatial_residual=decltype(use_spatial_residual_constant)::value;
            if constexpr(time_ref_num==0)
                one_point_set::spatial::compress<predictor>(*this);
            else
                one_point_set::temporal::compress<time_ref_num,use_spatial_residual,predictor>(*this);
        });
    }

    template<typename T_coord,typename T_data>
    template<point_set_codec_e predictor>
    void param_c<T_coord,T_data>::dispatch_staged_label_compression()
    {
        prediction_framework_c framework=get_prediction_framework(2);
        compression_dispatch::label_mode(framework.label_mode,[&](auto time_ref_num_constant)
        {
            constexpr uint32_t time_ref_num=decltype(time_ref_num_constant)::value;
            if constexpr(time_ref_num==0)
                two_point_sets::staged::label_spatial::compress<predictor>(*this);
            else
                two_point_sets::staged::label_temporal::compress<time_ref_num,predictor>(*this);
        });
    }

    template<typename T_coord,typename T_data>
    template<point_set_codec_e predictor>
    void param_c<T_coord,T_data>::dispatch_staged_value_compression()
    {
        prediction_framework_c framework=get_prediction_framework(2);
        compression_dispatch::value_mode(framework.value_mode,[&](auto time_ref_num_constant,auto use_spatial_residual_constant)
        {
            constexpr uint32_t time_ref_num=decltype(time_ref_num_constant)::value;
            constexpr bool use_spatial_residual=decltype(use_spatial_residual_constant)::value;
            if constexpr(time_ref_num==0)
                two_point_sets::staged::value_spatial::compress<predictor>(*this);
            else
                two_point_sets::staged::value_temporal::compress<time_ref_num,use_spatial_residual,predictor>(*this);
        });
    }

    template<typename T_coord,typename T_data>
    template<point_set_codec_e predictor>
    void param_c<T_coord,T_data>::dispatch_fused_compression()
    {
        prediction_framework_c framework=get_prediction_framework(2);
        compression_dispatch::label_mode(framework.label_mode,[&](auto label_time_ref_num_constant)
        {
            compression_dispatch::value_mode(framework.value_mode,[&](auto value_time_ref_num_constant,auto use_spatial_residual_constant)
            {
                constexpr uint32_t label_time_ref_num=decltype(label_time_ref_num_constant)::value;
                constexpr uint32_t value_time_ref_num=decltype(value_time_ref_num_constant)::value;
                constexpr bool use_spatial_residual=decltype(use_spatial_residual_constant)::value;
                if constexpr(label_time_ref_num==0&&value_time_ref_num==0)
                    two_point_sets::fused::label_spatial_value_spatial::compress<predictor>(*this);
                else if constexpr(label_time_ref_num==0)
                    two_point_sets::fused::label_spatial_value_temporal::compress<value_time_ref_num,use_spatial_residual,predictor>(*this);
                else if constexpr(value_time_ref_num==0)
                    two_point_sets::fused::label_temporal_value_spatial::compress<label_time_ref_num,predictor>(*this);
                else if constexpr(label_time_ref_num==value_time_ref_num)
                    two_point_sets::fused::label_temporal_value_temporal::compress<label_time_ref_num,use_spatial_residual,predictor>(*this);
                else
                    throw runtime_error("融合两点集压缩要求label和value使用相同数量的时间参考，或其中一方使用空间预测");
            });
        });
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets>
    void param_c<T_coord,T_data>::classify_current_point_sets(compression_file_state_c&state)
    {
        uint32_t &point_set_num=state.point_set_num;
        auto stage_start_time=chrono::steady_clock::now();
        uint32_t configured_point_set_num=(uint32_t)point_set_config_vec.size();
        point_set_point_num_vec.assign(configured_point_set_num,0);
        point_set_min_data_vec.assign(configured_point_set_num,numeric_limits<T_data>::infinity());
        point_set_max_data_vec.assign(configured_point_set_num,-numeric_limits<T_data>::infinity());
        if constexpr(!two_point_sets)
        {
            point_set_num=1;
            point_set_id_vec.clear();
            point_set_point_num_vec[0]=point_num;
            T_data minimum[4]={numeric_limits<T_data>::infinity(),numeric_limits<T_data>::infinity(),numeric_limits<T_data>::infinity(),numeric_limits<T_data>::infinity()};
            T_data maximum[4]={-numeric_limits<T_data>::infinity(),-numeric_limits<T_data>::infinity(),-numeric_limits<T_data>::infinity(),-numeric_limits<T_data>::infinity()};
            uint32_t point_id=0;
            for(;point_id+3<point_num;point_id+=4)
            {
                for(uint32_t lane=0;lane<4;lane++)
                {
                    T_data value=current_data[point_id+lane];
                    minimum[lane]=min(minimum[lane],value);
                    maximum[lane]=max(maximum[lane],value);
                }
            }
            for(;point_id<point_num;point_id++)
            {
                minimum[0]=min(minimum[0],current_data[point_id]);
                maximum[0]=max(maximum[0],current_data[point_id]);
            }
            for(uint32_t lane=1;lane<4;lane++)
            {
                minimum[0]=min(minimum[0],minimum[lane]);
                maximum[0]=max(maximum[0],maximum[lane]);
            }
            point_set_min_data_vec[0]=minimum[0];
            point_set_max_data_vec[0]=maximum[0];
        }
        else if(point_set_rule_vec.size()==1)
        {
            point_set_id_vec.resize(point_num);
            const point_set_rule_c &rule=point_set_rule_vec[0];
            auto classify_two_point_sets=[&](auto matches)
            {
                uint32_t point_num_per_set[4][2]={{0,0},{0,0},{0,0},{0,0}};
                T_data minimum[4][2],maximum[4][2];
                for(uint32_t lane=0;lane<4;lane++)
                    for(uint32_t point_set_id=0;point_set_id<2;point_set_id++)
                    {
                        minimum[lane][point_set_id]=numeric_limits<T_data>::infinity();
                        maximum[lane][point_set_id]=-numeric_limits<T_data>::infinity();
                    }
                uint32_t point_id=0;
                for(;point_id+3<point_num;point_id+=4)
                {
                    for(uint32_t lane=0;lane<4;lane++)
                    {
                        T_data value=current_data[point_id+lane];
                        uint32_t point_set_id=matches((double)value)?0:1;
                        point_set_id_vec[point_id+lane]=(uint8_t)point_set_id;
                        point_num_per_set[lane][point_set_id]++;
                        minimum[lane][point_set_id]=min(minimum[lane][point_set_id],value);
                        maximum[lane][point_set_id]=max(maximum[lane][point_set_id],value);
                    }
                }
                for(;point_id<point_num;point_id++)
                {
                    T_data value=current_data[point_id];
                    uint32_t point_set_id=matches((double)value)?0:1;
                    point_set_id_vec[point_id]=(uint8_t)point_set_id;
                    point_num_per_set[0][point_set_id]++;
                    minimum[0][point_set_id]=min(minimum[0][point_set_id],value);
                    maximum[0][point_set_id]=max(maximum[0][point_set_id],value);
                }
                for(uint32_t lane=1;lane<4;lane++)
                    for(uint32_t point_set_id=0;point_set_id<2;point_set_id++)
                    {
                        point_num_per_set[0][point_set_id]+=point_num_per_set[lane][point_set_id];
                        minimum[0][point_set_id]=min(minimum[0][point_set_id],minimum[lane][point_set_id]);
                        maximum[0][point_set_id]=max(maximum[0][point_set_id],maximum[lane][point_set_id]);
                    }
                for(uint32_t point_set_id=0;point_set_id<2;point_set_id++)
                {
                    point_set_point_num_vec[point_set_id]=point_num_per_set[0][point_set_id];
                    point_set_min_data_vec[point_set_id]=minimum[0][point_set_id];
                    point_set_max_data_vec[point_set_id]=maximum[0][point_set_id];
                }
                point_set_num=2;
            };
            double threshold=rule.value;
            switch(rule.operator_type)
            {
                case point_set_rule_operator_e::less:
                    classify_two_point_sets([threshold](double value){return value<threshold;});
                    break;
                case point_set_rule_operator_e::less_equal:
                    classify_two_point_sets([threshold](double value){return value<=threshold;});
                    break;
                case point_set_rule_operator_e::greater:
                    classify_two_point_sets([threshold](double value){return value>threshold;});
                    break;
                case point_set_rule_operator_e::greater_equal:
                    classify_two_point_sets([threshold](double value){return value>=threshold;});
                    break;
                case point_set_rule_operator_e::equal:
                    classify_two_point_sets([threshold](double value){return value==threshold;});
                    break;
                case point_set_rule_operator_e::not_equal:
                    classify_two_point_sets([threshold](double value){return value!=threshold;});
                    break;
                default:
                    throw runtime_error("未知的点集分类操作符");
            }
        }
        else
            throw runtime_error("双点集压缩要求恰好一条分类规则");
        state.point_set_classification_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_current_point_set_codecs_and_eb(compression_file_state_c&state)
    {
        uint32_t point_set_num=state.point_set_num;
        auto stage_start_time=chrono::steady_clock::now();
        point_set_point_num_vec.resize(point_set_num);
        point_set_min_data_vec.resize(point_set_num);
        point_set_max_data_vec.resize(point_set_num);
        point_set_abs_eb_vec.resize(point_set_num);
        point_set_codec_vec.resize(point_set_num);
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            if (point_set_point_num_vec[point_set_id]==0)
            {
                point_set_min_data_vec[point_set_id]=0;
                point_set_max_data_vec[point_set_id]=0;
                point_set_abs_eb_vec[point_set_id]=0;
                point_set_codec_vec[point_set_id]=point_set_codec_e::constant;
                continue;
            }
            const point_set_config_c &config=point_set_config_vec[point_set_id];
            bool is_constant=point_set_min_data_vec[point_set_id]==point_set_max_data_vec[point_set_id];
            if (is_constant)
                point_set_codec_vec[point_set_id]=point_set_codec_e::constant;
            else
                point_set_codec_vec[point_set_id]=config.codec;
            if (!is_constant && point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                throw runtime_error("非常量点集不能使用常量编码");
            if (is_constant)
                point_set_abs_eb_vec[point_set_id]=0;
            else if (config.error_mode==point_set_error_mode_e::absolute)
                point_set_abs_eb_vec[point_set_id]=(T_data)config.eb;
            else
                point_set_abs_eb_vec[point_set_id]=(T_data)(config.eb*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]));
            if (!is_constant && !(point_set_abs_eb_vec[point_set_id]>0))
                throw runtime_error("点集"+to_string(point_set_id)+"的绝对误差上限必须大于0");
        }
        state.point_set_setup_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution>
    void param_c<T_coord,T_data>::prepare_current_prediction_state(compression_file_state_c&state)
    {
        prediction_framework_c prediction_framework;
        prediction_framework.has_label=two_point_sets;
        prediction_framework.value_mode=fixed_value_mode;
        prediction_framework.label_mode=fixed_label_mode;
        prediction_framework.execution=two_point_sets?fixed_execution:label_value_execution_e::no_label;
        current_prediction_framework=prediction_framework;
        constexpr bool use_fused_two_point_sets_prediction=two_point_sets&&fixed_execution==label_value_execution_e::fused;
        if(prediction_data_uses_prediction_order&&two_point_sets)
        {
            const vector<int>&target_ids=prediction_structure_vec.front().target_ids;
            prediction_order_point_set_id_vec.resize(point_num);
            for(uint32_t prediction_index=0;prediction_index<point_num;prediction_index++)
                prediction_order_point_set_id_vec[prediction_index]=point_set_id_vec[target_ids[prediction_index]];
        }
        prepare_temporal_prediction();
        if constexpr(use_fused_two_point_sets_prediction)
        {
            if(state.point_set_num!=2)
                throw runtime_error("融合两点集预测要求当前文件恰好包含两个点集");
            for(point_set_codec_e codec:point_set_codec_vec)
                if(codec!=point_set_codec_e::constant&&codec!=point_set_codec_e::idw&&codec!=point_set_codec_e::rbf)
                    throw runtime_error("融合两点集预测只支持constant、IDW或RBF预测器");
        }
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,label_value_execution_e fixed_execution>
    void param_c<T_coord,T_data>::initialize_current_compression_buffers(compression_file_state_c&state)
    {
        constexpr bool use_fused_two_point_sets_prediction=two_point_sets&&fixed_execution==label_value_execution_e::fused;
        uint32_t point_set_num=state.point_set_num;
        state.has_label=point_set_num==2;
        uint32_t structure_num=prediction_structure_vec.size();
        state.encoded_payload_vec.resize(structure_num);
        state.label_target_num=0;
        state.label_error_num=0;
        if constexpr(use_fused_two_point_sets_prediction)
        {
            current_decompressed_data.resize(point_num);
            if(prediction_data_uses_prediction_order)
                prediction_order_decompressed_data.resize(point_num);
        }
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution>
    void param_c<T_coord,T_data>::prepare_current_compression(compression_file_state_c&state)
    {
        classify_current_point_sets<two_point_sets>(state);
        prepare_current_point_set_codecs_and_eb(state);
        prepare_current_prediction_state<two_point_sets,fixed_value_mode,fixed_label_mode,fixed_execution>(state);
        initialize_current_compression_buffers<two_point_sets,fixed_execution>(state);
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e fixed_label_mode,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::compress_labels(compression_file_state_c&state)
    {
        double &label_prediction_time=state.label_prediction_time;
        const prediction_structure_c<T_calc>&structure=prediction_structure_vec.front();
        bool standard_reference_layout=structure.uses_standard_reference_layout(prediction_max_ref_num);
        auto predict_configured_label=[&](uint32_t prediction_index)
        {
            typename prediction_structure_c<T_calc>::reference_range_c reference_range=structure.get_reference_range(prediction_index,prediction_max_ref_num,standard_reference_layout);
            const vector<uint8_t>&prediction_point_set_ids=prediction_data_uses_prediction_order?prediction_order_point_set_id_vec:point_set_id_vec;
            return predict_label(reference_range.begin,reference_range.num,structure.reference_ids,structure.prediction_weights,prediction_point_set_ids);
        };
        auto stage_start_time=chrono::steady_clock::now();
        vector<uint8_t> label_symbols;
        if(!state.has_label)
        {
            vector<vector<int32_t>> empty_point_set_qbs(state.point_set_num);
            vector<T_data> empty_unpredictable_values;
            state.encoded_payload_vec.front()=encode_compression_payload(label_symbols,false,empty_point_set_qbs,empty_unpredictable_values);
            return;
        }
        bool use_causal_time_label=fixed_label_mode==time_dimension_mode_e::constant_extrapolation&&previous_time_label_available;
        constexpr bool try_time_label=fixed_label_mode==time_dimension_mode_e::constant_extrapolation||fixed_label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual;
        vector<uint8_t> label_symbol_by_point(point_num,0);
        for(uint32_t prediction_index=0;prediction_index<structure.target_ids.size();prediction_index++)
        {
            int pred_id=structure.target_ids[prediction_index];
            uint32_t storage_id=prediction_data_uses_prediction_order?prediction_index:(uint32_t)pred_id;
            uint32_t actual_label=point_set_id_vec[pred_id]==0?0:1;
            uint32_t predicted_label;
            if(use_causal_time_label)
            {
                if(!predict_temporal_label<fixed_label_mode>(storage_id,0,predicted_label))
                    throw runtime_error("因果时间标签预测缺少历史标签");
            }
            else
            {
                bool used_time_label=false;
                if constexpr(try_time_label)
                    used_time_label=predict_temporal_label<fixed_label_mode>(storage_id,0,predicted_label);
                if(!used_time_label)
                    predicted_label=predict_configured_label(prediction_index);
            }
            bool prediction_error=actual_label!=predicted_label;
            state.label_target_num++;
            state.label_error_num+=prediction_error;
            label_symbol_by_point[pred_id]=prediction_error?1:0;
        }
        label_symbols.reserve(state.label_target_num);
        for(uint32_t point_id=0;point_id<point_num;point_id++)
            label_symbols.push_back(label_symbol_by_point[point_id]);
        label_prediction_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        vector<vector<int32_t>> empty_point_set_qbs(state.point_set_num);
        vector<T_data> empty_unpredictable_values;
        auto encoding_start_time=chrono::steady_clock::now();
        state.encoded_payload_vec.front()=encode_compression_payload(label_symbols,true,empty_point_set_qbs,empty_unpredictable_values);
        state.lossless_encoding_time=chrono::duration<double>(chrono::steady_clock::now()-encoding_start_time).count();
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,point_set_codec_e fixed_predictor>
    value_compression_result_c<T_data> param_c<T_coord,T_data>::compress_structure_values(const prediction_structure_c<T_calc>&prediction_structure)
    {
        const vector<int>&target_ids=prediction_structure.target_ids;
        const vector<int>&reference_ids=prediction_structure.reference_ids;
        const vector<T_calc>&prediction_weights=prediction_structure.prediction_weights;
        const vector<uint8_t>&rbf_fallback=prediction_structure.rbf_fallback_flags;
        const vector<double>&pred_eb_multiplier_vec=prediction_structure.eb_multiplier;
        const vector<double>&inverse_eb_multiplier_vec=prediction_structure.inverse_eb_multiplier;
        bool standard_reference_layout=prediction_structure.uses_standard_reference_layout(prediction_max_ref_num);
        uint32_t point_set_num=(uint32_t)point_set_codec_vec.size();
        array<double,2> inverse_double_base_abs_eb{{0,0}};
        for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
            if(point_set_codec_vec[point_set_id]!=point_set_codec_e::constant)
                inverse_double_base_abs_eb[point_set_id]=0.5/point_set_abs_eb_vec[point_set_id];
        value_compression_result_c<T_data> result;
        result.point_set_qb_vec.resize(point_set_num);
        vector<T_data>&reconstructed_data=prediction_data_uses_prediction_order?prediction_order_decompressed_data:current_decompressed_data;
        if constexpr(!two_point_sets)
            if(point_set_codec_vec[0]!=point_set_codec_e::constant)
                result.point_set_qb_vec[0].resize(target_ids.size());
        if constexpr(!two_point_sets&&fixed_value_mode==time_dimension_mode_e::spatial)
        {
            if(point_set_codec_vec[0]==point_set_codec_e::constant)
            {
                for(uint32_t prediction_index=0;prediction_index<target_ids.size();prediction_index++)
                {
                    uint32_t reconstructed_id=prediction_data_uses_prediction_order?prediction_index:(uint32_t)target_ids[prediction_index];
                    reconstructed_data[reconstructed_id]=point_set_min_data_vec[0];
                }
                return result;
            }
            vector<int32_t>&qbs=result.point_set_qb_vec[0];
            double base_abs_eb=point_set_abs_eb_vec[0];
            auto quantize_prediction=[&](uint32_t prediction_index,T_data prediction)
            {
                int pred_id=target_ids[prediction_index];
                double point_abs_eb=base_abs_eb*pred_eb_multiplier_vec[prediction_index];
                uint32_t reconstructed_id=prediction_data_uses_prediction_order?prediction_index:(uint32_t)pred_id;
                double inverse_double_abs_eb=inverse_double_base_abs_eb[0]*inverse_eb_multiplier_vec[prediction_index];
                qbs[prediction_index]=quantize_or_store_unpredictable(current_data[pred_id],prediction,point_abs_eb,inverse_double_abs_eb,result.unpredictable_value_vec,reconstructed_data[reconstructed_id]);
            };
            uint32_t general_target_num=min<uint32_t>(prediction_max_ref_num,(uint32_t)target_ids.size());
            for(uint32_t prediction_index=0;prediction_index<general_target_num;prediction_index++)
            {
                typename prediction_structure_c<T_calc>::reference_range_c reference_range=prediction_structure.get_reference_range(prediction_index,prediction_max_ref_num,standard_reference_layout);
                quantize_prediction(prediction_index,predict_value_one_point_set(reference_range.begin,reference_range.num,reference_ids,prediction_weights,reconstructed_data));
            }
            uint64_t fixed_ref_begin=(uint64_t)prediction_max_ref_num*(prediction_max_ref_num-1)/2;
            if(prediction_max_ref_num==4&&standard_reference_layout)
            {
                for(uint32_t prediction_index=general_target_num;prediction_index<target_ids.size();prediction_index++,fixed_ref_begin+=4)
                    quantize_prediction(prediction_index,predict_value_one_point_set_4(fixed_ref_begin,reference_ids,prediction_weights,reconstructed_data));
            }
            else if(prediction_max_ref_num==10&&standard_reference_layout)
            {
                for(uint32_t prediction_index=general_target_num;prediction_index<target_ids.size();prediction_index++,fixed_ref_begin+=10)
                    quantize_prediction(prediction_index,predict_value_one_point_set_10(fixed_ref_begin,reference_ids,prediction_weights,reconstructed_data));
            }
            else
            {
                for(uint32_t prediction_index=general_target_num;prediction_index<target_ids.size();prediction_index++)
                {
                    typename prediction_structure_c<T_calc>::reference_range_c reference_range=prediction_structure.get_reference_range(prediction_index,prediction_max_ref_num,false);
                    quantize_prediction(prediction_index,predict_value_one_point_set(reference_range.begin,reference_range.num,reference_ids,prediction_weights,reconstructed_data));
                }
            }
            return result;
        }
        constexpr time_dimension_mode_e value_mode=fixed_value_mode;
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
                if constexpr(fixed_predictor==point_set_codec_e::idw)
                    return predict_value_two_point_sets_idw(ref_ids,ref_weights,data,point_set_id,valid_ref_mask);
                return predict_value_two_point_sets_rbf(!rbf_fallback[prediction_index],ref_ids,ref_weights,data,point_set_id,valid_ref_mask);
            };
            T_data pred=0,temporal_prediction=0;
            bool use_cross_time_prediction=false;
            if constexpr(value_mode==time_dimension_mode_e::constant_extrapolation||value_mode==time_dimension_mode_e::linear_interpolation||value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
            {
                use_cross_time_prediction=predict_temporal_value<value_mode>(storage_id,point_set_id,temporal_prediction);
                pred=temporal_prediction;
                if constexpr(value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                    if(use_cross_time_prediction)
                        pred+=predict_spatial_value(current_time_spatial_residual_data,&current_time_spatial_residual_valid);
            }
            if(!use_cross_time_prediction)
                pred=predict_spatial_value(reconstructed_data);
            result.cross_time_prediction_num+=use_cross_time_prediction;
            double point_abs_eb=point_set_abs_eb_vec[point_set_id]*pred_eb_multiplier_vec[prediction_index];
            double inverse_double_abs_eb=inverse_double_base_abs_eb[point_set_id]*inverse_eb_multiplier_vec[prediction_index];
            int32_t qb=quantize_or_store_unpredictable(current_data[pred_id],pred,point_abs_eb,inverse_double_abs_eb,result.unpredictable_value_vec,reconstructed_data[storage_id]);
            if constexpr(two_point_sets)
                result.point_set_qb_vec[point_set_id].push_back(qb);
            else
                result.point_set_qb_vec[0][prediction_index]=qb;
            store_temporal_residual<fixed_value_mode>(storage_id,use_cross_time_prediction,temporal_prediction);
        }
        return result;
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::compress_values(compression_file_state_c&state)
    {
        constexpr time_dimension_mode_e value_mode=fixed_value_mode;
        current_decompressed_data.resize(point_num);
        if(prediction_data_uses_prediction_order)
            prediction_order_decompressed_data.resize(point_num);
        auto stage_start_time=chrono::steady_clock::now();
        vector<uint64_t> cross_time_prediction_num_vec(prediction_structure_vec.size(),0);
        vector<double> prediction_duration_vec(prediction_structure_vec.size()),encoding_duration_vec(prediction_structure_vec.size());
        exception_ptr prediction_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(prediction_structure_vec.size()>1&&thread_num>1)
        for(int64_t structure_index=0;structure_index<(int64_t)prediction_structure_vec.size();structure_index++)
        {
            try
            {
                auto prediction_start_time=chrono::steady_clock::now();
                value_compression_result_c<T_data> result=compress_structure_values<two_point_sets,fixed_value_mode,fixed_predictor>(prediction_structure_vec[structure_index]);
                prediction_duration_vec[structure_index]=chrono::duration<double>(chrono::steady_clock::now()-prediction_start_time).count();
                cross_time_prediction_num_vec[structure_index]=result.cross_time_prediction_num;
                vector<uint8_t> empty_label_symbols;
                auto encoding_start_time=chrono::steady_clock::now();
                state.encoded_payload_vec[structure_index]=encode_compression_payload(empty_label_symbols,state.has_label,result.point_set_qb_vec,result.unpredictable_value_vec);
                encoding_duration_vec[structure_index]=chrono::duration<double>(chrono::steady_clock::now()-encoding_start_time).count();
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
        state.cross_time_prediction_num=accumulate(cross_time_prediction_num_vec.begin(),cross_time_prediction_num_vec.end(),(uint64_t)0);
        if constexpr(value_mode==time_dimension_mode_e::constant_extrapolation||value_mode==time_dimension_mode_e::linear_interpolation||value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
            printf("cross_time_prediction_num=%llu, ratio=%f\n",(unsigned long long)state.cross_time_prediction_num,(double)state.cross_time_prediction_num/point_num);
        double combined_wall_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        double prediction_cpu_time=accumulate(prediction_duration_vec.begin(),prediction_duration_vec.end(),0.0);
        double encoding_cpu_time=accumulate(encoding_duration_vec.begin(),encoding_duration_vec.end(),0.0);
        double measured_cpu_time=prediction_cpu_time+encoding_cpu_time;
        state.value_prediction_time=measured_cpu_time>0?combined_wall_time*prediction_cpu_time/measured_cpu_time:0;
        state.lossless_encoding_time=combined_wall_time-state.value_prediction_time;
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::compress_labels_and_values(compression_file_state_c&state)
    {
        constexpr bool try_time_label=fixed_label_mode==time_dimension_mode_e::constant_extrapolation||fixed_label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual;
        constexpr bool try_time_value=fixed_value_mode==time_dimension_mode_e::constant_extrapolation||fixed_value_mode==time_dimension_mode_e::linear_interpolation||fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual;
        constexpr bool use_spatial_residual=fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual;
        array<double,2> inverse_double_base_abs_eb{{0,0}};
        for(uint32_t point_set_id=0;point_set_id<point_set_codec_vec.size();point_set_id++)
            if(point_set_codec_vec[point_set_id]!=point_set_codec_e::constant)
                inverse_double_base_abs_eb[point_set_id]=0.5/point_set_abs_eb_vec[point_set_id];
        vector<T_data>&reconstructed_data=prediction_data_uses_prediction_order?prediction_order_decompressed_data:current_decompressed_data;
        bool label_symbols_use_original_order=prediction_structure_vec.size()==1;
        auto stage_start_time=chrono::steady_clock::now();
        auto compress_targets=[&](const prediction_structure_c<T_calc>&prediction_structure,vector<uint8_t>&label_symbols,vector<vector<int32_t>>&point_set_qbs,vector<T_data>&unpredictable_values)
        {
            const vector<int>&target_ids=prediction_structure.target_ids;
            const vector<int>&ref_ids_vec=prediction_structure.reference_ids;
            const vector<T_calc>&weights=prediction_structure.prediction_weights;
            const vector<uint8_t>&rbf_fallback=prediction_structure.rbf_fallback_flags;
            const vector<double>&eb_multipliers=prediction_structure.eb_multiplier;
            const vector<double>&inverse_eb_multipliers=prediction_structure.inverse_eb_multiplier;
            bool standard_reference_layout=prediction_structure.uses_standard_reference_layout(prediction_max_ref_num);
            if(label_symbols_use_original_order)
                label_symbols.assign(target_ids.size(),0);
            else
                label_symbols.reserve(target_ids.size());
            uint64_t local_label_error_num=0,local_cross_time_prediction_num=0;
            for(uint32_t prediction_index=0;prediction_index<target_ids.size();prediction_index++)
            {
                int pred_id=target_ids[prediction_index];
                uint32_t storage_id=prediction_data_uses_prediction_order?prediction_index:(uint32_t)pred_id;
                uint32_t actual_point_set_id=point_set_id_vec[pred_id];
                bool value_requires_prediction=point_set_codec_vec[actual_point_set_id]!=point_set_codec_e::constant;
                uint32_t predicted_label=0;
                T_data predicted_value=0,temporal_prediction=0;
                bool used_time_label=false,used_time_value=false;
                if constexpr(try_time_label)
                    used_time_label=predict_temporal_label<fixed_label_mode>(storage_id,0,predicted_label);
                if(used_time_label)
                {
                    if(value_requires_prediction)
                    {
                        bool has_rbf_weights=true;
                        if constexpr(fixed_predictor==point_set_codec_e::rbf)
                            has_rbf_weights=!rbf_fallback[prediction_index];
                        auto predict_spatial_value=[&](const vector<T_data>&data,const vector<uint8_t>*valid_ref_mask=nullptr)
                        {
                            typename prediction_structure_c<T_calc>::reference_range_c reference_range=prediction_structure.get_reference_range(prediction_index,prediction_max_ref_num,standard_reference_layout);
                            array_view_c<int> ref_ids(reference_range.num?ref_ids_vec.data()+reference_range.begin:nullptr,reference_range.num);
                            array_view_c<T_calc> ref_weights(reference_range.num?weights.data()+reference_range.begin:nullptr,reference_range.num);
                            if constexpr(fixed_predictor==point_set_codec_e::rbf)
                                return predict_value_two_point_sets_rbf(has_rbf_weights,ref_ids,ref_weights,data,actual_point_set_id,valid_ref_mask);
                            else
                                return predict_value_two_point_sets_idw(ref_ids,ref_weights,data,actual_point_set_id,valid_ref_mask);
                        };
                        if constexpr(try_time_value)
                            used_time_value=predict_temporal_value<fixed_value_mode>(storage_id,actual_point_set_id,temporal_prediction);
                        predicted_value=temporal_prediction;
                        if(used_time_value)
                        {
                            if constexpr(use_spatial_residual)
                                predicted_value+=predict_spatial_value(current_time_spatial_residual_data,&current_time_spatial_residual_valid);
                        }
                        else
                            predicted_value=predict_spatial_value(reconstructed_data);
                    }
                }
                else
                {
                    typename prediction_structure_c<T_calc>::reference_range_c reference_range=prediction_structure.get_reference_range(prediction_index,prediction_max_ref_num,standard_reference_layout);
                    array_view_c<int> ref_ids(reference_range.num?ref_ids_vec.data()+reference_range.begin:nullptr,reference_range.num);
                    array_view_c<T_calc> ref_weights(reference_range.num?weights.data()+reference_range.begin:nullptr,reference_range.num);
                    bool has_rbf_weights=true;
                    if constexpr(fixed_predictor==point_set_codec_e::rbf)
                        has_rbf_weights=!rbf_fallback[prediction_index];
                    if constexpr(try_time_value)
                        if(value_requires_prediction)
                            used_time_value=predict_temporal_value<fixed_value_mode>(storage_id,actual_point_set_id,temporal_prediction);
                    predicted_value=temporal_prediction;
                    array<spatial_value_source_c<T_data>,2> spatial_value_sources;
                    if(value_requires_prediction)
                    {
                        if(used_time_value)
                        {
                            if constexpr(use_spatial_residual)
                                spatial_value_sources[actual_point_set_id]={&current_time_spatial_residual_data,&current_time_spatial_residual_valid};
                        }
                        else
                            spatial_value_sources[actual_point_set_id].data=&reconstructed_data;
                    }
                    bool needs_spatial_value_prediction=spatial_value_sources[actual_point_set_id].data!=nullptr;
                    fused_spatial_prediction_c<T_data> spatial_prediction;
                    if constexpr(fixed_predictor==point_set_codec_e::rbf)
                        spatial_prediction=predict_fused_spatial_rbf(has_rbf_weights,ref_ids,ref_weights,true,spatial_value_sources);
                    else
                        spatial_prediction=predict_fused_spatial_idw(ref_ids,ref_weights,true,spatial_value_sources);
                    predicted_label=spatial_prediction.label;
                    if(needs_spatial_value_prediction)
                    {
                        if(used_time_value)
                            predicted_value+=spatial_prediction.values[actual_point_set_id];
                        else
                            predicted_value=spatial_prediction.values[actual_point_set_id];
                    }
                }
                local_cross_time_prediction_num+=used_time_value;
                bool prediction_error=actual_point_set_id!=predicted_label;
                if(label_symbols_use_original_order)
                    label_symbols[pred_id]=prediction_error?1:0;
                else
                    label_symbols.push_back(prediction_error?1:0);
                local_label_error_num+=prediction_error;
                if(!value_requires_prediction)
                    reconstructed_data[storage_id]=point_set_min_data_vec[actual_point_set_id];
                else
                {
                    double point_abs_eb=point_set_abs_eb_vec[actual_point_set_id]*eb_multipliers[prediction_index];
                    double inverse_double_abs_eb=inverse_double_base_abs_eb[actual_point_set_id]*inverse_eb_multipliers[prediction_index];
                    point_set_qbs[actual_point_set_id].push_back(quantize_or_store_unpredictable(current_data[pred_id],predicted_value,point_abs_eb,inverse_double_abs_eb,unpredictable_values,reconstructed_data[storage_id]));
                }
                if(value_requires_prediction)
                    store_temporal_residual<fixed_value_mode>(storage_id,used_time_value,temporal_prediction);
            }
            return pair<uint64_t,uint64_t>(local_label_error_num,local_cross_time_prediction_num);
        };
        vector<pair<uint64_t,uint64_t>> counts(prediction_structure_vec.size());
        vector<double> prediction_duration_vec(prediction_structure_vec.size()),encoding_duration_vec(prediction_structure_vec.size());
        exception_ptr compression_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(prediction_structure_vec.size()>1&&thread_num>1)
        for(int64_t structure_index=0;structure_index<(int64_t)prediction_structure_vec.size();structure_index++)
        {
            try
            {
                vector<uint8_t> label_symbols;
                vector<vector<int32_t>> point_set_qbs(state.point_set_num);
                vector<T_data> unpredictable_values;
                auto prediction_start_time=chrono::steady_clock::now();
                counts[structure_index]=compress_targets(prediction_structure_vec[structure_index],label_symbols,point_set_qbs,unpredictable_values);
                prediction_duration_vec[structure_index]=chrono::duration<double>(chrono::steady_clock::now()-prediction_start_time).count();
                auto encoding_start_time=chrono::steady_clock::now();
                state.encoded_payload_vec[structure_index]=encode_compression_payload(label_symbols,true,point_set_qbs,unpredictable_values);
                encoding_duration_vec[structure_index]=chrono::duration<double>(chrono::steady_clock::now()-encoding_start_time).count();
            }
            catch(...)
            {
#pragma omp critical
                if(!compression_exception)
                    compression_exception=current_exception();
            }
        }
        if(compression_exception)
            rethrow_exception(compression_exception);
        for(uint32_t structure_id=0;structure_id<counts.size();structure_id++)
        {
            state.label_error_num+=counts[structure_id].first;
            state.cross_time_prediction_num+=counts[structure_id].second;
        }
        state.label_target_num=point_num;
        double combined_wall_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        double prediction_cpu_time=accumulate(prediction_duration_vec.begin(),prediction_duration_vec.end(),0.0);
        double encoding_cpu_time=accumulate(encoding_duration_vec.begin(),encoding_duration_vec.end(),0.0);
        double measured_cpu_time=prediction_cpu_time+encoding_cpu_time;
        state.label_prediction_time=measured_cpu_time>0?combined_wall_time*prediction_cpu_time/measured_cpu_time:0;
        state.lossless_encoding_time=combined_wall_time-state.label_prediction_time;
        if constexpr(fixed_value_mode==time_dimension_mode_e::constant_extrapolation||fixed_value_mode==time_dimension_mode_e::linear_interpolation||fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
            printf("cross_time_prediction_num=%llu, ratio=%f\n",(unsigned long long)state.cross_time_prediction_num,(double)state.cross_time_prediction_num/point_num);
    }

}
