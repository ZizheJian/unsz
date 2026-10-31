#pragma once

namespace unsz
{
    template<typename T_data>
    __attribute__((always_inline)) inline int32_t quantize_or_store_unpredictable(T_data value,T_data prediction,T_data abs_eb,T_data inverse_double_abs_eb,vector<T_data>&unpredictable_values,T_data&reconstructed_value)
    {
        T_data scaled_error=(value-prediction)*inverse_double_abs_eb;
        constexpr T_data min_predictable_quantized_error=(T_data)numeric_limits<int32_t>::min(),max_predictable_quantized_error=(T_data)numeric_limits<int32_t>::max();
        if(scaled_error>min_predictable_quantized_error&&scaled_error<max_predictable_quantized_error)
        {
            int32_t quantized_error=(int32_t)lrintf(scaled_error);
            reconstructed_value=prediction+(T_data)quantized_error*((T_data)2*abs_eb);
            if(abs(value-reconstructed_value)<=abs_eb)
                return quantized_error;
        }
        unpredictable_values.push_back(value);
        reconstructed_value=value;
        return numeric_limits<int32_t>::min();
    }

    // 各prediction structure并行完成预测和编码，只能测到两者合计的墙钟时间。按两部分累计CPU时间的比例把墙钟时间拆开，返回（预测时间，编码时间）。
    inline pair<double,double> split_wall_time(double wall_time,const vector<double>&prediction_durations,const vector<double>&encoding_durations)
    {
        double prediction_cpu_time=accumulate(prediction_durations.begin(),prediction_durations.end(),0.0);
        double encoding_cpu_time=accumulate(encoding_durations.begin(),encoding_durations.end(),0.0);
        double measured_cpu_time=prediction_cpu_time+encoding_cpu_time;
        double prediction_time=measured_cpu_time>0?wall_time*prediction_cpu_time/measured_cpu_time:0;
        return {prediction_time,wall_time-prediction_time};
    }

    template<typename T_coord,typename T_data>
    template<bool two_labels,slice_prediction_mode_e value_mode,slice_prediction_mode_e label_mode>
    void param_c<T_coord,T_data>::compress_current_file()
    {
        compression_file_state_c state;
        prepare_current_compression<two_labels,value_mode,label_mode>(state);
        if constexpr(two_labels)
            compress_labels_and_values<value_mode,label_mode>(state);
        else
            compress_values<value_mode>(state);
        assemble_current_compression_file(state);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::compress_current_file_by_mode()
    {
        dispatch_prediction_framework(get_prediction_framework((uint32_t)label_config_vec.size()),[&](auto two_labels,auto value_mode,auto label_mode)
        {
            this->template compress_current_file<decltype(two_labels)::value,decltype(value_mode)::value,decltype(label_mode)::value>();
        });
    }

    // 把当前slice的点分到各标签，并统计每个标签的点数和min/max。
    // 循环手工展开成4路，各路用独立的min/max累加器，打破相邻迭代间的依赖，编译器才能向量化。
    // 实测（gcc 11.2，-O3 -march=native）单标签快约9倍，两种标签快约1.4倍，结果与普通循环逐位相同。
    template<typename T_coord,typename T_data>
    template<bool two_labels,bool collect_statistics>
    void param_c<T_coord,T_data>::classify_current_labels(compression_file_state_c&state)
    {
        uint32_t &label_num=state.label_num;
        auto stage_start_time=chrono::steady_clock::now();
        uint32_t configured_label_num=(uint32_t)label_config_vec.size();
        label_point_num_vec.assign(configured_label_num,0);
        if constexpr(collect_statistics)
        {
            label_min_data_vec.assign(configured_label_num,numeric_limits<T_data>::infinity());
            label_max_data_vec.assign(configured_label_num,-numeric_limits<T_data>::infinity());
        }
        if constexpr(!two_labels)
        {
            label_num=1;
            label_id_vec.clear();
            label_point_num_vec[0]=point_num;
            if constexpr(collect_statistics)
            {
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
                label_min_data_vec[0]=minimum[0];
                label_max_data_vec[0]=maximum[0];
            }
        }
        else if(current_active_level_counts)
        {
            label_id_vec.resize(point_num);
            uint32_t point_num_per_set[2]={0,0};
            T_data minimum[2]={numeric_limits<T_data>::infinity(),numeric_limits<T_data>::infinity()};
            T_data maximum[2]={-numeric_limits<T_data>::infinity(),-numeric_limits<T_data>::infinity()};
            const uint32_t*active_level_counts=current_active_level_counts->data();
            uint32_t height_index=current_slice%current_tensor_shape[1];
            for(uint32_t point_id=0;point_id<point_num;point_id++)
            {
                T_data value=current_data[point_id];
                uint32_t label_id=classify_structural_label(value,active_level_counts[point_id],height_index,current_active_level_count_offset);
                label_id_vec[point_id]=(uint8_t)label_id;
                point_num_per_set[label_id]++;
                if constexpr(collect_statistics)
                {
                    minimum[label_id]=min(minimum[label_id],value);
                    maximum[label_id]=max(maximum[label_id],value);
                }
            }
            for(uint32_t label_id=0;label_id<2;label_id++)
            {
                label_point_num_vec[label_id]=point_num_per_set[label_id];
                if constexpr(collect_statistics)
                {
                    label_min_data_vec[label_id]=minimum[label_id];
                    label_max_data_vec[label_id]=maximum[label_id];
                }
            }
            label_num=2;
        }
        else if(label_rule_vec.size()==1)
        {
            label_id_vec.resize(point_num);
            const label_rule_c &rule=label_rule_vec[0];
            auto classify_two_labels=[&](auto matches)
            {
                uint32_t point_num_per_set[4][2]={{0,0},{0,0},{0,0},{0,0}};
                T_data minimum[4][2],maximum[4][2];
                if constexpr(collect_statistics)
                    for(uint32_t lane=0;lane<4;lane++)
                        for(uint32_t label_id=0;label_id<2;label_id++)
                        {
                            minimum[lane][label_id]=numeric_limits<T_data>::infinity();
                            maximum[lane][label_id]=-numeric_limits<T_data>::infinity();
                        }
                uint32_t point_id=0;
                for(;point_id+3<point_num;point_id+=4)
                {
                    for(uint32_t lane=0;lane<4;lane++)
                    {
                        T_data value=current_data[point_id+lane];
                        uint32_t label_id=matches((double)value)?0:1;
                        label_id_vec[point_id+lane]=(uint8_t)label_id;
                        point_num_per_set[lane][label_id]++;
                        if constexpr(collect_statistics)
                        {
                            minimum[lane][label_id]=min(minimum[lane][label_id],value);
                            maximum[lane][label_id]=max(maximum[lane][label_id],value);
                        }
                    }
                }
                for(;point_id<point_num;point_id++)
                {
                    T_data value=current_data[point_id];
                    uint32_t label_id=matches((double)value)?0:1;
                    label_id_vec[point_id]=(uint8_t)label_id;
                    point_num_per_set[0][label_id]++;
                    if constexpr(collect_statistics)
                    {
                        minimum[0][label_id]=min(minimum[0][label_id],value);
                        maximum[0][label_id]=max(maximum[0][label_id],value);
                    }
                }
                for(uint32_t lane=1;lane<4;lane++)
                    for(uint32_t label_id=0;label_id<2;label_id++)
                    {
                        point_num_per_set[0][label_id]+=point_num_per_set[lane][label_id];
                        if constexpr(collect_statistics)
                        {
                            minimum[0][label_id]=min(minimum[0][label_id],minimum[lane][label_id]);
                            maximum[0][label_id]=max(maximum[0][label_id],maximum[lane][label_id]);
                        }
                    }
                for(uint32_t label_id=0;label_id<2;label_id++)
                {
                    label_point_num_vec[label_id]=point_num_per_set[0][label_id];
                    if constexpr(collect_statistics)
                    {
                        label_min_data_vec[label_id]=minimum[0][label_id];
                        label_max_data_vec[label_id]=maximum[0][label_id];
                    }
                }
                label_num=2;
            };
            double threshold=rule.value;
            switch(rule.operator_type)
            {
                case label_rule_operator_e::less:
                    classify_two_labels([threshold](double value){return value<threshold;});
                    break;
                case label_rule_operator_e::less_equal:
                    classify_two_labels([threshold](double value){return value<=threshold;});
                    break;
                case label_rule_operator_e::greater:
                    classify_two_labels([threshold](double value){return value>threshold;});
                    break;
                case label_rule_operator_e::greater_equal:
                    classify_two_labels([threshold](double value){return value>=threshold;});
                    break;
                case label_rule_operator_e::equal:
                    classify_two_labels([threshold](double value){return value==threshold;});
                    break;
                case label_rule_operator_e::not_equal:
                    classify_two_labels([threshold](double value){return value!=threshold;});
                    break;
                default:
                    throw runtime_error("未知的标签分类操作符");
            }
        }
        else
            throw runtime_error("双标签压缩要求恰好一条分类规则");
        state.label_classification_time=seconds_since(stage_start_time);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_current_label_codecs_and_eb(compression_file_state_c&state)
    {
        uint32_t label_num=state.label_num;
        auto stage_start_time=chrono::steady_clock::now();
        label_point_num_vec.resize(label_num);
        label_min_data_vec.resize(label_num);
        label_max_data_vec.resize(label_num);
        label_abs_eb_vec.resize(label_num);
        label_codec_vec.resize(label_num);
        for (uint32_t label_id=0;label_id<label_num;label_id++)
        {
            if (label_point_num_vec[label_id]==0)
            {
                label_min_data_vec[label_id]=0;
                label_max_data_vec[label_id]=0;
                label_abs_eb_vec[label_id]=0;
                label_codec_vec[label_id]=label_codec_e::constant;
                continue;
            }
            const label_config_c &config=label_config_vec[label_id];
            bool is_constant=label_min_data_vec[label_id]==label_max_data_vec[label_id];
            if (is_constant)
                label_codec_vec[label_id]=label_codec_e::constant;
            else
                label_codec_vec[label_id]=config.codec;
            if(!is_constant&&label_codec_vec[label_id]==label_codec_e::constant)
                throw runtime_error("非常量标签不能使用常量编码");
            label_abs_eb_vec[label_id]=is_constant?0:label_abs_eb(label_id,false);
            if(!is_constant&&!(label_abs_eb_vec[label_id]>0))
                throw runtime_error("标签"+to_string(label_id)+"的绝对误差上限必须大于0");
        }
        state.label_setup_time=seconds_since(stage_start_time);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::precompute_current_field_label_statistics()
    {
        compression_file_label_min_data_vec.resize(original_data_path_vec.size());
        compression_file_label_max_data_vec.resize(original_data_path_vec.size());
        compression_file_label_abs_eb_vec.resize(original_data_path_vec.size());
        compression_file_label_codec_vec.resize(original_data_path_vec.size());
        current_data.resize(point_num);
        bool two_labels=label_config_vec.size()==2;
        array<uint64_t,2> field_label_point_num{{0,0}};
        array<bool,2> label_is_always_constant{{true,true}};
        field_label_min_data={{numeric_limits<double>::infinity(),numeric_limits<double>::infinity()}};
        field_label_max_data={{-numeric_limits<double>::infinity(),-numeric_limits<double>::infinity()}};
        for(uint32_t slice_index=0;slice_index<original_data_path_vec.size();slice_index++)
        {
            current_slice=slice_vec[slice_index];
            auto input_start_time=chrono::steady_clock::now();
            read_binary_file_exact(original_data_path_vec[slice_index],current_data.data(),point_num);
            label_statistics_input_time+=seconds_since(input_start_time);
            compression_file_state_c state;
            if(two_labels)
                classify_current_labels<true,true>(state);
            else
                classify_current_labels<false,true>(state);
            prepare_current_label_codecs_and_eb(state);
            for(uint32_t label_id=0;label_id<state.label_num;label_id++)
            {
                field_label_point_num[label_id]+=label_point_num_vec[label_id];
                if(label_point_num_vec[label_id]>0&&label_codec_vec[label_id]!=label_codec_e::constant)
                    label_is_always_constant[label_id]=false;
                if(label_point_num_vec[label_id]>0)
                {
                    field_label_min_data[label_id]=min(field_label_min_data[label_id],(double)label_min_data_vec[label_id]);
                    field_label_max_data[label_id]=max(field_label_max_data[label_id],(double)label_max_data_vec[label_id]);
                }
                compression_file_label_min_data_vec[slice_index][label_id]=label_min_data_vec[label_id];
                compression_file_label_max_data_vec[slice_index][label_id]=label_max_data_vec[label_id];
                compression_file_label_abs_eb_vec[slice_index][label_id]=label_abs_eb_vec[label_id];
                compression_file_label_codec_vec[slice_index][label_id]=label_codec_vec[label_id];
            }
        }
        // FIELD_REL用当前field全部slice的整体range换算同一个ABS EB；常量slice仍按常量精确保存
        for(uint32_t label_id=0;label_id<label_config_vec.size();label_id++)
        {
            const label_config_c&config=label_config_vec[label_id];
            if(config.error_mode!=label_error_mode_e::field_relative||field_label_point_num[label_id]==0)
                continue;
            T_data field_abs_eb=label_abs_eb(label_id,true);
            for(uint32_t slice_index=0;slice_index<original_data_path_vec.size();slice_index++)
                if(compression_file_label_codec_vec[slice_index][label_id]!=label_codec_e::constant)
                    compression_file_label_abs_eb_vec[slice_index][label_id]=field_abs_eb;
        }
        if(two_labels&&(field_label_point_num[0]==0||field_label_point_num[1]==0))
        {
            uint32_t source_label_id=field_label_point_num[0]>0?0:1;
            field_label_min_data[0]=field_label_min_data[source_label_id];
            field_label_max_data[0]=field_label_max_data[source_label_id];
            for(uint32_t slice_index=0;slice_index<original_data_path_vec.size();slice_index++)
            {
                compression_file_label_min_data_vec[slice_index][0]=compression_file_label_min_data_vec[slice_index][source_label_id];
                compression_file_label_max_data_vec[slice_index][0]=compression_file_label_max_data_vec[slice_index][source_label_id];
                compression_file_label_abs_eb_vec[slice_index][0]=compression_file_label_abs_eb_vec[slice_index][source_label_id];
                compression_file_label_codec_vec[slice_index][0]=compression_file_label_codec_vec[slice_index][source_label_id];
            }
            label_config_vec={label_config_vec[source_label_id]};
            if(label_is_always_constant[source_label_id])
                label_config_vec[0].codec=label_codec_e::constant;
            label_rule_vec.clear();
            label_source_id_vec={(uint8_t)source_label_id};
            printf("effective label configuration: configured=2,effective=1,source=%u\n",source_label_id);
        }
        else
        {
            if(label_config_vec.size()==1&&label_is_always_constant[0])
                label_config_vec[0].codec=label_codec_e::constant;
            label_source_id_vec.resize(label_config_vec.size());
            iota(label_source_id_vec.begin(),label_source_id_vec.end(),0);
            printf("effective label configuration: configured=%u,effective=%u\n",(uint32_t)configured_label_config_vec.size(),(uint32_t)label_config_vec.size());
        }
        if(label_config_vec.size()==1)
        {
            label_pipeline="independent";
            if(label_config_vec[0].codec==label_codec_e::constant)
                value_pipeline="independent";
            else
                resolve_single_label_pipeline();
        }
        if(current_field_uses_tensor&&value_pipeline=="independent")
        {
            selected_tensor_pipeline={};
            tensor_pipeline_selected=true;
        }
        current_data.clear();
        label_id_vec.clear();
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::load_current_label_statistics(compression_file_state_c&state)
    {
        auto stage_start_time=chrono::steady_clock::now();
        uint32_t label_num=state.label_num;
        if(current_slice_index>=compression_file_label_abs_eb_vec.size())
            throw runtime_error("当前slice缺少预计算的标签统计结果");
        label_min_data_vec.resize(label_num);
        label_max_data_vec.resize(label_num);
        label_abs_eb_vec.resize(label_num);
        label_codec_vec.resize(label_num);
        for(uint32_t label_id=0;label_id<label_num;label_id++)
        {
            label_min_data_vec[label_id]=compression_file_label_min_data_vec[current_slice_index][label_id];
            label_max_data_vec[label_id]=compression_file_label_max_data_vec[current_slice_index][label_id];
            label_abs_eb_vec[label_id]=compression_file_label_abs_eb_vec[current_slice_index][label_id];
            label_codec_vec[label_id]=compression_file_label_codec_vec[current_slice_index][label_id];
        }
        state.label_setup_time=seconds_since(stage_start_time);
    }

    template<typename T_coord,typename T_data>
    template<bool two_labels,slice_prediction_mode_e fixed_value_mode,slice_prediction_mode_e fixed_label_mode>
    void param_c<T_coord,T_data>::prepare_current_prediction_state(compression_file_state_c&state)
    {
        prediction_framework_c prediction_framework;
        prediction_framework.has_label=two_labels;
        prediction_framework.value_mode=fixed_value_mode;
        prediction_framework.label_mode=fixed_label_mode;
        current_prediction_framework=prediction_framework;
        if(prediction_data_uses_prediction_order&&two_labels)
        {
            prediction_order_label_id_vec.resize(point_num);
            for(const prediction_structure_c<T_calc>&structure:prediction_structure_vec)
                for(uint32_t prediction_index=0;prediction_index<structure.target_ids.size();prediction_index++)
                    prediction_order_label_id_vec[structure.prediction_order_offset+prediction_index]=label_id_vec[structure.target_ids[prediction_index]];
        }
        prepare_tensor_prediction();
        if constexpr(two_labels)
        {
            if(state.label_num!=2)
                throw runtime_error("融合两种标签预测要求当前文件恰好包含两个标签");
            for(label_codec_e codec:label_codec_vec)
                if(codec!=label_codec_e::constant&&codec!=label_codec_e::rbf)
                    throw runtime_error("融合两种标签预测只支持constant或RBF预测器");
        }
    }

    template<typename T_coord,typename T_data>
    template<bool two_labels>
    void param_c<T_coord,T_data>::initialize_current_compression_buffers(compression_file_state_c&state)
    {
        uint32_t label_num=state.label_num;
        state.has_label=label_num==2;
        uint32_t structure_num=prediction_structure_vec.size();
        state.encoded_payload_vec.resize(structure_num);
        state.label_target_num=0;
        state.label_error_num=0;
        if constexpr(two_labels)
        {
            current_decompressed_data.resize(point_num);
            if(prediction_data_uses_prediction_order)
                prediction_order_decompressed_data.resize(point_num);
        }
    }

    template<typename T_coord,typename T_data>
    template<bool two_labels,slice_prediction_mode_e fixed_value_mode,slice_prediction_mode_e fixed_label_mode>
    void param_c<T_coord,T_data>::prepare_current_compression(compression_file_state_c&state)
    {
        classify_current_labels<two_labels,false>(state);
        load_current_label_statistics(state);
        prepare_current_prediction_state<two_labels,fixed_value_mode,fixed_label_mode>(state);
        initialize_current_compression_buffers<two_labels>(state);
    }

    // 单标签时逐点压缩一个prediction structure。
    // 单标签的跨slice预测对整个slice要么都可用、要么都不可用（如第一个slice），因此在循环外判断一次。
    // 可用时预测值为跨slice预测，空间残差方式再加上对邻近点跨slice残差的空间预测。不可用时只用空间预测。
    template<typename T_coord,typename T_data>
    template<slice_prediction_mode_e value_mode>
    value_compression_result_c<T_data> param_c<T_coord,T_data>::compress_structure_values(const prediction_structure_c<T_calc>&prediction_structure)
    {
        constexpr bool use_spatial_residual=value_mode==slice_prediction_mode_e::extrapolation_spatial_residual||value_mode==slice_prediction_mode_e::interpolation_spatial_residual;
        const vector<int>&target_ids=prediction_structure.target_ids;
        const vector<int>&reference_ids=prediction_structure.reference_ids;
        const vector<T_calc>&prediction_weights=prediction_structure.prediction_weights;
        const vector<T_calc>&pred_eb_multiplier_vec=prediction_structure.eb_multiplier;
        const vector<T_calc>&inverse_eb_multiplier_vec=prediction_structure.inverse_eb_multiplier;
        bool standard_reference_layout=prediction_structure.uses_standard_reference_layout(rbf_reference_num);
        vector<T_data>&reconstructed_data=prediction_data_uses_prediction_order?prediction_order_decompressed_data:current_decompressed_data;
        value_compression_result_c<T_data> result;
        result.label_quantized_error_vec.resize(1);
        if(label_codec_vec[0]==label_codec_e::constant)
        {
            for(uint32_t prediction_index=0;prediction_index<target_ids.size();prediction_index++)
                reconstructed_data[prediction_structure.get_storage_id(prediction_index,target_ids[prediction_index],prediction_data_uses_prediction_order)]=label_min_data_vec[0];
            return result;
        }
        vector<int32_t>&quantized_errors=result.label_quantized_error_vec[0];
        quantized_errors.resize(target_ids.size());
        T_data base_abs_eb=label_abs_eb_vec[0],inverse_double_base_abs_eb=(T_data)0.5/label_abs_eb_vec[0];
        auto cross_slice_predictor=get_single_label_tensor_predictor<value_mode>();
        bool use_cross_slice_prediction=cross_slice_predictor.first;
        const auto&predict_cross_slice=cross_slice_predictor.second;
        // 跨slice预测是否可用对整个slice相同，因此生成可用和不可用两个循环版本，循环内不再判断。
        auto compress_targets=[&](auto use_cross_slice_constant)
        {
            constexpr bool use_cross_slice=decltype(use_cross_slice_constant)::value;
            auto compress_target=[&](uint32_t prediction_index,uint64_t ref_begin,uint32_t ref_num) __attribute__((always_inline))
            {
                int pred_id=target_ids[prediction_index];
                uint32_t storage_id=prediction_structure.get_storage_id(prediction_index,pred_id,prediction_data_uses_prediction_order);
                T_data cross_slice_prediction=0,prediction;
                if constexpr(use_cross_slice)
                {
                    cross_slice_prediction=predict_cross_slice(storage_id);
                    prediction=cross_slice_prediction;
                    if constexpr(use_spatial_residual)
                        prediction+=predict_value_single_label(ref_begin,ref_num,reference_ids,prediction_weights,current_slice_spatial_residual_data);
                }
                else
                    prediction=predict_value_single_label(ref_begin,ref_num,reference_ids,prediction_weights,reconstructed_data);
                T_data point_abs_eb=base_abs_eb*pred_eb_multiplier_vec[prediction_index];
                T_data inverse_double_abs_eb=inverse_double_base_abs_eb*inverse_eb_multiplier_vec[prediction_index];
                quantized_errors[prediction_index]=quantize_or_store_unpredictable(current_data[pred_id],prediction,point_abs_eb,inverse_double_abs_eb,result.unpredictable_value_vec,reconstructed_data[storage_id]);
                // 单标签时同一slice的点全部使用或全部不用跨slice预测，残差有效标志只有两种标签会读取，这里不必写。
                if constexpr(use_spatial_residual&&use_cross_slice)
                    current_slice_spatial_residual_data[storage_id]=reconstructed_data[storage_id]-cross_slice_prediction;
            };
            // 标准参考布局时，前rbf_reference_num个目标之后每个目标恰好有rbf_reference_num个参考点，第二个循环直接递推参考区间，并固定使用10点预测。
            uint32_t target_num=(uint32_t)target_ids.size(),general_target_num=standard_reference_layout?min<uint32_t>(rbf_reference_num,target_num):target_num;
            for(uint32_t prediction_index=0;prediction_index<general_target_num;prediction_index++)
            {
                typename prediction_structure_c<T_calc>::reference_range_c reference_range=prediction_structure.get_reference_range(prediction_index,rbf_reference_num,false);
                compress_target(prediction_index,reference_range.begin,reference_range.num);
            }
            uint64_t ref_begin=(uint64_t)rbf_reference_num*(rbf_reference_num-1)/2;
            for(uint32_t prediction_index=general_target_num;prediction_index<target_num;prediction_index++,ref_begin+=rbf_reference_num)
                compress_target(prediction_index,ref_begin,rbf_reference_num);
        };
        if(use_cross_slice_prediction)
            compress_targets(true_type());
        else
            compress_targets(false_type());
        result.tensor_prediction_num=use_cross_slice_prediction?target_ids.size():0;
        return result;
    }

    template<typename T_coord,typename T_data>
    template<slice_prediction_mode_e value_mode>
    void param_c<T_coord,T_data>::compress_values(compression_file_state_c&state)
    {
        current_decompressed_data.resize(point_num);
        if(prediction_data_uses_prediction_order)
            prediction_order_decompressed_data.resize(point_num);
        auto stage_start_time=chrono::steady_clock::now();
        vector<uint64_t> tensor_prediction_num_vec(prediction_structure_vec.size(),0);
        vector<double> prediction_duration_vec(prediction_structure_vec.size()),encoding_duration_vec(prediction_structure_vec.size());
        parallel_for_tasks((uint32_t)prediction_structure_vec.size(),thread_num,[&](uint32_t structure_index)
        {
            auto prediction_start_time=chrono::steady_clock::now();
            value_compression_result_c<T_data> result=compress_structure_values<value_mode>(prediction_structure_vec[structure_index]);
            prediction_duration_vec[structure_index]=seconds_since(prediction_start_time);
            tensor_prediction_num_vec[structure_index]=result.tensor_prediction_num;
            vector<uint8_t> empty_label_symbols;
            auto encoding_start_time=chrono::steady_clock::now();
            state.encoded_payload_vec[structure_index]=encode_compression_payload(empty_label_symbols,state.has_label,result.label_quantized_error_vec,result.unpredictable_value_vec);
            encoding_duration_vec[structure_index]=seconds_since(encoding_start_time);
        });
        state.tensor_prediction_num=accumulate(tensor_prediction_num_vec.begin(),tensor_prediction_num_vec.end(),(uint64_t)0);
        tie(state.value_prediction_time,state.lossless_encoding_time)=split_wall_time(seconds_since(stage_start_time),prediction_duration_vec,encoding_duration_vec);
    }

    template<typename T_coord,typename T_data>
    template<slice_prediction_mode_e fixed_value_mode,slice_prediction_mode_e fixed_label_mode>
    void param_c<T_coord,T_data>::compress_labels_and_values(compression_file_state_c&state)
    {
        constexpr bool try_tensor_label=fixed_label_mode==slice_prediction_mode_e::extrapolation||fixed_label_mode==slice_prediction_mode_e::label_interpolation_spatial_residual;
        constexpr bool try_tensor_value=fixed_value_mode==slice_prediction_mode_e::extrapolation||fixed_value_mode==slice_prediction_mode_e::interpolation||fixed_value_mode==slice_prediction_mode_e::extrapolation_spatial_residual||fixed_value_mode==slice_prediction_mode_e::interpolation_spatial_residual;
        constexpr bool use_spatial_residual=fixed_value_mode==slice_prediction_mode_e::extrapolation_spatial_residual||fixed_value_mode==slice_prediction_mode_e::interpolation_spatial_residual;
        array<T_data,2> inverse_double_base_abs_eb{{0,0}};
        for(uint32_t label_id=0;label_id<label_codec_vec.size();label_id++)
            if(label_codec_vec[label_id]!=label_codec_e::constant)
                inverse_double_base_abs_eb[label_id]=(T_data)0.5/label_abs_eb_vec[label_id];
        vector<T_data>&reconstructed_data=prediction_data_uses_prediction_order?prediction_order_decompressed_data:current_decompressed_data;
        bool label_symbols_use_original_order=prediction_structure_vec.size()==1;
        auto stage_start_time=chrono::steady_clock::now();
        auto compress_targets=[&](const prediction_structure_c<T_calc>&prediction_structure,vector<uint8_t>&label_symbols,vector<vector<int32_t>>&label_quantized_errors,vector<T_data>&unpredictable_values)
        {
            const vector<int>&target_ids=prediction_structure.target_ids;
            const vector<int>&ref_ids_vec=prediction_structure.reference_ids;
            const vector<T_calc>&weights=prediction_structure.prediction_weights;
            const vector<uint8_t>&rbf_fallback=prediction_structure.rbf_fallback_flags;
            const vector<T_calc>&eb_multipliers=prediction_structure.eb_multiplier;
            const vector<T_calc>&inverse_eb_multipliers=prediction_structure.inverse_eb_multiplier;
            bool standard_reference_layout=prediction_structure.uses_standard_reference_layout(rbf_reference_num);
            if(label_symbols_use_original_order)
                label_symbols.assign(target_ids.size(),0);
            else
                label_symbols.reserve(target_ids.size());
            uint64_t local_label_error_num=0,local_tensor_prediction_num=0;
            for(uint32_t prediction_index=0;prediction_index<target_ids.size();prediction_index++)
            {
                int pred_id=target_ids[prediction_index];
                uint32_t storage_id=prediction_structure.get_storage_id(prediction_index,pred_id,prediction_data_uses_prediction_order);
                uint32_t actual_label_id=label_id_vec[pred_id];
                bool value_requires_prediction=label_codec_vec[actual_label_id]!=label_codec_e::constant;
                uint32_t predicted_label=0;
                T_data predicted_value=0,tensor_prediction=0;
                bool used_tensor_label=false,used_tensor_value=false;
                if constexpr(try_tensor_label)
                    used_tensor_label=predict_tensor_label<fixed_label_mode>(storage_id,0,predicted_label);
                if(used_tensor_label)
                {
                    if(value_requires_prediction)
                    {
                        bool has_rbf_weights=!rbf_fallback[prediction_index];
                        auto predict_spatial_value=[&](const vector<T_data>&data,const vector<uint8_t>*valid_ref_mask=nullptr)
                        {
                            typename prediction_structure_c<T_calc>::reference_range_c reference_range=prediction_structure.get_reference_range(prediction_index,rbf_reference_num,standard_reference_layout);
                            array_view_c<int> ref_ids(reference_range.num?ref_ids_vec.data()+reference_range.begin:nullptr,reference_range.num);
                            array_view_c<T_calc> ref_weights(reference_range.num?weights.data()+reference_range.begin:nullptr,reference_range.num);
                            return predict_value_with_labels_rbf(has_rbf_weights,ref_ids,ref_weights,data,actual_label_id,valid_ref_mask);
                        };
                        if constexpr(try_tensor_value)
                            used_tensor_value=predict_tensor_value<fixed_value_mode>(storage_id,actual_label_id,tensor_prediction);
                        predicted_value=tensor_prediction;
                        if(used_tensor_value)
                        {
                            if constexpr(use_spatial_residual)
                                predicted_value+=predict_spatial_value(current_slice_spatial_residual_data,&current_slice_spatial_residual_valid);
                        }
                        else
                            predicted_value=predict_spatial_value(reconstructed_data);
                    }
                }
                else
                {
                    typename prediction_structure_c<T_calc>::reference_range_c reference_range=prediction_structure.get_reference_range(prediction_index,rbf_reference_num,standard_reference_layout);
                    array_view_c<int> ref_ids(reference_range.num?ref_ids_vec.data()+reference_range.begin:nullptr,reference_range.num);
                    array_view_c<T_calc> ref_weights(reference_range.num?weights.data()+reference_range.begin:nullptr,reference_range.num);
                    bool has_rbf_weights=!rbf_fallback[prediction_index];
                    if constexpr(try_tensor_value)
                        if(value_requires_prediction)
                            used_tensor_value=predict_tensor_value<fixed_value_mode>(storage_id,actual_label_id,tensor_prediction);
                    predicted_value=tensor_prediction;
                    array<spatial_value_source_c<T_data>,2> spatial_value_sources;
                    if(value_requires_prediction)
                    {
                        if(used_tensor_value)
                        {
                            if constexpr(use_spatial_residual)
                                spatial_value_sources[actual_label_id]={&current_slice_spatial_residual_data,&current_slice_spatial_residual_valid};
                        }
                        else
                            spatial_value_sources[actual_label_id].data=&reconstructed_data;
                    }
                    bool needs_spatial_value_prediction=spatial_value_sources[actual_label_id].data!=nullptr;
                    fused_spatial_prediction_c<T_data> spatial_prediction;
                    spatial_prediction=predict_fused_spatial_rbf(has_rbf_weights,ref_ids,ref_weights,true,spatial_value_sources);
                    predicted_label=spatial_prediction.label;
                    if(needs_spatial_value_prediction)
                    {
                        if(used_tensor_value)
                            predicted_value+=spatial_prediction.values[actual_label_id];
                        else
                            predicted_value=spatial_prediction.values[actual_label_id];
                    }
                }
                local_tensor_prediction_num+=used_tensor_value;
                bool prediction_error=actual_label_id!=predicted_label;
                if(label_symbols_use_original_order)
                    label_symbols[pred_id]=prediction_error?1:0;
                else
                    label_symbols.push_back(prediction_error?1:0);
                local_label_error_num+=prediction_error;
                if(!value_requires_prediction)
                    reconstructed_data[storage_id]=label_min_data_vec[actual_label_id];
                else
                {
                    T_data point_abs_eb=label_abs_eb_vec[actual_label_id]*eb_multipliers[prediction_index];
                    T_data inverse_double_abs_eb=inverse_double_base_abs_eb[actual_label_id]*inverse_eb_multipliers[prediction_index];
                    label_quantized_errors[actual_label_id].push_back(quantize_or_store_unpredictable(current_data[pred_id],predicted_value,point_abs_eb,inverse_double_abs_eb,unpredictable_values,reconstructed_data[storage_id]));
                }
                if(value_requires_prediction)
                    store_tensor_residual<fixed_value_mode>(storage_id,used_tensor_value,tensor_prediction);
            }
            return pair<uint64_t,uint64_t>(local_label_error_num,local_tensor_prediction_num);
        };
        vector<pair<uint64_t,uint64_t>> counts(prediction_structure_vec.size());
        vector<double> prediction_duration_vec(prediction_structure_vec.size()),encoding_duration_vec(prediction_structure_vec.size());
        parallel_for_tasks((uint32_t)prediction_structure_vec.size(),thread_num,[&](uint32_t structure_index)
        {
            vector<uint8_t> label_symbols;
            vector<vector<int32_t>> label_quantized_errors(state.label_num);
            vector<T_data> unpredictable_values;
            auto prediction_start_time=chrono::steady_clock::now();
            counts[structure_index]=compress_targets(prediction_structure_vec[structure_index],label_symbols,label_quantized_errors,unpredictable_values);
            prediction_duration_vec[structure_index]=seconds_since(prediction_start_time);
            auto encoding_start_time=chrono::steady_clock::now();
            state.encoded_payload_vec[structure_index]=encode_compression_payload(label_symbols,true,label_quantized_errors,unpredictable_values);
            encoding_duration_vec[structure_index]=seconds_since(encoding_start_time);
        });
        for(uint32_t structure_id=0;structure_id<counts.size();structure_id++)
        {
            state.label_error_num+=counts[structure_id].first;
            state.tensor_prediction_num+=counts[structure_id].second;
        }
        state.label_target_num=point_num;
        tie(state.label_prediction_time,state.lossless_encoding_time)=split_wall_time(seconds_since(stage_start_time),prediction_duration_vec,encoding_duration_vec);
    }

}
