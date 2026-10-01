#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    template<typename T_system>
    bool param_c<T_coord,T_data>::solve_linear_system(vector<vector<T_system>> &A,vector<T_system> &b,vector<T_system> &x)
    {
        uint32_t n=A.size();
        if (n==0 || b.size()!=n)
            return false;
        for (auto &row:A)
            if (row.size()!=n)
                return false;
        for (uint32_t col=0;col<n;col++)
        {
            uint32_t pivot=col;
            for (uint32_t row=col+1;row<n;row++)
                if (abs(A[row][col])>abs(A[pivot][col]))
                    pivot=row;
            if(abs(A[pivot][col])<(T_system)1e-12)
                return false;
            if (pivot!=col)
            {
                swap(A[col],A[pivot]);
                swap(b[col],b[pivot]);
            }
            T_system divisor=A[col][col];
            for (uint32_t i=col;i<n;i++)
                A[col][i]/=divisor;
            b[col]/=divisor;
            for (uint32_t row=0;row<n;row++)
            {
                if (row==col)
                    continue;
                T_system factor=A[row][col];
                for (uint32_t i=col;i<n;i++)
                    A[row][i]-=factor*A[col][i];
                b[row]-=factor*b[col];
            }
        }
        x.assign(b.begin(),b.end());
        return true;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::dispatch_two_point_set_compression_by_mode()
    {
        if(point_set_config_vec.size()!=2)
            throw runtime_error("两点集压缩要求field恰好配置两个点集");
        prediction_framework_c framework=get_prediction_framework(2);
        if(framework.execution==label_value_execution_e::staged)
        {
            auto dispatch_value=[&](auto label_mode_constant)
            {
                constexpr time_dimension_mode_e label_mode=decltype(label_mode_constant)::value;
                switch(framework.value_mode)
                {
                    case time_dimension_mode_e::spatial:
                        return dispatch_two_point_set_compression_by_predictor<label_mode,time_dimension_mode_e::spatial,label_value_execution_e::staged>();
                    case time_dimension_mode_e::constant_extrapolation:
                        return dispatch_two_point_set_compression_by_predictor<label_mode,time_dimension_mode_e::constant_extrapolation,label_value_execution_e::staged>();
                    case time_dimension_mode_e::linear_interpolation:
                        return dispatch_two_point_set_compression_by_predictor<label_mode,time_dimension_mode_e::linear_interpolation,label_value_execution_e::staged>();
                    case time_dimension_mode_e::constant_extrapolation_spatial_residual:
                        return dispatch_two_point_set_compression_by_predictor<label_mode,time_dimension_mode_e::constant_extrapolation_spatial_residual,label_value_execution_e::staged>();
                    case time_dimension_mode_e::linear_interpolation_spatial_residual:
                        return dispatch_two_point_set_compression_by_predictor<label_mode,time_dimension_mode_e::linear_interpolation_spatial_residual,label_value_execution_e::staged>();
                    default:
                        throw runtime_error("分阶段两点集压缩不支持当前数值流程");
                }
            };
            if(framework.label_mode==time_dimension_mode_e::spatial)
                return dispatch_value(integral_constant<time_dimension_mode_e,time_dimension_mode_e::spatial>());
            if(framework.label_mode==time_dimension_mode_e::constant_extrapolation)
                return dispatch_value(integral_constant<time_dimension_mode_e,time_dimension_mode_e::constant_extrapolation>());
            if(framework.label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual)
                return dispatch_value(integral_constant<time_dimension_mode_e,time_dimension_mode_e::label_linear_interpolation_spatial_residual>());
        }
        else if(framework.execution==label_value_execution_e::fused)
        {
            if(framework.label_mode==time_dimension_mode_e::spatial&&framework.value_mode==time_dimension_mode_e::spatial)
                return dispatch_two_point_set_compression_by_predictor<time_dimension_mode_e::spatial,time_dimension_mode_e::spatial,label_value_execution_e::fused>();
            if(framework.label_mode==time_dimension_mode_e::constant_extrapolation&&framework.value_mode==time_dimension_mode_e::spatial)
                return dispatch_two_point_set_compression_by_predictor<time_dimension_mode_e::constant_extrapolation,time_dimension_mode_e::spatial,label_value_execution_e::fused>();
            if(framework.label_mode==time_dimension_mode_e::constant_extrapolation&&framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
                return dispatch_two_point_set_compression_by_predictor<time_dimension_mode_e::constant_extrapolation,time_dimension_mode_e::constant_extrapolation_spatial_residual,label_value_execution_e::fused>();
            if(framework.label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual&&framework.value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                return dispatch_two_point_set_compression_by_predictor<time_dimension_mode_e::label_linear_interpolation_spatial_residual,time_dimension_mode_e::linear_interpolation_spatial_residual,label_value_execution_e::fused>();
        }
        throw runtime_error("两点集压缩尚未实现当前标签、数值和执行方式的组合");
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::dispatch_single_point_set_compression()
    {
        if(point_set_config_vec.size()!=1)
            throw runtime_error("单点集压缩要求field恰好配置一个点集");
        prediction_framework_c framework=get_prediction_framework(1);
        point_set_codec_e predictor=point_set_config_vec[0].codec;
        auto dispatch_predictor=[&](auto mode_constant)
        {
            constexpr time_dimension_mode_e mode=decltype(mode_constant)::value;
            if(predictor==point_set_codec_e::idw)
                return compress_current_file<false,mode,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>();
            if(predictor==point_set_codec_e::rbf)
                return compress_current_file<false,mode,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>();
            throw runtime_error("单点集压缩只支持IDW或RBF空间预测器");
        };
        if(framework.value_mode==time_dimension_mode_e::spatial)
            return dispatch_predictor(integral_constant<time_dimension_mode_e,time_dimension_mode_e::spatial>());
        if(framework.value_mode==time_dimension_mode_e::constant_extrapolation)
            return dispatch_predictor(integral_constant<time_dimension_mode_e,time_dimension_mode_e::constant_extrapolation>());
        if(framework.value_mode==time_dimension_mode_e::linear_interpolation)
            return dispatch_predictor(integral_constant<time_dimension_mode_e,time_dimension_mode_e::linear_interpolation>());
        if(framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
            return dispatch_predictor(integral_constant<time_dimension_mode_e,time_dimension_mode_e::constant_extrapolation_spatial_residual>());
        if(framework.value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
            return dispatch_predictor(integral_constant<time_dimension_mode_e,time_dimension_mode_e::linear_interpolation_spatial_residual>());
        throw runtime_error("单点集压缩不支持当前锚点内插阶段");
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e label_mode,time_dimension_mode_e value_mode,label_value_execution_e execution>
    void param_c<T_coord,T_data>::dispatch_two_point_set_compression_by_predictor()
    {
        point_set_codec_e predictor=point_set_codec_e::constant;
        for(const point_set_config_c &config:point_set_config_vec)
            if(config.codec!=point_set_codec_e::constant)
            {
                if(predictor!=point_set_codec_e::constant&&predictor!=config.codec)
                    throw runtime_error("同一次两点集压缩中的非常量点集必须使用同一种空间预测器");
                predictor=config.codec;
            }
        if(predictor==point_set_codec_e::idw)
            return compress_current_file<true,value_mode,label_mode,execution,point_set_codec_e::idw>();
        if(predictor==point_set_codec_e::rbf)
            return compress_current_file<true,value_mode,label_mode,execution,point_set_codec_e::rbf>();
        throw runtime_error("两点集压缩至少需要一个使用IDW或RBF的非常量点集");
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::compress_current_file()
    {
        constexpr bool use_normalized_idw_weights=fixed_predictor==point_set_codec_e::idw;
        if constexpr(two_point_sets)
            point_set_predictor=fixed_predictor==point_set_codec_e::rbf?"rbf":"idw";
        auto stage_start_time=chrono::steady_clock::now();
        double point_set_time=0,point_set_classification_time=0,point_set_setup_time=0,point_set_eb_schedule_time=0,label_prediction_time=0,value_prediction_time=0,lossless_encoding_time=0,validation_time=0;
        if (current_data.size()!=point_num)
            throw runtime_error("当前数据点数与坐标点数不一致");
        if(point_set_config_vec.empty())
            throw runtime_error("尚未配置点集");
        if(point_set_config_vec.size()!=point_set_rule_vec.size()+1)
            throw runtime_error("点集配置数量必须等于有序分类规则数量加1");
        if(eb_schedule!="none"&&eb_schedule!="round"&&eb_schedule!="point"&&eb_schedule!="inverse_rank_log2"&&eb_schedule!="round_point"&&eb_schedule!="round_ratio"&&eb_schedule!="round_point_ratio")
            throw runtime_error("未知的EB序列模式："+eb_schedule);
        if (!(eb_schedule_start_ratio>0) || eb_schedule_start_ratio>1)
            throw runtime_error("EB序列起始倍率必须在(0,1]内");
        point_set_id_vec.resize(point_num);
        uint32_t point_set_num=0,configured_point_set_num=(uint32_t)point_set_config_vec.size();
        point_set_point_num_vec.assign(configured_point_set_num,0);
        point_set_min_data_vec.assign(configured_point_set_num,numeric_limits<T_data>::infinity());
        point_set_max_data_vec.assign(configured_point_set_num,-numeric_limits<T_data>::infinity());
        if constexpr(!two_point_sets)
        {
            point_set_num=1;
            point_set_id_vec.assign(point_num,0);
            point_set_point_num_vec[0]=point_num;
            for(T_data value:current_data)
            {
                point_set_min_data_vec[0]=min(point_set_min_data_vec[0],value);
                point_set_max_data_vec[0]=max(point_set_max_data_vec[0],value);
            }
        }
        else if(point_set_rule_vec.size()==1)
        {
            const point_set_rule_c &rule=point_set_rule_vec[0];
            auto classify_two_point_sets=[&](auto matches)
            {
                uint32_t point_num_per_set[2]={0,0};
                T_data minimum[2]={numeric_limits<T_data>::infinity(),numeric_limits<T_data>::infinity()};
                T_data maximum[2]={-numeric_limits<T_data>::infinity(),-numeric_limits<T_data>::infinity()};
                for(uint32_t point_id=0;point_id<point_num;point_id++)
                {
                    T_data value=current_data[point_id];
                    uint32_t point_set_id=matches((double)value)?0:1;
                    point_set_id_vec[point_id]=(uint8_t)point_set_id;
                    point_num_per_set[point_set_id]++;
                    minimum[point_set_id]=min(minimum[point_set_id],value);
                    maximum[point_set_id]=max(maximum[point_set_id],value);
                }
                for(uint32_t point_set_id=0;point_set_id<2;point_set_id++)
                {
                    point_set_point_num_vec[point_set_id]=point_num_per_set[point_set_id];
                    point_set_min_data_vec[point_set_id]=minimum[point_set_id];
                    point_set_max_data_vec[point_set_id]=maximum[point_set_id];
                }
                point_set_num=point_num_per_set[1]>0?2:1;
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
        point_set_classification_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        stage_start_time=chrono::steady_clock::now();
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
        uint32_t primary_point_set_id=0;
        while (point_set_point_num_vec[primary_point_set_id]==0)
            primary_point_set_id++;
        current_min_data=point_set_min_data_vec[primary_point_set_id];
        current_max_data=point_set_max_data_vec[primary_point_set_id];
        current_abs_eb=point_set_abs_eb_vec[primary_point_set_id];
        current_valid_node_num=point_set_point_num_vec[0];
        select_partition_prediction_configs();
        point_set_setup_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        stage_start_time=chrono::steady_clock::now();
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
            if(has_ten_prediction_references(pred_ref_id))
                return predict_label_10(pred_ref_begin[pred_ref_id],pred_ref_prediction_weights,label_vec);
            return predict_label(pred_ref_id,pred_ref_prediction_weights,label_vec);
        };
        constexpr bool use_fused_two_point_set_prediction=two_point_sets&&fixed_execution==label_value_execution_e::fused;
        constexpr bool use_selected_fused_temporal_flow=use_fused_two_point_set_prediction&&((fixed_label_mode==time_dimension_mode_e::constant_extrapolation&&fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)||(fixed_label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual&&fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual));
        if(cached_pred_eb_multiplier_vec.size()!=point_num)
            throw runtime_error("预处理没有生成完整的EB倍率");
        if(use_fused_two_point_set_prediction&&cached_pred_eb_inverse_multiplier_vec.size()!=point_num)
        {
            cached_pred_eb_inverse_multiplier_vec.resize(point_num);
            for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
                cached_pred_eb_inverse_multiplier_vec[pred_ref_id]=1/cached_pred_eb_multiplier_vec[pred_ref_id];
        }
        const vector<double> &pred_eb_multiplier_vec=cached_pred_eb_multiplier_vec;
        const vector<double> &pred_eb_inverse_multiplier_vec=cached_pred_eb_inverse_multiplier_vec;
        point_set_eb_schedule_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        point_set_time=point_set_classification_time+point_set_setup_time+point_set_eb_schedule_time;

        stage_start_time=chrono::steady_clock::now();
        uint32_t label_channel_num=point_set_num>0?point_set_num-1:0;
        bool use_partition_stream_format=!partition_pred_ref_ids_vec.empty();
        vector<vector<uint8_t>> label_symbol_file_order_vec_vec(label_channel_num);
        uint32_t partition_stream_num=0;
        if(use_partition_stream_format)
            partition_stream_num=(uint32_t)partition_pred_ref_ids_vec.size();
        vector<vector<vector<uint8_t>>> partition_label_symbol_vec_vec(partition_stream_num,vector<vector<uint8_t>>(label_channel_num));
        vector<uint64_t> label_channel_target_num_vec(label_channel_num,0),label_channel_error_num_vec(label_channel_num,0);
        uint64_t label_prediction_error_num=0;
        if(point_set_num>1&&point_set_predictor!="idw"&&point_set_predictor!="rbf")
            throw runtime_error("未知的点集标签预测器："+point_set_predictor);
        prepare_linear_interpolation();
        bool use_previous_time_label=prediction_framework.label_mode==time_dimension_mode_e::constant_extrapolation && previous_time_point_set_id_vec.size()==point_num;
        bool use_linear_previous_label=use_linear_previous_label_experimental && use_fused_two_point_set_prediction && use_previous_time_label;
        bool use_causal_time_label=prediction_framework.label_mode==time_dimension_mode_e::constant_extrapolation&&previous_time_point_set_id_vec.size()==point_num;
        bool use_original_order_constant_label_phase=use_selected_fused_temporal_flow&&prediction_framework.label_mode==time_dimension_mode_e::constant_extrapolation&&value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual&&use_previous_time_label;
        constexpr bool try_time_label=fixed_label_mode==time_dimension_mode_e::constant_extrapolation||fixed_label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual;
        if(use_fused_two_point_set_prediction)
        {
            if(point_set_num!=2)
                throw runtime_error("融合两点集预测要求当前文件恰好包含两个点集");
            if(use_partition_stream_format || thread_num>1)
                throw runtime_error("融合两点集预测目前只支持单线程非partition路径");
            if(point_set_predictor!="idw"&&point_set_predictor!="rbf")
                throw runtime_error("融合两点集预测当前流程不支持所选标签预测器");
            if(use_partition_adaptive_predictor_experimental)
                throw runtime_error("融合两点集预测目前不能和自适应预测器同时使用");
            if(use_selected_fused_temporal_flow&&use_linear_previous_label_experimental)
                throw runtime_error("选定的融合时间流程不使用线性上一时刻标签修正流");
            if(!use_selected_fused_temporal_flow&&value_mode!=time_dimension_mode_e::spatial)
                throw runtime_error("当前融合两点集流程不支持所选数值预测方式");
            for(point_set_codec_e codec:point_set_codec_vec)
                if(codec!=point_set_codec_e::constant&&codec!=point_set_codec_e::idw&&codec!=point_set_codec_e::rbf)
                    throw runtime_error("融合两点集预测只支持constant、IDW或RBF预测器");
        }
        vector<vector<int32_t>> point_set_qb_vec(point_set_num);
        vector<vector<vector<int32_t>>> partition_point_set_qb_vec(partition_stream_num,vector<vector<int32_t>>(point_set_num));
        vector<uint64_t> point_set_qb_num_vec(point_set_num,0);
        vector<int32_t> point_set_qb_min_vec(point_set_num,numeric_limits<int32_t>::max()),point_set_qb_max_vec(point_set_num,numeric_limits<int32_t>::min());
        if(use_fused_two_point_set_prediction)
            for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                if(point_set_codec_vec[point_set_id]!=point_set_codec_e::constant)
                    point_set_qb_vec[point_set_id].reserve(point_set_point_num_vec[point_set_id]);
        vector<double> point_set_inverse_double_abs_eb_vec(point_set_num,0);
        if(use_fused_two_point_set_prediction)
            for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                if(point_set_codec_vec[point_set_id]!=point_set_codec_e::constant)
                    point_set_inverse_double_abs_eb_vec[point_set_id]=0.5/point_set_abs_eb_vec[point_set_id];
        if(use_fused_two_point_set_prediction)
            current_decompressed_data.resize(point_num);
        for (uint32_t label_channel_id=0;label_channel_id<label_channel_num;label_channel_id++)
        {
            vector<uint8_t> label_symbol_internal_vec(point_num,0);
            vector<uint8_t> binary_label_vec;
            if(!(use_fused_two_point_set_prediction&&!use_selected_fused_temporal_flow&&use_normalized_idw_weights))
                binary_label_vec.assign(point_num,1);
            if(!use_fused_two_point_set_prediction)
                for (uint32_t point_id=0;point_id<point_num;point_id++)
                    if (point_set_id_vec[point_id]==label_channel_id)
                        binary_label_vec[point_id]=0;
            if(use_causal_time_label&&!use_fused_two_point_set_prediction)
            {
                uint32_t pred_ref_id=0;
                for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
                    for(uint32_t round_target_id=0;round_target_id<prediction_target_id_groups[round_id].size();round_target_id++,pred_ref_id++)
                    {
                    int pred_id=pred_target_ids[pred_ref_id];
                    uint32_t actual_point_set_id=point_set_id_vec[pred_id];
                    if(actual_point_set_id<label_channel_id)
                        continue;
                    uint32_t original_id=pred_id;
                    uint32_t predicted_label;
                    if(!predict_time_label_fixed<fixed_label_mode>(original_id,label_channel_id,predicted_label))
                        throw runtime_error("因果时间标签预测缺少历史标签");
                    uint32_t actual_label=actual_point_set_id==label_channel_id?0:1;
                    bool prediction_error=actual_label!=predicted_label;
                    label_channel_target_num_vec[label_channel_id]++;
                    label_channel_error_num_vec[label_channel_id]+=prediction_error;
                    label_prediction_error_num+=prediction_error;
                    label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                }
            }
            else if(use_partition_stream_format)
            {
                if(partition_pred_ref_ids_vec.empty())
                    throw runtime_error("多线程标签预测需要独立partition预测DAG");
                vector<uint64_t> partition_target_num_vec(partition_pred_ref_ids_vec.size(),0),partition_error_num_vec(partition_pred_ref_ids_vec.size(),0);
                exception_ptr label_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
                for(int64_t partition_index=0;partition_index<(int64_t)partition_pred_ref_ids_vec.size();partition_index++)
                {
                    try
                    {
                        uint32_t partition_id=(uint32_t)partition_index;
                        uint64_t local_target_num=0,local_error_num=0;
                        for(uint32_t pred_ref_id:partition_pred_ref_ids_vec[partition_id])
                        {
                            int pred_id=pred_target_ids[pred_ref_id];
                            uint32_t actual_point_set_id=point_set_id_vec[pred_id];
                            if(actual_point_set_id<label_channel_id)
                                continue;
                            uint32_t original_id=pred_id;
                            uint32_t predicted_label;
                            bool used_time_label=try_time_label&&predict_time_label_fixed<fixed_label_mode>(original_id,label_channel_id,predicted_label);
                            if(!used_time_label)
                                predicted_label=predict_configured_label(pred_ref_id,binary_label_vec);
                            uint32_t actual_label=actual_point_set_id==label_channel_id?0:1;
                            bool prediction_error=actual_label!=predicted_label;
                            local_target_num++;
                            local_error_num+=prediction_error;
                            label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                        }
                        partition_target_num_vec[partition_id]=local_target_num;
                        partition_error_num_vec[partition_id]=local_error_num;
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
                label_channel_target_num_vec[label_channel_id]=accumulate(partition_target_num_vec.begin(),partition_target_num_vec.end(),(uint64_t)0);
                label_channel_error_num_vec[label_channel_id]=accumulate(partition_error_num_vec.begin(),partition_error_num_vec.end(),(uint64_t)0);
                label_prediction_error_num+=label_channel_error_num_vec[label_channel_id];
            }
            else if(use_original_order_constant_label_phase)
            {
                if(label_channel_id!=0)
                    throw runtime_error("标签零阶外插只能处理一个标签通道");
                for(uint32_t original_id=0;original_id<point_num;original_id++)
                {
                    uint32_t internal_id=original_id;
                    uint32_t actual_label=point_set_id_vec[internal_id];
                    bool prediction_error=actual_label!=previous_time_point_set_id_vec[original_id];
                    label_symbol_internal_vec[internal_id]=prediction_error?1:0;
                    label_channel_error_num_vec[0]+=prediction_error;
                }
                label_channel_target_num_vec[0]=point_num;
                label_prediction_error_num+=label_channel_error_num_vec[0];
            }
            else if(use_selected_fused_temporal_flow)
            {
                if(label_channel_id!=0)
                    throw runtime_error("融合两点集时间预测只能处理一个标签通道");
                uint32_t pred_ref_id=0;
                uint64_t ref_begin=pred_ref_begin[0];
                for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
                    for(uint32_t round_target_id=0;round_target_id<prediction_target_id_groups[round_id].size();round_target_id++,pred_ref_id++)
                    {
                        int pred_id=pred_target_ids[pred_ref_id];
                        uint32_t actual_point_set_id=point_set_id_vec[pred_id];
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
                        bool prediction_error=actual_point_set_id!=predicted_label;
                        label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                        label_channel_error_num_vec[0]+=prediction_error;
                        binary_label_vec[pred_id]=(uint8_t)actual_point_set_id;
                        if(point_set_codec_vec[actual_point_set_id]==point_set_codec_e::constant)
                            current_decompressed_data[pred_id]=point_set_min_data_vec[actual_point_set_id];
                        else
                        {
                            T_data pred=0;
                            bool use_time_prediction=false;
                            if(value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                            {
                                use_time_prediction=predict_time_spatial_residual_baseline_fixed<fixed_value_mode>(original_id,actual_point_set_id,pred);
                                if(use_time_prediction)
                                    pred+=predict_configured_value(pred_ref_id,pred_id,ref_ids,ref_weights,actual_point_set_id,current_time_spatial_residual_data);
                            }
                            if(!use_time_prediction)
                                pred=predict_configured_value(pred_ref_id,pred_id,ref_ids,ref_weights,actual_point_set_id,current_decompressed_data);
                            double point_abs_eb=point_set_abs_eb_vec[actual_point_set_id]*pred_eb_multiplier_vec[pred_ref_id];
                            double qb_double=round(((double)current_data[pred_id]-pred)*point_set_inverse_double_abs_eb_vec[actual_point_set_id]*pred_eb_inverse_multiplier_vec[pred_ref_id]);
                            if(!isfinite(qb_double)||qb_double<numeric_limits<int32_t>::min()||qb_double>numeric_limits<int32_t>::max())
                                throw runtime_error("量化索引超出int32_t范围，请增大误差上限");
                            int32_t qb=(int32_t)qb_double;
                            point_set_qb_vec[actual_point_set_id].push_back(qb);
                            current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                        }
                        update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,actual_point_set_id);
                    }
                if(pred_ref_id!=pred_target_ids.size())
                    throw runtime_error("融合两点集时间预测没有处理全部目标");
                label_channel_target_num_vec[0]=pred_ref_id;
                label_prediction_error_num+=label_channel_error_num_vec[0];
                for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                {
                    point_set_qb_num_vec[point_set_id]=point_set_qb_vec[point_set_id].size();
                }
            }
            else if(use_fused_two_point_set_prediction)
            {
                if(label_channel_id!=0)
                    throw runtime_error("融合两点集预测只能处理一个标签通道");
                if(use_linear_previous_label)
                {
                    for(uint32_t original_id=0;original_id<point_num;original_id++)
                    {
                        uint32_t internal_id=original_id;
                        bool prediction_error=point_set_id_vec[internal_id]!=previous_time_point_set_id_vec[original_id];
                        label_symbol_internal_vec[internal_id]=prediction_error?1:0;
                        label_channel_error_num_vec[0]+=prediction_error;
                    }
                    label_channel_target_num_vec[0]=point_num;
                }
                uint32_t pred_ref_id=0;
                uint64_t ref_begin=pred_ref_begin[0];
                for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
                    for(uint32_t round_target_id=0;round_target_id<prediction_target_id_groups[round_id].size();round_target_id++,pred_ref_id++)
                    {
                        int pred_id=pred_target_ids[pred_ref_id];
                        uint32_t actual_point_set_id=point_set_id_vec[pred_id];
                        uint64_t ref_end=pred_ref_begin[pred_ref_id+1];
                        uint32_t ref_num=(uint32_t)(ref_end-ref_begin);
                        array_view_c<int> ref_ids(ref_num?pred_ref_ids.data()+ref_begin:nullptr,ref_num);
                        auto ref_weights=get_pred_ref_weights(pred_ref_id);
                        ref_begin=ref_end;
                        T_data predicted_value;
                        bool value_requires_prediction=point_set_codec_vec[actual_point_set_id]!=point_set_codec_e::constant;
                        if(use_previous_time_label)
                        {
                            if(point_set_codec_vec[actual_point_set_id]==point_set_codec_e::rbf)
                            {
                                if(value_requires_prediction)
                                    predicted_value=predict_configured_value(pred_ref_id,pred_id,ref_ids,ref_weights,actual_point_set_id,current_decompressed_data);
                                else
                                    predicted_value=0;
                            }
                            else
                                if(value_requires_prediction)
                                    predicted_value=predict_value_two_point_sets_idw(ref_ids,ref_weights,current_decompressed_data,actual_point_set_id);
                                else
                                    predicted_value=0;
                            if(!use_linear_previous_label)
                            {
                                uint32_t original_id=pred_id;
                                uint32_t predicted_label=previous_time_point_set_id_vec[original_id];
                                bool prediction_error=actual_point_set_id!=predicted_label;
                                label_channel_error_num_vec[0]+=prediction_error;
                                label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                            }
                        }
                        else
                        {
                            pair<uint32_t,T_data> prediction;
                            if constexpr(fixed_predictor==point_set_codec_e::rbf)
                                prediction=predict_label_and_value_two_point_sets_rbf(pred_ref_id,ref_ids,ref_weights,binary_label_vec,current_decompressed_data,actual_point_set_id,value_requires_prediction);
                            else
                                prediction=predict_label_and_value_two_point_sets_idw(ref_ids,ref_weights,current_decompressed_data,actual_point_set_id,value_requires_prediction);
                            uint32_t predicted_label=prediction.first;
                            predicted_value=prediction.second;
                            bool prediction_error=actual_point_set_id!=predicted_label;
                            label_channel_error_num_vec[0]+=prediction_error;
                            label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                            if(!use_normalized_idw_weights)
                                binary_label_vec[pred_id]=(uint8_t)actual_point_set_id;
                        }
                        if(!value_requires_prediction)
                            current_decompressed_data[pred_id]=point_set_min_data_vec[actual_point_set_id];
                        else
                        {
                            double point_abs_eb=point_set_abs_eb_vec[actual_point_set_id]*pred_eb_multiplier_vec[pred_ref_id];
                            T_data pred=predicted_value;
                            double qb_double=round(((double)current_data[pred_id]-pred)*point_set_inverse_double_abs_eb_vec[actual_point_set_id]*pred_eb_inverse_multiplier_vec[pred_ref_id]);
                            if(!isfinite(qb_double) || qb_double<numeric_limits<int32_t>::min() || qb_double>numeric_limits<int32_t>::max())
                                throw runtime_error("量化索引超出int32_t范围，请增大误差上限");
                            int32_t qb=(int32_t)qb_double;
                            point_set_qb_vec[actual_point_set_id].push_back(qb);
                            current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                        }
                    }
                if(pred_ref_id!=pred_target_ids.size())
                    throw runtime_error("融合两点集预测没有处理全部目标");
                if(!use_linear_previous_label)
                    label_channel_target_num_vec[0]=pred_ref_id;
                label_prediction_error_num+=label_channel_error_num_vec[0];
                for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                {
                    point_set_qb_num_vec[point_set_id]=point_set_qb_vec[point_set_id].size();
                }
            }
            else
            {
                for(uint32_t pred_ref_id=0;pred_ref_id<pred_target_ids.size();pred_ref_id++)
                {
                    int pred_id=pred_target_ids[pred_ref_id];
                    uint32_t actual_point_set_id=point_set_id_vec[pred_id];
                    if(actual_point_set_id<label_channel_id)
                        continue;
                    uint32_t original_id=pred_id;
                    uint32_t predicted_label;
                    bool used_time_label=try_time_label&&predict_time_label_fixed<fixed_label_mode>(original_id,label_channel_id,predicted_label);
                    if(!used_time_label)
                        predicted_label=predict_configured_label(pred_ref_id,binary_label_vec);
                    uint32_t actual_label=actual_point_set_id==label_channel_id?0:1;
                    bool prediction_error=actual_label!=predicted_label;
                    label_channel_target_num_vec[label_channel_id]++;
                    label_channel_error_num_vec[label_channel_id]+=prediction_error;
                    label_prediction_error_num+=prediction_error;
                    label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                }
            }
            if(!use_partition_stream_format)
            {
                if(use_fused_two_point_set_prediction&&label_channel_id==0)
                    label_symbol_file_order_vec_vec[label_channel_id]=move(label_symbol_internal_vec);
                else
                {
                    label_symbol_file_order_vec_vec[label_channel_id].reserve(label_channel_target_num_vec[label_channel_id]);
                    for (uint32_t original_id=0;original_id<point_num;original_id++)
                    {
                        uint32_t internal_id=original_id;
                        if (point_set_id_vec[internal_id]>=label_channel_id)
                            label_symbol_file_order_vec_vec[label_channel_id].push_back(label_symbol_internal_vec[internal_id]);
                    }
                }
            }
            if(use_partition_stream_format)
                for(uint32_t partition_id=0;partition_id<partition_pred_ref_ids_vec.size();partition_id++)
                    for(uint32_t pred_ref_id:partition_pred_ref_ids_vec[partition_id])
                    {
                        int pred_id=pred_target_ids[pred_ref_id];
                        if(point_set_id_vec[pred_id]>=label_channel_id)
                            partition_label_symbol_vec_vec[partition_id][label_channel_id].push_back(label_symbol_internal_vec[pred_id]);
                    }
        }
        label_prediction_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        stage_start_time=chrono::steady_clock::now();
        if(!use_fused_two_point_set_prediction)
            current_decompressed_data.resize(point_num);
        uint64_t value_huffman_byte_num=0;
        uint32_t pred_ref_index=0;
        constexpr bool use_normalized_value_weights=!two_point_sets&&use_normalized_idw_weights;
        bool has_previous_time=value_mode==time_dimension_mode_e::constant_extrapolation&&previous_time_decompressed_data.size()==point_num&&previous_time_point_set_id_vec.size()==point_num;
        prepare_linear_interpolation();
        uint64_t cross_time_prediction_num=0;
        bool use_single_point_set_rbf_fast_path=!two_point_sets&&point_set_codec_vec[0]==point_set_codec_e::rbf&&value_mode==time_dimension_mode_e::spatial&&!use_partition_stream_format&&!use_partition_adaptive_predictor_experimental&&pred_ref_prediction_weights.size()==pred_ref_ids.size();
        constexpr bool supports_two_point_set_constant_residual_rbf_fast_path=two_point_sets&&fixed_execution==label_value_execution_e::fused&&fixed_label_mode==time_dimension_mode_e::constant_extrapolation&&fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual&&fixed_predictor==point_set_codec_e::rbf;
        bool use_two_point_set_constant_residual_rbf_fast_path=supports_two_point_set_constant_residual_rbf_fast_path&&use_original_order_constant_label_phase&&point_set_codec_vec.size()==2&&point_set_codec_vec[0]==point_set_codec_e::rbf&&point_set_codec_vec[1]==point_set_codec_e::constant&&previous_time_decompressed_data.size()==point_num&&current_time_spatial_residual_data.size()==point_num&&!use_partition_stream_format&&!use_partition_adaptive_predictor_experimental&&pred_ref_prediction_weights.size()==pred_ref_ids.size();
        bool use_multihead_execution=!multihead_execution_pred_ref_ids.empty();
        if(use_multihead_execution&&!use_single_point_set_rbf_fast_path)
            throw runtime_error("多头执行要求单点集纯空间RBF快速路径");
        if(use_fused_two_point_set_prediction&&!use_original_order_constant_label_phase)
            pred_ref_index=point_num;
        else if(use_two_point_set_constant_residual_rbf_fast_path)
        {
            vector<int32_t> &qb_vec=point_set_qb_vec[0];
            uint64_t ref_begin=pred_ref_begin[0];
            for(uint32_t current_pred_ref_id=0;current_pred_ref_id<point_num;current_pred_ref_id++)
            {
                int pred_id=pred_target_ids[current_pred_ref_id];
                uint64_t ref_end=pred_ref_begin[current_pred_ref_id+1];
                if(point_set_id_vec[pred_id]==1)
                    current_decompressed_data[pred_id]=point_set_min_data_vec[1];
                else
                {
                    uint32_t ref_num=(uint32_t)(ref_end-ref_begin);
                    array_view_c<int> ref_ids(ref_num?pred_ref_ids.data()+ref_begin:nullptr,ref_num);
                    auto ref_weights=get_pred_ref_weights(current_pred_ref_id);
                    bool use_time_prediction=previous_time_point_set_id_vec[pred_id]==0;
                    T_data time_baseline=use_time_prediction?previous_time_decompressed_data[pred_id]:0;
                    vector<T_data> &prediction_data=use_time_prediction?current_time_spatial_residual_data:current_decompressed_data;
                    T_data pred=time_baseline+predict_configured_value(current_pred_ref_id,pred_id,ref_ids,ref_weights,0,prediction_data);
                    double point_abs_eb=point_set_abs_eb_vec[0]*pred_eb_multiplier_vec[current_pred_ref_id];
                    double qb_double=round(((double)current_data[pred_id]-pred)/(2*point_abs_eb));
                    if(!isfinite(qb_double)||qb_double<numeric_limits<int32_t>::min()||qb_double>numeric_limits<int32_t>::max())
                        throw runtime_error("量化索引超出int32_t范围，请增大误差上限");
                    int32_t qb=(int32_t)qb_double;
                    qb_vec.push_back(qb);
                    point_set_qb_min_vec[0]=min(point_set_qb_min_vec[0],qb);
                    point_set_qb_max_vec[0]=max(point_set_qb_max_vec[0],qb);
                    current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                    if(use_time_prediction)
                    {
                        current_time_spatial_residual_data[pred_id]=current_decompressed_data[pred_id]-time_baseline;
                        cross_time_prediction_num++;
                    }
                }
                ref_begin=ref_end;
            }
            point_set_qb_num_vec[0]=qb_vec.size();
            pred_ref_index=point_num;
        }
        else if(use_single_point_set_rbf_fast_path)
        {
            vector<int32_t> &qb_vec=point_set_qb_vec[0];
            if(use_multihead_execution)
                qb_vec.resize(point_num);
            else
                qb_vec.reserve(point_num);
            double base_abs_eb=point_set_abs_eb_vec[0];
            auto encode_prediction=[&](uint32_t pred_ref_id,T_data pred)
            {
                uint32_t pred_id=(uint32_t)pred_target_ids[pred_ref_id];
                double point_abs_eb=base_abs_eb*pred_eb_multiplier_vec[pred_ref_id];
                double qb_double=round(((double)current_data[pred_id]-pred)/(2*point_abs_eb));
                if(!isfinite(qb_double)||qb_double<numeric_limits<int32_t>::min()||qb_double>numeric_limits<int32_t>::max())
                    throw runtime_error("量化索引超出int32_t范围，请增大误差上限");
                int32_t qb=(int32_t)qb_double;
                if(use_multihead_execution)
                    qb_vec[multihead_qb_index_by_pred_ref_id[pred_ref_id]]=qb;
                else
                    qb_vec.push_back(qb);
                point_set_qb_min_vec[0]=min(point_set_qb_min_vec[0],qb);
                point_set_qb_max_vec[0]=max(point_set_qb_max_vec[0],qb);
                current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
            };
            if(use_multihead_execution)
            {
                for(uint32_t pred_ref_id:multihead_execution_pred_ref_ids)
                {
                    if(has_ten_prediction_references(pred_ref_id))
                        encode_prediction(pred_ref_id,predict_value_one_point_set_10(pred_ref_begin[pred_ref_id],pred_ref_prediction_weights,current_decompressed_data));
                    else
                        encode_prediction(pred_ref_id,predict_value_one_point_set(pred_ref_id,pred_ref_prediction_weights,current_decompressed_data));
                }
            }
            else
                for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
                    if(has_ten_prediction_references(pred_ref_id))
                        encode_prediction(pred_ref_id,predict_value_one_point_set_10(pred_ref_begin[pred_ref_id],pred_ref_prediction_weights,current_decompressed_data));
                    else
                        encode_prediction(pred_ref_id,predict_value_one_point_set(pred_ref_id,pred_ref_prediction_weights,current_decompressed_data));
            point_set_qb_num_vec[0]=point_num;
            pred_ref_index=point_num;
        }
        else if(use_partition_stream_format)
        {
            if(partition_pred_ref_ids_vec.empty())
                throw runtime_error("多线程数值预测需要独立partition预测DAG");
            vector<vector<uint64_t>> partition_point_set_qb_num_vec(partition_pred_ref_ids_vec.size(),vector<uint64_t>(point_set_num,0));
            vector<vector<int32_t>> partition_point_set_qb_min_vec(partition_pred_ref_ids_vec.size(),vector<int32_t>(point_set_num,numeric_limits<int32_t>::max())),partition_point_set_qb_max_vec(partition_pred_ref_ids_vec.size(),vector<int32_t>(point_set_num,numeric_limits<int32_t>::min()));
            vector<uint64_t> partition_cross_time_prediction_num_vec(partition_pred_ref_ids_vec.size(),0);
            exception_ptr prediction_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
            for(int64_t partition_index=0;partition_index<(int64_t)partition_pred_ref_ids_vec.size();partition_index++)
            {
                try
                {
                    uint32_t partition_id=(uint32_t)partition_index;
                    uint64_t local_cross_time_prediction_num=0;
                    for(uint32_t current_pred_ref_id:partition_pred_ref_ids_vec[partition_id])
                    {
                        auto ref_ids=get_pred_ref_ids(current_pred_ref_id);
                        auto ref_weights=get_pred_ref_weights(current_pred_ref_id);
                        int pred_id=pred_target_ids[current_pred_ref_id];
                        uint32_t point_set_id=two_point_sets?point_set_id_vec[pred_id]:0;
                        if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                        {
                            current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                            uint32_t original_id=pred_id;
                            update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,point_set_id);
                            continue;
                        }
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
                        local_cross_time_prediction_num+=use_cross_time_prediction;
                        double point_abs_eb=point_set_abs_eb_vec[point_set_id]*pred_eb_multiplier_vec[current_pred_ref_id];
                        double qb_double=round(((double)current_data[pred_id]-pred)/(2*point_abs_eb));
                        if(!isfinite(qb_double) || qb_double<numeric_limits<int32_t>::min() || qb_double>numeric_limits<int32_t>::max())
                            throw runtime_error("量化索引超出int32_t范围，请增大误差上限");
                        int32_t qb=(int32_t)qb_double;
                        partition_point_set_qb_vec[partition_id][point_set_id].push_back(qb);
                        partition_point_set_qb_num_vec[partition_id][point_set_id]++;
                        partition_point_set_qb_min_vec[partition_id][point_set_id]=min(partition_point_set_qb_min_vec[partition_id][point_set_id],qb);
                        partition_point_set_qb_max_vec[partition_id][point_set_id]=max(partition_point_set_qb_max_vec[partition_id][point_set_id],qb);
                        current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                        update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,point_set_id);
                    }
                    partition_cross_time_prediction_num_vec[partition_id]=local_cross_time_prediction_num;
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
            cross_time_prediction_num=accumulate(partition_cross_time_prediction_num_vec.begin(),partition_cross_time_prediction_num_vec.end(),(uint64_t)0);
            pred_ref_index=point_num;
            for(uint32_t partition_id=0;partition_id<partition_pred_ref_ids_vec.size();partition_id++)
                for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                {
                    uint64_t local_qb_num=partition_point_set_qb_num_vec[partition_id][point_set_id];
                    if(local_qb_num==0)
                        continue;
                    point_set_qb_num_vec[point_set_id]+=local_qb_num;
                    point_set_qb_min_vec[point_set_id]=min(point_set_qb_min_vec[point_set_id],partition_point_set_qb_min_vec[partition_id][point_set_id]);
                    point_set_qb_max_vec[point_set_id]=max(point_set_qb_max_vec[point_set_id],partition_point_set_qb_max_vec[partition_id][point_set_id]);
                }
        }
        else
        {
        for (uint32_t round_id=0;round_id<prediction_round_num;round_id++)
        {
            for (uint32_t i=0;i<prediction_target_id_groups[round_id].size();i++)
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
                    continue;
                }
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
                    if(use_partition_config)
                        pred=predict_partition_adaptive_value(pred_id,ref_ids,current_decompressed_data,partition_prediction_config_id_vec[point_partition_id_vec[pred_id]]);
                    else if(use_normalized_value_weights)
                        pred=predict_value_one_point_set(current_pred_ref_id,pred_ref_prediction_weights,current_decompressed_data);
                    else
                        pred=predict_configured_value(current_pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_decompressed_data);
                }
                cross_time_prediction_num+=use_cross_time_prediction;
                double point_abs_eb=point_set_abs_eb_vec[point_set_id]*eb_multiplier;
                double qb_double=round(((double)current_data[pred_id]-pred)/(2*point_abs_eb));
                if (!isfinite(qb_double) || qb_double<numeric_limits<int32_t>::min() || qb_double>numeric_limits<int32_t>::max())
                    throw runtime_error("量化索引超出int32_t范围，请增大误差上限");
                int32_t qb=(int32_t)qb_double;
                point_set_qb_vec[point_set_id].push_back(qb);
                point_set_qb_num_vec[point_set_id]++;
                point_set_qb_min_vec[point_set_id]=min(point_set_qb_min_vec[point_set_id],qb);
                point_set_qb_max_vec[point_set_id]=max(point_set_qb_max_vec[point_set_id],qb);
                current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,point_set_id);
            }
        }
        }
        if (pred_ref_index!=pred_target_ids.size())
            throw runtime_error("逐轮预测目标数量与预测参考结构不一致");
        if(value_mode==time_dimension_mode_e::constant_extrapolation||value_mode==time_dimension_mode_e::linear_interpolation||value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
            printf("cross_time_prediction_num=%llu, ratio=%f\n",(unsigned long long)cross_time_prediction_num,(double)cross_time_prediction_num/point_num);
        value_prediction_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();

        stage_start_time=chrono::steady_clock::now();
        vector<unsigned char> container;
        uint64_t label_position_delta_byte_num=0,label_zstd_byte_num=0,value_zstd_byte_num=0;
        if(use_partition_stream_format)
        {
            const unsigned char magic_v9[8]={'U','N','S','Z','P','S','0','9'};
            container.insert(container.end(),magic_v9,magic_v9+8);
            append_binary<uint32_t>(container,9);
            append_binary<uint32_t>(container,point_num);
            append_binary<uint32_t>(container,sizeof(T_data));
            append_binary<uint32_t>(container,point_set_num);
            if(use_partition_adaptive_predictor_experimental)
            {
                append_binary<uint32_t>(container,partition_prediction_config_id_vec.size());
                container.insert(container.end(),partition_prediction_config_id_vec.begin(),partition_prediction_config_id_vec.end());
            }
            for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
            {
                append_binary<uint8_t>(container,(uint8_t)point_set_codec_vec[point_set_id]);
                append_binary<uint32_t>(container,point_set_point_num_vec[point_set_id]);
                append_binary<T_data>(container,point_set_min_data_vec[point_set_id]);
                append_binary<T_data>(container,point_set_max_data_vec[point_set_id]);
            }
            uint32_t partition_num=(uint32_t)partition_pred_ref_ids_vec.size();
            vector<vector<unsigned char>> partition_payload_vec(partition_num);
            vector<uint64_t> partition_label_delta_byte_num_vec(partition_num,0),partition_label_zstd_byte_num_vec(partition_num,0),partition_huffman_byte_num_vec(partition_num,0),partition_value_zstd_byte_num_vec(partition_num,0);
            exception_ptr lossless_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
            for(int64_t partition_index=0;partition_index<(int64_t)partition_num;partition_index++)
            {
                try
                {
                    uint32_t partition_id=(uint32_t)partition_index;
                    vector<unsigned char> &payload=partition_payload_vec[partition_id];
                    for(uint32_t label_channel_id=0;label_channel_id<label_channel_num;label_channel_id++)
                    {
                        vector<uint8_t> &symbol_vec=partition_label_symbol_vec_vec[partition_id][label_channel_id];
                        vector<unsigned char> delta_bytes=symbol_vec.empty()?vector<unsigned char>():binary_one_position_delta_vec(symbol_vec);
                        vector<unsigned char> zstd_bytes;
                        if(!delta_bytes.empty())
                            zstd_bytes=zstd_compress(delta_bytes,zstd_compression_level);
                        append_binary<uint64_t>(payload,symbol_vec.size());
                        append_binary<uint64_t>(payload,zstd_bytes.size());
                        payload.insert(payload.end(),zstd_bytes.begin(),zstd_bytes.end());
                        partition_label_delta_byte_num_vec[partition_id]+=delta_bytes.size();
                        partition_label_zstd_byte_num_vec[partition_id]+=zstd_bytes.size();
                    }
                    for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                    {
                        vector<int32_t> &qb_vec=partition_point_set_qb_vec[partition_id][point_set_id];
                        vector<unsigned char> huffman_bytes=huffman_encode(qb_vec);
                        vector<unsigned char> zstd_bytes;
                        if(!huffman_bytes.empty())
                            zstd_bytes=zstd_compress(huffman_bytes,zstd_compression_level);
                        append_binary<uint64_t>(payload,qb_vec.size());
                        append_binary<uint64_t>(payload,zstd_bytes.size());
                        payload.insert(payload.end(),zstd_bytes.begin(),zstd_bytes.end());
                        partition_huffman_byte_num_vec[partition_id]+=huffman_bytes.size();
                        partition_value_zstd_byte_num_vec[partition_id]+=zstd_bytes.size();
                    }
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
            label_position_delta_byte_num=accumulate(partition_label_delta_byte_num_vec.begin(),partition_label_delta_byte_num_vec.end(),(uint64_t)0);
            label_zstd_byte_num=accumulate(partition_label_zstd_byte_num_vec.begin(),partition_label_zstd_byte_num_vec.end(),(uint64_t)0);
            value_huffman_byte_num=accumulate(partition_huffman_byte_num_vec.begin(),partition_huffman_byte_num_vec.end(),(uint64_t)0);
            value_zstd_byte_num=accumulate(partition_value_zstd_byte_num_vec.begin(),partition_value_zstd_byte_num_vec.end(),(uint64_t)0);
            append_binary<uint32_t>(container,partition_num);
            uint64_t partition_payload_begin=0;
            append_binary<uint64_t>(container,partition_payload_begin);
            for(uint32_t partition_id=0;partition_id<partition_num;partition_id++)
            {
                partition_payload_begin+=partition_payload_vec[partition_id].size();
                append_binary<uint64_t>(container,partition_payload_begin);
            }
            for(vector<unsigned char> &payload:partition_payload_vec)
                container.insert(container.end(),payload.begin(),payload.end());
            current_zstd_bs_vec=move(container);
            printf("partition_stream_num=%u, offset_table_bytes=%llu\n",partition_num,(unsigned long long)(partition_num+1)*sizeof(uint64_t));
        }
        else
        {
        bool use_legacy_point_set_format=point_set_num<=2;
        const unsigned char magic_v7[8]={'U','N','S','Z','P','S','0','7'};
        const unsigned char magic_v8[8]={'U','N','S','Z','P','S','0','8'};
        const unsigned char *magic=use_legacy_point_set_format?magic_v7:magic_v8;
        container.insert(container.end(),magic,magic+8);
        append_binary<uint32_t>(container,use_legacy_point_set_format?7:8);
        append_binary<uint32_t>(container,point_num);
        append_binary<uint32_t>(container,sizeof(T_data));
        append_binary<uint32_t>(container,point_set_num);
        if (use_partition_adaptive_predictor_experimental)
        {
            append_binary<uint32_t>(container,partition_prediction_config_id_vec.size());
            container.insert(container.end(),partition_prediction_config_id_vec.begin(),partition_prediction_config_id_vec.end());
        }
        vector<vector<unsigned char>> label_position_delta_bytes_vec(label_channel_num),label_zstd_bytes_vec(label_channel_num);
        for (uint32_t label_channel_id=0;label_channel_id<label_channel_num;label_channel_id++)
        {
            label_position_delta_bytes_vec[label_channel_id]=binary_one_position_delta_vec(label_symbol_file_order_vec_vec[label_channel_id]);
            if (!label_position_delta_bytes_vec[label_channel_id].empty())
                label_zstd_bytes_vec[label_channel_id]=zstd_compress(label_position_delta_bytes_vec[label_channel_id],zstd_compression_level);
            label_position_delta_byte_num+=label_position_delta_bytes_vec[label_channel_id].size();
            label_zstd_byte_num+=label_zstd_bytes_vec[label_channel_id].size();
            if (!use_legacy_point_set_format)
            {
                double label_error_ratio=0;
                if(label_channel_target_num_vec[label_channel_id]>0)
                    label_error_ratio=(double)label_channel_error_num_vec[label_channel_id]/label_channel_target_num_vec[label_channel_id];
                printf("label_channel%u_target_num=%llu, error_ratio=%f, position_delta_bytes=%llu, zstd_bytes=%llu\n",label_channel_id,(unsigned long long)label_channel_target_num_vec[label_channel_id],label_error_ratio,(unsigned long long)label_position_delta_bytes_vec[label_channel_id].size(),(unsigned long long)label_zstd_bytes_vec[label_channel_id].size());
            }
        }
        if (use_legacy_point_set_format)
        {
            uint64_t label_zstd_bytes=label_channel_num==0?0:label_zstd_bytes_vec[0].size();
            append_binary<uint64_t>(container,label_zstd_bytes);
            if (label_channel_num>0)
                container.insert(container.end(),label_zstd_bytes_vec[0].begin(),label_zstd_bytes_vec[0].end());
        }
        else
        {
            append_binary<uint32_t>(container,label_channel_num);
            for (uint32_t label_channel_id=0;label_channel_id<label_channel_num;label_channel_id++)
            {
                append_binary<uint64_t>(container,label_channel_target_num_vec[label_channel_id]);
                append_binary<uint64_t>(container,label_zstd_bytes_vec[label_channel_id].size());
                container.insert(container.end(),label_zstd_bytes_vec[label_channel_id].begin(),label_zstd_bytes_vec[label_channel_id].end());
            }
        }
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            append_binary<uint8_t>(container,(uint8_t)point_set_codec_vec[point_set_id]);
            append_binary<uint32_t>(container,point_set_point_num_vec[point_set_id]);
            append_binary<T_data>(container,point_set_min_data_vec[point_set_id]);
            append_binary<T_data>(container,point_set_max_data_vec[point_set_id]);
            append_binary<uint64_t>(container,point_set_qb_num_vec[point_set_id]);
            vector<unsigned char> qb_huffman_bytes=huffman_encode(point_set_qb_vec[point_set_id],&point_set_qb_min_vec[point_set_id],&point_set_qb_max_vec[point_set_id]);
            vector<unsigned char> qb_zstd_bytes;
            if(!qb_huffman_bytes.empty())
                qb_zstd_bytes=zstd_compress(qb_huffman_bytes,zstd_compression_level);
            value_huffman_byte_num+=qb_huffman_bytes.size();
            value_zstd_byte_num+=qb_zstd_bytes.size();
            append_binary<uint64_t>(container,qb_zstd_bytes.size());
            container.insert(container.end(),qb_zstd_bytes.begin(),qb_zstd_bytes.end());
            int32_t qb_min=point_set_qb_num_vec[point_set_id]==0?0:point_set_qb_min_vec[point_set_id];
            int32_t qb_max=point_set_qb_num_vec[point_set_id]==0?0:point_set_qb_max_vec[point_set_id];
            printf("set%u_qb_num=%llu, set%u_qb_range=[%d,%d], set%u_huffman_bytes=%llu, set%u_zstd_bytes=%llu\n",point_set_id,
                (unsigned long long)point_set_qb_num_vec[point_set_id],point_set_id,qb_min,qb_max,point_set_id,(unsigned long long)qb_huffman_bytes.size(),
                point_set_id,(unsigned long long)qb_zstd_bytes.size());
        }
        current_zstd_bs_vec=move(container);
        }
        lossless_encoding_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        uint64_t label_target_num=accumulate(label_channel_target_num_vec.begin(),label_channel_target_num_vec.end(),(uint64_t)0);
        double label_prediction_error_ratio=0;
        if(label_target_num>0)
            label_prediction_error_ratio=(double)label_prediction_error_num/label_target_num;
        printf("label_prediction_error_ratio=%f, label_position_delta_bytes=%llu, label_zstd_bytes=%llu, zstd_level=%d\n",label_prediction_error_ratio,(unsigned long long)label_position_delta_byte_num,(unsigned long long)label_zstd_byte_num,zstd_compression_level);
        printf("value_huffman_bytes=%llu, value_zstd_bytes=%llu, file_bytes=%llu\n",(unsigned long long)value_huffman_byte_num,
            (unsigned long long)value_zstd_byte_num,(unsigned long long)current_zstd_bs_vec.size());

        printf("point_set_num=%u",point_set_num);
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            const char *codec_name;
            if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                codec_name="constant";
            else if(point_set_codec_vec[point_set_id]==point_set_codec_e::rbf)
                codec_name="rbf";
            else
                codec_name="idw";
            printf(", set%u_num=%u, set%u_codec=%s",point_set_id,point_set_point_num_vec[point_set_id],point_set_id,codec_name);
        }
        printf("\n");
        if (print_compression_stage_time)
        {
            printf("compression stage time: point_set=%f s, label_prediction=%f s, value_prediction=%f s, lossless_encoding=%f s, validation=%f s\n",point_set_time,label_prediction_time,value_prediction_time,lossless_encoding_time,validation_time);
            printf("point set stage time: classification=%f s, setup=%f s, eb_schedule=%f s\n",point_set_classification_time,point_set_setup_time,point_set_eb_schedule_time);
        }
    }

}
