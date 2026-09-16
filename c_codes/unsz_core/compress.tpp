#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::solve_4x4(double A_in[4][4],double b_in[4],double x[4])
    {
        double A[4][5];
        for (uint32_t i=0;i<4;i++)
        {
            for (uint32_t j=0;j<4;j++)
                A[i][j]=A_in[i][j];
            A[i][4]=b_in[i];
        }
        for (uint32_t col=0;col<4;col++)
        {
            uint32_t pivot=col;
            for (uint32_t row=col+1;row<4;row++)
                if (abs(A[row][col])>abs(A[pivot][col]))
                    pivot=row;
            if (abs(A[pivot][col])<1e-12)
                return false;
            if (pivot!=col)
                for (uint32_t i=col;i<5;i++)
                    swap(A[col][i],A[pivot][i]);
            double divisor=A[col][col];
            for (uint32_t i=col;i<5;i++)
                A[col][i]/=divisor;
            for (uint32_t row=0;row<4;row++)
            {
                if (row==col)
                    continue;
                double factor=A[row][col];
                for (uint32_t i=col;i<5;i++)
                    A[row][i]-=factor*A[col][i];
            }
        }
        for (uint32_t i=0;i<4;i++)
            x[i]=A[i][4];
        return true;
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::solve_linear_system(vector<vector<double>> &A,vector<double> &b,vector<double> &x)
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
            if (abs(A[pivot][col])<1e-12)
                return false;
            if (pivot!=col)
            {
                swap(A[col],A[pivot]);
                swap(b[col],b[pivot]);
            }
            double divisor=A[col][col];
            for (uint32_t i=col;i<n;i++)
                A[col][i]/=divisor;
            b[col]/=divisor;
            for (uint32_t row=0;row<n;row++)
            {
                if (row==col)
                    continue;
                double factor=A[row][col];
                for (uint32_t i=col;i<n;i++)
                    A[row][i]-=factor*A[col][i];
                b[row]-=factor*b[col];
            }
        }
        x.assign(b.begin(),b.end());
        return true;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::compress_two_point_sets()
    {
        if(point_set_config_vec.size()!=2)
            throw runtime_error("两点集压缩要求field恰好配置两个点集");
        prediction_framework_c framework=get_prediction_framework(2);
        if(framework.execution==label_value_execution_e::staged)
        {
            if(framework.label_mode==time_dimension_mode_e::spatial&&framework.value_mode==time_dimension_mode_e::spatial)
                return compress_two_point_sets_fixed<time_dimension_mode_e::spatial,time_dimension_mode_e::spatial,label_value_execution_e::staged>();
            if(framework.label_mode==time_dimension_mode_e::constant_extrapolation&&framework.value_mode==time_dimension_mode_e::spatial)
                return compress_two_point_sets_fixed<time_dimension_mode_e::constant_extrapolation,time_dimension_mode_e::spatial,label_value_execution_e::staged>();
            if(framework.label_mode==time_dimension_mode_e::constant_extrapolation&&framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
                return compress_two_point_sets_fixed<time_dimension_mode_e::constant_extrapolation,time_dimension_mode_e::constant_extrapolation_spatial_residual,label_value_execution_e::staged>();
            if(framework.label_mode==time_dimension_mode_e::endpoint_agreement&&framework.value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                return compress_two_point_sets_fixed<time_dimension_mode_e::endpoint_agreement,time_dimension_mode_e::linear_interpolation_spatial_residual,label_value_execution_e::staged>();
        }
        else if(framework.execution==label_value_execution_e::fused)
        {
            if(framework.label_mode==time_dimension_mode_e::spatial&&framework.value_mode==time_dimension_mode_e::spatial)
                return compress_two_point_sets_fixed<time_dimension_mode_e::spatial,time_dimension_mode_e::spatial,label_value_execution_e::fused>();
            if(framework.label_mode==time_dimension_mode_e::constant_extrapolation&&framework.value_mode==time_dimension_mode_e::spatial)
                return compress_two_point_sets_fixed<time_dimension_mode_e::constant_extrapolation,time_dimension_mode_e::spatial,label_value_execution_e::fused>();
            if(framework.label_mode==time_dimension_mode_e::constant_extrapolation&&framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
                return compress_two_point_sets_fixed<time_dimension_mode_e::constant_extrapolation,time_dimension_mode_e::constant_extrapolation_spatial_residual,label_value_execution_e::fused>();
            if(framework.label_mode==time_dimension_mode_e::endpoint_agreement&&framework.value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                return compress_two_point_sets_fixed<time_dimension_mode_e::endpoint_agreement,time_dimension_mode_e::linear_interpolation_spatial_residual,label_value_execution_e::fused>();
        }
        throw runtime_error("两点集压缩尚未实现当前标签、数值和执行方式的组合");
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e label_mode,time_dimension_mode_e value_mode,label_value_execution_e execution>
    void param_c<T_coord,T_data>::compress_two_point_sets_fixed()
    {
        point_set_codec_e predictor=point_set_codec_e::constant;
        for(const point_set_config_c &config:point_set_config_vec)
            if(config.codec!=point_set_codec_e::constant)
            {
                if(predictor!=point_set_codec_e::constant&&predictor!=config.codec)
                    throw runtime_error("同一次两点集压缩中的非常量点集必须使用同一种空间预测器");
                predictor=config.codec;
            }
        if(predictor==point_set_codec_e::inverse_proportion)
            return compress_point_sets<true,value_mode,label_mode,execution,point_set_codec_e::inverse_proportion>();
        if(predictor==point_set_codec_e::rbf_fitting)
            return compress_point_sets<true,value_mode,label_mode,execution,point_set_codec_e::rbf_fitting>();
        throw runtime_error("两点集压缩至少需要一个使用IDW或RBF的非常量点集");
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
    void param_c<T_coord,T_data>::compress_point_sets()
    {
        auto stage_start_time=chrono::steady_clock::now();
        double input_reorder_time=0,point_set_time=0,point_set_classification_time=0,point_set_setup_time=0,point_set_eb_schedule_time=0,label_prediction_time=0,value_prediction_time=0,lossless_encoding_time=0,validation_time=0;
        last_reorder_data_by_prediction_order_time=0;
        if (current_data.size()!=point_num)
            throw runtime_error("当前数据点数与坐标点数不一致");
        if(point_set_config_vec.empty())
            throw runtime_error("尚未配置点集");
        if(point_set_classification==point_set_classification_e::ordered_rules && point_set_config_vec.size()!=point_set_rule_vec.size()+1)
            throw runtime_error("点集配置数量必须等于有序分类规则数量加1");
        if(eb_schedule_experimental!="none" && eb_schedule_experimental!="round" && eb_schedule_experimental!="point" && eb_schedule_experimental!="point_log2_ratio" && eb_schedule_experimental!="round_point" && eb_schedule_experimental!="round_ratio" && eb_schedule_experimental!="round_point_ratio" && eb_schedule_experimental!="round_point_ratio_dag" && eb_schedule_experimental!="round_point_ratio_dag_direct" && eb_schedule_experimental!="round_point_ratio_dag_variance_direct")
            throw runtime_error("未知的实验性EB序列模式："+eb_schedule_experimental);
        if (!(eb_schedule_start_ratio>0) || eb_schedule_start_ratio>1)
            throw runtime_error("实验性EB序列起始倍率必须在(0,1]内");
        if (reorder_data_by_prediction_order_experimental)
        {
            vector<T_data> original_order_data=move(current_data);
            current_data.resize(point_num);
            for (uint32_t prediction_id=0;prediction_id<point_num;prediction_id++)
                current_data[prediction_id]=original_order_data[prediction_id_to_original_id[prediction_id]];
            input_reorder_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
            last_reorder_data_by_prediction_order_time=input_reorder_time;
            stage_start_time=chrono::steady_clock::now();
        }

        vector<uint8_t> external_point_set_id_vec;
        if constexpr(two_point_sets)
            if(point_set_classification==point_set_classification_e::label_file)
                external_point_set_id_vec=load_point_set_label_ids();
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
        else if(point_set_classification==point_set_classification_e::ordered_rules && point_set_rule_vec.size()==1)
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
            if(rule.operator_type==point_set_rule_operator_e::less)
                classify_two_point_sets([threshold](double value){return value<threshold;});
            else if(rule.operator_type==point_set_rule_operator_e::less_equal)
                classify_two_point_sets([threshold](double value){return value<=threshold;});
            else if(rule.operator_type==point_set_rule_operator_e::greater)
                classify_two_point_sets([threshold](double value){return value>threshold;});
            else if(rule.operator_type==point_set_rule_operator_e::greater_equal)
                classify_two_point_sets([threshold](double value){return value>=threshold;});
            else if(rule.operator_type==point_set_rule_operator_e::equal)
                classify_two_point_sets([threshold](double value){return value==threshold;});
            else
                classify_two_point_sets([threshold](double value){return value!=threshold;});
        }
        else
            for (uint32_t i=0;i<point_num;i++)
            {
                uint32_t original_id=reorder_data_by_prediction_order_experimental?prediction_id_to_original_id[i]:i;
                uint32_t point_set_id=point_set_classification==point_set_classification_e::label_file?external_point_set_id_vec[original_id]:get_point_set_id(current_data[i]);
                if (point_set_id>=configured_point_set_num)
                    throw runtime_error("分类函数返回的点集编号越界");
                point_set_id_vec[i]=(uint8_t)point_set_id;
                point_set_num=max(point_set_num,point_set_id+1);
                point_set_point_num_vec[point_set_id]++;
                point_set_min_data_vec[point_set_id]=min(point_set_min_data_vec[point_set_id],current_data[i]);
                point_set_max_data_vec[point_set_id]=max(point_set_max_data_vec[point_set_id],current_data[i]);
            }
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
        auto predict_configured_value=[&](uint32_t pred_ref_id,int pred_id,array_view_c<int> ref_ids,array_view_c<double> ref_weights,uint32_t point_set_id,vector<T_data> &data)
        {
            static_assert(fixed_predictor==point_set_codec_e::inverse_proportion||fixed_predictor==point_set_codec_e::rbf_fitting,"固定数值预测器必须是IDW或RBF");
            if constexpr(fixed_predictor==point_set_codec_e::inverse_proportion)
                return predict_value_fixed<point_set_codec_e::inverse_proportion>(pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,data);
            else
                return predict_value_fixed<point_set_codec_e::rbf_fitting>(pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,data);
        };
        constexpr bool use_fused_two_point_set_prediction=two_point_sets&&fixed_execution==label_value_execution_e::fused;
        constexpr bool use_selected_fused_temporal_flow=use_fused_two_point_set_prediction&&((fixed_label_mode==time_dimension_mode_e::constant_extrapolation&&fixed_value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)||(fixed_label_mode==time_dimension_mode_e::endpoint_agreement&&fixed_value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual));
        bool data_dependent_eb_schedule=eb_schedule_experimental=="round_point_ratio_dag" || eb_schedule_experimental=="round_point_ratio_dag_direct" || eb_schedule_experimental=="round_point_ratio_dag_variance_direct";
        if(data_dependent_eb_schedule || cached_pred_eb_multiplier_vec.size()!=point_num)
        {
            cached_pred_eb_multiplier_vec=generate_pred_eb_multiplier_vec(true);
            cached_pred_eb_inverse_multiplier_vec.clear();
        }
        if(use_fused_two_point_set_prediction && !data_dependent_eb_schedule && cached_pred_eb_inverse_multiplier_vec.size()!=point_num)
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
        if(use_partition_stream_format && multi_huffman_mode_experimental!="none")
            throw runtime_error("partition独立码流不能和-multi-huffman-experimental同时使用");
        vector<vector<uint8_t>> label_symbol_vec_vec(label_channel_num);
        vector<vector<uint8_t>> label_symbol_file_order_vec_vec(label_channel_num);
        vector<vector<vector<uint8_t>>> partition_label_symbol_vec_vec(use_partition_stream_format?partition_pred_ref_ids_vec.size():0,vector<vector<uint8_t>>(label_channel_num));
        vector<uint64_t> label_channel_target_num_vec(label_channel_num,0),label_channel_error_num_vec(label_channel_num,0);
        vector<uint8_t> actual_label_prediction_order_vec;
        uint64_t label_prediction_error_num=0;
        if (point_set_num>1 && point_set_predictor!="inverse_proportion" && point_set_predictor!="majority" && point_set_predictor!="continuous_actual" && point_set_predictor!="continuous_reflect")
            throw runtime_error("未知的点集标签预测器："+point_set_predictor);
        bool continuous=point_set_predictor=="continuous_actual" || point_set_predictor=="continuous_reflect";
        bool use_majority=point_set_predictor=="majority";
        bool use_normalized_label_weights=use_normalized_weight_experimental && continuous;
        prepare_linear_interpolation();
        bool use_previous_time_label=prediction_framework.label_mode==time_dimension_mode_e::constant_extrapolation && previous_time_point_set_id_vec.size()==point_num;
        bool use_linear_previous_label=use_linear_previous_label_experimental && use_fused_two_point_set_prediction && use_previous_time_label;
        bool use_causal_time_label=prediction_framework.label_mode==time_dimension_mode_e::constant_extrapolation&&previous_time_point_set_id_vec.size()==point_num;
        bool use_original_order_constant_label_phase=use_selected_fused_temporal_flow&&prediction_framework.label_mode==time_dimension_mode_e::constant_extrapolation&&value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual&&use_previous_time_label;
        constexpr bool try_time_label=fixed_label_mode==time_dimension_mode_e::constant_extrapolation||fixed_label_mode==time_dimension_mode_e::endpoint_agreement;
        if(use_fused_two_point_set_prediction)
        {
            if(point_set_num!=2)
                throw runtime_error("融合两点集预测要求当前文件恰好包含两个点集");
            if(use_partition_stream_format || thread_num>1)
                throw runtime_error("融合两点集预测目前只支持单线程非partition路径");
            if(point_set_predictor!="inverse_proportion")
                throw runtime_error("融合两点集预测目前只支持反比例标签预测器");
            if(use_partition_adaptive_predictor_experimental || print_pointwise_oracle_statistics_experimental || print_prediction_residual_statistics)
                throw runtime_error("融合两点集预测目前不能和自适应预测器、逐点oracle或预测残差统计同时使用");
            if(use_selected_fused_temporal_flow&&use_linear_previous_label_experimental)
                throw runtime_error("选定的融合时间流程不使用线性上一时刻标签修正流");
            if(use_original_order_constant_label_phase&&use_prediction_order_label_experimental)
                throw runtime_error("标签零阶外插固定按原始点顺序存储，不能启用预测顺序标签码流");
            if(!use_selected_fused_temporal_flow&&value_mode!=time_dimension_mode_e::spatial)
                throw runtime_error("当前融合两点集流程不支持所选数值预测方式");
            for(point_set_codec_e codec:point_set_codec_vec)
                if(codec!=point_set_codec_e::constant&&codec!=point_set_codec_e::inverse_proportion&&!(use_selected_fused_temporal_flow&&codec==point_set_codec_e::rbf_fitting)&&!(use_linear_previous_label_experimental&&codec==point_set_codec_e::rbf_fitting))
                    throw runtime_error("融合两点集预测只支持constant、IDW以及选定时间流程中的RBF预测器");
        }
        uint32_t multi_huffman_group_num=get_multi_huffman_group_num();
        bool single_huffman_group=multi_huffman_group_num==1;
        vector<vector<vector<int32_t>>> point_set_group_qb_vec(point_set_num,vector<vector<int32_t>>(multi_huffman_group_num));
        vector<vector<vector<int32_t>>> partition_point_set_qb_vec(use_partition_stream_format?partition_pred_ref_ids_vec.size():0,vector<vector<int32_t>>(point_set_num));
        vector<uint64_t> point_set_qb_num_vec(point_set_num,0);
        vector<int32_t> point_set_qb_min_vec(point_set_num,numeric_limits<int32_t>::max()),point_set_qb_max_vec(point_set_num,numeric_limits<int32_t>::min());
        vector<vector<int32_t>> round_qb_vec(print_round_statistics?predict_round_num:0);
        if(use_fused_two_point_set_prediction && multi_huffman_group_num==1)
            for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                if(point_set_codec_vec[point_set_id]!=point_set_codec_e::constant)
                    point_set_group_qb_vec[point_set_id][0].reserve(point_set_point_num_vec[point_set_id]);
        uint64_t fused_value_prediction_num=0,fused_normalized_value_prediction_num=0;
        vector<double> point_set_inverse_double_abs_eb_vec(point_set_num,0);
        if(use_fused_two_point_set_prediction && !data_dependent_eb_schedule)
            for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                if(point_set_codec_vec[point_set_id]!=point_set_codec_e::constant)
                    point_set_inverse_double_abs_eb_vec[point_set_id]=0.5/point_set_abs_eb_vec[point_set_id];
        if(use_fused_two_point_set_prediction)
            current_decompressed_data.resize(point_num);
        for (uint32_t label_channel_id=0;label_channel_id<label_channel_num;label_channel_id++)
        {
            vector<uint8_t> label_symbol_internal_vec(point_num,0);
            vector<uint8_t> binary_label_vec=use_fused_two_point_set_prediction&&!use_selected_fused_temporal_flow?vector<uint8_t>():vector<uint8_t>(point_num,1);
            if(!use_fused_two_point_set_prediction)
                for (uint32_t point_id=0;point_id<point_num;point_id++)
                    if (point_set_id_vec[point_id]==label_channel_id)
                        binary_label_vec[point_id]=0;
            vector<double> binary_label_value_vec(continuous?point_num:0,0);
            if (continuous)
                for (uint32_t point_id=0;point_id<point_num;point_id++)
                    if (point_set_id_vec[point_id]<label_channel_id)
                        binary_label_value_vec[point_id]=1;
            if (print_label_compression_statistics || (!use_partition_stream_format && use_prediction_order_label_experimental))
                label_symbol_vec_vec[label_channel_id].reserve(point_num);
            if (print_label_compression_statistics && point_set_num==2)
                actual_label_prediction_order_vec.reserve(point_num);
            if(use_causal_time_label&&!use_fused_two_point_set_prediction)
            {
                uint32_t pred_ref_id=0;
                for(uint32_t round_id=0;round_id<predict_round_num;round_id++)
                    for(uint32_t round_target_id=0;round_target_id<round_target_ids[round_id].size();round_target_id++,pred_ref_id++)
                    {
                    int pred_id=pred_target_ids[pred_ref_id];
                    uint32_t actual_point_set_id=point_set_id_vec[pred_id];
                    if(actual_point_set_id<label_channel_id)
                        continue;
                    uint32_t original_id=reorder_data_by_prediction_order_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                    uint32_t predicted_label;
                    if(!predict_time_label_fixed<fixed_label_mode>(original_id,label_channel_id,predicted_label))
                        throw runtime_error("因果时间标签预测缺少历史标签");
                    uint32_t actual_label=actual_point_set_id==label_channel_id?0:1;
                    bool prediction_error=actual_label!=predicted_label;
                    label_channel_target_num_vec[label_channel_id]++;
                    label_channel_error_num_vec[label_channel_id]+=prediction_error;
                    label_prediction_error_num+=prediction_error;
                    label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                    if(print_label_compression_statistics || (!use_partition_stream_format && use_prediction_order_label_experimental))
                        label_symbol_vec_vec[label_channel_id].push_back(prediction_error?1:0);
                    if(print_label_compression_statistics && point_set_num==2)
                        actual_label_prediction_order_vec.push_back(actual_label);
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
                            auto ref_ids=get_pred_ref_ids(pred_ref_id);
                            auto ref_weights=get_pred_ref_weights(pred_ref_id,use_normalized_label_weights,false);
                            uint32_t original_id=reorder_data_by_prediction_order_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                            uint32_t predicted_label;
                            bool used_time_label=try_time_label&&predict_time_label_fixed<fixed_label_mode>(original_id,label_channel_id,predicted_label);
                            double predicted_value=used_time_label?predicted_label:(continuous?(use_normalized_label_weights?predict_binary_label_value_normalized(ref_ids,ref_weights,binary_label_value_vec):predict_binary_label_value(ref_ids,ref_weights,binary_label_value_vec)):0);
                            if(!used_time_label)
                                predicted_label=continuous?(predicted_value>=0.5):predict_binary_label(ref_ids,ref_weights,binary_label_vec,use_majority);
                            uint32_t actual_label=actual_point_set_id==label_channel_id?0:1;
                            bool prediction_error=actual_label!=predicted_label;
                            local_target_num++;
                            local_error_num+=prediction_error;
                            label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                            if(continuous)
                            {
                                if(!prediction_error)
                                    binary_label_value_vec[pred_id]=predicted_value;
                                else if(point_set_predictor=="continuous_actual" || predicted_value==0.5)
                                    binary_label_value_vec[pred_id]=actual_label;
                                else
                                    binary_label_value_vec[pred_id]=1-predicted_value;
                                uint32_t rounded_label=binary_label_value_vec[pred_id]>=0.5;
                                if(rounded_label!=actual_label)
                                    throw runtime_error("连续点集标签状态无法恢复真实标签");
                            }
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
                if(print_label_compression_statistics || (!use_partition_stream_format && use_prediction_order_label_experimental))
                    for(uint32_t pred_ref_id=0;pred_ref_id<pred_target_ids.size();pred_ref_id++)
                    {
                        int pred_id=pred_target_ids[pred_ref_id];
                        if(point_set_id_vec[pred_id]>=label_channel_id)
                            label_symbol_vec_vec[label_channel_id].push_back(label_symbol_internal_vec[pred_id]);
                    }
                if(print_label_compression_statistics && point_set_num==2)
                    for(int pred_id:pred_target_ids)
                        actual_label_prediction_order_vec.push_back(point_set_id_vec[pred_id]==label_channel_id?0:1);
            }
            else if(use_original_order_constant_label_phase)
            {
                if(label_channel_id!=0)
                    throw runtime_error("标签零阶外插只能处理一个标签通道");
                for(uint32_t original_id=0;original_id<point_num;original_id++)
                {
                    uint32_t internal_id=reorder_data_by_prediction_order_experimental?original_id_to_prediction_id[original_id]:original_id;
                    uint32_t actual_label=point_set_id_vec[internal_id];
                    bool prediction_error=actual_label!=previous_time_point_set_id_vec[original_id];
                    label_symbol_internal_vec[internal_id]=prediction_error?1:0;
                    label_channel_error_num_vec[0]+=prediction_error;
                    if(print_label_compression_statistics)
                        actual_label_prediction_order_vec.push_back((uint8_t)actual_label);
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
                for(uint32_t round_id=0;round_id<predict_round_num;round_id++)
                    for(uint32_t round_target_id=0;round_target_id<round_target_ids[round_id].size();round_target_id++,pred_ref_id++)
                    {
                        int pred_id=pred_target_ids[pred_ref_id];
                        uint32_t actual_point_set_id=point_set_id_vec[pred_id];
                        uint32_t original_id=reorder_data_by_prediction_order_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                        uint64_t ref_end=pred_ref_begin[pred_ref_id+1];
                        uint32_t ref_num=(uint32_t)(ref_end-ref_begin);
                        array_view_c<int> ref_ids(ref_num?pred_ref_ids.data()+ref_begin:nullptr,ref_num);
                        array_view_c<double> ref_weights(ref_num?pred_ref_weights.data()+ref_begin:nullptr,ref_num);
                        ref_begin=ref_end;
                        uint32_t predicted_label;
                        bool used_time_label=try_time_label&&predict_time_label_fixed<fixed_label_mode>(original_id,0,predicted_label);
                        if(!used_time_label)
                            predicted_label=predict_binary_label(ref_ids,ref_weights,binary_label_vec,false);
                        bool prediction_error=actual_point_set_id!=predicted_label;
                        label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                        if(print_label_compression_statistics||use_prediction_order_label_experimental)
                            label_symbol_vec_vec[0].push_back(prediction_error?1:0);
                        label_channel_error_num_vec[0]+=prediction_error;
                        binary_label_vec[pred_id]=(uint8_t)actual_point_set_id;
                        if(print_label_compression_statistics)
                            actual_label_prediction_order_vec.push_back((uint8_t)actual_point_set_id);
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
                            double qb_double=data_dependent_eb_schedule?round(((double)current_data[pred_id]-pred)/(2*point_abs_eb)):round(((double)current_data[pred_id]-pred)*point_set_inverse_double_abs_eb_vec[actual_point_set_id]*pred_eb_inverse_multiplier_vec[pred_ref_id]);
                            if(!isfinite(qb_double)||qb_double<numeric_limits<int32_t>::min()||qb_double>numeric_limits<int32_t>::max())
                                throw runtime_error("量化索引超出int32_t范围，请增大误差上限");
                            int32_t qb=(int32_t)qb_double;
                            uint32_t huffman_group_id=single_huffman_group?0:get_multi_huffman_group_id(round_id,pred_id);
                            point_set_group_qb_vec[actual_point_set_id][huffman_group_id].push_back(qb);
                            if(print_round_statistics)
                                round_qb_vec[round_id].push_back(qb);
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
                    if(point_set_codec_vec[point_set_id]!=point_set_codec_e::constant)
                        fused_value_prediction_num+=point_set_point_num_vec[point_set_id];
                    for(const vector<int32_t> &group_qb_vec:point_set_group_qb_vec[point_set_id])
                        point_set_qb_num_vec[point_set_id]+=group_qb_vec.size();
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
                        uint32_t internal_id=reorder_data_by_prediction_order_experimental?original_id_to_prediction_id[original_id]:original_id;
                        bool prediction_error=point_set_id_vec[internal_id]!=previous_time_point_set_id_vec[original_id];
                        label_symbol_internal_vec[internal_id]=prediction_error?1:0;
                        label_channel_error_num_vec[0]+=prediction_error;
                    }
                    label_channel_target_num_vec[0]=point_num;
                }
                uint32_t pred_ref_id=0;
                uint64_t ref_begin=pred_ref_begin[0];
                for(uint32_t round_id=0;round_id<predict_round_num;round_id++)
                    for(uint32_t round_target_id=0;round_target_id<round_target_ids[round_id].size();round_target_id++,pred_ref_id++)
                    {
                        int pred_id=pred_target_ids[pred_ref_id];
                        uint32_t actual_point_set_id=point_set_id_vec[pred_id];
                        uint64_t ref_end=pred_ref_begin[pred_ref_id+1];
                        uint32_t ref_num=(uint32_t)(ref_end-ref_begin);
                        array_view_c<int> ref_ids(ref_num?pred_ref_ids.data()+ref_begin:nullptr,ref_num);
                        array_view_c<double> ref_weights(ref_num?pred_ref_weights.data()+ref_begin:nullptr,ref_num);
                        array_view_c<double> normalized_ref_weights=use_normalized_weight_experimental?array_view_c<double>(ref_num?pred_ref_value_normalized_weights.data()+ref_begin:nullptr,ref_num):array_view_c<double>();
                        ref_begin=ref_end;
                        T_data predicted_value;
                        bool predict_value=point_set_codec_vec[actual_point_set_id]!=point_set_codec_e::constant;
                        bool used_normalized_value;
                        if(use_previous_time_label)
                        {
                            if(point_set_codec_vec[actual_point_set_id]==point_set_codec_e::rbf_fitting)
                            {
                                predicted_value=predict_value?predict_configured_value(pred_ref_id,pred_id,ref_ids,ref_weights,actual_point_set_id,current_decompressed_data):(T_data)0;
                                used_normalized_value=false;
                            }
                            else
                                used_normalized_value=predict_known_point_set_inverse_value(ref_ids,ref_weights,normalized_ref_weights,current_decompressed_data,actual_point_set_id,predict_value,predicted_value);
                            if(!use_linear_previous_label)
                            {
                                uint32_t original_id=reorder_data_by_prediction_order_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                                uint32_t predicted_label=previous_time_point_set_id_vec[original_id];
                                bool prediction_error=actual_point_set_id!=predicted_label;
                                label_channel_error_num_vec[0]+=prediction_error;
                                if(print_label_compression_statistics || use_prediction_order_label_experimental)
                                    label_symbol_vec_vec[0].push_back(prediction_error?1:0);
                                label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                            }
                        }
                        else
                        {
                            uint32_t predicted_label;
                            if(point_set_codec_vec[actual_point_set_id]==point_set_codec_e::rbf_fitting)
                            {
                                T_data unused_inverse_prediction;
                                predict_binary_label_and_inverse_value(ref_ids,ref_weights,normalized_ref_weights,current_decompressed_data,actual_point_set_id,false,predicted_label,unused_inverse_prediction);
                                predicted_value=predict_value?predict_configured_value(pred_ref_id,pred_id,ref_ids,ref_weights,actual_point_set_id,current_decompressed_data):(T_data)0;
                                used_normalized_value=false;
                            }
                            else
                                used_normalized_value=predict_binary_label_and_inverse_value(ref_ids,ref_weights,normalized_ref_weights,current_decompressed_data,actual_point_set_id,predict_value,predicted_label,predicted_value);
                            bool prediction_error=actual_point_set_id!=predicted_label;
                            label_channel_error_num_vec[0]+=prediction_error;
                            if(print_label_compression_statistics || use_prediction_order_label_experimental)
                                label_symbol_vec_vec[0].push_back(prediction_error?1:0);
                            label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                        }
                        fused_normalized_value_prediction_num+=used_normalized_value;
                        if(print_label_compression_statistics)
                            actual_label_prediction_order_vec.push_back(actual_point_set_id);
                        if(!predict_value)
                            current_decompressed_data[pred_id]=point_set_min_data_vec[actual_point_set_id];
                        else
                        {
                            double point_abs_eb=point_set_abs_eb_vec[actual_point_set_id]*pred_eb_multiplier_vec[pred_ref_id];
                            T_data pred=predicted_value;
                            double qb_double=data_dependent_eb_schedule?round(((double)current_data[pred_id]-pred)/(2*point_abs_eb)):round(((double)current_data[pred_id]-pred)*point_set_inverse_double_abs_eb_vec[actual_point_set_id]*pred_eb_inverse_multiplier_vec[pred_ref_id]);
                            if(!isfinite(qb_double) || qb_double<numeric_limits<int32_t>::min() || qb_double>numeric_limits<int32_t>::max())
                                throw runtime_error("量化索引超出int32_t范围，请增大误差上限");
                            int32_t qb=(int32_t)qb_double;
                            uint32_t huffman_group_id=single_huffman_group?0:get_multi_huffman_group_id(round_id,pred_id);
                            point_set_group_qb_vec[actual_point_set_id][huffman_group_id].push_back(qb);
                            if(print_round_statistics)
                                round_qb_vec[round_id].push_back(qb);
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
                    if(point_set_codec_vec[point_set_id]!=point_set_codec_e::constant)
                        fused_value_prediction_num+=point_set_point_num_vec[point_set_id];
                    for(const vector<int32_t> &group_qb_vec:point_set_group_qb_vec[point_set_id])
                        point_set_qb_num_vec[point_set_id]+=group_qb_vec.size();
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
                    auto ref_ids=get_pred_ref_ids(pred_ref_id);
                    auto ref_weights=get_pred_ref_weights(pred_ref_id,use_normalized_label_weights,false);
                    uint32_t original_id=reorder_data_by_prediction_order_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                    uint32_t predicted_label;
                    bool used_time_label=try_time_label&&predict_time_label_fixed<fixed_label_mode>(original_id,label_channel_id,predicted_label);
                    double predicted_value=used_time_label?predicted_label:(continuous?(use_normalized_label_weights?predict_binary_label_value_normalized(ref_ids,ref_weights,binary_label_value_vec):predict_binary_label_value(ref_ids,ref_weights,binary_label_value_vec)):0);
                    if(!used_time_label)
                        predicted_label=continuous?(predicted_value>=0.5):predict_binary_label(ref_ids,ref_weights,binary_label_vec,use_majority);
                    uint32_t actual_label=actual_point_set_id==label_channel_id?0:1;
                    bool prediction_error=actual_label!=predicted_label;
                    label_channel_target_num_vec[label_channel_id]++;
                    label_channel_error_num_vec[label_channel_id]+=prediction_error;
                    label_prediction_error_num+=prediction_error;
                    if(print_label_compression_statistics && point_set_num==2)
                        actual_label_prediction_order_vec.push_back(actual_label);
                    if(print_label_compression_statistics || use_prediction_order_label_experimental)
                        label_symbol_vec_vec[label_channel_id].push_back(prediction_error?1:0);
                    label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                    if(continuous)
                    {
                        if(!prediction_error)
                            binary_label_value_vec[pred_id]=predicted_value;
                        else if(point_set_predictor=="continuous_actual" || predicted_value==0.5)
                            binary_label_value_vec[pred_id]=actual_label;
                        else
                            binary_label_value_vec[pred_id]=1-predicted_value;
                        uint32_t rounded_label=binary_label_value_vec[pred_id]>=0.5;
                        if(rounded_label!=actual_label)
                            throw runtime_error("连续点集标签状态无法恢复真实标签");
                    }
                }
            }
            if ((!use_partition_stream_format && !use_prediction_order_label_experimental) || print_label_compression_statistics)
            {
                if(use_fused_two_point_set_prediction && label_channel_id==0 && !reorder_data_by_prediction_order_experimental)
                    label_symbol_file_order_vec_vec[label_channel_id]=move(label_symbol_internal_vec);
                else
                {
                    label_symbol_file_order_vec_vec[label_channel_id].reserve(label_channel_target_num_vec[label_channel_id]);
                    for (uint32_t original_id=0;original_id<point_num;original_id++)
                    {
                        uint32_t internal_id=reorder_data_by_prediction_order_experimental?original_id_to_prediction_id[original_id]:original_id;
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
        if(print_compression_stage_time && use_fused_two_point_set_prediction && use_normalized_weight_experimental)
            printf("fused normalized value fast path: hits=%llu, value_predictions=%llu, ratio=%f\n",(unsigned long long)fused_normalized_value_prediction_num,(unsigned long long)fused_value_prediction_num,fused_value_prediction_num?(double)fused_normalized_value_prediction_num/fused_value_prediction_num:0);

        stage_start_time=chrono::steady_clock::now();
        if(!use_fused_two_point_set_prediction)
            current_decompressed_data.resize(point_num);
        uint64_t value_huffman_byte_num=0;
        uint32_t pred_ref_index=0;
        vector<T_data> oracle_decompressed_data;
        vector<int32_t> oracle_qb_vec,oracle_selector_int_vec;
        vector<unsigned char> oracle_selector_vec;
        vector<partition_prediction_config_c> oracle_config_vec;
        vector<uint64_t> oracle_config_count_vec;
        double oracle_squared_error=0;
        if (print_pointwise_oracle_statistics_experimental)
        {
            if (point_set_num!=1 || point_set_codec_vec[0]==point_set_codec_e::constant)
                throw runtime_error("逐点oracle统计当前只支持单个非常量点集");
            oracle_config_vec=get_partition_prediction_config_vec();
            if (oracle_config_vec.size()>256)
                throw runtime_error("逐点oracle候选配置超过256个");
            oracle_decompressed_data.assign(point_num,0);
            oracle_qb_vec.reserve(point_num);
            oracle_selector_int_vec.reserve(point_num);
            oracle_selector_vec.reserve(point_num);
            oracle_config_count_vec.assign(oracle_config_vec.size(),0);
        }
        bool use_normalized_value_weights=!two_point_sets&&use_normalized_weight_experimental&&point_set_codec_vec[0]==point_set_codec_e::inverse_proportion;
        bool has_previous_time=value_mode==time_dimension_mode_e::constant_extrapolation&&previous_time_decompressed_data.size()==point_num&&previous_time_point_set_id_vec.size()==point_num;
        prepare_linear_interpolation();
        uint64_t cross_time_prediction_num=0;
        double prediction_residual_squared_error=0,prediction_residual_absolute_error=0;
        uint64_t prediction_residual_num=0;
        bool use_single_point_set_rbf_fast_path=!two_point_sets&&point_set_codec_vec[0]==point_set_codec_e::rbf_fitting&&value_mode==time_dimension_mode_e::spatial&&!use_partition_stream_format&&!use_partition_adaptive_predictor_experimental&&!print_pointwise_oracle_statistics_experimental&&!print_round_statistics&&!print_prediction_residual_statistics&&single_huffman_group&&pred_ref_rbf_weights.size()==pred_ref_ids.size();
        bool use_multihead_execution=!multihead_execution_pred_ref_ids.empty();
        if(use_multihead_execution&&!use_single_point_set_rbf_fast_path)
            throw runtime_error("多头执行要求单点集纯空间RBF快速路径");
        bool store_qb_in_original_order=qb_storage_order=="original";
        if(store_qb_in_original_order&&!use_single_point_set_rbf_fast_path)
            throw runtime_error("原始顺序qb存储目前只支持单点集纯空间RBF路径");
        if(use_fused_two_point_set_prediction&&!use_original_order_constant_label_phase)
            pred_ref_index=point_num;
        else if(use_single_point_set_rbf_fast_path)
        {
            vector<int32_t> &qb_vec=point_set_group_qb_vec[0][0];
            if(store_qb_in_original_order||use_multihead_execution)
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
                else if(store_qb_in_original_order)
                {
                    uint32_t original_id=reorder_data_by_prediction_order_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                    qb_vec[original_id]=qb;
                }
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
                    if(pred_ref_id<single_point_set_rbf10_fast_begin)
                        encode_prediction(pred_ref_id,predict_single_point_set_rbf_value(pred_ref_id,current_decompressed_data));
                    else
                        encode_prediction(pred_ref_id,predict_single_point_set_rbf10_value(pred_ref_id,pred_ref_begin[pred_ref_id],current_decompressed_data));
                }
            }
            else
            {
                for(uint32_t pred_ref_id=0;pred_ref_id<single_point_set_rbf10_fast_begin;pred_ref_id++)
                    encode_prediction(pred_ref_id,predict_single_point_set_rbf_value(pred_ref_id,current_decompressed_data));
                uint64_t ref_begin=pred_ref_begin[single_point_set_rbf10_fast_begin];
                for(uint32_t pred_ref_id=single_point_set_rbf10_fast_begin;pred_ref_id<point_num;pred_ref_id++,ref_begin+=10)
                    encode_prediction(pred_ref_id,predict_single_point_set_rbf10_value(pred_ref_id,ref_begin,current_decompressed_data));
            }
            point_set_qb_num_vec[0]=point_num;
            pred_ref_index=point_num;
        }
        else if(use_partition_stream_format)
        {
            if(partition_pred_ref_ids_vec.empty())
                throw runtime_error("多线程数值预测需要独立partition预测DAG");
            if(print_pointwise_oracle_statistics_experimental)
                throw runtime_error("多线程数值预测暂不支持逐点oracle统计");
            vector<int32_t> parallel_qb_vec(print_round_statistics?point_num:0);
            vector<uint8_t> parallel_has_qb_vec(print_round_statistics?point_num:0,0);
            vector<vector<uint64_t>> partition_point_set_qb_num_vec(partition_pred_ref_ids_vec.size(),vector<uint64_t>(point_set_num,0));
            vector<vector<int32_t>> partition_point_set_qb_min_vec(partition_pred_ref_ids_vec.size(),vector<int32_t>(point_set_num,numeric_limits<int32_t>::max())),partition_point_set_qb_max_vec(partition_pred_ref_ids_vec.size(),vector<int32_t>(point_set_num,numeric_limits<int32_t>::min()));
            vector<uint64_t> partition_cross_time_prediction_num_vec(partition_pred_ref_ids_vec.size(),0);
            vector<double> partition_prediction_residual_squared_error_vec(partition_pred_ref_ids_vec.size(),0),partition_prediction_residual_absolute_error_vec(partition_pred_ref_ids_vec.size(),0);
            vector<uint64_t> partition_prediction_residual_num_vec(partition_pred_ref_ids_vec.size(),0);
            exception_ptr prediction_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num)
            for(int64_t partition_index=0;partition_index<(int64_t)partition_pred_ref_ids_vec.size();partition_index++)
            {
                try
                {
                    uint32_t partition_id=(uint32_t)partition_index;
                    uint64_t local_cross_time_prediction_num=0;
                    double local_prediction_residual_squared_error=0,local_prediction_residual_absolute_error=0;
                    uint64_t local_prediction_residual_num=0;
                    for(uint32_t current_pred_ref_id:partition_pred_ref_ids_vec[partition_id])
                    {
                        auto ref_ids=get_pred_ref_ids(current_pred_ref_id);
                        auto ref_weights=get_pred_ref_weights(current_pred_ref_id,false,use_normalized_value_weights);
                        int pred_id=pred_target_ids[current_pred_ref_id];
                        uint32_t point_set_id=two_point_sets?point_set_id_vec[pred_id]:0;
                        if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                        {
                            current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                            uint32_t original_id=reorder_data_by_prediction_order_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                            update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,point_set_id);
                            continue;
                        }
                        bool use_partition_config=use_partition_adaptive_predictor_experimental && !partition_adaptive_sample_mask[pred_id];
                        uint32_t original_id=reorder_data_by_prediction_order_experimental?prediction_id_to_original_id[pred_id]:pred_id;
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
                                pred+=use_partition_config?predict_partition_adaptive_value(pred_id,ref_ids,current_time_spatial_residual_data,partition_prediction_config_id_vec[partition_id]):(use_normalized_value_weights?predict_value_normalized(ref_ids,ref_weights,current_time_spatial_residual_data):predict_configured_value(current_pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_time_spatial_residual_data));
                        }
                        if(!use_cross_time_prediction)
                            pred=use_partition_config?predict_partition_adaptive_value(pred_id,ref_ids,current_decompressed_data,partition_prediction_config_id_vec[partition_id]):(use_normalized_value_weights?predict_value_normalized(ref_ids,ref_weights,current_decompressed_data):predict_configured_value(current_pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_decompressed_data));
                        local_cross_time_prediction_num+=use_cross_time_prediction;
                        double prediction_residual=(double)current_data[pred_id]-pred;
                        local_prediction_residual_squared_error+=prediction_residual*prediction_residual;
                        local_prediction_residual_absolute_error+=abs(prediction_residual);
                        local_prediction_residual_num++;
                        double point_abs_eb=point_set_abs_eb_vec[point_set_id]*pred_eb_multiplier_vec[current_pred_ref_id];
                        double qb_double=round(((double)current_data[pred_id]-pred)/(2*point_abs_eb));
                        if(!isfinite(qb_double) || qb_double<numeric_limits<int32_t>::min() || qb_double>numeric_limits<int32_t>::max())
                            throw runtime_error("量化索引超出int32_t范围，请增大误差上限");
                        int32_t qb=(int32_t)qb_double;
                        if(print_round_statistics)
                        {
                            parallel_qb_vec[current_pred_ref_id]=qb;
                            parallel_has_qb_vec[current_pred_ref_id]=1;
                        }
                        partition_point_set_qb_vec[partition_id][point_set_id].push_back(qb);
                        partition_point_set_qb_num_vec[partition_id][point_set_id]++;
                        partition_point_set_qb_min_vec[partition_id][point_set_id]=min(partition_point_set_qb_min_vec[partition_id][point_set_id],qb);
                        partition_point_set_qb_max_vec[partition_id][point_set_id]=max(partition_point_set_qb_max_vec[partition_id][point_set_id],qb);
                        current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                        update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,point_set_id);
                    }
                    partition_cross_time_prediction_num_vec[partition_id]=local_cross_time_prediction_num;
                    partition_prediction_residual_squared_error_vec[partition_id]=local_prediction_residual_squared_error;
                    partition_prediction_residual_absolute_error_vec[partition_id]=local_prediction_residual_absolute_error;
                    partition_prediction_residual_num_vec[partition_id]=local_prediction_residual_num;
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
            prediction_residual_squared_error=accumulate(partition_prediction_residual_squared_error_vec.begin(),partition_prediction_residual_squared_error_vec.end(),0.0);
            prediction_residual_absolute_error=accumulate(partition_prediction_residual_absolute_error_vec.begin(),partition_prediction_residual_absolute_error_vec.end(),0.0);
            prediction_residual_num=accumulate(partition_prediction_residual_num_vec.begin(),partition_prediction_residual_num_vec.end(),(uint64_t)0);
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
            if(print_round_statistics)
            {
                uint32_t assemble_pred_ref_id=0;
                for(uint32_t round_id=0;round_id<predict_round_num;round_id++)
                    for(uint32_t i=0;i<round_target_ids[round_id].size();i++)
                    {
                        uint32_t current_pred_ref_id=assemble_pred_ref_id++;
                        if(parallel_has_qb_vec[current_pred_ref_id])
                            round_qb_vec[round_id].push_back(parallel_qb_vec[current_pred_ref_id]);
                    }
            }
        }
        else
        {
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            for (uint32_t i=0;i<round_target_ids[round_id].size();i++)
            {
                uint32_t current_pred_ref_id=pred_ref_index++;
                auto ref_ids=get_pred_ref_ids(current_pred_ref_id);
                auto ref_weights=get_pred_ref_weights(current_pred_ref_id,false,use_normalized_value_weights);
                int pred_id=pred_target_ids[current_pred_ref_id];
                uint32_t point_set_id=two_point_sets?point_set_id_vec[pred_id]:0;
                double eb_multiplier=pred_eb_multiplier_vec[current_pred_ref_id];
                if (point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                {
                    current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                    uint32_t original_id=reorder_data_by_prediction_order_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                    update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,point_set_id);
                    if (print_pointwise_oracle_statistics_experimental)
                        oracle_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                    continue;
                }
                bool use_partition_config=use_partition_adaptive_predictor_experimental && !partition_adaptive_sample_mask[pred_id];
                uint32_t original_id=reorder_data_by_prediction_order_experimental?prediction_id_to_original_id[pred_id]:pred_id;
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
                        pred+=use_partition_config?predict_partition_adaptive_value(pred_id,ref_ids,current_time_spatial_residual_data,partition_prediction_config_id_vec[point_partition_id_vec[pred_id]]):(use_normalized_value_weights?predict_value_normalized(ref_ids,ref_weights,current_time_spatial_residual_data):predict_configured_value(current_pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_time_spatial_residual_data));
                }
                if (!use_cross_time_prediction)
                    pred=use_partition_config?predict_partition_adaptive_value(pred_id,ref_ids,current_decompressed_data,partition_prediction_config_id_vec[point_partition_id_vec[pred_id]]):(use_normalized_value_weights?predict_value_normalized(ref_ids,ref_weights,current_decompressed_data):predict_configured_value(current_pred_ref_id,pred_id,ref_ids,ref_weights,point_set_id,current_decompressed_data));
                cross_time_prediction_num+=use_cross_time_prediction;
                double prediction_residual=(double)current_data[pred_id]-pred;
                prediction_residual_squared_error+=prediction_residual*prediction_residual;
                prediction_residual_absolute_error+=abs(prediction_residual);
                prediction_residual_num++;
                double point_abs_eb=point_set_abs_eb_vec[point_set_id]*eb_multiplier;
                if (print_pointwise_oracle_statistics_experimental)
                {
                    uint8_t best_config_id=partition_prediction_config_id_vec[point_partition_id_vec[pred_id]];
                    T_data best_prediction=predict_partition_adaptive_value(pred_id,ref_ids,oracle_decompressed_data,oracle_config_vec[best_config_id]);
                    double best_error=abs((double)current_data[pred_id]-best_prediction);
                    for (uint32_t config_id=0;config_id<oracle_config_vec.size();config_id++)
                    {
                        if (config_id==best_config_id)
                            continue;
                        T_data candidate_prediction=predict_partition_adaptive_value(pred_id,ref_ids,oracle_decompressed_data,oracle_config_vec[config_id]);
                        double candidate_error=abs((double)current_data[pred_id]-candidate_prediction);
                        if (candidate_error<best_error)
                        {
                            best_error=candidate_error;
                            best_prediction=candidate_prediction;
                            best_config_id=(uint8_t)config_id;
                        }
                    }
                    double oracle_qb_double=round(((double)current_data[pred_id]-best_prediction)/(2*point_abs_eb));
                    if (!isfinite(oracle_qb_double) || oracle_qb_double<numeric_limits<int32_t>::min() || oracle_qb_double>numeric_limits<int32_t>::max())
                        throw runtime_error("逐点oracle量化索引超出int32_t范围");
                    int32_t oracle_qb=(int32_t)oracle_qb_double;
                    oracle_qb_vec.push_back(oracle_qb);
                    oracle_selector_int_vec.push_back(best_config_id);
                    oracle_selector_vec.push_back(best_config_id);
                    oracle_config_count_vec[best_config_id]++;
                    oracle_decompressed_data[pred_id]=best_prediction+oracle_qb*2*point_abs_eb;
                    double oracle_error=(double)current_data[pred_id]-oracle_decompressed_data[pred_id];
                    oracle_squared_error+=oracle_error*oracle_error;
                }
                double qb_double=round(((double)current_data[pred_id]-pred)/(2*point_abs_eb));
                if (!isfinite(qb_double) || qb_double<numeric_limits<int32_t>::min() || qb_double>numeric_limits<int32_t>::max())
                    throw runtime_error("量化索引超出int32_t范围，请增大误差上限");
                int32_t qb=(int32_t)qb_double;
                uint32_t huffman_group_id=single_huffman_group?0:get_multi_huffman_group_id(round_id,pred_id);
                point_set_group_qb_vec[point_set_id][huffman_group_id].push_back(qb);
                point_set_qb_num_vec[point_set_id]++;
                point_set_qb_min_vec[point_set_id]=min(point_set_qb_min_vec[point_set_id],qb);
                point_set_qb_max_vec[point_set_id]=max(point_set_qb_max_vec[point_set_id],qb);
                if (print_round_statistics)
                    round_qb_vec[round_id].push_back(qb);
                current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
                update_time_spatial_residual_fixed<fixed_value_mode>(pred_id,original_id,point_set_id);
            }
        }
        }
        if (pred_ref_index!=pred_target_ids.size())
            throw runtime_error("逐轮预测目标数量与预测参考结构不一致");
        if(value_mode==time_dimension_mode_e::constant_extrapolation||value_mode==time_dimension_mode_e::linear_interpolation||value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
            printf("cross_time_prediction_num=%llu, ratio=%f\n",(unsigned long long)cross_time_prediction_num,(double)cross_time_prediction_num/point_num);
        if(print_prediction_residual_statistics)
        {
            map<int32_t,uint64_t> qb_count;
            uint64_t qb_num=0,qb_zero_num=0,qb_same_as_previous_num=0,qb_adjacent_pair_num=0;
            auto add_qb_vec=[&](const vector<int32_t> &current_qb_vec)
            {
                for(int32_t qb:current_qb_vec)
                {
                    qb_count[qb]++;
                    qb_zero_num+=qb==0;
                }
                qb_num+=current_qb_vec.size();
                for(uint64_t qb_id=1;qb_id<current_qb_vec.size();qb_id++)
                    qb_same_as_previous_num+=current_qb_vec[qb_id]==current_qb_vec[qb_id-1];
                if(!current_qb_vec.empty())
                    qb_adjacent_pair_num+=current_qb_vec.size()-1;
            };
            if(use_partition_stream_format)
                for(const auto &partition_qb_vec:partition_point_set_qb_vec)
                    for(const auto &current_qb_vec:partition_qb_vec)
                        add_qb_vec(current_qb_vec);
            else
                for(const auto &point_set_qb_vec:point_set_group_qb_vec)
                    for(const auto &current_qb_vec:point_set_qb_vec)
                        add_qb_vec(current_qb_vec);
            double qb_entropy_br=0;
            for(const auto &[qb,count]:qb_count)
            {
                double probability=(double)count/qb_num;
                qb_entropy_br-=probability*log2(probability);
            }
            printf("prediction_residual_stats: num=%llu, mse=%.17g, mae=%.17g, qb_entropy_br=%.17g, qb_zero_ratio=%.17g, qb_same_as_previous_ratio=%.17g\n",(unsigned long long)prediction_residual_num,prediction_residual_num?prediction_residual_squared_error/prediction_residual_num:0,prediction_residual_num?prediction_residual_absolute_error/prediction_residual_num:0,qb_entropy_br,qb_num?(double)qb_zero_num/qb_num:0,qb_adjacent_pair_num?(double)qb_same_as_previous_num/qb_adjacent_pair_num:0);
        }
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
                        vector<unsigned char> zstd_bytes=delta_bytes.empty()?vector<unsigned char>():zstd_compress(delta_bytes,zstd_compression_level);
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
                        vector<unsigned char> zstd_bytes=huffman_bytes.empty()?vector<unsigned char>():zstd_compress(huffman_bytes,zstd_compression_level);
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
            label_position_delta_bytes_vec[label_channel_id]=use_prediction_order_label_experimental?binary_one_position_delta_vec(label_symbol_vec_vec[label_channel_id]):binary_one_position_delta_vec(label_symbol_file_order_vec_vec[label_channel_id]);
            if (!label_position_delta_bytes_vec[label_channel_id].empty())
                label_zstd_bytes_vec[label_channel_id]=zstd_compress(label_position_delta_bytes_vec[label_channel_id],zstd_compression_level);
            label_position_delta_byte_num+=label_position_delta_bytes_vec[label_channel_id].size();
            label_zstd_byte_num+=label_zstd_bytes_vec[label_channel_id].size();
            if (!use_legacy_point_set_format)
                printf("label_channel%u_target_num=%llu, error_ratio=%f, position_delta_bytes=%llu, zstd_bytes=%llu\n",label_channel_id,(unsigned long long)label_channel_target_num_vec[label_channel_id],label_channel_target_num_vec[label_channel_id]?(double)label_channel_error_num_vec[label_channel_id]/label_channel_target_num_vec[label_channel_id]:0,(unsigned long long)label_position_delta_bytes_vec[label_channel_id].size(),(unsigned long long)label_zstd_bytes_vec[label_channel_id].size());
        }
        if (point_set_num==2 && print_label_compression_statistics)
        {
            const vector<uint8_t> &label_symbol_vec=label_symbol_vec_vec[0];
            const vector<uint8_t> &label_symbol_file_order_vec=label_symbol_file_order_vec_vec[0];
            const vector<unsigned char> &label_position_delta_bytes=label_position_delta_bytes_vec[0];
            for (int zstd_level:{1,3,6,9,15,19})
                printf("label_position_delta_zstd_level=%d, bytes=%llu\n",zstd_level,(unsigned long long)zstd_compress(label_position_delta_bytes,zstd_level).size());
            auto print_label_candidate=[&](const char *name,vector<unsigned char> bytes)
            {
                printf("label_candidate=%s, bytes_before_zstd=%llu, bytes_after_zstd=%llu\n",name,
                    (unsigned long long)bytes.size(),(unsigned long long)zstd_size(bytes));
            };
            vector<uint8_t> point_set_id_file_order_vec(point_num);
            if (reorder_data_by_prediction_order_experimental)
                for (uint32_t prediction_id=0;prediction_id<point_num;prediction_id++)
                    point_set_id_file_order_vec[prediction_id_to_original_id[prediction_id]]=point_set_id_vec[prediction_id];
            else
                point_set_id_file_order_vec=point_set_id_vec;
            print_label_candidate("actual_original_byte",binary_to_byte_vec(point_set_id_file_order_vec));
            print_label_candidate("actual_original_bit",binary_to_bit_vec(point_set_id_file_order_vec));
            print_label_candidate("actual_original_rle",binary_run_length_vec(point_set_id_file_order_vec));
            print_label_candidate("actual_prediction_order_byte",binary_to_byte_vec(actual_label_prediction_order_vec));
            print_label_candidate("actual_prediction_order_bit",binary_to_bit_vec(actual_label_prediction_order_vec));
            print_label_candidate("error_byte",binary_to_byte_vec(label_symbol_vec));
            print_label_candidate("error_bit",binary_to_bit_vec(label_symbol_vec));
            print_label_candidate("error_position_delta",binary_one_position_delta_vec(label_symbol_vec));
            print_label_candidate("error_rle",binary_run_length_vec(label_symbol_vec));
            print_label_candidate("error_original_order_bit",binary_to_bit_vec(label_symbol_file_order_vec));
            print_label_candidate("error_original_order_position_delta",binary_one_position_delta_vec(label_symbol_file_order_vec));
            print_label_candidate("error_original_order_rle",binary_run_length_vec(label_symbol_file_order_vec));
            uint64_t round_bit_bytes=0,round_zstd_bytes=0,label_begin=0;
            for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
            {
                uint64_t label_end=label_begin+round_target_ids[round_id].size();
                vector<uint8_t> round_label_vec(label_symbol_vec.begin()+label_begin,label_symbol_vec.begin()+label_end);
                vector<unsigned char> round_bit_vec=binary_to_bit_vec(round_label_vec);
                round_bit_bytes+=round_bit_vec.size();
                round_zstd_bytes+=zstd_size(round_bit_vec);
                label_begin=label_end;
            }
            printf("label_candidate=error_bit_by_round, bytes_before_zstd=%llu, bytes_after_zstd=%llu\n",
                (unsigned long long)round_bit_bytes,(unsigned long long)round_zstd_bytes);
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
            uint64_t huffman_payload_byte_num=0,multi_huffman_metadata_byte_num=0;
            vector<unsigned char> qb_huffman_bytes;
            if (multi_huffman_mode_experimental=="none")
            {
                qb_huffman_bytes=huffman_encode(point_set_group_qb_vec[point_set_id][0],&point_set_qb_min_vec[point_set_id],&point_set_qb_max_vec[point_set_id]);
                huffman_payload_byte_num=qb_huffman_bytes.size();
            }
            else
            {
                qb_huffman_bytes=encode_multi_huffman_groups(point_set_group_qb_vec[point_set_id],huffman_payload_byte_num,multi_huffman_metadata_byte_num);
                if(use_fused_two_point_set_prediction && point_set_qb_num_vec[point_set_id]>0)
                    for(const vector<int32_t> &group_qb_vec:point_set_group_qb_vec[point_set_id])
                        if(!group_qb_vec.empty())
                        {
                            auto range=minmax_element(group_qb_vec.begin(),group_qb_vec.end());
                            point_set_qb_min_vec[point_set_id]=min(point_set_qb_min_vec[point_set_id],*range.first);
                            point_set_qb_max_vec[point_set_id]=max(point_set_qb_max_vec[point_set_id],*range.second);
                        }
            }
            vector<unsigned char> qb_zstd_bytes=qb_huffman_bytes.empty()?vector<unsigned char>():zstd_compress(qb_huffman_bytes,zstd_compression_level);
            if (!qb_huffman_bytes.empty() && print_label_compression_statistics)
                for (int zstd_level:{1,3,6,9,15,19})
                    printf("set%u_zstd_level=%d, bytes=%llu\n",point_set_id,zstd_level,(unsigned long long)zstd_compress(qb_huffman_bytes,zstd_level).size());
            value_huffman_byte_num+=qb_huffman_bytes.size();
            value_zstd_byte_num+=qb_zstd_bytes.size();
            append_binary<uint64_t>(container,qb_zstd_bytes.size());
            container.insert(container.end(),qb_zstd_bytes.begin(),qb_zstd_bytes.end());
            int32_t qb_min=point_set_qb_num_vec[point_set_id]==0?0:point_set_qb_min_vec[point_set_id];
            int32_t qb_max=point_set_qb_num_vec[point_set_id]==0?0:point_set_qb_max_vec[point_set_id];
            printf("set%u_qb_num=%llu, set%u_qb_range=[%d,%d], set%u_huffman_bytes=%llu, set%u_zstd_bytes=%llu\n",point_set_id,
                (unsigned long long)point_set_qb_num_vec[point_set_id],point_set_id,qb_min,qb_max,point_set_id,(unsigned long long)qb_huffman_bytes.size(),
                point_set_id,(unsigned long long)qb_zstd_bytes.size());
            if (multi_huffman_mode_experimental!="none" && point_set_qb_num_vec[point_set_id]>0)
                printf("set%u_multi_huffman_mode=%s, group_num=%u, metadata_bytes=%llu, huffman_payload_bytes=%llu\n",point_set_id,multi_huffman_mode_experimental.c_str(),multi_huffman_group_num,(unsigned long long)multi_huffman_metadata_byte_num,(unsigned long long)huffman_payload_byte_num);
        }
        current_zstd_bs_vec=move(container);
        }
        if (print_pointwise_oracle_statistics_experimental)
        {
            vector<unsigned char> oracle_qb_huffman_bytes=huffman_encode(oracle_qb_vec);
            vector<unsigned char> oracle_qb_zstd_bytes=zstd_compress(oracle_qb_huffman_bytes,zstd_compression_level);
            vector<unsigned char> oracle_selector_byte_zstd=zstd_compress(oracle_selector_vec,zstd_compression_level);
            vector<unsigned char> oracle_selector_huffman_bytes=huffman_encode(oracle_selector_int_vec);
            vector<unsigned char> oracle_selector_huffman_zstd=zstd_compress(oracle_selector_huffman_bytes,zstd_compression_level);
            uint64_t oracle_selector_byte_num=min(oracle_selector_byte_zstd.size(),oracle_selector_huffman_zstd.size());
            double selector_entropy_br=0;
            for (uint64_t count:oracle_config_count_vec)
                if (count>0)
                {
                    double probability=(double)count/oracle_selector_vec.size();
                    selector_entropy_br-=probability*log2(probability);
                }
            uint64_t oracle_no_selector_file_bytes=current_zstd_bs_vec.size()-value_zstd_byte_num+oracle_qb_zstd_bytes.size();
            uint64_t oracle_with_selector_file_bytes=oracle_no_selector_file_bytes+sizeof(uint64_t)+oracle_selector_byte_num;
            double oracle_psnr=oracle_squared_error==0?numeric_limits<double>::infinity():10*log10(pow((double)current_max_data-current_min_data,2)/(oracle_squared_error/point_num));
            printf("pointwise_oracle_config_count:");
            for (uint32_t config_id=0;config_id<oracle_config_count_vec.size();config_id++)
                if (oracle_config_count_vec[config_id]>0)
                    printf(" %u:%llu",config_id,(unsigned long long)oracle_config_count_vec[config_id]);
            printf("\n");
            printf("pointwise_oracle_qb_huffman_bytes=%llu, qb_zstd_bytes=%llu, selector_byte_zstd_bytes=%llu, selector_huffman_zstd_bytes=%llu, selector_entropy_br=%f\n",(unsigned long long)oracle_qb_huffman_bytes.size(),(unsigned long long)oracle_qb_zstd_bytes.size(),(unsigned long long)oracle_selector_byte_zstd.size(),(unsigned long long)oracle_selector_huffman_zstd.size(),selector_entropy_br);
            printf("pointwise_oracle_no_selector_file_br=%f, with_selector_file_br=%f, psnr=%f\n",(double)oracle_no_selector_file_bytes*8/point_num,(double)oracle_with_selector_file_bytes*8/point_num,oracle_psnr);
        }
        lossless_encoding_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        uint64_t label_target_num=accumulate(label_channel_target_num_vec.begin(),label_channel_target_num_vec.end(),(uint64_t)0);
        printf("label_prediction_error_ratio=%f, label_position_delta_bytes=%llu, label_zstd_bytes=%llu, zstd_level=%d\n",label_target_num?(double)label_prediction_error_num/label_target_num:0,(unsigned long long)label_position_delta_byte_num,(unsigned long long)label_zstd_byte_num,zstd_compression_level);
        printf("value_huffman_bytes=%llu, value_zstd_bytes=%llu, file_bytes=%llu\n",(unsigned long long)value_huffman_byte_num,
            (unsigned long long)value_zstd_byte_num,(unsigned long long)current_zstd_bs_vec.size());

        if (print_round_statistics)
        {
            printf("逐轮量化索引统计（不包含点集标签和文件头）：\n");
            for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
            {
                vector<unsigned char> round_huffman=huffman_encode(round_qb_vec[round_id]);
                vector<unsigned char> round_zstd=round_huffman.empty()?vector<unsigned char>():zstd_compress(round_huffman,zstd_compression_level);
                double squared_error=0;
                for (int pred_id:round_target_ids[round_id])
                {
                    double error=(double)current_data[pred_id]-current_decompressed_data[pred_id];
                    squared_error+=error*error;
                }
                double mse=squared_error/round_target_ids[round_id].size();
                double psnr=mse==0?numeric_limits<double>::infinity():10*log10(pow((double)current_max_data-current_min_data,2)/mse);
                double br=(double)round_zstd.size()*8/round_target_ids[round_id].size();
                printf("Round %u: point_num=%llu, qb_num=%llu, qb_br=%f, psnr=%f\n",round_id,(unsigned long long)round_target_ids[round_id].size(),(unsigned long long)round_qb_vec[round_id].size(),br,psnr);
            }
        }

        printf("point_set_num=%u",point_set_num);
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            const char *codec_name=point_set_codec_vec[point_set_id]==point_set_codec_e::constant?"constant":point_set_codec_vec[point_set_id]==point_set_codec_e::rbf_fitting?"rbf_fitting":"inverse_proportion";
            printf(", set%u_num=%u, set%u_codec=%s",point_set_id,point_set_point_num_vec[point_set_id],point_set_id,codec_name);
        }
        printf("\n");
        if (print_compression_stage_time)
        {
            printf("compression stage time: input_reorder=%f s, point_set=%f s, label_prediction=%f s, value_prediction=%f s, lossless_encoding=%f s, validation=%f s\n",input_reorder_time,point_set_time,label_prediction_time,value_prediction_time,lossless_encoding_time,validation_time);
            printf("point set stage time: classification=%f s, setup=%f s, eb_schedule=%f s\n",point_set_classification_time,point_set_setup_time,point_set_eb_schedule_time);
        }
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::restore_compression_result_to_original_order()
    {
        if (!reorder_data_by_prediction_order_experimental)
            return;
        vector<T_data> prediction_order_data=move(current_data),prediction_order_decompressed_data=move(current_decompressed_data);
        vector<uint8_t> prediction_order_point_set_id_vec=move(point_set_id_vec);
        current_data.resize(point_num);
        current_decompressed_data.resize(point_num);
        point_set_id_vec.resize(point_num);
        for (uint32_t prediction_id=0;prediction_id<point_num;prediction_id++)
        {
            uint32_t original_id=prediction_id_to_original_id[prediction_id];
            current_data[original_id]=prediction_order_data[prediction_id];
            current_decompressed_data[original_id]=prediction_order_decompressed_data[prediction_id];
            point_set_id_vec[original_id]=prediction_order_point_set_id_vec[prediction_id];
        }
    }
}
