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
    bool param_c<T_coord,T_data>::solve_linear_system(vector<vector<double>> A,vector<double> b,vector<double> &x)
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
        x=move(b);
        return true;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::compress()
    {
        auto stage_start_time=chrono::steady_clock::now();
        double layout_reorder_time=0,point_set_time=0,label_prediction_time=0,value_prediction_time=0,lossless_encoding_time=0,validation_time=0;
        if (current_data.size()!=point_num)
            throw runtime_error("当前数据点数与坐标点数不一致");
        if (point_set_config_vec.size()!=point_set_classifier_vec.size()+1)
            throw runtime_error("点集配置数量必须等于分类函数数量加1");
        bool has_point_set_error_bound=false;
        for (const point_set_config_c &config:point_set_config_vec)
            has_point_set_error_bound|=config.error_mode!=point_set_error_mode_e::use_global;
        if (!got_rel_eb && !got_abs_eb && !has_point_set_error_bound)
            throw runtime_error("没有指定误差上限");
        if (eb_schedule_experimental!="none" && eb_schedule_experimental!="round" && eb_schedule_experimental!="point" && eb_schedule_experimental!="round_point" && eb_schedule_experimental!="round_ratio" && eb_schedule_experimental!="round_point_ratio")
            throw runtime_error("未知的实验性EB序列模式："+eb_schedule_experimental);
        if (!(eb_schedule_start_ratio>0) || eb_schedule_start_ratio>1)
            throw runtime_error("实验性EB序列起始倍率必须在(0,1]内");
        if (use_prediction_order_layout_experimental)
        {
            vector<T_data> original_order_data=move(current_data);
            current_data.resize(point_num);
            for (uint32_t prediction_id=0;prediction_id<point_num;prediction_id++)
                current_data[prediction_id]=original_order_data[prediction_id_to_original_id[prediction_id]];
            layout_reorder_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
            stage_start_time=chrono::steady_clock::now();
        }

        point_set_id_vec.resize(point_num);
        uint32_t point_set_num=0,configured_point_set_num=(uint32_t)point_set_classifier_vec.size()+1;
        point_set_point_num_vec.assign(configured_point_set_num,0);
        point_set_min_data_vec.assign(configured_point_set_num,numeric_limits<T_data>::infinity());
        point_set_max_data_vec.assign(configured_point_set_num,-numeric_limits<T_data>::infinity());
        for (uint32_t i=0;i<point_num;i++)
        {
            uint32_t point_set_id=get_point_set_id(current_data[i]);
            if (point_set_id>=configured_point_set_num)
                throw runtime_error("分类函数返回的点集编号越界");
            point_set_id_vec[i]=point_set_id;
            point_set_num=max(point_set_num,point_set_id+1);
            point_set_point_num_vec[point_set_id]++;
            point_set_min_data_vec[point_set_id]=min(point_set_min_data_vec[point_set_id],current_data[i]);
            point_set_max_data_vec[point_set_id]=max(point_set_max_data_vec[point_set_id],current_data[i]);
        }
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
            if (config.codec!=point_set_codec_e::use_default && config.codec!=point_set_codec_e::inverse_proportion && config.codec!=point_set_codec_e::fitting && config.codec!=point_set_codec_e::constant && config.codec!=point_set_codec_e::hybrid_loo)
                throw runtime_error("点集"+to_string(point_set_id)+"配置了未知的压缩方法");
            if (config.error_mode!=point_set_error_mode_e::use_global && config.error_mode!=point_set_error_mode_e::relative && config.error_mode!=point_set_error_mode_e::absolute)
                throw runtime_error("点集"+to_string(point_set_id)+"配置了未知的误差模式");
            bool is_constant=point_set_min_data_vec[point_set_id]==point_set_max_data_vec[point_set_id];
            if (is_constant)
                point_set_codec_vec[point_set_id]=point_set_codec_e::constant;
            else if (config.codec!=point_set_codec_e::use_default)
                point_set_codec_vec[point_set_id]=config.codec;
            else if (predictor=="inverse_proportion")
                point_set_codec_vec[point_set_id]=point_set_codec_e::inverse_proportion;
            else if (predictor=="fitting")
                point_set_codec_vec[point_set_id]=point_set_codec_e::fitting;
            else if (predictor=="hybrid_loo")
                point_set_codec_vec[point_set_id]=point_set_codec_e::hybrid_loo;
            else
                throw runtime_error("未知的默认数值预测器："+predictor);
            if (!is_constant && point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                throw runtime_error("非常量点集不能使用常量编码");
            if (is_constant)
                point_set_abs_eb_vec[point_set_id]=0;
            else if (config.error_mode==point_set_error_mode_e::absolute)
                point_set_abs_eb_vec[point_set_id]=(T_data)config.error_bound;
            else if (config.error_mode==point_set_error_mode_e::relative)
                point_set_abs_eb_vec[point_set_id]=(T_data)(config.error_bound*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]));
            else
            {
                if (!got_rel_eb && !got_abs_eb)
                    throw runtime_error("点集"+to_string(point_set_id)+"没有指定误差上限");
                point_set_abs_eb_vec[point_set_id]=got_abs_eb?abs_eb:rel_eb*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]);
            }
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
        select_block_prediction_configs();
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
        if (eb_schedule_experimental!="none")
            printf("experimental eb schedule: mode=%s, start_ratio=%.9g, common_ratio=%.12g\n",eb_schedule_experimental.c_str(),eb_schedule_start_ratio,schedule_common_ratio);
        point_set_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();

        stage_start_time=chrono::steady_clock::now();
        uint32_t label_channel_num=point_set_num>0?point_set_num-1:0;
        vector<vector<uint32_t>> label_symbol_vec_vec(label_channel_num);
        vector<vector<uint8_t>> label_symbol_file_order_vec_vec(label_channel_num);
        vector<uint64_t> label_channel_target_num_vec(label_channel_num,0),label_channel_error_num_vec(label_channel_num,0);
        vector<uint32_t> actual_label_prediction_order_vec;
        uint64_t label_prediction_error_num=0;
        if (point_set_num>1 && point_set_predictor!="inverse_proportion" && point_set_predictor!="majority" && point_set_predictor!="continuous_actual" && point_set_predictor!="continuous_reflect")
            throw runtime_error("未知的点集标签预测器："+point_set_predictor);
        bool continuous=point_set_predictor=="continuous_actual" || point_set_predictor=="continuous_reflect";
        bool use_majority=point_set_predictor=="majority";
        bool use_normalized_label_weights=use_normalized_weight_experimental && continuous;
        for (uint32_t label_channel_id=0;label_channel_id<label_channel_num;label_channel_id++)
        {
            vector<uint8_t> label_symbol_internal_vec(point_num,0);
            vector<uint8_t> binary_label_vec(point_num,1);
            for (uint32_t point_id=0;point_id<point_num;point_id++)
                if (point_set_id_vec[point_id]==label_channel_id)
                    binary_label_vec[point_id]=0;
            vector<double> binary_label_value_vec(continuous?point_num:0,0);
            if (continuous)
                for (uint32_t point_id=0;point_id<point_num;point_id++)
                    if (point_set_id_vec[point_id]<label_channel_id)
                        binary_label_value_vec[point_id]=1;
            if (print_label_compression_statistics || use_prediction_order_label_experimental)
                label_symbol_vec_vec[label_channel_id].reserve(point_num);
            if (print_label_compression_statistics && point_set_num==2)
                actual_label_prediction_order_vec.reserve(point_num);
            for (uint32_t pred_ref_id=0;pred_ref_id<pred_target_ids.size();pred_ref_id++)
            {
                int pred_id=pred_target_ids[pred_ref_id];
                uint32_t actual_point_set_id=point_set_id_vec[pred_id];
                if (actual_point_set_id<label_channel_id)
                    continue;
                auto ref_ids=get_pred_ref_ids(pred_ref_id);
                auto ref_weights=get_pred_ref_weights(pred_ref_id,use_normalized_label_weights,false);
                double predicted_value=continuous?(use_normalized_label_weights?predict_binary_label_value_normalized(ref_ids,ref_weights,binary_label_value_vec):predict_binary_label_value(ref_ids,ref_weights,binary_label_value_vec)):0;
                uint32_t predicted_label=continuous?(predicted_value>=0.5):predict_binary_label(ref_ids,ref_weights,binary_label_vec,use_majority);
                uint32_t actual_label=actual_point_set_id==label_channel_id?0:1;
                bool prediction_error=actual_label!=predicted_label;
                label_channel_target_num_vec[label_channel_id]++;
                label_channel_error_num_vec[label_channel_id]+=prediction_error;
                label_prediction_error_num+=prediction_error;
                if (print_label_compression_statistics && point_set_num==2)
                    actual_label_prediction_order_vec.push_back(actual_label);
                if (print_label_compression_statistics || use_prediction_order_label_experimental)
                    label_symbol_vec_vec[label_channel_id].push_back(prediction_error?1:0);
                label_symbol_internal_vec[pred_id]=prediction_error?1:0;
                if (continuous)
                {
                    if (!prediction_error)
                        binary_label_value_vec[pred_id]=predicted_value;
                    else if (point_set_predictor=="continuous_actual" || predicted_value==0.5)
                        binary_label_value_vec[pred_id]=actual_label;
                    else
                        binary_label_value_vec[pred_id]=1-predicted_value;
                    uint32_t rounded_label=binary_label_value_vec[pred_id]>=0.5;
                    if (rounded_label!=actual_label)
                        throw runtime_error("连续点集标签状态无法恢复真实标签");
                }
            }
            if (!use_prediction_order_label_experimental || print_label_compression_statistics)
            {
                label_symbol_file_order_vec_vec[label_channel_id].reserve(label_channel_target_num_vec[label_channel_id]);
                for (uint32_t original_id=0;original_id<point_num;original_id++)
                {
                    uint32_t internal_id=use_prediction_order_layout_experimental?original_id_to_prediction_id[original_id]:original_id;
                    if (point_set_id_vec[internal_id]>=label_channel_id)
                        label_symbol_file_order_vec_vec[label_channel_id].push_back(label_symbol_internal_vec[internal_id]);
                }
            }
        }
        label_prediction_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();

        stage_start_time=chrono::steady_clock::now();
        current_decompressed_data.assign(point_num,0);
        uint32_t multi_huffman_group_num=get_multi_huffman_group_num();
        vector<vector<vector<int32_t>>> point_set_group_qb_vec(point_set_num,vector<vector<int32_t>>(multi_huffman_group_num));
        vector<uint64_t> point_set_qb_num_vec(point_set_num,0);
        vector<int32_t> point_set_qb_min_vec(point_set_num,numeric_limits<int32_t>::max()),point_set_qb_max_vec(point_set_num,numeric_limits<int32_t>::min());
        uint64_t value_huffman_byte_num=0;
        vector<vector<int32_t>> round_qb_vec(print_round_statistics?predict_round_num:0);
        uint32_t pred_ref_index=0;
        hybrid_prediction_num=0;
        hybrid_fitting_prediction_num=0;
        if (hybrid_selector_path.empty())
            hybrid_selector_vec.clear();
        else
            hybrid_selector_vec.assign(point_num,255);
        vector<T_data> oracle_decompressed_data;
        vector<int32_t> oracle_qb_vec,oracle_selector_int_vec;
        vector<unsigned char> oracle_selector_vec;
        vector<block_prediction_config_c> oracle_config_vec;
        vector<uint64_t> oracle_config_count_vec;
        double oracle_squared_error=0;
        if (print_pointwise_oracle_statistics_experimental)
        {
            if (point_set_num!=1 || point_set_codec_vec[0]==point_set_codec_e::constant)
                throw runtime_error("逐点oracle统计当前只支持单个非常量点集");
            oracle_config_vec=get_block_prediction_config_vec();
            if (oracle_config_vec.size()>256)
                throw runtime_error("逐点oracle候选配置超过256个");
            oracle_decompressed_data.assign(point_num,0);
            oracle_qb_vec.reserve(point_num);
            oracle_selector_int_vec.reserve(point_num);
            oracle_selector_vec.reserve(point_num);
            oracle_config_count_vec.assign(oracle_config_vec.size(),0);
        }
        bool use_normalized_value_weights=use_normalized_weight_experimental && point_set_num==1 && point_set_codec_vec[0]==point_set_codec_e::inverse_proportion;
        bool has_previous_time=cross_time_predictor_experimental=="previous" && previous_time_decompressed_data.size()==point_num && previous_time_point_set_id_vec.size()==point_num;
        prepare_time_interpolation();
        uint64_t cross_time_prediction_num=0;
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            double round_point_eb_multiplier=round_eb_multiplier_vec[round_id];
            for (uint32_t i=0;i<round_target_ids[round_id].size();i++)
            {
                uint32_t current_pred_ref_id=pred_ref_index++;
                auto ref_ids=get_pred_ref_ids(current_pred_ref_id);
                auto ref_weights=get_pred_ref_weights(current_pred_ref_id,false,use_normalized_value_weights);
                int pred_id=pred_target_ids[current_pred_ref_id];
                uint32_t point_set_id=point_set_id_vec[pred_id];
                double eb_multiplier=(eb_schedule_experimental=="round" || eb_schedule_experimental=="round_ratio")?round_eb_multiplier_vec[round_id]:(use_round_point_schedule?round_point_eb_multiplier:point_eb_multiplier);
                if (eb_schedule_experimental=="point")
                    point_eb_multiplier=min(1.0,point_eb_multiplier*schedule_common_ratio);
                else if (use_round_point_schedule)
                    round_point_eb_multiplier=min(round_eb_multiplier_vec[min(round_id+1,predict_round_num-1)],round_point_eb_multiplier*round_point_common_ratio_vec[round_id]);
                if (point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                {
                    current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                    if (print_pointwise_oracle_statistics_experimental)
                        oracle_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                    continue;
                }
                bool use_block_config=use_block_adaptive_predictor_experimental && !block_adaptive_sample_mask[pred_id];
                uint32_t original_id=use_prediction_order_layout_experimental?prediction_id_to_original_id[pred_id]:pred_id;
                T_data pred=0;
                bool use_cross_time_prediction=false;
                if (has_previous_time && previous_time_point_set_id_vec[original_id]==point_set_id)
                {
                    pred=previous_time_decompressed_data[original_id];
                    use_cross_time_prediction=true;
                }
                else if (cross_time_predictor_experimental=="interpolation")
                    use_cross_time_prediction=predict_time_interpolation(original_id,point_set_id,pred);
                if (!use_cross_time_prediction)
                    pred=use_block_config?predict_block_adaptive_value(pred_id,ref_ids,current_decompressed_data,block_prediction_config_id_vec[point_independent_block_id_vec[pred_id]]):(use_normalized_value_weights?predict_value_normalized(ref_ids,ref_weights,current_decompressed_data):predict_value(pred_id,ref_ids,ref_weights,point_set_id,current_decompressed_data));
                cross_time_prediction_num+=use_cross_time_prediction;
                double point_abs_eb=point_set_abs_eb_vec[point_set_id]*eb_multiplier;
                if (print_pointwise_oracle_statistics_experimental)
                {
                    uint8_t best_config_id=block_prediction_config_id_vec[point_independent_block_id_vec[pred_id]];
                    T_data best_prediction=predict_block_adaptive_value(pred_id,ref_ids,oracle_decompressed_data,oracle_config_vec[best_config_id]);
                    double best_error=abs((double)current_data[pred_id]-best_prediction);
                    for (uint32_t config_id=0;config_id<oracle_config_vec.size();config_id++)
                    {
                        if (config_id==best_config_id)
                            continue;
                        T_data candidate_prediction=predict_block_adaptive_value(pred_id,ref_ids,oracle_decompressed_data,oracle_config_vec[config_id]);
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
                uint32_t huffman_group_id=get_multi_huffman_group_id(round_id,pred_id);
                point_set_group_qb_vec[point_set_id][huffman_group_id].push_back(qb);
                point_set_qb_num_vec[point_set_id]++;
                point_set_qb_min_vec[point_set_id]=min(point_set_qb_min_vec[point_set_id],qb);
                point_set_qb_max_vec[point_set_id]=max(point_set_qb_max_vec[point_set_id],qb);
                if (print_round_statistics)
                    round_qb_vec[round_id].push_back(qb);
                current_decompressed_data[pred_id]=pred+qb*2*point_abs_eb;
            }
        }
        if (pred_ref_index!=pred_target_ids.size())
            throw runtime_error("逐轮预测目标数量与预测参考结构不一致");
        if (hybrid_prediction_num>0)
            printf("hybrid_loo_fitting_ratio=%f, fitting_num=%llu, prediction_num=%llu\n",(double)hybrid_fitting_prediction_num/hybrid_prediction_num,(unsigned long long)hybrid_fitting_prediction_num,(unsigned long long)hybrid_prediction_num);
        if (cross_time_predictor_experimental!="none")
            printf("cross_time_prediction_num=%llu, ratio=%f\n",(unsigned long long)cross_time_prediction_num,(double)cross_time_prediction_num/point_num);
        value_prediction_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();

        stage_start_time=chrono::steady_clock::now();
        vector<unsigned char> container;
        bool use_legacy_point_set_format=point_set_num<=2;
        const unsigned char magic_v7[8]={'U','N','S','Z','P','S','0','7'};
        const unsigned char magic_v8[8]={'U','N','S','Z','P','S','0','8'};
        const unsigned char *magic=use_legacy_point_set_format?magic_v7:magic_v8;
        container.insert(container.end(),magic,magic+8);
        append_binary<uint32_t>(container,use_legacy_point_set_format?7:8);
        append_binary<uint32_t>(container,point_num);
        append_binary<uint32_t>(container,sizeof(T_data));
        append_binary<uint32_t>(container,point_set_num);
        if (use_block_adaptive_predictor_experimental)
        {
            append_binary<uint32_t>(container,block_prediction_config_id_vec.size());
            container.insert(container.end(),block_prediction_config_id_vec.begin(),block_prediction_config_id_vec.end());
        }
        vector<vector<unsigned char>> label_position_delta_bytes_vec(label_channel_num),label_zstd_bytes_vec(label_channel_num);
        uint64_t label_position_delta_byte_num=0,label_zstd_byte_num=0;
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
            const vector<uint32_t> &label_symbol_vec=label_symbol_vec_vec[0];
            const vector<uint8_t> &label_symbol_file_order_vec=label_symbol_file_order_vec_vec[0];
            const vector<unsigned char> &label_position_delta_bytes=label_position_delta_bytes_vec[0];
            for (int zstd_level:{1,3,6,9,15,19})
                printf("label_position_delta_zstd_level=%d, bytes=%llu\n",zstd_level,(unsigned long long)zstd_compress(label_position_delta_bytes,zstd_level).size());
            auto print_label_candidate=[&](const char *name,vector<unsigned char> bytes)
            {
                printf("label_candidate=%s, bytes_before_zstd=%llu, bytes_after_zstd=%llu\n",name,
                    (unsigned long long)bytes.size(),(unsigned long long)zstd_size(bytes));
            };
            vector<uint32_t> point_set_id_file_order_vec(point_num);
            if (use_prediction_order_layout_experimental)
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
                vector<uint32_t> round_label_vec(label_symbol_vec.begin()+label_begin,label_symbol_vec.begin()+label_end);
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
        uint64_t value_zstd_byte_num=0;
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
                qb_huffman_bytes=huffman_encode(point_set_group_qb_vec[point_set_id][0]);
                huffman_payload_byte_num=qb_huffman_bytes.size();
            }
            else
                qb_huffman_bytes=encode_multi_huffman_groups(point_set_group_qb_vec[point_set_id],huffman_payload_byte_num,multi_huffman_metadata_byte_num);
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

        stage_start_time=chrono::steady_clock::now();
        double squared_error=0;
        for (uint32_t i=0;i<point_num;i++)
        {
            double error=(double)current_data[i]-current_decompressed_data[i];
            squared_error+=error*error;
        }
        double mse=squared_error/point_num;
        double psnr=mse==0?numeric_limits<double>::infinity():10*log10(pow((double)current_max_data-current_min_data,2)/mse);
        double cr=(double)point_num*sizeof(T_data)/current_zstd_bs_vec.size();
        double br=sizeof(T_data)*8/cr;
        printf("point_set_num=%u",point_set_num);
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            const char *codec_name=point_set_codec_vec[point_set_id]==point_set_codec_e::constant?"constant":point_set_codec_vec[point_set_id]==point_set_codec_e::fitting?"fitting":point_set_codec_vec[point_set_id]==point_set_codec_e::hybrid_loo?"hybrid_loo":"inverse_proportion";
            printf(", set%u_num=%u, set%u_codec=%s",point_set_id,point_set_point_num_vec[point_set_id],point_set_id,codec_name);
        }
        printf("\ncr= %f, br= %f, psnr= %f\n",cr,br,psnr);
        validation_time=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        if (print_compression_stage_time)
            printf("compression stage time: layout_reorder=%f s, point_set=%f s, label_prediction=%f s, value_prediction=%f s, lossless_encoding=%f s, validation=%f s\n",layout_reorder_time,point_set_time,label_prediction_time,value_prediction_time,lossless_encoding_time,validation_time);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::restore_compression_result_to_original_order()
    {
        if (!use_prediction_order_layout_experimental)
            return;
        vector<T_data> prediction_order_data=move(current_data),prediction_order_decompressed_data=move(current_decompressed_data);
        vector<uint32_t> prediction_order_point_set_id_vec=move(point_set_id_vec);
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
