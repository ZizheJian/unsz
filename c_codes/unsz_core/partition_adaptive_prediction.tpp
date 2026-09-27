#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    vector<partition_prediction_config_c> param_c<T_coord,T_data>::get_partition_prediction_config_vec()
    {
        vector<partition_prediction_config_c> config_vec;
        for (double power:{2.5,3.0,3.5,5.0,7.5})
            for (uint32_t ref_num:{4u,6u,8u,10u})
                config_vec.push_back({ref_num,power,false});
        config_vec.push_back({10,3.5,true});
        bool include_weighted=partition_fitting_candidates_experimental=="weighted" || partition_fitting_candidates_experimental=="mixed" || partition_fitting_candidates_experimental=="extended" || partition_fitting_candidates_experimental=="all";
        if (include_weighted)
        {
            config_vec.push_back({10,3.5,true,0,0,true});
            for (double weight_power:{1.0,2.0,3.5})
                config_vec.push_back({10,3.5,true,weight_power,0,true});
        }
        if (partition_fitting_candidates_experimental=="ridge" || partition_fitting_candidates_experimental=="all")
            for (double ridge:{1e-6,1e-4,1e-2})
                config_vec.push_back({10,3.5,true,0,ridge,true});
        if (partition_fitting_candidates_experimental=="mixed" || partition_fitting_candidates_experimental=="extended" || partition_fitting_candidates_experimental=="all")
            for (double weight_power:{1.0,2.0,3.5})
                config_vec.push_back({10,3.5,true,weight_power,0,true,partition_fitting_method_e::continuous_mix});
        if (partition_fitting_candidates_experimental=="residual" || partition_fitting_candidates_experimental=="extended" || partition_fitting_candidates_experimental=="all")
            for (double weight_power:{0.0,1.0,2.0,3.5})
                config_vec.push_back({10,3.5,true,weight_power,0,true,partition_fitting_method_e::idw_residual});
        if (partition_fitting_candidates_experimental!="original" && partition_fitting_candidates_experimental!="weighted" && partition_fitting_candidates_experimental!="ridge" && partition_fitting_candidates_experimental!="mixed" && partition_fitting_candidates_experimental!="residual" && partition_fitting_candidates_experimental!="extended" && partition_fitting_candidates_experimental!="all")
            throw runtime_error("未知的partition拟合候选模式："+partition_fitting_candidates_experimental);
        return config_vec;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_partition_adaptive_sample_mask()
    {
        if (!use_partition_adaptive_predictor_experimental)
            return;
        if (partition_point_ids_vec.empty())
            throw runtime_error("partition自适应预测器需要启用partition");
        partition_adaptive_sample_mask.assign(point_num,0);
        for (auto &point_ids:partition_point_ids_vec)
        {
            uint32_t sample_num=min<uint32_t>(partition_adaptive_sample_num_experimental,point_ids.size());
            for (uint32_t sample_id=0;sample_id<sample_num;sample_id++)
            {
                uint64_t point_index=((uint64_t)sample_id*2+1)*point_ids.size()/(sample_num*2);
                partition_adaptive_sample_mask[point_ids[point_index]]=1;
            }
        }
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_partition_adaptive_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,uint8_t config_id)
    {
        vector<partition_prediction_config_c> config_vec=get_partition_prediction_config_vec();
        if (config_id>=config_vec.size())
            throw runtime_error("partition预测配置编号越界");
        return predict_partition_adaptive_value(pred_id,ref_ids,data,config_vec[config_id]);
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_partition_adaptive_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,const partition_prediction_config_c &config)
    {
        uint32_t ref_num=min<uint32_t>(config.ref_num,ref_ids.size());
        if (ref_num==0)
            return 0;
        vector<int> selected_ref_ids(ref_ids.begin(),ref_ids.begin()+ref_num);
        if (config.use_fitting)
        {
            T_data fitting_prediction;
            bool fitting_succeeded=false;
            if (config.fitting_method==partition_fitting_method_e::direct)
                fitting_succeeded=predict_fitting_value(pred_id,selected_ref_ids,data,fitting_prediction,config.fitting_weight_power,config.fitting_ridge,config.normalize_fitting_coord);
            else if (config.fitting_method==partition_fitting_method_e::continuous_mix)
                fitting_succeeded=predict_continuous_mix_value(pred_id,selected_ref_ids,data,config.idw_power,config.fitting_weight_power,fitting_prediction);
            else if (config.fitting_method==partition_fitting_method_e::idw_residual)
                fitting_succeeded=predict_idw_residual_fitting_value(pred_id,selected_ref_ids,data,config.idw_power,config.fitting_weight_power,fitting_prediction);
            if (fitting_succeeded)
                return fitting_prediction;
        }
        double total_weight=0,prediction=0;
        for (int ref_id:selected_ref_ids)
        {
            double squared_distance=(double)get_dist_to_coord(coord_array[pred_id],coord_array[ref_id],dim);
            if (squared_distance==0)
                return data[ref_id];
            double weight=pow(squared_distance,-config.idw_power/2);
            if (isinf(weight))
                return data[ref_id];
            total_weight+=weight;
            prediction+=weight*data[ref_id];
        }
        return total_weight>0?(T_data)(prediction/total_weight):(T_data)0;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::select_partition_prediction_configs()
    {
        if (!use_partition_adaptive_predictor_experimental)
            return;
        if (point_set_point_num_vec.size()!=1)
            throw runtime_error("partition自适应预测器当前只支持单点集数据");
        if (reorder_by_prediction_order)
            throw runtime_error("partition自适应预测器暂不支持预测顺序内部重排");
        vector<partition_prediction_config_c> config_vec=get_partition_prediction_config_vec();
        partition_prediction_config_id_vec.assign(partition_point_ids_vec.size(),0);
        prepare_partition_adaptive_sample_mask();
        vector<vector<double>> partition_score_vec(partition_point_ids_vec.size(),vector<double>(config_vec.size(),0));
        vector<uint32_t> partition_sample_num_vec(partition_point_ids_vec.size());
        for (uint32_t partition_id=0;partition_id<partition_point_ids_vec.size();partition_id++)
        {
            auto &point_ids=partition_point_ids_vec[partition_id];
            vector<int> sample_ids;
            sample_ids.reserve(min<uint32_t>(partition_adaptive_sample_num_experimental,point_ids.size()));
            for (int point_id:point_ids)
                if (partition_adaptive_sample_mask[point_id])
                    sample_ids.push_back(point_id);
            if (sample_ids.size()<=10)
                throw runtime_error("partition自适应预测器的采样点数必须大于10");
            partition_sample_num_vec[partition_id]=(uint32_t)sample_ids.size();
            vector<double> &score_vec=partition_score_vec[partition_id];
            for (int target_id:sample_ids)
            {
                vector<pair<double,int>> distance_id_vec;
                distance_id_vec.reserve(sample_ids.size()-1);
                for (int ref_id:sample_ids)
                    if (ref_id!=target_id)
                        distance_id_vec.push_back({(double)get_dist_to_coord(coord_array[target_id],coord_array[ref_id],dim),ref_id});
                uint32_t nearest_num=min<uint32_t>(10,distance_id_vec.size());
                partial_sort(distance_id_vec.begin(),distance_id_vec.begin()+nearest_num,distance_id_vec.end());
                vector<int> ref_ids;
                ref_ids.reserve(nearest_num);
                for (uint32_t ref_index=0;ref_index<nearest_num;ref_index++)
                    ref_ids.push_back(distance_id_vec[ref_index].second);
                for (uint32_t config_id=0;config_id<config_vec.size();config_id++)
                {
                    T_data prediction=predict_partition_adaptive_value(target_id,ref_ids,current_data,config_id);
                    double qb=round(((double)current_data[target_id]-prediction)/(2*point_set_abs_eb_vec[0]));
                    score_vec[config_id]+=isfinite(qb)?log2(1+abs(qb)):numeric_limits<double>::max()/sample_ids.size();
                }
            }
        }
        vector<double> field_score_vec(config_vec.size(),0);
        for (const vector<double> &score_vec:partition_score_vec)
            for (uint32_t config_id=0;config_id<config_vec.size();config_id++)
                field_score_vec[config_id]+=score_vec[config_id];
        uint32_t field_config_id=min_element(field_score_vec.begin(),field_score_vec.end())-field_score_vec.begin();
        printf("partition_adaptive_field_config: config=%u,score=%.6f,min_local_improvement=%.6f\n",field_config_id,field_score_vec[field_config_id],partition_adaptive_min_improvement_experimental);
        for (uint32_t partition_id=0;partition_id<partition_point_ids_vec.size();partition_id++)
        {
            vector<double> &score_vec=partition_score_vec[partition_id];
            uint32_t local_config_id=min_element(score_vec.begin(),score_vec.end())-score_vec.begin();
            double field_config_local_score=score_vec[field_config_id];
            double local_improvement=field_config_local_score>0?(field_config_local_score-score_vec[local_config_id])/field_config_local_score:0;
            uint32_t selected_config_id=local_config_id!=field_config_id&&local_improvement<partition_adaptive_min_improvement_experimental?field_config_id:local_config_id;
            partition_prediction_config_id_vec[partition_id]=(uint8_t)selected_config_id;
            auto &selected_config=config_vec[selected_config_id];
            const char *predictor_name=!selected_config.use_fitting?"idw":selected_config.fitting_method==partition_fitting_method_e::continuous_mix?"continuous_mix":selected_config.fitting_method==partition_fitting_method_e::idw_residual?"idw_residual_fitting":selected_config.fitting_weight_power>0?"weighted_fitting":selected_config.fitting_ridge>0?"ridge_fitting":"fitting";
            printf("partition_adaptive_config: partition=%u,sample_num=%u,config=%u,local_config=%u,field_config=%u,local_improvement=%.6f,predictor=%s,ref_num=%u,power=%.6g,fitting_weight_power=%.6g,ridge=%.6g,normalize_coord=%u,score=%.6f\n",partition_id,partition_sample_num_vec[partition_id],selected_config_id,local_config_id,field_config_id,local_improvement,predictor_name,selected_config.ref_num,selected_config.idw_power,selected_config.fitting_weight_power,selected_config.fitting_ridge,selected_config.normalize_fitting_coord,score_vec[selected_config_id]);
        }
    }
}
