#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    vector<double> param_c<T_coord,T_data>::generate_pred_eb_multiplier_vec(bool print_statistics)
    {
        bool use_dag_rank=eb_schedule=="round_point_ratio_dag";
        bool use_dag_direct=eb_schedule=="round_point_ratio_dag_direct";
        bool use_dag_variance_direct=eb_schedule=="round_point_ratio_dag_variance_direct";
        bool use_dag_influence=use_dag_rank || use_dag_direct || use_dag_variance_direct;
        bool use_inverse_rank_log2=eb_schedule=="inverse_rank_log2";
        if(eb_schedule!="none" && eb_schedule!="round" && eb_schedule!="point" && eb_schedule!="round_point" && eb_schedule!="round_ratio" && eb_schedule!="round_point_ratio" && !use_inverse_rank_log2 && !use_dag_influence)
            throw runtime_error("未知的EB序列模式："+eb_schedule);
        if(!(eb_schedule_start_ratio>0) || eb_schedule_start_ratio>1)
            throw runtime_error("EB序列起始倍率必须在(0,1]内");
        vector<double> result(point_num,1),round_eb_multiplier_vec(prediction_round_num,1),round_point_common_ratio_vec(prediction_round_num,1);
        double schedule_common_ratio=1,point_eb_multiplier=1;
        bool use_round_schedule=eb_schedule=="round" || eb_schedule=="round_point" || eb_schedule=="round_ratio" || eb_schedule=="round_point_ratio" || use_dag_influence;
        bool use_round_point_schedule=eb_schedule=="round_point" || eb_schedule=="round_point_ratio" || use_dag_influence;
        bool parameter_is_round_ratio=eb_schedule=="round_ratio" || eb_schedule=="round_point_ratio" || use_dag_influence;
        if(use_round_schedule && prediction_round_num>1)
        {
            schedule_common_ratio=parameter_is_round_ratio?1/eb_schedule_start_ratio:pow(1/eb_schedule_start_ratio,1.0/(prediction_round_num-1));
            for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
                round_eb_multiplier_vec[round_id]=parameter_is_round_ratio?pow(eb_schedule_start_ratio,prediction_round_num-1-round_id):pow(eb_schedule_start_ratio,(double)(prediction_round_num-1-round_id)/(prediction_round_num-1));
            if(use_round_point_schedule)
                for(uint32_t round_id=0;round_id+1<prediction_round_num;round_id++)
                    if(round_target_ids[round_id].size()>1)
                        round_point_common_ratio_vec[round_id]=pow(round_eb_multiplier_vec[round_id+1]/round_eb_multiplier_vec[round_id],1.0/(round_target_ids[round_id].size()-1));
        }
        else if(eb_schedule=="point")
        {
            point_eb_multiplier=eb_schedule_start_ratio;
            if(point_num>1)
                schedule_common_ratio=pow(1/eb_schedule_start_ratio,1.0/(point_num-1));
        }
        uint32_t pred_ref_id=0;
        for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
        {
            double round_point_eb_multiplier=round_eb_multiplier_vec[round_id];
            for(uint32_t i=0;i<round_target_ids[round_id].size();i++)
            {
                double log2_exponent=use_inverse_rank_log2?clamp(log2((double)point_num/(pred_ref_id+1))-eb_schedule_log2_shift,0.0,eb_schedule_max_log2_exponent):0;
                result[pred_ref_id]=use_inverse_rank_log2?pow(eb_schedule_start_ratio,log2_exponent):((eb_schedule=="round" || eb_schedule=="round_ratio")?round_eb_multiplier_vec[round_id]:(use_round_point_schedule?round_point_eb_multiplier:point_eb_multiplier));
                pred_ref_id++;
                if(eb_schedule=="point")
                    point_eb_multiplier=min(1.0,point_eb_multiplier*schedule_common_ratio);
                else if(use_round_point_schedule)
                    round_point_eb_multiplier=min(round_eb_multiplier_vec[min(round_id+1,prediction_round_num-1)],round_point_eb_multiplier*round_point_common_ratio_vec[round_id]);
            }
        }
        if(pred_ref_id!=point_num || pred_target_ids.size()!=point_num || pred_ref_begin.size()!=(uint64_t)point_num+1)
            throw runtime_error("无法为全部预测目标生成EB倍率");
        if(use_dag_influence)
        {
            if(value_flow!="none")
                throw runtime_error("DAG影响力EB暂不支持独立的跨time step预测器");
            if(use_partition_adaptive_predictor_experimental)
                throw runtime_error("DAG影响力EB暂不支持partition自适应预测器");
            auto dag_start_time=chrono::steady_clock::now();
            for(point_set_codec_e codec:point_set_codec_vec)
                if(codec!=point_set_codec_e::idw && codec!=point_set_codec_e::constant)
                    throw runtime_error("DAG影响力EB目前只支持反比例预测和常量点集");
            vector<double> influence_vec(point_num,1);
            vector<uint32_t> valid_ref_index_vec;
            valid_ref_index_vec.reserve(prediction_max_ref_num);
            uint64_t direct_edge_num=0;
            for(uint32_t reverse_target_pred_ref_id=point_num;reverse_target_pred_ref_id>0;reverse_target_pred_ref_id--)
            {
                uint32_t target_pred_ref_id=reverse_target_pred_ref_id-1;
                uint32_t target_id=pred_target_ids[target_pred_ref_id];
                uint32_t point_set_id=point_set_id_vec[target_id];
                if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                    continue;
                auto ref_ids=get_pred_ref_ids(target_pred_ref_id);
                auto ref_weights=get_pred_ref_weights(target_pred_ref_id);
                valid_ref_index_vec.clear();
                for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
                    if(point_set_id_vec[ref_ids[ref_index]]==point_set_id)
                    {
                        if(isinf(ref_weights[ref_index]))
                        {
                            valid_ref_index_vec.clear();
                            valid_ref_index_vec.push_back(ref_index);
                            break;
                        }
                        valid_ref_index_vec.push_back(ref_index);
                        if(valid_ref_index_vec.size()==prediction_max_ref_num)
                            break;
                    }
                if(valid_ref_index_vec.empty())
                    continue;
                direct_edge_num+=valid_ref_index_vec.size();
                if(isinf(ref_weights[valid_ref_index_vec[0]]))
                    influence_vec[ref_ids[valid_ref_index_vec[0]]]+=influence_vec[target_id];
                else
                {
                    double weight_sum=0;
                    for(uint32_t ref_index:valid_ref_index_vec)
                        weight_sum+=ref_weights[ref_index];
                    if(!(weight_sum>0) || !isfinite(weight_sum))
                        throw runtime_error("DAG影响力EB遇到无效的反比例预测权重和");
                    for(uint32_t ref_index:valid_ref_index_vec)
                    {
                        double normalized_weight=ref_weights[ref_index]/weight_sum;
                        double propagated_weight=use_dag_variance_direct?normalized_weight*normalized_weight:normalized_weight;
                        influence_vec[ref_ids[ref_index]]+=propagated_weight*influence_vec[target_id];
                    }
                }
            }
            uint32_t round_begin=0,changed_num=0,affected_point_num=0;
            double max_influence=1;
            for(double influence:influence_vec)
            {
                affected_point_num+=influence>1;
                max_influence=max(max_influence,influence);
            }
            for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
            {
                uint32_t round_size=round_target_ids[round_id].size();
                vector<double> original_multiplier_vec(result.begin()+round_begin,result.begin()+round_begin+round_size);
                if(use_dag_rank)
                {
                    vector<uint32_t> ranked_pred_ref_id_vec(round_size);
                    for(uint32_t i=0;i<round_size;i++)
                        ranked_pred_ref_id_vec[i]=round_begin+i;
                    stable_sort(ranked_pred_ref_id_vec.begin(),ranked_pred_ref_id_vec.end(),[&](uint32_t left,uint32_t right){return influence_vec[pred_target_ids[left]]>influence_vec[pred_target_ids[right]];});
                    for(uint32_t rank=0;rank<round_size;rank++)
                        result[ranked_pred_ref_id_vec[rank]]=original_multiplier_vec[rank];
                }
                else if(round_size>0)
                {
                    double min_log_influence=numeric_limits<double>::infinity(),max_log_influence=-numeric_limits<double>::infinity();
                    for(uint32_t i=0;i<round_size;i++)
                    {
                        double log_influence=log(influence_vec[pred_target_ids[round_begin+i]]);
                        min_log_influence=min(min_log_influence,log_influence);
                        max_log_influence=max(max_log_influence,log_influence);
                    }
                    double min_multiplier=original_multiplier_vec.front(),max_multiplier=original_multiplier_vec.back();
                    double log_multiplier_range=log(max_multiplier/min_multiplier);
                    for(uint32_t i=0;i<round_size;i++)
                    {
                        double normalized_influence=max_log_influence>min_log_influence?(log(influence_vec[pred_target_ids[round_begin+i]])-min_log_influence)/(max_log_influence-min_log_influence):0.5;
                        result[round_begin+i]=max_multiplier*exp(-normalized_influence*log_multiplier_range);
                    }
                }
                for(uint32_t i=0;i<round_size;i++)
                    changed_num+=result[round_begin+i]!=original_multiplier_vec[i];
                round_begin+=round_size;
            }
            if(print_statistics)
                printf("DAG influence EB: mapping=%s, propagation=%s, direct_edges=%llu, affected_points=%u, max_transitive_influence=%.9g, reassigned_points=%u, time=%.6f s\n",use_dag_rank?"rank":"direct",use_dag_variance_direct?"variance":"absolute",(unsigned long long)direct_edge_num,affected_point_num,max_influence,changed_num,chrono::duration<double>(chrono::steady_clock::now()-dag_start_time).count());
        }
        if(print_statistics && eb_schedule!="none")
            printf("eb schedule: mode=%s, start_ratio=%.9g, log2_shift=%.9g, common_ratio=%.12g\n",eb_schedule.c_str(),eb_schedule_start_ratio,eb_schedule_log2_shift,schedule_common_ratio);
        return result;
    }
}
