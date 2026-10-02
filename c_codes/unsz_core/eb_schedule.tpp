#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    vector<double> param_c<T_coord,T_data>::generate_pred_eb_multiplier_vec(const vector<vector<int>>&target_id_groups,uint32_t target_num,bool print_statistics)
    {
        bool use_inverse_rank_log2=eb_schedule=="inverse_rank_log2";
        if(eb_schedule!="none"&&eb_schedule!="round"&&eb_schedule!="point"&&eb_schedule!="round_point"&&eb_schedule!="round_ratio"&&eb_schedule!="round_point_ratio"&&!use_inverse_rank_log2)
            throw runtime_error("未知的EB序列模式："+eb_schedule);
        if(!(eb_schedule_start_ratio>0) || eb_schedule_start_ratio>1)
            throw runtime_error("EB序列起始倍率必须在(0,1]内");
        if(target_id_groups.size()!=prediction_round_num)
            throw runtime_error("EB倍率的预测轮数量不正确");
        vector<double> result(target_num,1),round_eb_multiplier_vec(prediction_round_num,1),round_point_common_ratio_vec(prediction_round_num,1);
        double schedule_common_ratio=1,point_eb_multiplier=1;
        bool use_round_schedule=eb_schedule=="round"||eb_schedule=="round_point"||eb_schedule=="round_ratio"||eb_schedule=="round_point_ratio";
        bool use_round_point_schedule=eb_schedule=="round_point"||eb_schedule=="round_point_ratio";
        bool parameter_is_round_ratio=eb_schedule=="round_ratio"||eb_schedule=="round_point_ratio";
        if(use_round_schedule && prediction_round_num>1)
        {
            if(parameter_is_round_ratio)
                schedule_common_ratio=1/eb_schedule_start_ratio;
            else
                schedule_common_ratio=pow(1/eb_schedule_start_ratio,1.0/(prediction_round_num-1));
            for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
                if(parameter_is_round_ratio)
                    round_eb_multiplier_vec[round_id]=pow(eb_schedule_start_ratio,prediction_round_num-1-round_id);
                else
                    round_eb_multiplier_vec[round_id]=pow(eb_schedule_start_ratio,(double)(prediction_round_num-1-round_id)/(prediction_round_num-1));
            if(use_round_point_schedule)
                for(uint32_t round_id=0;round_id+1<prediction_round_num;round_id++)
                    if(target_id_groups[round_id].size()>1)
                        round_point_common_ratio_vec[round_id]=pow(round_eb_multiplier_vec[round_id+1]/round_eb_multiplier_vec[round_id],1.0/(target_id_groups[round_id].size()-1));
        }
        else if(eb_schedule=="point")
        {
            point_eb_multiplier=eb_schedule_start_ratio;
            if(target_num>1)
                schedule_common_ratio=pow(1/eb_schedule_start_ratio,1.0/(target_num-1));
        }
        uint32_t prediction_index=0;
        for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
        {
            double round_point_eb_multiplier=round_eb_multiplier_vec[round_id];
            for(uint32_t i=0;i<target_id_groups[round_id].size();i++)
            {
                if(use_inverse_rank_log2)
                {
                    double log2_exponent=clamp(log2((double)target_num/(prediction_index+1))-eb_schedule_log2_shift,0.0,eb_schedule_max_log2_exponent);
                    result[prediction_index]=pow(eb_schedule_start_ratio,log2_exponent);
                }
                else if(eb_schedule=="round"||eb_schedule=="round_ratio")
                    result[prediction_index]=round_eb_multiplier_vec[round_id];
                else if(use_round_point_schedule)
                    result[prediction_index]=round_point_eb_multiplier;
                else
                    result[prediction_index]=point_eb_multiplier;
                prediction_index++;
                if(eb_schedule=="point")
                    point_eb_multiplier=min(1.0,point_eb_multiplier*schedule_common_ratio);
                else if(use_round_point_schedule)
                    round_point_eb_multiplier=min(round_eb_multiplier_vec[min(round_id+1,prediction_round_num-1)],round_point_eb_multiplier*round_point_common_ratio_vec[round_id]);
            }
        }
        if(prediction_index!=target_num)
            throw runtime_error("无法为全部预测目标生成EB倍率");
        if(print_statistics&&eb_schedule!="none")
            printf("eb schedule: mode=%s, start_ratio=%.9g, log2_shift=%.9g, common_ratio=%.12g\n",eb_schedule.c_str(),eb_schedule_start_ratio,eb_schedule_log2_shift,schedule_common_ratio);
        return result;
    }
}
