#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    vector<typename param_c<T_coord,T_data>::T_calc> param_c<T_coord,T_data>::generate_pred_eb_multiplier_vec(uint32_t target_num,bool print_statistics)
    {
        if(eb_schedule!="none"&&eb_schedule!="inverse_rank_log2")
            throw runtime_error("未知的EB序列模式："+eb_schedule);
        if(!(eb_schedule_start_ratio>0)||eb_schedule_start_ratio>1)
            throw runtime_error("EB序列起始倍率必须在(0,1]内");
        vector<T_calc> result(target_num,1);
        if(eb_schedule=="inverse_rank_log2")
            for(uint32_t prediction_index=0;prediction_index<target_num;prediction_index++)
            {
                double log2_exponent=clamp(log2((double)target_num/(prediction_index+1))-eb_schedule_log2_shift,0.0,eb_schedule_max_log2_exponent);
                result[prediction_index]=(T_calc)pow(eb_schedule_start_ratio,log2_exponent);
            }
        if(print_statistics&&eb_schedule!="none")
            printf("eb schedule: mode=%s, base=%.9g, log2_shift=%.9g, max_log2_exponent=%.9g\n",eb_schedule.c_str(),eb_schedule_start_ratio,eb_schedule_log2_shift,eb_schedule_max_log2_exponent);
        return result;
    }
}
