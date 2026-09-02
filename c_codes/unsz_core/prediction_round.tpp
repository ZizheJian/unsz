#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    vector<uint32_t> param_c<T_coord,T_data>::get_round_target_num_vec(uint32_t local_point_num) const
    {
        if(predict_round_num==0 || predict_round_num>63)
            throw runtime_error("predict_round_num必须在[1,63]内");
        if(!isfinite(round_size_growth_ratio_experimental) || round_size_growth_ratio_experimental<1 || round_size_growth_ratio_experimental>16)
            throw runtime_error("预测轮目标点数增长比例必须在[1,16]内");
        vector<long double> round_weight_vec(predict_round_num,1);
        for(uint32_t round_id=2;round_id<predict_round_num;round_id++)
            round_weight_vec[round_id]=round_weight_vec[round_id-1]*(long double)round_size_growth_ratio_experimental;
        long double total_weight=accumulate(round_weight_vec.begin(),round_weight_vec.end(),(long double)0),cumulative_weight=0;
        vector<uint32_t> result(predict_round_num);
        uint64_t previous_end=0;
        for(uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            cumulative_weight+=round_weight_vec[round_id];
            uint64_t current_end=round_id+1==predict_round_num?local_point_num:(uint64_t)ceil((long double)local_point_num*cumulative_weight/total_weight);
            current_end=min<uint64_t>(max(current_end,previous_end),local_point_num);
            result[round_id]=(uint32_t)(current_end-previous_end);
            previous_end=current_end;
        }
        if(previous_end!=local_point_num)
            throw runtime_error("预测轮次没有覆盖全部点");
        return result;
    }
}
