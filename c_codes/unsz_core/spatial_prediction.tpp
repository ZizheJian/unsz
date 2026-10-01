#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    typename param_c<T_coord,T_data>::T_calc param_c<T_coord,T_data>::predict_value_from_weighted_sum(array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,const vector<T_data> &data)
    {
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,ref_ids.size());
        T_calc prediction=0;
        if(ref_num==4)
        {
            T_calc weight0=ref_weights[0],weight1=ref_weights[1],weight2=ref_weights[2],weight3=ref_weights[3];
            prediction=weight0*(T_calc)data[ref_ids[0]]+weight1*(T_calc)data[ref_ids[1]]+weight2*(T_calc)data[ref_ids[2]]+weight3*(T_calc)data[ref_ids[3]];
        }
        else
            for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                prediction+=ref_weights[ref_index]*(T_calc)data[ref_ids[ref_index]];
        return prediction;
    }

    template<typename T_coord,typename T_data>
    typename param_c<T_coord,T_data>::T_calc param_c<T_coord,T_data>::predict_value_from_weighted_average(array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,const vector<T_data> &data)
    {
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,ref_ids.size());
        T_calc weighted_sum=0,weight_sum=0;
        for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
        {
            T_calc weight=ref_weights[ref_index];
            weighted_sum+=weight*(T_calc)data[ref_ids[ref_index]];
            weight_sum+=weight;
        }
        return weight_sum!=(T_calc)0?weighted_sum/weight_sum:(T_calc)0;
    }

    template<typename T_coord,typename T_data>
    __attribute__((always_inline)) inline T_data param_c<T_coord,T_data>::predict_value_one_point_set(uint32_t pred_ref_id,const vector<T_calc> &weights,const vector<T_data> &data)
    {
        uint64_t ref_begin=pred_ref_begin[pred_ref_id],ref_end=pred_ref_begin[pred_ref_id+1];
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,(uint32_t)(ref_end-ref_begin));
        T_calc prediction=0;
        for(uint32_t local_ref_index=0;local_ref_index<ref_num;local_ref_index++)
        {
            uint64_t ref_index=ref_begin+local_ref_index;
            prediction+=(T_calc)weights[ref_index]*(T_calc)data[pred_ref_ids[ref_index]];
        }
        return (T_data)prediction;
    }

    template<typename T_coord,typename T_data>
    __attribute__((always_inline)) inline T_data param_c<T_coord,T_data>::predict_value_one_point_set_10(uint64_t ref_begin,const vector<T_calc> &weights,const vector<T_data> &data)
    {
        const int *ref_ids=pred_ref_ids.data()+ref_begin;
        const T_calc *ref_weights=weights.data()+ref_begin;
        T_calc prediction=(T_calc)ref_weights[0]*(T_calc)data[ref_ids[0]]+(T_calc)ref_weights[1]*(T_calc)data[ref_ids[1]]+(T_calc)ref_weights[2]*(T_calc)data[ref_ids[2]]+(T_calc)ref_weights[3]*(T_calc)data[ref_ids[3]]+(T_calc)ref_weights[4]*(T_calc)data[ref_ids[4]]+(T_calc)ref_weights[5]*(T_calc)data[ref_ids[5]]+(T_calc)ref_weights[6]*(T_calc)data[ref_ids[6]]+(T_calc)ref_weights[7]*(T_calc)data[ref_ids[7]]+(T_calc)ref_weights[8]*(T_calc)data[ref_ids[8]]+(T_calc)ref_weights[9]*(T_calc)data[ref_ids[9]];
        return (T_data)prediction;
    }

    template<typename T_coord,typename T_data>
    __attribute__((always_inline)) inline uint32_t param_c<T_coord,T_data>::predict_label(uint32_t pred_ref_id,const vector<T_calc> &weights,const vector<uint8_t> &label_vec)
    {
        uint64_t ref_begin=pred_ref_begin[pred_ref_id],ref_end=pred_ref_begin[pred_ref_id+1];
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,(uint32_t)(ref_end-ref_begin));
        T_calc prediction=0;
        for(uint32_t local_ref_index=0;local_ref_index<ref_num;local_ref_index++)
        {
            uint64_t ref_index=ref_begin+local_ref_index;
            prediction+=(T_calc)weights[ref_index]*(T_calc)label_vec[pred_ref_ids[ref_index]];
        }
        return prediction>(T_calc)0.5;
    }

    template<typename T_coord,typename T_data>
    __attribute__((always_inline)) inline uint32_t param_c<T_coord,T_data>::predict_label_10(uint64_t ref_begin,const vector<T_calc> &weights,const vector<uint8_t> &label_vec)
    {
        const int *ref_ids=pred_ref_ids.data()+ref_begin;
        const T_calc *ref_weights=weights.data()+ref_begin;
        T_calc prediction=(T_calc)ref_weights[0]*(T_calc)label_vec[ref_ids[0]]+(T_calc)ref_weights[1]*(T_calc)label_vec[ref_ids[1]]+(T_calc)ref_weights[2]*(T_calc)label_vec[ref_ids[2]]+(T_calc)ref_weights[3]*(T_calc)label_vec[ref_ids[3]]+(T_calc)ref_weights[4]*(T_calc)label_vec[ref_ids[4]]+(T_calc)ref_weights[5]*(T_calc)label_vec[ref_ids[5]]+(T_calc)ref_weights[6]*(T_calc)label_vec[ref_ids[6]]+(T_calc)ref_weights[7]*(T_calc)label_vec[ref_ids[7]]+(T_calc)ref_weights[8]*(T_calc)label_vec[ref_ids[8]]+(T_calc)ref_weights[9]*(T_calc)label_vec[ref_ids[9]];
        return prediction>(T_calc)0.5;
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_value_two_point_sets_idw(array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,const vector<T_data> &data,uint32_t value_point_set_id)
    {
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,ref_ids.size());
        bool all_value_refs_match=ref_num>0;
        T_calc value_ref_weight_sum=0,weighted_value_sum=0;
        uint32_t value_ref_num=0;
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            if(point_set_id_vec[ref_id]!=value_point_set_id)
            {
                if(ref_index<ref_num)
                    all_value_refs_match=false;
                continue;
            }
            T_calc weight=ref_weights[ref_index];
            value_ref_weight_sum+=weight;
            weighted_value_sum+=weight*data[ref_id];
            if(++value_ref_num==prediction_max_ref_num)
                break;
        }
        if(all_value_refs_match)
            return (T_data)weighted_value_sum;
        if(!(value_ref_weight_sum>0)||!isfinite(value_ref_weight_sum)||!isfinite(weighted_value_sum))
            return 0;
        T_calc prediction=weighted_value_sum/value_ref_weight_sum;
        return isfinite(prediction)?(T_data)prediction:(T_data)0;
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_value_two_point_sets_rbf(uint32_t pred_ref_id,array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,const vector<T_data> &data,uint32_t value_point_set_id)
    {
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,ref_ids.size());
        bool has_rbf_weights=!pred_ref_rbf_fallback_vec[pred_ref_id];
        bool all_value_refs_match=ref_num>0;
        int nearest_value_ref_id=-1;
        T_calc value_ref_weight_sum=0,weighted_value_sum=0;
        uint32_t value_ref_num=0;
        for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            if(point_set_id_vec[ref_id]!=value_point_set_id)
            {
                all_value_refs_match=false;
                continue;
            }
            if(nearest_value_ref_id<0)
                nearest_value_ref_id=ref_id;
            value_ref_num++;
            if(has_rbf_weights)
            {
                T_calc weight=ref_weights[ref_index];
                value_ref_weight_sum+=weight;
                weighted_value_sum+=weight*data[ref_id];
            }
        }
        if(nearest_value_ref_id<0)
            return 0;
        if(!has_rbf_weights||value_ref_num<prediction_min_ref_num||!isfinite(weighted_value_sum))
            return data[nearest_value_ref_id];
        if(all_value_refs_match)
            return (T_data)weighted_value_sum;
        if(!isfinite(value_ref_weight_sum)||value_ref_weight_sum==0)
            return 0;
        T_calc prediction=weighted_value_sum/value_ref_weight_sum;
        return isfinite(prediction)?(T_data)prediction:(T_data)0;
    }

    template<typename T_coord,typename T_data>
    pair<uint32_t,T_data> param_c<T_coord,T_data>::predict_label_and_value_two_point_sets_idw(array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,const vector<T_data> &data,uint32_t value_point_set_id,bool value_requires_prediction)
    {
        T_calc label_weight_sum[2]={0,0};
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,ref_ids.size());
        if(!value_requires_prediction)
        {
            for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
            {
                uint32_t point_set_id=point_set_id_vec[ref_ids[ref_index]];
                label_weight_sum[point_set_id]+=ref_weights[ref_index];
            }
            uint32_t predicted_label=label_weight_sum[1]>label_weight_sum[0];
            return {predicted_label,(T_data)0};
        }

        bool all_value_refs_match=ref_num>0;
        T_calc value_ref_weight_sum=0,weighted_value_sum=0;
        uint32_t value_ref_num=0;
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            uint32_t point_set_id=point_set_id_vec[ref_id];
            if(ref_index<ref_num)
            {
                label_weight_sum[point_set_id]+=ref_weights[ref_index];
                if(point_set_id!=value_point_set_id)
                    all_value_refs_match=false;
            }
            if(point_set_id!=value_point_set_id||value_ref_num==prediction_max_ref_num)
                continue;
            T_calc weight=ref_weights[ref_index];
            value_ref_weight_sum+=weight;
            weighted_value_sum+=weight*data[ref_id];
            value_ref_num++;
            if(value_ref_num==prediction_max_ref_num&&ref_index+1>=ref_num)
                break;
        }
        uint32_t predicted_label=label_weight_sum[1]>label_weight_sum[0];
        if(all_value_refs_match)
            return {predicted_label,(T_data)weighted_value_sum};
        if(!(value_ref_weight_sum>0)||!isfinite(value_ref_weight_sum)||!isfinite(weighted_value_sum))
            return {predicted_label,(T_data)0};
        T_calc predicted_value=weighted_value_sum/value_ref_weight_sum;
        return {predicted_label,isfinite(predicted_value)?(T_data)predicted_value:(T_data)0};
    }

    template<typename T_coord,typename T_data>
    pair<uint32_t,T_data> param_c<T_coord,T_data>::predict_label_and_value_two_point_sets_rbf(uint32_t pred_ref_id,array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,const vector<uint8_t> &label_vec,const vector<T_data> &data,uint32_t value_point_set_id,bool value_requires_prediction)
    {
        if(!value_requires_prediction)
        {
            uint32_t predicted_label;
            if(ref_ids.size()==10)
                predicted_label=predict_label_10(pred_ref_begin[pred_ref_id],pred_ref_prediction_weights,label_vec);
            else
                predicted_label=predict_label(pred_ref_id,pred_ref_prediction_weights,label_vec);
            return {predicted_label,(T_data)0};
        }

        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,ref_ids.size());
        bool has_rbf_weights=!pred_ref_rbf_fallback_vec[pred_ref_id];
        bool all_value_refs_match=ref_num>0;
        int nearest_value_ref_id=-1;
        T_calc label_prediction=0,value_ref_weight_sum=0,weighted_value_sum=0;
        uint32_t value_ref_num=0;
        for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            T_calc weight=ref_weights[ref_index];
            label_prediction+=weight*label_vec[ref_id];
            if(point_set_id_vec[ref_id]!=value_point_set_id)
            {
                all_value_refs_match=false;
                continue;
            }
            if(nearest_value_ref_id<0)
                nearest_value_ref_id=ref_id;
            value_ref_num++;
            if(has_rbf_weights)
            {
                value_ref_weight_sum+=weight;
                weighted_value_sum+=weight*data[ref_id];
            }
        }
        uint32_t predicted_label=label_prediction>(T_calc)0.5;
        if(nearest_value_ref_id<0)
            return {predicted_label,(T_data)0};
        if(!has_rbf_weights||value_ref_num<prediction_min_ref_num||!isfinite(weighted_value_sum))
            return {predicted_label,data[nearest_value_ref_id]};
        if(all_value_refs_match)
            return {predicted_label,(T_data)weighted_value_sum};
        if(!isfinite(value_ref_weight_sum)||value_ref_weight_sum==0)
            return {predicted_label,(T_data)0};
        T_calc predicted_value=weighted_value_sum/value_ref_weight_sum;
        return {predicted_label,isfinite(predicted_value)?(T_data)predicted_value:(T_data)0};
    }

}
