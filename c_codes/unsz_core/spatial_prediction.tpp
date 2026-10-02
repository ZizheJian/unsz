#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    __attribute__((always_inline)) inline T_data param_c<T_coord,T_data>::predict_value_one_point_set(uint32_t prediction_index,const vector<int>&ref_ids,const vector<uint64_t>&ref_begin,const vector<T_calc>&weights,const vector<T_data>&data)
    {
        uint64_t begin=ref_begin[prediction_index],end=ref_begin[prediction_index+1];
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,(uint32_t)(end-begin));
        if(ref_num==4)
            return predict_value_one_point_set_4(begin,ref_ids,weights,data);
        if(ref_num==10)
            return predict_value_one_point_set_10(begin,ref_ids,weights,data);
        T_calc prediction=0;
        for(uint32_t local_ref_index=0;local_ref_index<ref_num;local_ref_index++)
        {
            uint64_t ref_index=begin+local_ref_index;
            prediction+=(T_calc)weights[ref_index]*(T_calc)data[ref_ids[ref_index]];
        }
        return (T_data)prediction;
    }

    template<typename T_coord,typename T_data>
    __attribute__((always_inline)) inline T_data param_c<T_coord,T_data>::predict_value_one_point_set_4(uint64_t ref_begin,const vector<int>&reference_ids,const vector<T_calc>&weights,const vector<T_data>&data)
    {
        const int *ref_ids=reference_ids.data()+ref_begin;
        const T_calc *ref_weights=weights.data()+ref_begin;
        T_calc prediction=(T_calc)ref_weights[0]*(T_calc)data[ref_ids[0]]+(T_calc)ref_weights[1]*(T_calc)data[ref_ids[1]]+(T_calc)ref_weights[2]*(T_calc)data[ref_ids[2]]+(T_calc)ref_weights[3]*(T_calc)data[ref_ids[3]];
        return (T_data)prediction;
    }

    template<typename T_coord,typename T_data>
    __attribute__((always_inline)) inline T_data param_c<T_coord,T_data>::predict_value_one_point_set_10(uint64_t ref_begin,const vector<int>&reference_ids,const vector<T_calc>&weights,const vector<T_data>&data)
    {
        const int *ref_ids=reference_ids.data()+ref_begin;
        const T_calc *ref_weights=weights.data()+ref_begin;
        T_calc prediction=(T_calc)ref_weights[0]*(T_calc)data[ref_ids[0]]+(T_calc)ref_weights[1]*(T_calc)data[ref_ids[1]]+(T_calc)ref_weights[2]*(T_calc)data[ref_ids[2]]+(T_calc)ref_weights[3]*(T_calc)data[ref_ids[3]]+(T_calc)ref_weights[4]*(T_calc)data[ref_ids[4]]+(T_calc)ref_weights[5]*(T_calc)data[ref_ids[5]]+(T_calc)ref_weights[6]*(T_calc)data[ref_ids[6]]+(T_calc)ref_weights[7]*(T_calc)data[ref_ids[7]]+(T_calc)ref_weights[8]*(T_calc)data[ref_ids[8]]+(T_calc)ref_weights[9]*(T_calc)data[ref_ids[9]];
        return (T_data)prediction;
    }

    template<typename T_coord,typename T_data>
    __attribute__((always_inline)) inline uint32_t param_c<T_coord,T_data>::predict_label(uint32_t prediction_index,const vector<int>&ref_ids,const vector<uint64_t>&ref_begin,const vector<T_calc>&weights,const vector<uint8_t>&label_vec)
    {
        uint64_t begin=ref_begin[prediction_index],end=ref_begin[prediction_index+1];
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,(uint32_t)(end-begin));
        if(ref_num==4)
            return predict_label_4(begin,ref_ids,weights,label_vec);
        if(ref_num==10)
            return predict_label_10(begin,ref_ids,weights,label_vec);
        T_calc prediction=0;
        for(uint32_t local_ref_index=0;local_ref_index<ref_num;local_ref_index++)
        {
            uint64_t ref_index=begin+local_ref_index;
            prediction+=(T_calc)weights[ref_index]*(T_calc)label_vec[ref_ids[ref_index]];
        }
        return prediction>(T_calc)0.5;
    }

    template<typename T_coord,typename T_data>
    __attribute__((always_inline)) inline uint32_t param_c<T_coord,T_data>::predict_label_4(uint64_t ref_begin,const vector<int>&reference_ids,const vector<T_calc>&weights,const vector<uint8_t>&label_vec)
    {
        const int *ref_ids=reference_ids.data()+ref_begin;
        const T_calc *ref_weights=weights.data()+ref_begin;
        T_calc prediction=(T_calc)ref_weights[0]*(T_calc)label_vec[ref_ids[0]]+(T_calc)ref_weights[1]*(T_calc)label_vec[ref_ids[1]]+(T_calc)ref_weights[2]*(T_calc)label_vec[ref_ids[2]]+(T_calc)ref_weights[3]*(T_calc)label_vec[ref_ids[3]];
        return prediction>(T_calc)0.5;
    }

    template<typename T_coord,typename T_data>
    __attribute__((always_inline)) inline uint32_t param_c<T_coord,T_data>::predict_label_10(uint64_t ref_begin,const vector<int>&reference_ids,const vector<T_calc>&weights,const vector<uint8_t>&label_vec)
    {
        const int *ref_ids=reference_ids.data()+ref_begin;
        const T_calc *ref_weights=weights.data()+ref_begin;
        T_calc prediction=(T_calc)ref_weights[0]*(T_calc)label_vec[ref_ids[0]]+(T_calc)ref_weights[1]*(T_calc)label_vec[ref_ids[1]]+(T_calc)ref_weights[2]*(T_calc)label_vec[ref_ids[2]]+(T_calc)ref_weights[3]*(T_calc)label_vec[ref_ids[3]]+(T_calc)ref_weights[4]*(T_calc)label_vec[ref_ids[4]]+(T_calc)ref_weights[5]*(T_calc)label_vec[ref_ids[5]]+(T_calc)ref_weights[6]*(T_calc)label_vec[ref_ids[6]]+(T_calc)ref_weights[7]*(T_calc)label_vec[ref_ids[7]]+(T_calc)ref_weights[8]*(T_calc)label_vec[ref_ids[8]]+(T_calc)ref_weights[9]*(T_calc)label_vec[ref_ids[9]];
        return prediction>(T_calc)0.5;
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_value_two_point_sets_idw(array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,const vector<T_data> &data,uint32_t value_point_set_id,const vector<uint8_t> *valid_ref_mask)
    {
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,ref_ids.size());
        bool all_value_refs_match=ref_num>0;
        T_calc value_ref_weight_sum=0,weighted_value_sum=0;
        uint32_t value_ref_num=0;
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            if(point_set_id_vec[ref_id]!=value_point_set_id||(valid_ref_mask&&!(*valid_ref_mask)[ref_id]))
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
    T_data param_c<T_coord,T_data>::predict_value_two_point_sets_rbf(bool has_rbf_weights,array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,const vector<T_data> &data,uint32_t value_point_set_id,const vector<uint8_t> *valid_ref_mask)
    {
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,ref_ids.size());
        bool all_value_refs_match=ref_num>0;
        int nearest_value_ref_id=-1;
        T_calc value_ref_weight_sum=0,weighted_value_sum=0;
        uint32_t value_ref_num=0;
        for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            if(point_set_id_vec[ref_id]!=value_point_set_id||(valid_ref_mask&&!(*valid_ref_mask)[ref_id]))
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
    fused_spatial_prediction_c<T_data> param_c<T_coord,T_data>::predict_fused_spatial_idw(array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,bool predict_label,const array<spatial_value_source_c<T_data>,2>&spatial_value_sources)
    {
        fused_spatial_prediction_c<T_data> result;
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,ref_ids.size());
        T_calc label_weight_sum[2]={0,0},value_weight_sum[2]={0,0},weighted_value_sum[2]={0,0};
        uint32_t value_ref_num[2]={0,0};
        bool all_refs_match[2]={ref_num>0,ref_num>0};
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            uint32_t point_set_id=point_set_id_vec[ref_id];
            T_calc weight=ref_weights[ref_index];
            if(predict_label&&ref_index<ref_num)
                label_weight_sum[point_set_id]+=weight;
            uint32_t other_point_set_id=1-point_set_id;
            if(ref_index<ref_num&&spatial_value_sources[other_point_set_id].data)
                all_refs_match[other_point_set_id]=false;
            const spatial_value_source_c<T_data>&source=spatial_value_sources[point_set_id];
            if(!source.data)
                continue;
            if(source.valid_ref_mask&&!(*source.valid_ref_mask)[ref_id])
            {
                if(ref_index<ref_num)
                    all_refs_match[point_set_id]=false;
                continue;
            }
            if(value_ref_num[point_set_id]==prediction_max_ref_num)
                continue;
            value_weight_sum[point_set_id]+=weight;
            weighted_value_sum[point_set_id]+=weight*(*source.data)[ref_id];
            value_ref_num[point_set_id]++;
        }
        if(predict_label)
            result.label=label_weight_sum[1]>label_weight_sum[0];
        for(uint32_t point_set_id=0;point_set_id<2;point_set_id++)
        {
            if(!spatial_value_sources[point_set_id].data)
                continue;
            if(all_refs_match[point_set_id])
                result.values[point_set_id]=(T_data)weighted_value_sum[point_set_id];
            else if(value_weight_sum[point_set_id]>0&&isfinite(value_weight_sum[point_set_id])&&isfinite(weighted_value_sum[point_set_id]))
            {
                T_calc prediction=weighted_value_sum[point_set_id]/value_weight_sum[point_set_id];
                if(isfinite(prediction))
                    result.values[point_set_id]=(T_data)prediction;
            }
        }
        return result;
    }

    template<typename T_coord,typename T_data>
    fused_spatial_prediction_c<T_data> param_c<T_coord,T_data>::predict_fused_spatial_rbf(bool has_rbf_weights,array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,bool predict_label,const array<spatial_value_source_c<T_data>,2>&spatial_value_sources)
    {
        fused_spatial_prediction_c<T_data> result;
        uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,ref_ids.size());
        T_calc label_prediction=0,value_weight_sum[2]={0,0},weighted_value_sum[2]={0,0};
        uint32_t value_ref_num[2]={0,0};
        int nearest_ref_id[2]={-1,-1};
        bool all_refs_match[2]={ref_num>0,ref_num>0};
        for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            uint32_t point_set_id=point_set_id_vec[ref_id];
            T_calc weight=ref_weights[ref_index];
            if(predict_label)
                label_prediction+=weight*point_set_id;
            uint32_t other_point_set_id=1-point_set_id;
            if(spatial_value_sources[other_point_set_id].data)
                all_refs_match[other_point_set_id]=false;
            const spatial_value_source_c<T_data>&source=spatial_value_sources[point_set_id];
            if(!source.data)
                continue;
            if(source.valid_ref_mask&&!(*source.valid_ref_mask)[ref_id])
            {
                all_refs_match[point_set_id]=false;
                continue;
            }
            if(nearest_ref_id[point_set_id]<0)
                nearest_ref_id[point_set_id]=ref_id;
            value_ref_num[point_set_id]++;
            if(has_rbf_weights)
            {
                value_weight_sum[point_set_id]+=weight;
                weighted_value_sum[point_set_id]+=weight*(*source.data)[ref_id];
            }
        }
        if(predict_label)
            result.label=label_prediction>(T_calc)0.5;
        for(uint32_t point_set_id=0;point_set_id<2;point_set_id++)
        {
            if(!spatial_value_sources[point_set_id].data)
                continue;
            if(nearest_ref_id[point_set_id]<0)
                continue;
            if(!has_rbf_weights||value_ref_num[point_set_id]<prediction_min_ref_num||!isfinite(weighted_value_sum[point_set_id]))
                result.values[point_set_id]=(*spatial_value_sources[point_set_id].data)[nearest_ref_id[point_set_id]];
            else if(all_refs_match[point_set_id])
                result.values[point_set_id]=(T_data)weighted_value_sum[point_set_id];
            else if(isfinite(value_weight_sum[point_set_id])&&value_weight_sum[point_set_id]!=0)
            {
                T_calc prediction=weighted_value_sum[point_set_id]/value_weight_sum[point_set_id];
                if(isfinite(prediction))
                    result.values[point_set_id]=(T_data)prediction;
            }
        }
        return result;
    }

}
