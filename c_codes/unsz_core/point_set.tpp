#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    uint32_t param_c<T_coord,T_data>::get_point_set_id(T_data val)
    {
        // 当前数据的点集划分规则。以后可以在这里扩展为更多点集，返回值必须从0开始且连续。
        return val<-1000?1:0;
    }

    template<typename T_coord,typename T_data>
    uint32_t param_c<T_coord,T_data>::predict_point_set_id(int pred_id,vector<int> &ref_ids,uint32_t point_set_num)
    {
        if (ref_ids.empty())
            return 0;
        if (point_set_predictor=="majority")
        {
            uint32_t ref_num=min<uint32_t>(point_set_ref_num,ref_ids.size());
            vector<uint32_t> vote_num(point_set_num,0);
            for (uint32_t i=0;i<ref_num;i++)
                vote_num[point_set_id_vec[ref_ids[i]]]++;
            uint32_t result=point_set_id_vec[ref_ids[0]];//票数相同时采用最近参考点所属的点集
            for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                if (vote_num[point_set_id]>vote_num[result])
                    result=point_set_id;
            return result;
        }
        if (point_set_predictor!="inverse_proportion")
            throw runtime_error("未知的点集标签预测器："+point_set_predictor);
        vector<double> weight_sum(point_set_num,0);
        uint32_t ref_num=min<uint32_t>(point_set_ref_num,ref_ids.size());
        for (uint32_t i=0;i<ref_num;i++)
        {
            int ref_id=ref_ids[i];
            double dist=(double)get_dist_to_coord(coord_array[pred_id],coord_array[ref_id],dim);
            if (dist==0)
                return point_set_id_vec[ref_id];
            weight_sum[point_set_id_vec[ref_id]]+=pow(dist,-inverse_distance_power/2);
        }
        uint32_t result=0;
        for (uint32_t point_set_id=1;point_set_id<point_set_num;point_set_id++)
            if (weight_sum[point_set_id]>weight_sum[result])
                result=point_set_id;
        return result;
    }

    template<typename T_coord,typename T_data>
    double param_c<T_coord,T_data>::predict_point_set_value(int pred_id,vector<int> &ref_ids)
    {
        if (ref_ids.empty())
            return 0;
        double total_weight=0,pred=0;
        uint32_t ref_num=min<uint32_t>(point_set_ref_num,ref_ids.size());
        for (uint32_t i=0;i<ref_num;i++)
        {
            int ref_id=ref_ids[i];
            double dist=(double)get_dist_to_coord(coord_array[pred_id],coord_array[ref_id],dim);
            if (dist==0)
                return point_set_prediction_value_vec[ref_id];
            double weight=pow(dist,-inverse_distance_power/2);
            total_weight+=weight;
            pred+=weight*point_set_prediction_value_vec[ref_id];
        }
        if (total_weight==0)
            return 0;
        return min(1.0,max(0.0,pred/total_weight));
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_value(int pred_id,vector<int> &ref_ids,uint32_t point_set_id,vector<T_data> &data)
    {
        vector<int> valid_ref_ids;
        valid_ref_ids.reserve(actual_ref_num);
        for (int ref_id:ref_ids)
            if (point_set_id_vec[ref_id]==point_set_id)
            {
                valid_ref_ids.push_back(ref_id);
                if (valid_ref_ids.size()==actual_ref_num)
                    break;
            }
        if (valid_ref_ids.empty())
            return 0;

        point_set_codec_e codec=point_set_codec_vec[point_set_id];
        if (codec==point_set_codec_e::fitting && dim==3 && valid_ref_ids.size()>dim)
        {
            uint32_t ref_num=valid_ref_ids.size();
            double A[4][4]={0},b[4]={0},x[4];
            for (uint32_t i=0;i<ref_num;i++)
            {
                int ref_id=valid_ref_ids[i];
                double dc[3];
                for (uint32_t d=0;d<3;d++)
                    dc[d]=(double)coord_array[ref_id][d]-coord_array[pred_id][d];
                for (uint32_t row=0;row<3;row++)
                {
                    for (uint32_t col=0;col<3;col++)
                        A[row][col]+=dc[row]*dc[col];
                    A[row][3]+=dc[row];
                    A[3][row]+=dc[row];
                    b[row]+=dc[row]*data[ref_id];
                }
                A[3][3]++;
                b[3]+=data[ref_id];
            }
            if (solve_4x4(A,b,x))
                return (T_data)x[3];
        }
        else if (codec!=point_set_codec_e::inverse_proportion && codec!=point_set_codec_e::fitting)
            throw runtime_error("点集使用了不支持的数值预测器");

        double total_weight=0,pred=0;
        for (int ref_id:valid_ref_ids)
        {
            double dist=(double)get_dist_to_coord(coord_array[pred_id],coord_array[ref_id],dim);
            if (dist==0)
                return data[ref_id];
            double weight=pow(dist,-inverse_distance_power/2);
            total_weight+=weight;
            pred+=weight*data[ref_id];
        }
        return (T_data)(pred/total_weight);
    }
}
