#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_time_interpolation()
    {
        interpolation_left_history_id=-1;
        interpolation_right_history_id=-1;
        interpolation_right_weight=0;
        if (cross_time_predictor_experimental!="interpolation" && cross_time_label_predictor_experimental!="endpoint_agreement")
            return;
        for (uint32_t history_id=0;history_id<reconstructed_time_step_vec.size();history_id++)
        {
            uint32_t history_time_step=reconstructed_time_step_vec[history_id];
            if (history_time_step<current_time_step && (interpolation_left_history_id<0 || history_time_step>reconstructed_time_step_vec[interpolation_left_history_id]))
                interpolation_left_history_id=history_id;
            if (history_time_step>current_time_step && (interpolation_right_history_id<0 || history_time_step<reconstructed_time_step_vec[interpolation_right_history_id]))
                interpolation_right_history_id=history_id;
        }
        if (interpolation_left_history_id<0 || interpolation_right_history_id<0)
            return;
        uint32_t left_time_step=reconstructed_time_step_vec[interpolation_left_history_id],right_time_step=reconstructed_time_step_vec[interpolation_right_history_id];
        interpolation_right_weight=(double)(current_time_step-left_time_step)/(right_time_step-left_time_step);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_time_label(uint32_t original_id,uint32_t label_channel_id,uint32_t &prediction)
    {
        auto get_binary_label=[label_channel_id,original_id](const vector<uint8_t> &point_set_ids){return point_set_ids[original_id]==label_channel_id?0:1;};
        if(cross_time_label_predictor_experimental=="endpoint_agreement")
        {
            if(interpolation_left_history_id<0 || interpolation_right_history_id<0)
                return false;
            const vector<uint8_t> &left_point_set_ids=reconstructed_time_point_set_id_vec[interpolation_left_history_id];
            const vector<uint8_t> &right_point_set_ids=reconstructed_time_point_set_id_vec[interpolation_right_history_id];
            if(left_point_set_ids.size()!=point_num || right_point_set_ids.size()!=point_num)
                throw runtime_error("时间端点标签历史的点数不正确");
            uint32_t left_label=get_binary_label(left_point_set_ids),right_label=get_binary_label(right_point_set_ids);
            if(left_label!=right_label)
                return false;
            prediction=left_label;
            return true;
        }
        if(previous_time_point_set_id_vec.size()!=point_num)
            return false;
        uint32_t label1=get_binary_label(previous_time_point_set_id_vec);
        if(cross_time_label_predictor_experimental=="previous" || older_time_point_set_id_vec_vec.empty())
        {
            prediction=label1;
            return true;
        }
        if(cross_time_label_predictor_experimental=="lorenzo3")
        {
            if(older_time_point_set_id_vec_vec.size()<2)
            {
                prediction=label1;
                return true;
            }
            int value=3*(int)label1-3*(int)get_binary_label(older_time_point_set_id_vec_vec[0])+(int)get_binary_label(older_time_point_set_id_vec_vec[1]);
            prediction=value>0;
            return true;
        }
        if(cross_time_label_predictor_experimental=="lorenzo4")
        {
            if(older_time_point_set_id_vec_vec.size()<3)
            {
                prediction=label1;
                return true;
            }
            int value=4*(int)label1-6*(int)get_binary_label(older_time_point_set_id_vec_vec[0])+4*(int)get_binary_label(older_time_point_set_id_vec_vec[1])-(int)get_binary_label(older_time_point_set_id_vec_vec[2]);
            prediction=value>0;
            return true;
        }
        return false;
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_time_interpolation(uint32_t original_id,uint32_t point_set_id,T_data &prediction)
    {
        if (interpolation_left_history_id<0 || interpolation_right_history_id<0)
            return false;
        auto &left_data=reconstructed_time_data_vec[interpolation_left_history_id];
        auto &right_data=reconstructed_time_data_vec[interpolation_right_history_id];
        auto &left_point_set_id_vec=reconstructed_time_point_set_id_vec[interpolation_left_history_id];
        auto &right_point_set_id_vec=reconstructed_time_point_set_id_vec[interpolation_right_history_id];
        if (left_data.size()!=point_num || right_data.size()!=point_num || left_point_set_id_vec.size()!=point_num || right_point_set_id_vec.size()!=point_num)
            throw runtime_error("时间插值历史数据的点数不正确");
        if (left_point_set_id_vec[original_id]!=point_set_id || right_point_set_id_vec[original_id]!=point_set_id)
            return false;
        prediction=(T_data)((1-interpolation_right_weight)*left_data[original_id]+interpolation_right_weight*right_data[original_id]);
        return true;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::save_time_reconstruction()
    {
        if (cross_time_predictor_experimental!="interpolation" && cross_time_label_predictor_experimental!="endpoint_agreement")
            return;
        if (current_decompressed_data.size()!=point_num || point_set_id_vec.size()!=point_num)
            throw runtime_error("无法保存不完整的时间插值重构数据");
        reconstructed_time_step_vec.push_back(current_time_step);
        reconstructed_time_data_vec.push_back(cross_time_predictor_experimental=="interpolation"?current_decompressed_data:vector<T_data>());
        reconstructed_time_point_set_id_vec.push_back(point_set_id_vec);
    }
}
