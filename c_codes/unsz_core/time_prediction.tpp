#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_linear_interpolation()
    {
        linear_interpolation_left_history_id=-1;
        linear_interpolation_right_history_id=-1;
        linear_interpolation_right_weight=0;
        if(value_flow=="constant_extrapolation_spatial_residual"||value_flow=="linear_interpolation_spatial_residual"||uses_anchor_interpolation_stride_8_flow())
            current_time_spatial_residual_data.assign(point_num,0);
        const prediction_framework_c &framework=current_prediction_framework;
        if(framework.value_mode!=time_dimension_mode_e::linear_interpolation&&framework.value_mode!=time_dimension_mode_e::linear_interpolation_spatial_residual&&framework.label_mode!=time_dimension_mode_e::label_linear_interpolation_spatial_residual)
            return;
        for (uint32_t history_id=0;history_id<reconstructed_time_step_vec.size();history_id++)
        {
            uint32_t history_time_step=reconstructed_time_step_vec[history_id];
            if (history_time_step<current_time_step && (linear_interpolation_left_history_id<0 || history_time_step>reconstructed_time_step_vec[linear_interpolation_left_history_id]))
                linear_interpolation_left_history_id=history_id;
            if (history_time_step>current_time_step && (linear_interpolation_right_history_id<0 || history_time_step<reconstructed_time_step_vec[linear_interpolation_right_history_id]))
                linear_interpolation_right_history_id=history_id;
        }
        if (linear_interpolation_left_history_id<0 || linear_interpolation_right_history_id<0)
            return;
        uint32_t left_time_step=reconstructed_time_step_vec[linear_interpolation_left_history_id],right_time_step=reconstructed_time_step_vec[linear_interpolation_right_history_id];
        if(time_step_coordinate_vec.empty())
            linear_interpolation_right_weight=(T_calc)(current_time_step-left_time_step)/(right_time_step-left_time_step);
        else
        {
            T_coord left_coordinate=time_step_coordinate_vec[left_time_step],right_coordinate=time_step_coordinate_vec[right_time_step],current_coordinate=time_step_coordinate_vec[current_time_step];
            linear_interpolation_right_weight=(T_calc)((current_coordinate-left_coordinate)/(right_coordinate-left_coordinate));
        }
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e label_mode>
    bool param_c<T_coord,T_data>::predict_time_label_fixed(uint32_t original_id,uint32_t label_channel_id,uint32_t &prediction)
    {
        auto get_binary_label=[label_channel_id,original_id](const vector<uint8_t> &point_set_ids)
        {
            if(point_set_ids[original_id]==label_channel_id)
                return 0;
            return 1;
        };
        if constexpr(label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual)
        {
            if(linear_interpolation_left_history_id<0 || linear_interpolation_right_history_id<0)
                return false;
            const vector<uint8_t> &left_point_set_ids=reconstructed_time_point_set_id_vec[linear_interpolation_left_history_id];
            const vector<uint8_t> &right_point_set_ids=reconstructed_time_point_set_id_vec[linear_interpolation_right_history_id];
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
        if constexpr(label_mode==time_dimension_mode_e::constant_extrapolation)
        {
            prediction=label1;
            return true;
        }
        return false;
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_linear_interpolation(uint32_t original_id,uint32_t point_set_id,T_data &prediction)
    {
        if (linear_interpolation_left_history_id<0 || linear_interpolation_right_history_id<0)
            return false;
        auto &left_data=reconstructed_time_data_vec[linear_interpolation_left_history_id];
        auto &right_data=reconstructed_time_data_vec[linear_interpolation_right_history_id];
        if(left_data.size()!=point_num||right_data.size()!=point_num)
            throw runtime_error("时间插值历史数据的点数不正确");
        if(point_set_config_vec.size()>1)
        {
            auto &left_point_set_id_vec=reconstructed_time_point_set_id_vec[linear_interpolation_left_history_id];
            auto &right_point_set_id_vec=reconstructed_time_point_set_id_vec[linear_interpolation_right_history_id];
            if(left_point_set_id_vec.size()!=point_num||right_point_set_id_vec.size()!=point_num)
                throw runtime_error("时间插值点集历史的点数不正确");
            if(left_point_set_id_vec[original_id]!=point_set_id||right_point_set_id_vec[original_id]!=point_set_id)
                return false;
        }
        T_calc right_weight=linear_interpolation_right_weight;
        prediction=(T_data)(((T_calc)1-right_weight)*(T_calc)left_data[original_id]+right_weight*(T_calc)right_data[original_id]);
        return true;
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e value_mode>
    bool param_c<T_coord,T_data>::predict_time_spatial_residual_baseline_fixed(uint32_t original_id,uint32_t point_set_id,T_data &prediction)
    {
        if constexpr(value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
        {
            if(previous_time_decompressed_data.size()!=point_num || previous_time_point_set_id_vec.size()!=point_num || previous_time_point_set_id_vec[original_id]!=point_set_id)
                return false;
            prediction=previous_time_decompressed_data[original_id];
            return true;
        }
        if constexpr(value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
            return predict_linear_interpolation(original_id,point_set_id,prediction);
        return false;
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e value_mode>
    void param_c<T_coord,T_data>::update_time_spatial_residual_fixed(int pred_id,uint32_t original_id,uint32_t point_set_id)
    {
        if constexpr(value_mode!=time_dimension_mode_e::constant_extrapolation_spatial_residual&&value_mode!=time_dimension_mode_e::linear_interpolation_spatial_residual)
            return;
        if(current_time_spatial_residual_data.size()!=point_num)
            return;
        T_data baseline=0;
        if(predict_time_spatial_residual_baseline_fixed<value_mode>(original_id,point_set_id,baseline))
            current_time_spatial_residual_data[pred_id]=current_decompressed_data[pred_id]-baseline;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::update_time_prediction_state()
    {
        const prediction_framework_c &framework=current_prediction_framework;
        bool first_time_anchor=uses_anchor_interpolation_stride_8_flow()&&current_time_step==0;
        bool save_interpolation_history=value_flow=="linear_interpolation"||value_flow=="linear_interpolation_spatial_residual"||label_flow=="linear_interpolation_spatial_residual"||uses_anchor_interpolation_stride_8_flow();
        bool save_value_history=value_flow=="linear_interpolation"||value_flow=="linear_interpolation_spatial_residual"||uses_anchor_interpolation_stride_8_flow();
        bool save_label_history=save_interpolation_history&&point_set_config_vec.size()>1;
        bool keep_previous_value=framework.value_mode==time_dimension_mode_e::constant_extrapolation||framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||first_time_anchor;
        bool keep_previous_label=keep_previous_value||framework.label_mode==time_dimension_mode_e::constant_extrapolation||first_time_anchor;
        if(save_interpolation_history)
        {
            if(current_decompressed_data.size()!=point_num||point_set_id_vec.size()!=point_num)
                throw runtime_error("无法保存不完整的时间预测状态");
            reconstructed_time_step_vec.push_back(current_time_step);
            if(save_value_history)
            {
                if(keep_previous_value)
                    reconstructed_time_data_vec.push_back(current_decompressed_data);
                else
                    reconstructed_time_data_vec.push_back(move(current_decompressed_data));
            }
            else
                reconstructed_time_data_vec.emplace_back();
            if(save_label_history)
            {
                if(keep_previous_label)
                    reconstructed_time_point_set_id_vec.push_back(point_set_id_vec);
                else
                    reconstructed_time_point_set_id_vec.push_back(move(point_set_id_vec));
            }
            else
                reconstructed_time_point_set_id_vec.emplace_back();
        }
        if(keep_previous_value)
            previous_time_decompressed_data.swap(current_decompressed_data);
        if(keep_previous_label)
            previous_time_point_set_id_vec.swap(point_set_id_vec);
    }
}
