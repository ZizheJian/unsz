#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_temporal_prediction()
    {
        temporal_history_ids={{-1,-1}};
        temporal_weights={{0,0}};
        previous_time_value_available=previous_time_decompressed_data.size()==point_num;
        previous_time_label_available=previous_time_point_set_id_vec.size()==point_num;
        temporal_interpolation_available=false;
        if(value_flow=="constant_extrapolation_spatial_residual"||value_flow=="linear_interpolation_spatial_residual"||uses_anchor_interpolation_stride_8_flow())
        {
            current_time_spatial_residual_data.assign(point_num,0);
            current_time_spatial_residual_valid.assign(point_num,0);
        }
        const prediction_framework_c &framework=current_prediction_framework;
        if(framework.value_mode!=time_dimension_mode_e::linear_interpolation&&framework.value_mode!=time_dimension_mode_e::linear_interpolation_spatial_residual&&framework.label_mode!=time_dimension_mode_e::label_linear_interpolation_spatial_residual)
            return;
        for (uint32_t history_id=0;history_id<reconstructed_time_step_vec.size();history_id++)
        {
            uint32_t history_time_step=reconstructed_time_step_vec[history_id];
            if(history_time_step<current_time_step&&(temporal_history_ids[0]<0||history_time_step>reconstructed_time_step_vec[temporal_history_ids[0]]))
                temporal_history_ids[0]=history_id;
            if(history_time_step>current_time_step&&(temporal_history_ids[1]<0||history_time_step<reconstructed_time_step_vec[temporal_history_ids[1]]))
                temporal_history_ids[1]=history_id;
        }
        if(temporal_history_ids[0]<0||temporal_history_ids[1]<0)
            return;
        bool interpolate_value=framework.value_mode==time_dimension_mode_e::linear_interpolation||framework.value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual;
        bool interpolate_label=framework.label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual;
        const vector<T_data>&left_data=reconstructed_time_data_vec[temporal_history_ids[0]],&right_data=reconstructed_time_data_vec[temporal_history_ids[1]];
        if(interpolate_value&&(left_data.size()!=point_num||right_data.size()!=point_num))
            throw runtime_error("时间插值历史数据的点数不正确");
        const vector<uint8_t>&left_point_set_ids=reconstructed_time_point_set_id_vec[temporal_history_ids[0]],&right_point_set_ids=reconstructed_time_point_set_id_vec[temporal_history_ids[1]];
        if((interpolate_label||(interpolate_value&&point_set_config_vec.size()>1))&&(left_point_set_ids.size()!=point_num||right_point_set_ids.size()!=point_num))
            throw runtime_error("时间插值点集历史的点数不正确");
        temporal_interpolation_available=true;
        uint32_t left_time_step=reconstructed_time_step_vec[temporal_history_ids[0]],right_time_step=reconstructed_time_step_vec[temporal_history_ids[1]];
        T_calc right_weight;
        if(time_step_coordinate_vec.empty())
            right_weight=(T_calc)(current_time_step-left_time_step)/(right_time_step-left_time_step);
        else
        {
            T_coord left_coordinate=time_step_coordinate_vec[left_time_step],right_coordinate=time_step_coordinate_vec[right_time_step],current_coordinate=time_step_coordinate_vec[current_time_step];
            right_weight=(T_calc)((current_coordinate-left_coordinate)/(right_coordinate-left_coordinate));
        }
        temporal_weights[0]=(T_calc)1-right_weight;
        temporal_weights[1]=right_weight;
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e label_mode>
    bool param_c<T_coord,T_data>::predict_temporal_label(uint32_t original_id,uint32_t label_channel_id,uint32_t &prediction)
    {
        auto get_binary_label=[label_channel_id,original_id](const vector<uint8_t> &point_set_ids)
        {
            if(point_set_ids[original_id]==label_channel_id)
                return 0;
            return 1;
        };
        if constexpr(label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual)
        {
            if(!temporal_interpolation_available)
                return false;
            const vector<uint8_t> &left_point_set_ids=reconstructed_time_point_set_id_vec[temporal_history_ids[0]];
            const vector<uint8_t> &right_point_set_ids=reconstructed_time_point_set_id_vec[temporal_history_ids[1]];
            uint32_t left_label=get_binary_label(left_point_set_ids),right_label=get_binary_label(right_point_set_ids);
            if(left_label!=right_label)
                return false;
            prediction=left_label;
            return true;
        }
        if(!previous_time_label_available)
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
        if(!temporal_interpolation_available)
            return false;
        auto &left_data=reconstructed_time_data_vec[temporal_history_ids[0]];
        auto &right_data=reconstructed_time_data_vec[temporal_history_ids[1]];
        if(point_set_config_vec.size()>1)
        {
            auto &left_point_set_id_vec=reconstructed_time_point_set_id_vec[temporal_history_ids[0]];
            auto &right_point_set_id_vec=reconstructed_time_point_set_id_vec[temporal_history_ids[1]];
            if(left_point_set_id_vec[original_id]!=point_set_id||right_point_set_id_vec[original_id]!=point_set_id)
                return false;
        }
        prediction=(T_data)(temporal_weights[0]*(T_calc)left_data[original_id]+temporal_weights[1]*(T_calc)right_data[original_id]);
        return true;
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e value_mode>
    bool param_c<T_coord,T_data>::predict_temporal_value_baseline(uint32_t original_id,uint32_t point_set_id,T_data &prediction)
    {
        if constexpr(value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
        {
            if(!previous_time_value_available)
                return false;
            if(point_set_config_vec.size()>1&&(!previous_time_label_available||previous_time_point_set_id_vec[original_id]!=point_set_id))
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
    void param_c<T_coord,T_data>::update_temporal_residual(int pred_id,uint32_t original_id,uint32_t point_set_id)
    {
        if constexpr(value_mode!=time_dimension_mode_e::constant_extrapolation_spatial_residual&&value_mode!=time_dimension_mode_e::linear_interpolation_spatial_residual)
            return;
        T_data baseline=0;
        if(predict_temporal_value_baseline<value_mode>(original_id,point_set_id,baseline))
        {
            current_time_spatial_residual_data[pred_id]=current_decompressed_data[pred_id]-baseline;
            current_time_spatial_residual_valid[pred_id]=1;
        }
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
        bool keep_previous_label=point_set_config_vec.size()>1&&(keep_previous_value||framework.label_mode==time_dimension_mode_e::constant_extrapolation||first_time_anchor);
        if(save_interpolation_history)
        {
            if(save_value_history&&current_decompressed_data.size()!=point_num)
                throw runtime_error("无法保存不完整的数值时间预测状态");
            if(save_label_history&&point_set_id_vec.size()!=point_num)
                throw runtime_error("无法保存不完整的标签时间预测状态");
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
