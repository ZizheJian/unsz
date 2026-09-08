#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    prediction_framework_c param_c<T_coord,T_data>::get_prediction_framework(uint32_t point_set_num) const
    {
        if(point_set_num==0)
            throw runtime_error("点集数量不能为0");
        prediction_framework_c framework;
        framework.has_label=point_set_num>1;
        bool use_joint_spatiotemporal=cross_time_spatial_point_num_experimental>0;
        if(use_joint_spatiotemporal && (cross_time_predictor_experimental!="none" || cross_time_label_predictor_experimental!="none"))
            throw runtime_error("统一时空DAG不能再叠加上一时刻或时间插值预测");
        if(use_joint_spatiotemporal)
        {
            framework.value_mode=time_dimension_mode_e::joint_spatiotemporal;
            if(framework.has_label)
                framework.label_mode=time_dimension_mode_e::joint_spatiotemporal;
        }
        else
        {
            if(cross_time_predictor_experimental=="none")
                framework.value_mode=time_dimension_mode_e::spatial;
            else if(cross_time_predictor_experimental=="previous")
                framework.value_mode=time_dimension_mode_e::previous;
            else if(cross_time_predictor_experimental=="interpolation")
                framework.value_mode=time_dimension_mode_e::interpolation;
            else
                throw runtime_error("未知的数值时间预测方式："+cross_time_predictor_experimental);
            if(framework.has_label)
            {
                if(cross_time_label_predictor_experimental=="none")
                    framework.label_mode=time_dimension_mode_e::spatial;
                else if(cross_time_label_predictor_experimental=="previous")
                    framework.label_mode=time_dimension_mode_e::previous;
                else if(cross_time_label_predictor_experimental=="lorenzo3")
                    framework.label_mode=time_dimension_mode_e::lorenzo3;
                else if(cross_time_label_predictor_experimental=="lorenzo4")
                    framework.label_mode=time_dimension_mode_e::lorenzo4;
                else if(cross_time_label_predictor_experimental=="endpoint_agreement")
                    framework.label_mode=time_dimension_mode_e::endpoint_agreement;
                else
                    throw runtime_error("未知的标签时间预测方式："+cross_time_label_predictor_experimental);
            }
        }
        if(!framework.has_label)
            framework.execution=label_value_execution_e::no_label;
        else if(use_fused_two_point_set_prediction_experimental)
            framework.execution=label_value_execution_e::fused;
        else
            framework.execution=label_value_execution_e::staged;
        if(framework.execution==label_value_execution_e::fused && framework.value_mode!=time_dimension_mode_e::spatial)
            throw runtime_error("标签与数值融合执行目前只支持空间数值预测");
        if(framework.execution==label_value_execution_e::fused && framework.label_mode!=time_dimension_mode_e::spatial && framework.label_mode!=time_dimension_mode_e::previous)
            throw runtime_error("标签与数值融合执行目前只支持空间或上一时刻标签预测");
        if(use_linear_previous_label_experimental && (framework.execution!=label_value_execution_e::fused || framework.label_mode!=time_dimension_mode_e::previous))
            throw runtime_error("线性上一时刻标签修正流要求启用融合两点集预测和previous标签预测");
        if(use_linear_previous_label_experimental && use_prediction_order_label_experimental)
            throw runtime_error("线性上一时刻标签修正流不能使用预测顺序标签码流");
        return framework;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::advance_time_prediction_buffers()
    {
        prediction_framework_c framework=get_prediction_framework((uint32_t)point_set_config_vec.size());
        if(framework.value_mode==time_dimension_mode_e::previous)
            previous_time_decompressed_data.swap(current_decompressed_data);
        bool use_causal_time_label=framework.label_mode==time_dimension_mode_e::previous || framework.label_mode==time_dimension_mode_e::lorenzo3 || framework.label_mode==time_dimension_mode_e::lorenzo4;
        bool use_higher_order_time_label=framework.label_mode==time_dimension_mode_e::lorenzo3 || framework.label_mode==time_dimension_mode_e::lorenzo4;
        if(use_higher_order_time_label && !previous_time_point_set_id_vec.empty())
        {
            older_time_point_set_id_vec_vec.insert(older_time_point_set_id_vec_vec.begin(),move(previous_time_point_set_id_vec));
            if(older_time_point_set_id_vec_vec.size()>3)
                older_time_point_set_id_vec_vec.pop_back();
        }
        if(framework.value_mode==time_dimension_mode_e::previous || use_causal_time_label)
            previous_time_point_set_id_vec.swap(point_set_id_vec);
    }
}
