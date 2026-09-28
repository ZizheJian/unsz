#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::uses_anchor_interpolation_stride_8_flow() const
    {
        return label_flow=="anchor_interpolation_stride_8"||value_flow=="anchor_interpolation_stride_8"||value_flow=="anchor_interpolation_stride_8_spatial_residual";
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::current_time_step_is_regular_anchor() const
    {
        return uses_anchor_interpolation_stride_8_flow()&&current_time_step%8==0;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::resolve_single_point_set_time_mode()
    {
        if(point_set_config_vec.size()!=1)
            throw runtime_error("单点集时间流程选择要求field恰好配置一个点集");
        if(value_flow=="joint_spatiotemporal")
            return;
        bool value_uses_anchor_flow=value_flow=="anchor_interpolation_stride_8"||value_flow=="anchor_interpolation_stride_8_spatial_residual";
        if(!value_uses_anchor_flow&&value_flow!="none"&&value_flow!="constant_extrapolation"&&value_flow!="linear_interpolation"&&value_flow!="constant_extrapolation_spatial_residual"&&value_flow!="linear_interpolation_spatial_residual")
            throw runtime_error("未知的单点集数值预测流程："+value_flow);
        if(time_step_vec.empty())
            throw runtime_error("单点集压缩或解压至少需要一个time step");
        if(time_step_vec.size()==1)
            value_flow="none";
        else if(time_step_vec.size()==2&&value_flow=="linear_interpolation")
            value_flow="constant_extrapolation";
        else if(time_step_vec.size()==2&&value_flow=="linear_interpolation_spatial_residual")
            value_flow="constant_extrapolation_spatial_residual";
    }

    template<typename T_coord,typename T_data>
    prediction_framework_c param_c<T_coord,T_data>::get_prediction_framework(uint32_t point_set_num) const
    {
        if(point_set_num==0)
            throw runtime_error("点集数量不能为0");
        prediction_framework_c framework;
        framework.has_label=point_set_num>1;
        bool use_joint_spatiotemporal=value_flow=="joint_spatiotemporal";
        if(use_joint_spatiotemporal&&label_flow!="none")
            throw runtime_error("联合时空数值流程不能再叠加标签时间预测流程");
        if(use_joint_spatiotemporal)
        {
            framework.value_mode=time_dimension_mode_e::joint_spatiotemporal;
            if(framework.has_label)
                framework.label_mode=time_dimension_mode_e::joint_spatiotemporal;
        }
        else
        {
            bool uses_anchor_flow=uses_anchor_interpolation_stride_8_flow();
            bool first_anchor=uses_anchor_flow&&current_time_step==0;
            bool regular_anchor=uses_anchor_flow&&current_time_step_is_regular_anchor();
            bool has_left_endpoint=false,has_right_endpoint=false;
            if(uses_anchor_flow)
                for(uint32_t reconstructed_time_step:reconstructed_time_step_vec)
                {
                    has_left_endpoint|=reconstructed_time_step<current_time_step;
                    has_right_endpoint|=reconstructed_time_step>current_time_step;
                }
            bool can_interpolate=has_left_endpoint&&has_right_endpoint;
            bool value_uses_anchor_flow=value_flow=="anchor_interpolation_stride_8"||value_flow=="anchor_interpolation_stride_8_spatial_residual";
            bool label_uses_anchor_flow=label_flow=="anchor_interpolation_stride_8";
            if(value_uses_anchor_flow&&first_anchor)
                framework.value_mode=time_dimension_mode_e::spatial;
            else if(value_uses_anchor_flow&&(regular_anchor||!can_interpolate))
                framework.value_mode=value_flow=="anchor_interpolation_stride_8_spatial_residual"?time_dimension_mode_e::constant_extrapolation_spatial_residual:time_dimension_mode_e::constant_extrapolation;
            else if(value_uses_anchor_flow)
                framework.value_mode=value_flow=="anchor_interpolation_stride_8_spatial_residual"?time_dimension_mode_e::linear_interpolation_spatial_residual:time_dimension_mode_e::linear_interpolation;
            else if(value_flow=="none")
                framework.value_mode=time_dimension_mode_e::spatial;
            else if(value_flow=="constant_extrapolation")
                framework.value_mode=time_dimension_mode_e::constant_extrapolation;
            else if(value_flow=="linear_interpolation")
                framework.value_mode=time_dimension_mode_e::linear_interpolation;
            else if(value_flow=="linear_interpolation_spatial_residual")
                framework.value_mode=time_dimension_mode_e::linear_interpolation_spatial_residual;
            else if(value_flow=="constant_extrapolation_spatial_residual")
                framework.value_mode=time_dimension_mode_e::constant_extrapolation_spatial_residual;
            else
                throw runtime_error("未知的数值时间预测方式："+value_flow);
            if(framework.has_label)
            {
                if(label_uses_anchor_flow&&first_anchor)
                    framework.label_mode=time_dimension_mode_e::spatial;
                else if(label_uses_anchor_flow&&(regular_anchor||!can_interpolate))
                    framework.label_mode=time_dimension_mode_e::constant_extrapolation;
                else if(label_uses_anchor_flow)
                    framework.label_mode=time_dimension_mode_e::endpoint_agreement;
                else if(label_flow=="none")
                    framework.label_mode=time_dimension_mode_e::spatial;
                else if(label_flow=="constant_extrapolation")
                    framework.label_mode=time_dimension_mode_e::constant_extrapolation;
                else if(label_flow=="endpoint_agreement")
                    framework.label_mode=time_dimension_mode_e::endpoint_agreement;
                else
                    throw runtime_error("未知的标签时间预测方式："+label_flow);
            }
        }
        if(!framework.has_label)
            framework.execution=label_value_execution_e::no_label;
        else if(label_value_execution=="fused")
            framework.execution=label_value_execution_e::fused;
        else
            framework.execution=label_value_execution_e::staged;
        if(use_linear_previous_label_experimental && (framework.execution!=label_value_execution_e::fused || framework.label_mode!=time_dimension_mode_e::constant_extrapolation))
            throw runtime_error("线性上一时刻标签修正流要求启用融合两点集预测和previous标签预测");
        if(use_linear_previous_label_experimental && use_prediction_order_label_experimental)
            throw runtime_error("线性上一时刻标签修正流不能使用预测顺序标签码流");
        return framework;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::advance_time_prediction_buffers()
    {
        const prediction_framework_c &framework=current_prediction_framework;
        bool first_time_anchor=uses_anchor_interpolation_stride_8_flow()&&current_time_step==0;
        if(framework.value_mode==time_dimension_mode_e::constant_extrapolation || framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||first_time_anchor)
            previous_time_decompressed_data.swap(current_decompressed_data);
        bool use_causal_time_label=framework.label_mode==time_dimension_mode_e::constant_extrapolation;
        if(framework.value_mode==time_dimension_mode_e::constant_extrapolation || framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual || use_causal_time_label||first_time_anchor)
            previous_time_point_set_id_vec.swap(point_set_id_vec);
    }
}
