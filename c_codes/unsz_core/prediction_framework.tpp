#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    uint32_t param_c<T_coord,T_data>::get_last_regular_time_anchor() const
    {
        if(time_anchor_stride_experimental==0||time_step_vec.empty())
            throw runtime_error("固定步长锚点流程尚未配置步长或time step");
        return time_anchor_last_regular_step;
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::current_time_step_is_regular_anchor() const
    {
        if(time_anchor_stride_experimental==0||time_step_vec.empty())
            return false;
        return current_time_step>=time_anchor_first_step&&current_time_step<=time_anchor_last_regular_step&&(current_time_step-time_anchor_first_step)%time_anchor_stride_experimental==0;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::resolve_single_point_set_time_mode()
    {
        if(point_set_config_vec.size()!=1)
            throw runtime_error("单点集时间流程选择要求field恰好配置一个点集");
        if(cross_time_spatial_point_num_experimental>0)
            return;
        if(value_flow!="none"&&value_flow!="constant_extrapolation"&&value_flow!="linear_interpolation"&&value_flow!="constant_extrapolation_spatial_residual"&&value_flow!="linear_interpolation_spatial_residual")
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
        if(time_anchor_stride_experimental>0)
        {
            if(point_set_num!=2)
                throw runtime_error("固定步长锚点流程目前只支持两个点集");
            if(value_flow!="none"||label_flow!="none"||cross_time_spatial_point_num_experimental>0)
                throw runtime_error("固定步长锚点流程不能再叠加其他时间预测流程");
            bool first_anchor=current_time_step==time_anchor_first_step;
            bool regular_anchor=current_time_step_is_regular_anchor();
            bool trailing_endpoint=current_time_step==time_anchor_last_step&&time_anchor_last_step!=get_last_regular_time_anchor();
            if(first_anchor)
            {
                framework.label_mode=time_dimension_mode_e::spatial;
                framework.value_mode=time_dimension_mode_e::spatial;
            }
            else if(regular_anchor||trailing_endpoint)
            {
                framework.label_mode=time_dimension_mode_e::constant_extrapolation;
                framework.value_mode=time_dimension_mode_e::constant_extrapolation_spatial_residual;
            }
            else
            {
                framework.label_mode=time_dimension_mode_e::endpoint_agreement;
                framework.value_mode=time_dimension_mode_e::linear_interpolation_spatial_residual;
            }
            framework.execution=label_value_execution=="fused"?label_value_execution_e::fused:label_value_execution_e::staged;
            return framework;
        }
        bool use_joint_spatiotemporal=cross_time_spatial_point_num_experimental>0;
        if(use_joint_spatiotemporal && (value_flow!="none" || label_flow!="none"))
            throw runtime_error("统一时空DAG不能再叠加零阶外插或一阶内插");
        if(use_joint_spatiotemporal)
        {
            framework.value_mode=time_dimension_mode_e::joint_spatiotemporal;
            if(framework.has_label)
                framework.label_mode=time_dimension_mode_e::joint_spatiotemporal;
        }
        else
        {
            if(value_flow=="none")
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
                if(label_flow=="none")
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
        if(framework.execution==label_value_execution_e::fused)
        {
            bool constant_flow=framework.label_mode==time_dimension_mode_e::constant_extrapolation&&framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual;
            bool interpolation_flow=framework.label_mode==time_dimension_mode_e::endpoint_agreement&&framework.value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual;
            bool legacy_spatial_flow=framework.value_mode==time_dimension_mode_e::spatial&&(framework.label_mode==time_dimension_mode_e::spatial||framework.label_mode==time_dimension_mode_e::constant_extrapolation);
            if(!constant_flow&&!interpolation_flow&&!legacy_spatial_flow)
                throw runtime_error("标签与数值融合执行只实现了选定的零阶、内插和既有空间流程");
        }
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
        bool first_time_anchor=time_anchor_stride_experimental>0&&current_time_step==time_anchor_first_step;
        if(framework.value_mode==time_dimension_mode_e::constant_extrapolation || framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||first_time_anchor)
            previous_time_decompressed_data.swap(current_decompressed_data);
        bool use_causal_time_label=framework.label_mode==time_dimension_mode_e::constant_extrapolation;
        if(framework.value_mode==time_dimension_mode_e::constant_extrapolation || framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual || use_causal_time_label||first_time_anchor)
            previous_time_point_set_id_vec.swap(point_set_id_vec);
    }
}
