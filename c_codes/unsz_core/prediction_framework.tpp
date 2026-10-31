#pragma once

namespace unsz
{
    constexpr uint32_t cross_slice_reference_num(time_dimension_mode_e mode)
    {
        if(mode==time_dimension_mode_e::constant_extrapolation||mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
            return 1;
        if(mode==time_dimension_mode_e::linear_interpolation||mode==time_dimension_mode_e::linear_interpolation_spatial_residual||mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual)
            return 2;
        return 0;
    }

    // 每个slice的预测方式在运行时才确定，而压缩和解压的内层循环要针对每种方式单独编译。
    // 这里把运行时的点集数、数值和标签预测方式转换成编译期常量，调用function(two_point_sets,value_mode,label_mode)，三个参数都是integral_constant。
    // 两点集时，标签和数值要么有一方只用空间预测，要么使用相同数量的参考slice。
    template<typename Function>
    void dispatch_prediction_framework(const prediction_framework_c&framework,Function function)
    {
        using mode_e=time_dimension_mode_e;
        auto dispatch_value_mode=[&](auto two_point_sets,auto label_mode)
        {
            auto call=[&](auto value_mode)
            {
                constexpr uint32_t label_ref_num=cross_slice_reference_num(decltype(label_mode)::value),value_ref_num=cross_slice_reference_num(decltype(value_mode)::value);
                if constexpr(label_ref_num==0||value_ref_num==0||label_ref_num==value_ref_num)
                    function(two_point_sets,value_mode,label_mode);
                else
                    throw runtime_error("两点集的标签和数值必须使用相同数量的参考slice，或其中一方只用空间预测");
            };
            switch(framework.value_mode)
            {
                case mode_e::spatial:return call(integral_constant<mode_e,mode_e::spatial>());
                case mode_e::constant_extrapolation:return call(integral_constant<mode_e,mode_e::constant_extrapolation>());
                case mode_e::linear_interpolation:return call(integral_constant<mode_e,mode_e::linear_interpolation>());
                case mode_e::constant_extrapolation_spatial_residual:return call(integral_constant<mode_e,mode_e::constant_extrapolation_spatial_residual>());
                case mode_e::linear_interpolation_spatial_residual:return call(integral_constant<mode_e,mode_e::linear_interpolation_spatial_residual>());
                default:throw runtime_error("未知的数值预测方式");
            }
        };
        if(!framework.has_label)
            return dispatch_value_mode(false_type(),integral_constant<mode_e,mode_e::not_applicable>());
        switch(framework.label_mode)
        {
            case mode_e::spatial:return dispatch_value_mode(true_type(),integral_constant<mode_e,mode_e::spatial>());
            case mode_e::constant_extrapolation:return dispatch_value_mode(true_type(),integral_constant<mode_e,mode_e::constant_extrapolation>());
            case mode_e::label_linear_interpolation_spatial_residual:return dispatch_value_mode(true_type(),integral_constant<mode_e,mode_e::label_linear_interpolation_spatial_residual>());
            default:throw runtime_error("未知的标签预测方式");
        }
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::check_label_value_time_step_order() const
    {
        if(point_set_config_vec.size()==2&&label_flow!="none"&&value_flow!="none"&&get_time_step_order(label_flow)!=get_time_step_order(value_flow))
            throw runtime_error("两点集要求label和value使用相同time step顺序");
    }

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
        if(time_step_vec.empty())
            throw runtime_error("单点集压缩或解压至少需要一个time step");
        if(value_flow=="auto")
            return;
        bool value_uses_anchor_flow=value_flow=="anchor_interpolation_stride_8"||value_flow=="anchor_interpolation_stride_8_spatial_residual";
        if(!value_uses_anchor_flow&&value_flow!="none"&&value_flow!="constant_extrapolation"&&value_flow!="linear_interpolation"&&value_flow!="constant_extrapolation_spatial_residual"&&value_flow!="linear_interpolation_spatial_residual")
            throw runtime_error("未知的单点集数值预测流程："+value_flow);
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
        if(current_field_uses_tensor)
        {
            if(!tensor_flow_selected)
                throw runtime_error("尚未选择tensor flow");
            uint8_t value_ref_num=temporal_value_dependency_vec[current_time_step_index].source_num;
            uint8_t label_ref_num=temporal_label_dependency_vec[current_time_step_index].source_num;
            if(value_ref_num==0)
                framework.value_mode=time_dimension_mode_e::spatial;
            else if(value_ref_num==1)
                framework.value_mode=selected_tensor_flow.use_spatial_residual?time_dimension_mode_e::constant_extrapolation_spatial_residual:time_dimension_mode_e::constant_extrapolation;
            else
                framework.value_mode=selected_tensor_flow.use_spatial_residual?time_dimension_mode_e::linear_interpolation_spatial_residual:time_dimension_mode_e::linear_interpolation;
            if(framework.has_label)
            {
                if(label_ref_num==0)
                    framework.label_mode=time_dimension_mode_e::spatial;
                else if(label_ref_num==1)
                    framework.label_mode=time_dimension_mode_e::constant_extrapolation;
                else
                    framework.label_mode=time_dimension_mode_e::label_linear_interpolation_spatial_residual;
            }
            return framework;
        }
        bool uses_anchor_flow=uses_anchor_interpolation_stride_8_flow();
        bool first_anchor=uses_anchor_flow&&current_time_step==0;
        bool regular_anchor=uses_anchor_flow&&current_time_step_is_regular_anchor();
        bool value_uses_anchor_flow=value_flow=="anchor_interpolation_stride_8"||value_flow=="anchor_interpolation_stride_8_spatial_residual";
        bool label_uses_anchor_flow=label_flow=="anchor_interpolation_stride_8";
        uint8_t value_time_ref_num=temporal_value_dependency_vec[current_time_step_index].source_num;
        uint8_t label_time_ref_num=temporal_label_dependency_vec[current_time_step_index].source_num;
        if(value_uses_anchor_flow&&first_anchor)
            framework.value_mode=time_dimension_mode_e::spatial;
        else if(value_uses_anchor_flow&&(regular_anchor||value_time_ref_num<2))
            if(value_flow=="anchor_interpolation_stride_8_spatial_residual")
                framework.value_mode=time_dimension_mode_e::constant_extrapolation_spatial_residual;
            else
                framework.value_mode=time_dimension_mode_e::constant_extrapolation;
        else if(value_uses_anchor_flow)
            if(value_flow=="anchor_interpolation_stride_8_spatial_residual")
                framework.value_mode=time_dimension_mode_e::linear_interpolation_spatial_residual;
            else
                framework.value_mode=time_dimension_mode_e::linear_interpolation;
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
            else if(label_uses_anchor_flow&&(regular_anchor||label_time_ref_num<2))
                framework.label_mode=time_dimension_mode_e::constant_extrapolation;
            else if(label_uses_anchor_flow)
                framework.label_mode=time_dimension_mode_e::label_linear_interpolation_spatial_residual;
            else if(label_flow=="none")
                framework.label_mode=time_dimension_mode_e::spatial;
            else if(label_flow=="constant_extrapolation")
                framework.label_mode=time_dimension_mode_e::constant_extrapolation;
            else if(label_flow=="linear_interpolation_spatial_residual")
                framework.label_mode=time_dimension_mode_e::label_linear_interpolation_spatial_residual;
            else
                throw runtime_error("未知的标签时间预测方式："+label_flow);
        }
        return framework;
    }

}
