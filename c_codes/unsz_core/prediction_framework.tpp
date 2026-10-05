#pragma once

namespace unsz::time_flow
{
    template<uint32_t time_ref_num>
    constexpr time_dimension_mode_e label_mode()
    {
        if constexpr(time_ref_num==1)
            return time_dimension_mode_e::constant_extrapolation;
        return time_dimension_mode_e::label_linear_interpolation_spatial_residual;
    }

    template<uint32_t time_ref_num,bool use_spatial_residual>
    constexpr time_dimension_mode_e value_mode()
    {
        if constexpr(time_ref_num==1)
        {
            if constexpr(use_spatial_residual)
                return time_dimension_mode_e::constant_extrapolation_spatial_residual;
            return time_dimension_mode_e::constant_extrapolation;
        }
        if constexpr(use_spatial_residual)
            return time_dimension_mode_e::linear_interpolation_spatial_residual;
        return time_dimension_mode_e::linear_interpolation;
    }
}

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
