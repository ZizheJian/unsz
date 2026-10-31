#pragma once

namespace unsz
{
    constexpr uint32_t cross_slice_reference_num(slice_prediction_mode_e mode)
    {
        if(mode==slice_prediction_mode_e::extrapolation||mode==slice_prediction_mode_e::extrapolation_spatial_residual)
            return 1;
        if(mode==slice_prediction_mode_e::interpolation||mode==slice_prediction_mode_e::interpolation_spatial_residual||mode==slice_prediction_mode_e::label_interpolation_spatial_residual)
            return 2;
        return 0;
    }

    // 每个slice的预测方式在运行时才确定，而压缩和解压的内层循环要针对每种方式单独编译。
    // 这里把运行时的标签数、数值和标签预测方式转换成编译期常量，调用function(two_labels,value_mode,label_mode)，三个参数都是integral_constant。
    // 两种标签时，标签和数值要么有一方只用空间预测，要么使用相同数量的参考slice。
    template<typename Function>
    void dispatch_prediction_framework(const prediction_framework_c&framework,Function function)
    {
        using mode_e=slice_prediction_mode_e;
        auto dispatch_value_mode=[&](auto two_labels,auto label_mode)
        {
            auto call=[&](auto value_mode)
            {
                constexpr uint32_t label_ref_num=cross_slice_reference_num(decltype(label_mode)::value),value_ref_num=cross_slice_reference_num(decltype(value_mode)::value);
                if constexpr(label_ref_num==0||value_ref_num==0||label_ref_num==value_ref_num)
                    function(two_labels,value_mode,label_mode);
                else
                    throw runtime_error("两种标签的标签和数值必须使用相同数量的参考slice，或其中一方只用空间预测");
            };
            switch(framework.value_mode)
            {
                case mode_e::independent:return call(integral_constant<mode_e,mode_e::independent>());
                case mode_e::extrapolation:return call(integral_constant<mode_e,mode_e::extrapolation>());
                case mode_e::interpolation:return call(integral_constant<mode_e,mode_e::interpolation>());
                case mode_e::extrapolation_spatial_residual:return call(integral_constant<mode_e,mode_e::extrapolation_spatial_residual>());
                case mode_e::interpolation_spatial_residual:return call(integral_constant<mode_e,mode_e::interpolation_spatial_residual>());
                default:throw runtime_error("未知的数值预测方式");
            }
        };
        if(!framework.has_label)
            return dispatch_value_mode(false_type(),integral_constant<mode_e,mode_e::not_applicable>());
        switch(framework.label_mode)
        {
            case mode_e::independent:return dispatch_value_mode(true_type(),integral_constant<mode_e,mode_e::independent>());
            case mode_e::extrapolation:return dispatch_value_mode(true_type(),integral_constant<mode_e,mode_e::extrapolation>());
            case mode_e::label_interpolation_spatial_residual:return dispatch_value_mode(true_type(),integral_constant<mode_e,mode_e::label_interpolation_spatial_residual>());
            default:throw runtime_error("未知的标签预测方式");
        }
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::check_label_value_slice_order() const
    {
        if(label_config_vec.size()==2&&label_pipeline!="independent"&&value_pipeline!="independent"&&get_slice_order(label_pipeline)!=get_slice_order(value_pipeline))
            throw runtime_error("两种标签要求label和value使用相同slice顺序");
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::uses_anchor_interpolation_pipeline() const
    {
        return label_pipeline=="anchor_interpolation"||value_pipeline=="anchor_interpolation"||value_pipeline=="anchor_interpolation_spatial_residual";
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::current_slice_is_regular_anchor() const
    {
        return uses_anchor_interpolation_pipeline()&&current_slice%8==0;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::resolve_single_label_pipeline()
    {
        if(label_config_vec.size()!=1)
            throw runtime_error("单标签pipeline选择要求field恰好配置一个标签");
        if(slice_vec.empty())
            throw runtime_error("单标签压缩或解压至少需要一个slice");
        if(value_pipeline=="auto")
            return;
        bool value_uses_anchor_pipeline=value_pipeline=="anchor_interpolation"||value_pipeline=="anchor_interpolation_spatial_residual";
        if(!value_uses_anchor_pipeline&&value_pipeline!="independent"&&value_pipeline!="extrapolation"&&value_pipeline!="interpolation"&&value_pipeline!="extrapolation_spatial_residual"&&value_pipeline!="interpolation_spatial_residual")
            throw runtime_error("未知的单标签数值预测流程："+value_pipeline);
        if(slice_vec.size()==1)
            value_pipeline="independent";
        else if(slice_vec.size()==2&&value_pipeline=="interpolation")
            value_pipeline="extrapolation";
        else if(slice_vec.size()==2&&value_pipeline=="interpolation_spatial_residual")
            value_pipeline="extrapolation_spatial_residual";
    }

    template<typename T_coord,typename T_data>
    prediction_framework_c param_c<T_coord,T_data>::get_prediction_framework(uint32_t label_num) const
    {
        if(label_num==0)
            throw runtime_error("标签数量不能为0");
        prediction_framework_c framework;
        framework.has_label=label_num>1;
        if(current_field_uses_tensor)
        {
            if(!tensor_pipeline_selected)
                throw runtime_error("尚未选择tensor pipeline");
            uint8_t value_ref_num=value_reference_slices_vec[current_slice_index].source_num;
            uint8_t label_ref_num=label_reference_slices_vec[current_slice_index].source_num;
            if(value_ref_num==0)
                framework.value_mode=slice_prediction_mode_e::independent;
            else if(value_ref_num==1)
                framework.value_mode=selected_tensor_pipeline.use_spatial_residual?slice_prediction_mode_e::extrapolation_spatial_residual:slice_prediction_mode_e::extrapolation;
            else
                framework.value_mode=selected_tensor_pipeline.use_spatial_residual?slice_prediction_mode_e::interpolation_spatial_residual:slice_prediction_mode_e::interpolation;
            if(framework.has_label)
            {
                if(label_ref_num==0)
                    framework.label_mode=slice_prediction_mode_e::independent;
                else if(label_ref_num==1)
                    framework.label_mode=slice_prediction_mode_e::extrapolation;
                else
                    framework.label_mode=slice_prediction_mode_e::label_interpolation_spatial_residual;
            }
            return framework;
        }
        bool uses_anchor_pipeline=uses_anchor_interpolation_pipeline();
        bool first_anchor=uses_anchor_pipeline&&current_slice==0;
        bool regular_anchor=uses_anchor_pipeline&&current_slice_is_regular_anchor();
        bool value_uses_anchor_pipeline=value_pipeline=="anchor_interpolation"||value_pipeline=="anchor_interpolation_spatial_residual";
        bool label_uses_anchor_pipeline=label_pipeline=="anchor_interpolation";
        uint8_t value_reference_slice_num=value_reference_slices_vec[current_slice_index].source_num;
        uint8_t label_reference_slice_num=label_reference_slices_vec[current_slice_index].source_num;
        if(value_uses_anchor_pipeline&&first_anchor)
            framework.value_mode=slice_prediction_mode_e::independent;
        else if(value_uses_anchor_pipeline&&(regular_anchor||value_reference_slice_num<2))
            if(value_pipeline=="anchor_interpolation_spatial_residual")
                framework.value_mode=slice_prediction_mode_e::extrapolation_spatial_residual;
            else
                framework.value_mode=slice_prediction_mode_e::extrapolation;
        else if(value_uses_anchor_pipeline)
            if(value_pipeline=="anchor_interpolation_spatial_residual")
                framework.value_mode=slice_prediction_mode_e::interpolation_spatial_residual;
            else
                framework.value_mode=slice_prediction_mode_e::interpolation;
        else if(value_pipeline=="independent")
            framework.value_mode=slice_prediction_mode_e::independent;
        else if(value_pipeline=="extrapolation")
            framework.value_mode=slice_prediction_mode_e::extrapolation;
        else if(value_pipeline=="interpolation")
            framework.value_mode=slice_prediction_mode_e::interpolation;
        else if(value_pipeline=="interpolation_spatial_residual")
            framework.value_mode=slice_prediction_mode_e::interpolation_spatial_residual;
        else if(value_pipeline=="extrapolation_spatial_residual")
            framework.value_mode=slice_prediction_mode_e::extrapolation_spatial_residual;
        else
            throw runtime_error("未知的数值跨slice预测方式："+value_pipeline);
        if(framework.has_label)
        {
            if(label_uses_anchor_pipeline&&first_anchor)
                framework.label_mode=slice_prediction_mode_e::independent;
            else if(label_uses_anchor_pipeline&&(regular_anchor||label_reference_slice_num<2))
                framework.label_mode=slice_prediction_mode_e::extrapolation;
            else if(label_uses_anchor_pipeline)
                framework.label_mode=slice_prediction_mode_e::label_interpolation_spatial_residual;
            else if(label_pipeline=="independent")
                framework.label_mode=slice_prediction_mode_e::independent;
            else if(label_pipeline=="extrapolation")
                framework.label_mode=slice_prediction_mode_e::extrapolation;
            else if(label_pipeline=="interpolation_spatial_residual")
                framework.label_mode=slice_prediction_mode_e::label_interpolation_spatial_residual;
            else
                throw runtime_error("未知的标签跨slice预测方式："+label_pipeline);
        }
        return framework;
    }

}
