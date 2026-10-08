#pragma once

namespace unsz::one_point_set::spatial
{
    template<typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        param.template decompress_current_file<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable>();
    }
}

namespace unsz::one_point_set::temporal
{
    template<uint32_t time_ref_num,bool use_spatial_residual,typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        constexpr time_dimension_mode_e value_mode=time_flow::value_mode<time_ref_num,use_spatial_residual>();
        param.template decompress_current_file<false,value_mode,time_dimension_mode_e::not_applicable>();
    }
}

namespace unsz::two_point_sets::fused::label_spatial_value_spatial
{
    template<typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        param.template decompress_current_file<true,time_dimension_mode_e::spatial,time_dimension_mode_e::spatial>();
    }
}

namespace unsz::two_point_sets::fused::label_spatial_value_temporal
{
    template<uint32_t time_ref_num,bool use_spatial_residual,typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        constexpr time_dimension_mode_e value_mode=time_flow::value_mode<time_ref_num,use_spatial_residual>();
        param.template decompress_current_file<true,value_mode,time_dimension_mode_e::spatial>();
    }
}

namespace unsz::two_point_sets::fused::label_temporal_value_spatial
{
    template<uint32_t time_ref_num,typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        constexpr time_dimension_mode_e label_mode=time_flow::label_mode<time_ref_num>();
        param.template decompress_current_file<true,time_dimension_mode_e::spatial,label_mode>();
    }
}

namespace unsz::two_point_sets::fused::label_temporal_value_temporal
{
    template<uint32_t time_ref_num,bool use_spatial_residual,typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        constexpr time_dimension_mode_e label_mode=time_flow::label_mode<time_ref_num>();
        constexpr time_dimension_mode_e value_mode=time_flow::value_mode<time_ref_num,use_spatial_residual>();
        param.template decompress_current_file<true,value_mode,label_mode>();
    }
}

namespace unsz::decompression_flow
{
    template<typename T_coord,typename T_data>
    void dispatch_one_point_set(param_c<T_coord,T_data>&param,time_dimension_mode_e value_mode)
    {
        switch(value_mode)
        {
            case time_dimension_mode_e::spatial:
                return one_point_set::spatial::decompress(param);
            case time_dimension_mode_e::constant_extrapolation:
                return one_point_set::temporal::decompress<1,false>(param);
            case time_dimension_mode_e::constant_extrapolation_spatial_residual:
                return one_point_set::temporal::decompress<1,true>(param);
            case time_dimension_mode_e::linear_interpolation:
                return one_point_set::temporal::decompress<2,false>(param);
            case time_dimension_mode_e::linear_interpolation_spatial_residual:
                return one_point_set::temporal::decompress<2,true>(param);
            default:
                throw runtime_error("未知的单点集数值预测流程");
        }
    }

    template<typename T_coord,typename T_data>
    void dispatch_two_point_sets_fused(param_c<T_coord,T_data>&param,const prediction_framework_c&framework)
    {
        if(framework.label_mode==time_dimension_mode_e::spatial)
        {
            switch(framework.value_mode)
            {
                case time_dimension_mode_e::spatial:
                    return two_point_sets::fused::label_spatial_value_spatial::decompress(param);
                case time_dimension_mode_e::constant_extrapolation:
                    return two_point_sets::fused::label_spatial_value_temporal::decompress<1,false>(param);
                case time_dimension_mode_e::constant_extrapolation_spatial_residual:
                    return two_point_sets::fused::label_spatial_value_temporal::decompress<1,true>(param);
                case time_dimension_mode_e::linear_interpolation:
                    return two_point_sets::fused::label_spatial_value_temporal::decompress<2,false>(param);
                case time_dimension_mode_e::linear_interpolation_spatial_residual:
                    return two_point_sets::fused::label_spatial_value_temporal::decompress<2,true>(param);
                default:
                    throw runtime_error("标签空间预测不支持当前数值流程");
            }
        }
        if(framework.label_mode==time_dimension_mode_e::constant_extrapolation)
        {
            if(framework.value_mode==time_dimension_mode_e::spatial)
                return two_point_sets::fused::label_temporal_value_spatial::decompress<1>(param);
            if(framework.value_mode==time_dimension_mode_e::constant_extrapolation)
                return two_point_sets::fused::label_temporal_value_temporal::decompress<1,false>(param);
            if(framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
                return two_point_sets::fused::label_temporal_value_temporal::decompress<1,true>(param);
            throw runtime_error("标签时间外插要求数值使用空间预测或时间外插");
        }
        if(framework.label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual)
        {
            if(framework.value_mode==time_dimension_mode_e::spatial)
                return two_point_sets::fused::label_temporal_value_spatial::decompress<2>(param);
            if(framework.value_mode==time_dimension_mode_e::linear_interpolation)
                return two_point_sets::fused::label_temporal_value_temporal::decompress<2,false>(param);
            if(framework.value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                return two_point_sets::fused::label_temporal_value_temporal::decompress<2,true>(param);
            throw runtime_error("标签时间内插要求数值使用空间预测或时间内插");
        }
        throw runtime_error("未知的两点集标签流程");
    }
}

namespace unsz::one_point_set
{
    template<typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        if(param.point_set_config_vec.size()!=1)
            throw runtime_error("单点集解压要求field恰好配置一个点集");
        param.resolve_single_point_set_time_mode();
        prediction_framework_c framework=param.get_prediction_framework(1);
        if(param.point_set_config_vec[0].codec!=point_set_codec_e::rbf&&param.point_set_config_vec[0].codec!=point_set_codec_e::constant)
            throw runtime_error("单点集解压只支持constant或RBF空间预测器");
        return decompression_flow::dispatch_one_point_set(param,framework.value_mode);
    }
}

namespace unsz::two_point_sets::fused
{
    template<typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        if(param.point_set_config_vec.size()!=2)
            throw runtime_error("两点集解压要求field恰好配置两个点集");
        prediction_framework_c framework=param.get_prediction_framework(2);
        if(param.label_flow!="none"&&param.value_flow!="none"&&param.get_time_step_order(param.label_flow)!=param.get_time_step_order(param.value_flow))
            throw runtime_error("两点集解压要求label和value使用相同time step顺序");
        for(const point_set_config_c&config:param.point_set_config_vec)
            if(config.codec!=point_set_codec_e::constant&&config.codec!=point_set_codec_e::rbf)
                throw runtime_error("两点集解压只支持constant或RBF空间预测器");
        return decompression_flow::dispatch_two_point_sets_fused(param,framework);
    }
}
