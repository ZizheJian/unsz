#pragma once

namespace unsz::one_point_set::spatial
{
    template<point_set_codec_e predictor,typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        param.template decompress_current_file<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable,predictor>();
    }
}

namespace unsz::one_point_set::temporal
{
    template<uint32_t time_ref_num,bool use_spatial_residual,point_set_codec_e predictor,typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        constexpr time_dimension_mode_e value_mode=time_flow::value_mode<time_ref_num,use_spatial_residual>();
        param.template decompress_current_file<false,value_mode,time_dimension_mode_e::not_applicable,predictor>();
    }
}

namespace unsz::two_point_sets::fused::label_spatial_value_spatial
{
    template<point_set_codec_e predictor,typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        param.template decompress_current_file<true,time_dimension_mode_e::spatial,time_dimension_mode_e::spatial,predictor>();
    }
}

namespace unsz::two_point_sets::fused::label_spatial_value_temporal
{
    template<uint32_t time_ref_num,bool use_spatial_residual,point_set_codec_e predictor,typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        constexpr time_dimension_mode_e value_mode=time_flow::value_mode<time_ref_num,use_spatial_residual>();
        param.template decompress_current_file<true,value_mode,time_dimension_mode_e::spatial,predictor>();
    }
}

namespace unsz::two_point_sets::fused::label_temporal_value_spatial
{
    template<uint32_t time_ref_num,point_set_codec_e predictor,typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        constexpr time_dimension_mode_e label_mode=time_flow::label_mode<time_ref_num>();
        param.template decompress_current_file<true,time_dimension_mode_e::spatial,label_mode,predictor>();
    }
}

namespace unsz::two_point_sets::fused::label_temporal_value_temporal
{
    template<uint32_t time_ref_num,bool use_spatial_residual,point_set_codec_e predictor,typename T_coord,typename T_data>
    void decompress(param_c<T_coord,T_data>&param)
    {
        constexpr time_dimension_mode_e label_mode=time_flow::label_mode<time_ref_num>();
        constexpr time_dimension_mode_e value_mode=time_flow::value_mode<time_ref_num,use_spatial_residual>();
        param.template decompress_current_file<true,value_mode,label_mode,predictor>();
    }
}

namespace unsz::decompression_flow
{
    template<point_set_codec_e predictor,typename T_coord,typename T_data>
    void dispatch_one_point_set(param_c<T_coord,T_data>&param,time_dimension_mode_e value_mode)
    {
        switch(value_mode)
        {
            case time_dimension_mode_e::spatial:
                return one_point_set::spatial::decompress<predictor>(param);
            case time_dimension_mode_e::constant_extrapolation:
                return one_point_set::temporal::decompress<1,false,predictor>(param);
            case time_dimension_mode_e::constant_extrapolation_spatial_residual:
                return one_point_set::temporal::decompress<1,true,predictor>(param);
            case time_dimension_mode_e::linear_interpolation:
                return one_point_set::temporal::decompress<2,false,predictor>(param);
            case time_dimension_mode_e::linear_interpolation_spatial_residual:
                return one_point_set::temporal::decompress<2,true,predictor>(param);
            default:
                throw runtime_error("未知的单点集数值预测流程");
        }
    }

    template<point_set_codec_e predictor,typename T_coord,typename T_data>
    void dispatch_two_point_sets_fused(param_c<T_coord,T_data>&param,const prediction_framework_c&framework)
    {
        if(framework.label_mode==time_dimension_mode_e::spatial)
        {
            switch(framework.value_mode)
            {
                case time_dimension_mode_e::spatial:
                    return two_point_sets::fused::label_spatial_value_spatial::decompress<predictor>(param);
                case time_dimension_mode_e::constant_extrapolation:
                    return two_point_sets::fused::label_spatial_value_temporal::decompress<1,false,predictor>(param);
                case time_dimension_mode_e::constant_extrapolation_spatial_residual:
                    return two_point_sets::fused::label_spatial_value_temporal::decompress<1,true,predictor>(param);
                case time_dimension_mode_e::linear_interpolation:
                    return two_point_sets::fused::label_spatial_value_temporal::decompress<2,false,predictor>(param);
                case time_dimension_mode_e::linear_interpolation_spatial_residual:
                    return two_point_sets::fused::label_spatial_value_temporal::decompress<2,true,predictor>(param);
                default:
                    throw runtime_error("标签空间预测不支持当前数值流程");
            }
        }
        if(framework.label_mode==time_dimension_mode_e::constant_extrapolation)
        {
            if(framework.value_mode==time_dimension_mode_e::spatial)
                return two_point_sets::fused::label_temporal_value_spatial::decompress<1,predictor>(param);
            if(framework.value_mode==time_dimension_mode_e::constant_extrapolation)
                return two_point_sets::fused::label_temporal_value_temporal::decompress<1,false,predictor>(param);
            if(framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
                return two_point_sets::fused::label_temporal_value_temporal::decompress<1,true,predictor>(param);
            throw runtime_error("标签时间外插要求数值使用空间预测或时间外插");
        }
        if(framework.label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual)
        {
            if(framework.value_mode==time_dimension_mode_e::spatial)
                return two_point_sets::fused::label_temporal_value_spatial::decompress<2,predictor>(param);
            if(framework.value_mode==time_dimension_mode_e::linear_interpolation)
                return two_point_sets::fused::label_temporal_value_temporal::decompress<2,false,predictor>(param);
            if(framework.value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
                return two_point_sets::fused::label_temporal_value_temporal::decompress<2,true,predictor>(param);
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
        point_set_codec_e predictor=param.point_set_config_vec[0].codec;
        if(predictor==point_set_codec_e::idw)
            return decompression_flow::dispatch_one_point_set<point_set_codec_e::idw>(param,framework.value_mode);
        if(predictor==point_set_codec_e::rbf)
            return decompression_flow::dispatch_one_point_set<point_set_codec_e::rbf>(param,framework.value_mode);
        throw runtime_error("单点集解压只支持IDW或RBF空间预测器");
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
        point_set_codec_e predictor=point_set_codec_e::constant;
        for(const point_set_config_c&config:param.point_set_config_vec)
            if(config.codec!=point_set_codec_e::constant)
            {
                if(predictor!=point_set_codec_e::constant&&predictor!=config.codec)
                    throw runtime_error("同一次两点集解压中的非常量点集必须使用同一种空间预测器");
                predictor=config.codec;
            }
        if(predictor==point_set_codec_e::idw)
            return decompression_flow::dispatch_two_point_sets_fused<point_set_codec_e::idw>(param,framework);
        if(predictor==point_set_codec_e::rbf)
            return decompression_flow::dispatch_two_point_sets_fused<point_set_codec_e::rbf>(param,framework);
        throw runtime_error("两点集解压至少需要一个使用IDW或RBF的非常量点集");
    }
}
