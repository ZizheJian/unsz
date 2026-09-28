#pragma once

namespace unsz::one_point_set::spatial::idw
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);}
}

namespace unsz::one_point_set::spatial::rbf
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);}
}

namespace unsz::one_point_set::constant_extrapolation::idw
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::constant_extrapolation,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);}
}

namespace unsz::one_point_set::constant_extrapolation::rbf
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::constant_extrapolation,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);}
}

namespace unsz::one_point_set::linear_interpolation::idw
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::linear_interpolation,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);}
}

namespace unsz::one_point_set::linear_interpolation::rbf
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::linear_interpolation,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);}
}

namespace unsz::one_point_set::constant_extrapolation_then_spatial::idw
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::constant_extrapolation_spatial_residual,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);}
}

namespace unsz::one_point_set::constant_extrapolation_then_spatial::rbf
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::constant_extrapolation_spatial_residual,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);}
}

namespace unsz::one_point_set::linear_interpolation_then_spatial::idw
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::linear_interpolation_spatial_residual,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);}
}

namespace unsz::one_point_set::linear_interpolation_then_spatial::rbf
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::linear_interpolation_spatial_residual,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);}
}

namespace unsz::one_point_set::joint_spatiotemporal::idw
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param){param.compress_joint_spatiotemporal_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);}
}

namespace unsz::one_point_set::joint_spatiotemporal::rbf
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param){param.compress_joint_spatiotemporal_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);}
}

namespace unsz::one_point_set::spatial
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param)
{
    point_set_codec_e codec=param.point_set_config_vec[0].codec;
    if(codec==point_set_codec_e::idw)
        idw::compress(param);
    else if(codec==point_set_codec_e::rbf)
        rbf::compress(param);
    else
        throw runtime_error("单点集纯空间压缩只支持IDW或RBF预测器");
}
}

namespace unsz::one_point_set::constant_extrapolation
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param)
{
    point_set_codec_e codec=param.point_set_config_vec[0].codec;
    if(codec==point_set_codec_e::idw)
        idw::compress(param);
    else if(codec==point_set_codec_e::rbf)
        rbf::compress(param);
    else
        throw runtime_error("单点集零阶外插压缩只支持IDW或RBF空间预测器");
}
}

namespace unsz::one_point_set::linear_interpolation
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param)
{
    point_set_codec_e codec=param.point_set_config_vec[0].codec;
    if(codec==point_set_codec_e::idw)
        idw::compress(param);
    else if(codec==point_set_codec_e::rbf)
        rbf::compress(param);
    else
        throw runtime_error("单点集一阶内插压缩只支持IDW或RBF空间预测器");
}
}

namespace unsz::one_point_set::constant_extrapolation_then_spatial
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param)
{
    point_set_codec_e codec=param.point_set_config_vec[0].codec;
    if(codec==point_set_codec_e::idw)
        idw::compress(param);
    else if(codec==point_set_codec_e::rbf)
        rbf::compress(param);
    else
        throw runtime_error("单点集零阶外插加空间补偿只支持IDW或RBF空间预测器");
}
}

namespace unsz::one_point_set::linear_interpolation_then_spatial
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param)
{
    point_set_codec_e codec=param.point_set_config_vec[0].codec;
    if(codec==point_set_codec_e::idw)
        idw::compress(param);
    else if(codec==point_set_codec_e::rbf)
        rbf::compress(param);
    else
        throw runtime_error("单点集一阶内插加空间补偿只支持IDW或RBF空间预测器");
}
}

namespace unsz::one_point_set::joint_spatiotemporal
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param)
{
    point_set_codec_e codec=param.point_set_config_vec[0].codec;
    if(codec==point_set_codec_e::idw)
        idw::compress(param);
    else if(codec==point_set_codec_e::rbf)
        rbf::compress(param);
    else
        throw runtime_error("单点集联合时空压缩只支持IDW或RBF空间预测器");
}
}

namespace unsz::one_point_set
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
{
    if(param.point_set_config_vec.size()!=1)
        throw runtime_error("one_point_set::compress要求恰好配置一个点集");
    param.resolve_single_point_set_time_mode();
    if(param.value_flow=="joint_spatiotemporal")
    {
        param.prepare_joint_spatiotemporal_coordinates();
        param.prepare_compression_structure(preprocessing_start_time);
        joint_spatiotemporal::compress(param);
        return;
    }
    if(param.uses_anchor_interpolation_stride_8_flow())
    {
        param.prepare_compression_structure(preprocessing_start_time);
        param.compress_files(&param_c<T_coord,T_data>::compress_single_point_set);
        return;
    }
    param.prepare_compression_structure(preprocessing_start_time);
    if(param.value_flow=="none")
        spatial::compress(param);
    else if(param.value_flow=="constant_extrapolation")
        constant_extrapolation::compress(param);
    else if(param.value_flow=="linear_interpolation")
        linear_interpolation::compress(param);
    else if(param.value_flow=="constant_extrapolation_spatial_residual")
        constant_extrapolation_then_spatial::compress(param);
    else if(param.value_flow=="linear_interpolation_spatial_residual")
        linear_interpolation_then_spatial::compress(param);
    else
        throw runtime_error("未知的单点集数值预测流程："+param.value_flow);
}
}
