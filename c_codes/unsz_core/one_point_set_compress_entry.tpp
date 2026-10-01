#pragma once

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
        point_set_codec_e predictor=param.point_set_config_vec[0].codec;
        if(predictor==point_set_codec_e::idw)
            return param.compress_joint_spatiotemporal_files(&param_c<T_coord,T_data>::template compress_current_file<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);
        if(predictor==point_set_codec_e::rbf)
            return param.compress_joint_spatiotemporal_files(&param_c<T_coord,T_data>::template compress_current_file<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);
        throw runtime_error("单点集联合时空压缩只支持IDW或RBF空间预测器");
    }
    param.prepare_compression_structure(preprocessing_start_time);
    param.compress_files(&param_c<T_coord,T_data>::dispatch_single_point_set_compression);
}
}
