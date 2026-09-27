#pragma once

#include "two_point_sets/label_spatial.tpp"
#include "two_point_sets/label_constant_extrapolation.tpp"
#include "two_point_sets/label_endpoint_agreement.tpp"
#include "two_point_sets/label_joint_spatiotemporal.tpp"

namespace unsz
{
namespace two_point_sets
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
{
    if(param.point_set_config_vec.size()!=2)
        throw runtime_error("two_point_sets::compress要求恰好配置两个点集");
    if(param.time_anchor_stride_experimental>0)
    {
        if(param.value_flow!="none"||param.label_flow!="none"||param.cross_time_spatial_point_num_experimental>0)
            throw runtime_error("固定步长锚点流程不能再叠加其他时间预测流程");
        param.prepare_compression_structure(preprocessing_start_time);
        param.compress_files(&param_c<T_coord,T_data>::compress_two_point_sets);
        return;
    }
    prediction_framework_c framework=param.get_prediction_framework(2);
    switch(framework.label_mode)
    {
        case time_dimension_mode_e::spatial:
            switch(framework.value_mode)
            {
                case time_dimension_mode_e::spatial:label::spatial::value::spatial::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::constant_extrapolation:label::spatial::value::constant_extrapolation::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::linear_interpolation:label::spatial::value::linear_interpolation::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::constant_extrapolation_spatial_residual:label::spatial::value::constant_extrapolation_then_spatial::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::linear_interpolation_spatial_residual:label::spatial::value::linear_interpolation_then_spatial::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::joint_spatiotemporal:label::spatial::value::joint_spatiotemporal::compress(param,preprocessing_start_time);return;
                default:throw runtime_error("two_point_sets不支持当前数值流程");
            }
        case time_dimension_mode_e::constant_extrapolation:
            switch(framework.value_mode)
            {
                case time_dimension_mode_e::spatial:label::constant_extrapolation::value::spatial::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::constant_extrapolation:label::constant_extrapolation::value::constant_extrapolation::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::linear_interpolation:label::constant_extrapolation::value::linear_interpolation::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::constant_extrapolation_spatial_residual:label::constant_extrapolation::value::constant_extrapolation_then_spatial::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::linear_interpolation_spatial_residual:label::constant_extrapolation::value::linear_interpolation_then_spatial::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::joint_spatiotemporal:label::constant_extrapolation::value::joint_spatiotemporal::compress(param,preprocessing_start_time);return;
                default:throw runtime_error("two_point_sets不支持当前数值流程");
            }
        case time_dimension_mode_e::endpoint_agreement:
            switch(framework.value_mode)
            {
                case time_dimension_mode_e::spatial:label::endpoint_agreement::value::spatial::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::constant_extrapolation:label::endpoint_agreement::value::constant_extrapolation::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::linear_interpolation:label::endpoint_agreement::value::linear_interpolation::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::constant_extrapolation_spatial_residual:label::endpoint_agreement::value::constant_extrapolation_then_spatial::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::linear_interpolation_spatial_residual:label::endpoint_agreement::value::linear_interpolation_then_spatial::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::joint_spatiotemporal:label::endpoint_agreement::value::joint_spatiotemporal::compress(param,preprocessing_start_time);return;
                default:throw runtime_error("two_point_sets不支持当前数值流程");
            }
        case time_dimension_mode_e::joint_spatiotemporal:
            switch(framework.value_mode)
            {
                case time_dimension_mode_e::spatial:label::joint_spatiotemporal::value::spatial::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::constant_extrapolation:label::joint_spatiotemporal::value::constant_extrapolation::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::linear_interpolation:label::joint_spatiotemporal::value::linear_interpolation::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::constant_extrapolation_spatial_residual:label::joint_spatiotemporal::value::constant_extrapolation_then_spatial::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::linear_interpolation_spatial_residual:label::joint_spatiotemporal::value::linear_interpolation_then_spatial::compress(param,preprocessing_start_time);return;
                case time_dimension_mode_e::joint_spatiotemporal:label::joint_spatiotemporal::value::joint_spatiotemporal::compress(param,preprocessing_start_time);return;
                default:throw runtime_error("two_point_sets不支持当前数值流程");
            }
        default:throw runtime_error("two_point_sets不支持当前标签流程");
    }
}
}
}
