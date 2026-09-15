#pragma once

namespace unsz
{
namespace two_point_sets::label::spatial::value::spatial
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
{
    prediction_framework_c framework=param.get_prediction_framework(2);
    if(framework.execution==label_value_execution_e::staged)
    {
        param.prepare_compression_structure(preprocessing_start_time);
        param.compress_files(&param_c<T_coord,T_data>::compress_two_point_sets);
        return;
    }
    if(framework.execution==label_value_execution_e::fused)
    {
        param.prepare_compression_structure(preprocessing_start_time);
        param.compress_files(&param_c<T_coord,T_data>::compress_two_point_sets);
        return;
    }
    throw runtime_error("two_point_sets分支要求staged或fused执行方式");
}
}

namespace two_point_sets::label::spatial::value::constant_extrapolation
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
{
    prediction_framework_c framework=param.get_prediction_framework(2);
    if(framework.execution==label_value_execution_e::staged)
    {
        param.prepare_compression_structure(preprocessing_start_time);
        param.compress_files(&param_c<T_coord,T_data>::compress_two_point_sets);
        return;
    }
    if(framework.execution==label_value_execution_e::fused)
        throw runtime_error("two_point_sets::label::spatial::value::constant_extrapolation的fused执行尚未实现");
    throw runtime_error("two_point_sets分支要求staged或fused执行方式");
}
}

namespace two_point_sets::label::spatial::value::linear_interpolation
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
{
    prediction_framework_c framework=param.get_prediction_framework(2);
    if(framework.execution==label_value_execution_e::staged)
    {
        param.prepare_compression_structure(preprocessing_start_time);
        param.compress_files(&param_c<T_coord,T_data>::compress_two_point_sets);
        return;
    }
    if(framework.execution==label_value_execution_e::fused)
        throw runtime_error("two_point_sets::label::spatial::value::linear_interpolation的fused执行尚未实现");
    throw runtime_error("two_point_sets分支要求staged或fused执行方式");
}
}

namespace two_point_sets::label::spatial::value::constant_extrapolation_then_spatial
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
{
    prediction_framework_c framework=param.get_prediction_framework(2);
    if(framework.execution==label_value_execution_e::staged)
    {
        param.prepare_compression_structure(preprocessing_start_time);
        param.compress_files(&param_c<T_coord,T_data>::compress_two_point_sets);
        return;
    }
    if(framework.execution==label_value_execution_e::fused)
        throw runtime_error("two_point_sets::label::spatial::value::constant_extrapolation_then_spatial的fused执行尚未实现");
    throw runtime_error("two_point_sets分支要求staged或fused执行方式");
}
}

namespace two_point_sets::label::spatial::value::linear_interpolation_then_spatial
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
{
    prediction_framework_c framework=param.get_prediction_framework(2);
    if(framework.execution==label_value_execution_e::staged)
    {
        param.prepare_compression_structure(preprocessing_start_time);
        param.compress_files(&param_c<T_coord,T_data>::compress_two_point_sets);
        return;
    }
    if(framework.execution==label_value_execution_e::fused)
        throw runtime_error("two_point_sets::label::spatial::value::linear_interpolation_then_spatial的fused执行尚未实现");
    throw runtime_error("two_point_sets分支要求staged或fused执行方式");
}
}

namespace two_point_sets::label::spatial::value::joint_spatiotemporal
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
{
    prediction_framework_c framework=param.get_prediction_framework(2);
    if(framework.execution==label_value_execution_e::staged)
        throw runtime_error("two_point_sets::label::spatial::value::joint_spatiotemporal的staged执行尚未实现");
    if(framework.execution==label_value_execution_e::fused)
        throw runtime_error("two_point_sets::label::spatial::value::joint_spatiotemporal的fused执行尚未实现");
    throw runtime_error("two_point_sets分支要求staged或fused执行方式");
}
}

}

