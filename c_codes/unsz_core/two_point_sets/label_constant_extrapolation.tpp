#pragma once

namespace unsz::two_point_sets::fused::label::constant_extrapolation::value::spatial
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

namespace unsz::two_point_sets::fused::label::constant_extrapolation::value::constant_extrapolation
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
        throw runtime_error("two_point_sets::label::constant_extrapolation::value::constant_extrapolation的fused执行尚未实现");
    throw runtime_error("two_point_sets分支要求staged或fused执行方式");
}
}

namespace unsz::two_point_sets::fused::label::constant_extrapolation::value::linear_interpolation
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
        throw runtime_error("two_point_sets::label::constant_extrapolation::value::linear_interpolation的fused执行尚未实现");
    throw runtime_error("two_point_sets分支要求staged或fused执行方式");
}
}

namespace unsz::two_point_sets::fused::label::constant_extrapolation::value::constant_extrapolation_then_spatial
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

namespace unsz::two_point_sets::fused::label::constant_extrapolation::value::linear_interpolation_then_spatial
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
        throw runtime_error("two_point_sets::label::constant_extrapolation::value::linear_interpolation_then_spatial的fused执行尚未实现");
    throw runtime_error("two_point_sets分支要求staged或fused执行方式");
}
}

namespace unsz::two_point_sets::fused::label::constant_extrapolation::value::joint_spatiotemporal
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
{
    prediction_framework_c framework=param.get_prediction_framework(2);
    if(framework.execution==label_value_execution_e::staged)
        throw runtime_error("two_point_sets::label::constant_extrapolation::value::joint_spatiotemporal的staged执行尚未实现");
    if(framework.execution==label_value_execution_e::fused)
        throw runtime_error("two_point_sets::label::constant_extrapolation::value::joint_spatiotemporal的fused执行尚未实现");
    throw runtime_error("two_point_sets分支要求staged或fused执行方式");
}
}

namespace unsz::two_point_sets::fused::label::constant_extrapolation::value::anchor_interpolation_stride_8
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&,chrono::steady_clock::time_point){throw runtime_error("two_point_sets::fused::label::constant_extrapolation::value::anchor_interpolation_stride_8尚未实现");}
}

namespace unsz::two_point_sets::fused::label::constant_extrapolation::value::anchor_interpolation_stride_8_then_spatial
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&,chrono::steady_clock::time_point){throw runtime_error("two_point_sets::fused::label::constant_extrapolation::value::anchor_interpolation_stride_8_then_spatial尚未实现");}
}

namespace unsz::two_point_sets::fused::label::constant_extrapolation
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
{
    if(param.value_flow=="none")
        return value::spatial::compress(param,preprocessing_start_time);
    if(param.value_flow=="constant_extrapolation")
        return value::constant_extrapolation::compress(param,preprocessing_start_time);
    if(param.value_flow=="linear_interpolation")
        return value::linear_interpolation::compress(param,preprocessing_start_time);
    if(param.value_flow=="constant_extrapolation_spatial_residual")
        return value::constant_extrapolation_then_spatial::compress(param,preprocessing_start_time);
    if(param.value_flow=="linear_interpolation_spatial_residual")
        return value::linear_interpolation_then_spatial::compress(param,preprocessing_start_time);
    if(param.value_flow=="anchor_interpolation_stride_8")
        return value::anchor_interpolation_stride_8::compress(param,preprocessing_start_time);
    if(param.value_flow=="anchor_interpolation_stride_8_spatial_residual")
        return value::anchor_interpolation_stride_8_then_spatial::compress(param,preprocessing_start_time);
    if(param.value_flow=="joint_spatiotemporal")
        return value::joint_spatiotemporal::compress(param,preprocessing_start_time);
    throw runtime_error("未知的两点集数值流程："+param.value_flow);
}
}
