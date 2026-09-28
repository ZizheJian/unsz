#pragma once

#include "two_point_sets/label_spatial.tpp"
#include "two_point_sets/label_constant_extrapolation.tpp"
#include "two_point_sets/label_endpoint_agreement.tpp"
#include "two_point_sets/label_anchor_interpolation_stride_8.tpp"
#include "two_point_sets/label_joint_spatiotemporal.tpp"

namespace unsz::two_point_sets
{
    namespace staged
    {
        template<typename T_coord,typename T_data>
        void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
        {
            param.prepare_compression_structure(preprocessing_start_time);
            param.compress_files(&param_c<T_coord,T_data>::compress_two_point_sets);
        }
    }

    namespace fused
    {
        template<typename T_coord,typename T_data>
        void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
        {
            if(param.value_flow=="joint_spatiotemporal")
                return label::joint_spatiotemporal::compress(param,preprocessing_start_time);
            if(param.label_flow=="none")
                return label::spatial::compress(param,preprocessing_start_time);
            if(param.label_flow=="constant_extrapolation")
                return label::constant_extrapolation::compress(param,preprocessing_start_time);
            if(param.label_flow=="endpoint_agreement")
                return label::endpoint_agreement::compress(param,preprocessing_start_time);
            if(param.label_flow=="anchor_interpolation_stride_8")
                return label::anchor_interpolation_stride_8::compress(param,preprocessing_start_time);
            throw runtime_error("未知的两点集标签流程："+param.label_flow);
        }
    }

    template<typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
    {
        if(param.point_set_config_vec.size()!=2)
            throw runtime_error("two_point_sets::compress要求恰好配置两个点集");
        if(param.label_value_execution=="staged")
            return staged::compress(param,preprocessing_start_time);
        if(param.label_value_execution=="fused")
            return fused::compress(param,preprocessing_start_time);
        throw runtime_error("two_point_sets只支持staged或fused执行方式");
    }
}
