#pragma once

namespace unsz::one_point_set::spatial
{
    template<typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param)
    {
        compression_file_state_c state;
        param.template prepare_current_compression<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable>(state);
        param.template compress_values<false,time_dimension_mode_e::spatial>(state);
        param.assemble_current_compression_file(state);
    }
}

namespace unsz::one_point_set::temporal
{
    template<uint32_t time_ref_num,bool use_spatial_residual,typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param)
    {
        constexpr time_dimension_mode_e value_mode=time_flow::value_mode<time_ref_num,use_spatial_residual>();
        compression_file_state_c state;
        param.template prepare_current_compression<false,value_mode,time_dimension_mode_e::not_applicable>(state);
        param.template compress_values<false,value_mode>(state);
        param.assemble_current_compression_file(state);
    }
}

namespace unsz::two_point_sets::fused::label_spatial_value_spatial
{
    template<typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param)
    {
        compression_file_state_c state;
        param.template prepare_current_compression<true,time_dimension_mode_e::spatial,time_dimension_mode_e::spatial>(state);
        param.template compress_labels_and_values<time_dimension_mode_e::spatial,time_dimension_mode_e::spatial>(state);
        param.assemble_current_compression_file(state);
    }
}

namespace unsz::two_point_sets::fused::label_spatial_value_temporal
{
    template<uint32_t time_ref_num,bool use_spatial_residual,typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param)
    {
        constexpr time_dimension_mode_e value_mode=time_flow::value_mode<time_ref_num,use_spatial_residual>();
        compression_file_state_c state;
        param.template prepare_current_compression<true,value_mode,time_dimension_mode_e::spatial>(state);
        param.template compress_labels_and_values<value_mode,time_dimension_mode_e::spatial>(state);
        param.assemble_current_compression_file(state);
    }
}

namespace unsz::two_point_sets::fused::label_temporal_value_spatial
{
    template<uint32_t time_ref_num,typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param)
    {
        constexpr time_dimension_mode_e label_mode=time_flow::label_mode<time_ref_num>();
        compression_file_state_c state;
        param.template prepare_current_compression<true,time_dimension_mode_e::spatial,label_mode>(state);
        param.template compress_labels_and_values<time_dimension_mode_e::spatial,label_mode>(state);
        param.assemble_current_compression_file(state);
    }
}

namespace unsz::two_point_sets::fused::label_temporal_value_temporal
{
    template<uint32_t time_ref_num,bool use_spatial_residual,typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param)
    {
        constexpr time_dimension_mode_e label_mode=time_flow::label_mode<time_ref_num>();
        constexpr time_dimension_mode_e value_mode=time_flow::value_mode<time_ref_num,use_spatial_residual>();
        compression_file_state_c state;
        param.template prepare_current_compression<true,value_mode,label_mode>(state);
        param.template compress_labels_and_values<value_mode,label_mode>(state);
        param.assemble_current_compression_file(state);
    }
}
