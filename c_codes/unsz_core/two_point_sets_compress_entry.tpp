#pragma once

namespace unsz::two_point_sets
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
{
    if(param.point_set_config_vec.size()!=2)
        throw runtime_error("two_point_sets::compress要求恰好配置两个点集");
    if(param.label_value_execution=="staged")
    {
        param.prepare_compression_structure(preprocessing_start_time);
        param.compress_staged_two_point_set_files();
        return;
    }
    if(param.label_value_execution=="fused")
    {
        if(param.value_flow=="joint_spatiotemporal")
            throw runtime_error("融合两点集尚未实现联合时空数值流程");
        param.prepare_compression_structure(preprocessing_start_time);
        param.compress_files(&param_c<T_coord,T_data>::dispatch_two_point_set_compression_by_mode);
        return;
    }
    throw runtime_error("two_point_sets只支持staged或fused执行方式");
}
}
