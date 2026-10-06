#pragma once

namespace unsz::two_point_sets
{
    template<typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param)
    {
        if(param.label_flow!="none"&&param.value_flow!="none"&&param.get_time_step_order(param.label_flow)!=param.get_time_step_order(param.value_flow))
            throw runtime_error("两点集压缩要求label和value使用相同time step顺序");
        param.compress_files(&param_c<T_coord,T_data>::dispatch_fused_compression);
    }
}
