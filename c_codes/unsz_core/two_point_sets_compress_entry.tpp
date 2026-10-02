#pragma once

namespace unsz::two_point_sets
{
    template<point_set_codec_e predictor,typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param)
    {
        if(param.label_value_execution=="staged")
        {
            param.template compress_staged_two_point_sets_files<predictor>();
            return;
        }
        if(param.label_value_execution=="fused")
        {
            if(param.label_flow!="none"&&param.value_flow!="none"&&param.get_time_step_order(param.label_flow)!=param.get_time_step_order(param.value_flow))
                throw runtime_error("fused模式尚未实现label和value使用不同time step顺序");
            param.compress_files(&param_c<T_coord,T_data>::template dispatch_fused_compression<predictor>);
            return;
        }
        throw runtime_error("two_point_sets只支持staged或fused执行方式");
    }
}
