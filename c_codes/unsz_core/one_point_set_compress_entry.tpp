#pragma once

namespace unsz::one_point_set
{
template<point_set_codec_e predictor,typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param)
{
    if(param.value_flow=="joint_spatiotemporal")
        return param.template compress_joint_spatiotemporal_files<predictor>();
    param.compress_files(&param_c<T_coord,T_data>::template dispatch_single_point_set_compression<predictor>);
}
}
