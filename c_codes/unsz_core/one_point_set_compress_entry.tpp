#pragma once

namespace unsz::one_point_set
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data>&param)
{
    param.compress_files(&param_c<T_coord,T_data>::dispatch_single_point_set_compression);
}
}
