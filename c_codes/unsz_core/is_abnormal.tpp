#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::is_abnormal(T_data val)
    {
        return val<-1000;
    }
}