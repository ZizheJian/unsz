#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    class param_c;

    void parse_args(int argc,char**argv,param_c<float,float>&param);
    int parse_args_and_select_types(int argc,char**argv);
}
