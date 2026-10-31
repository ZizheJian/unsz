#include <string>
#include <cstdio>
#include "cli_common.hpp"
#include "compress_cli.hpp"

using namespace std;

namespace unsz
{
    void parse_args(int argc,char**argv,param_c<float,float>&param)
    {
        parse_common_args(argc,argv,param,[&](const string&option,auto&)
        {
            if(option!="-compression-stage-time")
                return false;
            param.print_compression_stage_time=true;
            return true;
        });
    }

    int parse_args_and_select_types(int argc,char**argv)
    {
        param_c<float,float> param;
        parse_args(argc,argv,param);
        param.init();
        printf("使用float处理坐标和距离\n");
        if(param.field_worker_num==1)
            unsz::compress(param);
        else
            run_field_workers(param,"compression",[](param_c<float,float>&worker_param){unsz::compress(worker_param);},[](const param_c<float,float>&worker_param){return worker_param.reported_compression_preprocessing_time+worker_param.reported_compression_time;});
        return 0;
    }
}
