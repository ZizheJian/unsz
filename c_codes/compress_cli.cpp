#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include "parsers.hpp"
#include "unsz_core/unsz.hpp"
#include "compress_cli.hpp"

using namespace std;

vector<uint32_t> parse_time_step_vec(string expression)
{
    vector<uint32_t> result;
    if (expression.size()>=5 && expression.front()=='[' && (expression.back()==']' || expression.back()==')'))
    {
        uint64_t comma_position=expression.find(',');
        if (comma_position==string::npos || expression.find(',',comma_position+1)!=string::npos)
            throw runtime_error("时间步区间必须包含且只包含一个逗号："+expression);
        uint32_t left=parse_uint32(expression.substr(1,comma_position-1),"时间步左端点");
        uint32_t right=parse_uint32(expression.substr(comma_position+1,expression.size()-comma_position-2),"时间步右端点");
        uint64_t end=(uint64_t)right+(expression.back()==']');
        if ((uint64_t)left>=end)
            throw runtime_error("时间步区间为空或端点顺序错误："+expression);
        result.reserve(end-left);
        for (uint64_t time_step=left;time_step<end;time_step++)
            result.push_back((uint32_t)time_step);
    }
    else if (expression.size()>=3 && expression.front()=='{' && expression.back()=='}')
    {
        if (expression[expression.size()-2]==',')
            throw runtime_error("时间步列表末尾不能有逗号："+expression);
        unordered_set<uint32_t> used_time_step_set;
        uint64_t begin=1,end=0;
        while (begin<expression.size()-1)
        {
            end=expression.find(',',begin);
            if (end==string::npos)
                end=expression.size()-1;
            uint32_t time_step=parse_uint32(expression.substr(begin,end-begin),"时间步");
            if (!used_time_step_set.insert(time_step).second)
                throw runtime_error("时间步列表中存在重复值："+to_string(time_step));
            result.push_back(time_step);
            begin=end+1;
        }
        if (result.empty())
            throw runtime_error("时间步列表不能为空");
    }
    else
        throw runtime_error("时间步必须写成区间'[a,b]'、'[a,b)'或列表'{a1,a2,...}'："+expression);
    return result;
}

namespace unsz
{
    void parse_args(int argc,char**argv,param_c<float,float>&param)
    {
        vector<uint32_t> time_step_vec;
        float error_bound=0;
        unordered_set<string> used_option_set;
        for (int arg_id=1;arg_id<argc;arg_id++)
        {
            if (!used_option_set.insert(argv[arg_id]).second)
                throw runtime_error("参数重复："+string(argv[arg_id]));
            if (strcmp(argv[arg_id],"-data-dir")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-data-dir后面缺少参数值");
                param.data_dir=argv[++arg_id];
                param.got_data_dir=true;
            }
            else if (strcmp(argv[arg_id],"-field")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-field后面缺少参数值");
                param.field_name=argv[++arg_id];
                param.got_field=true;
            }
            else if (strcmp(argv[arg_id],"-coordinate-dim")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-coordinate-dim后面缺少参数值");
                param.dim=parse_positive_uint32(argv[++arg_id],"-coordinate-dim");
            }
            else if (strcmp(argv[arg_id],"-time-step")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-time-step后面缺少参数值");
                time_step_vec=parse_time_step_vec(argv[++arg_id]);
                param.got_time_step=true;
            }
            else if(strcmp(argv[arg_id],"-time-step-coordinate-file")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-time-step-coordinate-file后面缺少文件路径");
                param.time_step_coordinate_path=argv[++arg_id];
            }
            else if (strcmp(argv[arg_id],"-M")==0)
            {
                if (arg_id+2>=argc)
                    throw runtime_error("-M后面需要误差模式和误差值");
                param.error_mode=argv[++arg_id];
                error_bound=parse_positive_float(argv[++arg_id],"-M");
                if (param.error_mode!="REL" && param.error_mode!="ABS")
                    throw runtime_error("-M的误差模式必须是REL或ABS："+param.error_mode);
                param.got_error_mode=true;
            }
            else if (strcmp(argv[arg_id],"-partition-size")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-partition-size后面缺少参数值");
                param.partition_size=parse_positive_uint32(argv[++arg_id],"-partition-size");
            }
            else if (strcmp(argv[arg_id],"-partition-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-partition-num后面缺少参数值");
                param.partition_num=parse_positive_uint32(argv[++arg_id],"-partition-num");
            }
            else if (strcmp(argv[arg_id],"-thread-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-thread-num后面缺少参数值");
                param.thread_num=parse_positive_uint32(argv[++arg_id],"-thread-num");
            }
            else if(strcmp(argv[arg_id],"-point-set-config")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-point-set-config后面缺少文件路径");
                param.point_set_config_path=argv[++arg_id];
            }
            else if (strcmp(argv[arg_id],"-value-flow")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-value-flow后面缺少模式");
                param.value_flow=argv[++arg_id];
                if(param.value_flow=="spatial")param.value_flow="none";
                if(param.value_flow!="none"&&param.value_flow!="auto"&&param.value_flow!="constant_extrapolation"&&param.value_flow!="linear_interpolation"&&param.value_flow!="constant_extrapolation_spatial_residual"&&param.value_flow!="linear_interpolation_spatial_residual"&&param.value_flow!="anchor_interpolation_stride_8"&&param.value_flow!="anchor_interpolation_stride_8_spatial_residual")
                    throw runtime_error("未知的value flow："+param.value_flow);
            }
            else if (strcmp(argv[arg_id],"-label-flow")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-label-flow后面缺少模式");
                param.label_flow=argv[++arg_id];
                if(param.label_flow=="spatial")param.label_flow="none";
                if(param.label_flow!="none"&&param.label_flow!="auto"&&param.label_flow!="constant_extrapolation"&&param.label_flow!="linear_interpolation_spatial_residual"&&param.label_flow!="anchor_interpolation_stride_8")
                    throw runtime_error("未知的label flow："+param.label_flow);
            }
            else if(strcmp(argv[arg_id],"-coordinate-transform")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-coordinate-transform后面缺少模式");
                param.coordinate_transform_mode=argv[++arg_id];
                if(param.coordinate_transform_mode!="off"&&param.coordinate_transform_mode!="on"&&param.coordinate_transform_mode!="auto")
                    throw runtime_error("-coordinate-transform必须是off、on或auto");
            }
            else if (strcmp(argv[arg_id],"-compression-stage-time")==0)
                param.print_compression_stage_time=true;
            else
                throw runtime_error("未知参数："+string(argv[arg_id]));
        }
        param.prepare_cli_parameters(time_step_vec,error_bound);
    }

    int parse_args_and_select_types(int argc,char**argv)
    {
        param_c<float,float> param;
        parse_args(argc,argv,param);
        param.init();
        printf("使用float处理坐标和距离\n");
        unsz::compress(param);
        return 0;
    }
}
