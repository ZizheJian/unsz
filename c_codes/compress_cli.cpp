#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <omp.h>
#include <cstring>
#include "cli_list_parser.hpp"
#include "unsz_core/unsz.hpp"
#include "compress_cli.hpp"

using namespace std;

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
            else if(strcmp(argv[arg_id],"-output-dir")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-output-dir后面缺少参数值");
                param.output_dir=argv[++arg_id];
            }
            else if (strcmp(argv[arg_id],"-field")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-field后面缺少参数值");
                if(param.got_field)
                    throw runtime_error("-field和-field-list不能同时使用");
                param.field_name=argv[++arg_id];
                param.got_field=true;
            }
            else if(strcmp(argv[arg_id],"-field-list")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-field-list后面缺少参数值");
                if(param.got_field)
                    throw runtime_error("-field和-field-list不能同时使用");
                param.field_name_vec=parse_field_name_list(argv[++arg_id]);
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
                time_step_vec=parse_time_step_list(argv[++arg_id]);
                param.got_time_step=true;
            }
            else if(strcmp(argv[arg_id],"-time-step-coordinate-file")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-time-step-coordinate-file后面缺少文件路径");
                param.time_step_coordinate_path=argv[++arg_id];
            }
            else if(strcmp(argv[arg_id],"-tensor-metadata")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-tensor-metadata后面缺少文件路径");
                param.tensor_metadata_path=argv[++arg_id];
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
            else if(strcmp(argv[arg_id],"-field-worker-num")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-field-worker-num后面缺少参数值");
                param.field_worker_num=parse_positive_uint32(argv[++arg_id],"-field-worker-num");
            }
            else if(strcmp(argv[arg_id],"-field-worker-map")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-field-worker-map后面缺少参数值");
                param.field_worker_id_vec=parse_worker_id_list(argv[++arg_id]);
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
        if(param.field_worker_num==1)
            unsz::compress(param);
        else
        {
            auto shared_dependency_plan_start_time=chrono::steady_clock::now();
            if(param.tensor_field_config_map.empty())
                param.precompute_temporal_dependency_plans();
            else
                param.precompute_all_tensor_dependency_plans();
            double shared_dependency_plan_time=chrono::duration<double>(chrono::steady_clock::now()-shared_dependency_plan_start_time).count();
            vector<vector<string>>worker_field_name_vec(param.field_worker_num);
            for(uint32_t field_id=0;field_id<param.field_name_vec.size();field_id++)
                worker_field_name_vec[param.field_worker_id_vec[field_id]].push_back(param.field_name_vec[field_id]);
            vector<unique_ptr<param_c<float,float>>>worker_param_vec;
            for(uint32_t worker_id=0;worker_id<param.field_worker_num;worker_id++)
            {
                if(worker_field_name_vec[worker_id].empty())
                    continue;
                printf("Field worker %u:",worker_id);
                for(const string&field_name:worker_field_name_vec[worker_id])
                    printf(" %s",field_name.c_str());
                printf("\n");
                worker_param_vec.push_back(make_unique<param_c<float,float>>(param));
                worker_param_vec.back()->field_name_vec=move(worker_field_name_vec[worker_id]);
            }
            vector<exception_ptr>worker_error_vec(worker_param_vec.size());
            auto parallel_start_time=chrono::steady_clock::now();
            omp_set_dynamic(0);
            omp_set_max_active_levels(2);
#pragma omp parallel for schedule(static,1) num_threads(param.field_worker_num)
            for(int64_t worker_id=0;worker_id<(int64_t)worker_param_vec.size();worker_id++)
            {
                try
                {
                    unsz::compress(*worker_param_vec[worker_id]);
                }
                catch(...)
                {
                    worker_error_vec[worker_id]=current_exception();
                }
            }
            double maximum_reported_time=0;
            for(uint32_t worker_id=0;worker_id<worker_param_vec.size();worker_id++)
            {
                double reported_time=worker_param_vec[worker_id]->reported_compression_preprocessing_time+worker_param_vec[worker_id]->reported_compression_time;
                printf("field worker %u reported compression-side time= %f s\n",worker_id,reported_time);
                maximum_reported_time=max(maximum_reported_time,reported_time);
            }
            maximum_reported_time+=shared_dependency_plan_time;
            printf("two-level shared dependency-plan preprocessing time= %f s\n",shared_dependency_plan_time);
            printf("two-level reported compression-side time= %f s\n",maximum_reported_time);
            printf("two-level compression wall time= %f s\n",chrono::duration<double>(chrono::steady_clock::now()-parallel_start_time).count());
            for(const exception_ptr&worker_error:worker_error_vec)
                if(worker_error)
                    rethrow_exception(worker_error);
        }
        return 0;
    }
}
