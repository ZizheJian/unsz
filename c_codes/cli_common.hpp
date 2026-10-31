#pragma once

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cstdio>
#include <exception>
#include <unordered_set>
#include <omp.h>
#include "cli_list_parser.hpp"
#include "unsz_core/unsz.hpp"

namespace unsz
{
    // 压缩器和解压器共用的命令行选项。只属于一端的选项交给parse_side_option(option,read_value)，它识别该选项时返回true。
    template<typename Side_option_parser>
    void parse_common_args(int argc,char**argv,param_c<float,float>&param,Side_option_parser parse_side_option)
    {
        vector<uint32_t> time_step_vec;
        float error_bound=0;
        unordered_set<string> used_option_set;
        for(int arg_id=1;arg_id<argc;arg_id++)
        {
            string option=argv[arg_id];
            if(!used_option_set.insert(option).second)
                throw runtime_error("参数重复："+option);
            auto read_value=[&]()
            {
                if(arg_id+1>=argc||argv[arg_id+1][0]=='-')
                    throw runtime_error(option+"后面缺少参数值");
                return string(argv[++arg_id]);
            };
            if(option=="-data-dir")
            {
                param.data_dir=read_value();
                param.got_data_dir=true;
            }
            else if(option=="-output-dir")
                param.output_dir=read_value();
            else if(option=="-field"||option=="-field-list")
            {
                if(param.got_field)
                    throw runtime_error("-field和-field-list不能同时使用");
                if(option=="-field")
                    param.field_name=read_value();
                else
                    param.field_name_vec=parse_field_name_list(read_value());
                param.got_field=true;
            }
            else if(option=="-coordinate-dim")
                param.dim=parse_positive_uint32(read_value(),option);
            else if(option=="-time-step")
            {
                time_step_vec=parse_time_step_list(read_value());
                param.got_time_step=true;
            }
            else if(option=="-time-step-coordinate-file")
                param.time_step_coordinate_path=read_value();
            else if(option=="-tensor-metadata")
                param.tensor_metadata_path=read_value();
            else if(option=="-M")
            {
                param.error_mode=read_value();
                error_bound=parse_positive_float(read_value(),option);
                if(param.error_mode!="REL"&&param.error_mode!="ABS")
                    throw runtime_error("-M的误差模式必须是REL或ABS："+param.error_mode);
                param.got_error_mode=true;
            }
            else if(option=="-partition-num")
                param.partition_num=parse_positive_uint32(read_value(),option);
            else if(option=="-thread-num")
                param.thread_num=parse_positive_uint32(read_value(),option);
            else if(option=="-field-worker-num")
                param.field_worker_num=parse_positive_uint32(read_value(),option);
            else if(option=="-field-worker-map")
                param.field_worker_id_vec=parse_worker_id_list(read_value());
            else if(option=="-point-set-config")
                param.point_set_config_path=read_value();
            else if(option=="-value-flow")
            {
                param.value_flow=read_value();
                if(param.value_flow=="spatial")param.value_flow="none";
                if(param.value_flow!="none"&&param.value_flow!="auto"&&param.value_flow!="constant_extrapolation"&&param.value_flow!="linear_interpolation"&&param.value_flow!="constant_extrapolation_spatial_residual"&&param.value_flow!="linear_interpolation_spatial_residual"&&param.value_flow!="anchor_interpolation_stride_8"&&param.value_flow!="anchor_interpolation_stride_8_spatial_residual")
                    throw runtime_error("未知的value flow："+param.value_flow);
            }
            else if(option=="-label-flow")
            {
                param.label_flow=read_value();
                if(param.label_flow=="spatial")param.label_flow="none";
                if(param.label_flow!="none"&&param.label_flow!="auto"&&param.label_flow!="constant_extrapolation"&&param.label_flow!="linear_interpolation_spatial_residual"&&param.label_flow!="anchor_interpolation_stride_8")
                    throw runtime_error("未知的label flow："+param.label_flow);
            }
            else if(option=="-coordinate-transform")
            {
                param.coordinate_transform_mode=read_value();
                if(param.coordinate_transform_mode!="off"&&param.coordinate_transform_mode!="on"&&param.coordinate_transform_mode!="auto")
                    throw runtime_error("-coordinate-transform必须是off、on或auto");
            }
            else if(!parse_side_option(option,read_value))
                throw runtime_error("未知参数："+option);
        }
        param.prepare_cli_parameters(time_step_vec,error_bound);
    }

    // 按-field-worker-map把field分给多个worker并行处理。每个worker持有param的一份副本，process_fields处理一个worker的全部field。
    // 各worker共用一次预先生成的跨slice依赖计划；报告时间取最慢worker的预处理与codec时间之和，再加上共用计划的生成时间。
    template<typename Process_fields,typename Reported_time>
    void run_field_workers(param_c<float,float>&param,const char*side,Process_fields process_fields,Reported_time reported_time)
    {
        auto shared_dependency_plan_start_time=chrono::steady_clock::now();
        if(param.tensor_field_config_map.empty())
            param.precompute_temporal_dependency_plans();
        else
            param.precompute_all_tensor_dependency_plans();
        double shared_dependency_plan_time=seconds_since(shared_dependency_plan_start_time);
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
                process_fields(*worker_param_vec[worker_id]);
            }
            catch(...)
            {
                worker_error_vec[worker_id]=current_exception();
            }
        }
        double maximum_reported_time=0;
        for(uint32_t worker_id=0;worker_id<worker_param_vec.size();worker_id++)
        {
            double worker_reported_time=reported_time(*worker_param_vec[worker_id]);
            printf("field worker %u reported %s-side time= %f s\n",worker_id,side,worker_reported_time);
            maximum_reported_time=max(maximum_reported_time,worker_reported_time);
        }
        maximum_reported_time+=shared_dependency_plan_time;
        printf("two-level shared dependency-plan preprocessing time= %f s\n",shared_dependency_plan_time);
        printf("two-level reported %s-side time= %f s\n",side,maximum_reported_time);
        printf("two-level %s wall time= %f s\n",side,seconds_since(parallel_start_time));
        for(const exception_ptr&worker_error:worker_error_vec)
            if(worker_error)
                rethrow_exception(worker_error);
    }
}
