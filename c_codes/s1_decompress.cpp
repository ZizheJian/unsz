#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>
#include <cstdio>
#include "huffman.hpp"
#include "cli_list_parser.hpp"
#include "third_party/zstd/lib/zstd.h"
#include "unsz_core/unsz.hpp"

using namespace std;

string read_option_value(int argc,char **argv,int &arg_id,string option)
{
    if (arg_id+1>=argc || argv[arg_id+1][0]=='-')
        throw runtime_error(option+"后面缺少参数值");
    return argv[++arg_id];
}

template<typename T_coord>
void prepare_current_decompression_geometry(unsz::param_c<T_coord,float>&param)
{
    param.restore_original_coordinates();
    param.apply_coordinate_transform();
    bool uses_spatial_prediction=any_of(param.point_set_config_vec.begin(),param.point_set_config_vec.end(),[](const unsz::point_set_config_c&config){return config.codec!=unsz::point_set_codec_e::constant;});
    if(uses_spatial_prediction&&!param.use_coordinate_transform&&param.activate_shared_spatial_prediction_state())
        printf("Reusing shared spatial prediction DAG and weights\n");
    else
    {
        param.generate_prediction_structures_and_weights();
        param.finalize_current_spatial_prediction_state(true);
        param.current_spatial_prediction_state_is_shared=uses_spatial_prediction&&!param.use_coordinate_transform;
    }
}

template<typename T_coord>
double prepare_current_decompression_field(unsz::param_c<T_coord,float>&param)
{
    param.read_compressed_field_metadata();
    auto preprocessing_start_time=chrono::steady_clock::now();
    param.prepare_time_step_order_from_flows();
    return chrono::duration<double>(chrono::steady_clock::now()-preprocessing_start_time).count();
}

template<typename T_coord>
void run_decompression_field(unsz::param_c<T_coord,float>&param)
{
    uint32_t validated_file_num=0;
    double total_decompression_time=0,total_time_prediction_state_update_time=0,total_prediction_order_restore_time=0;
    unsz::reconstruction_statistics_c total_statistics;
    string max_error_data_path;
    using param_type=unsz::param_c<T_coord,float>;
    if(param.point_set_config_vec.size()!=1&&param.point_set_config_vec.size()!=2)
        throw runtime_error("解压只支持一个或两个点集");
    void(param_type::*decompress_field)();
    if(param.point_set_config_vec.size()==1)
        decompress_field=&param_type::dispatch_single_point_set_decompression;
    else
        decompress_field=&param_type::dispatch_two_point_sets_decompression_by_mode;
    param.prepare_temporal_dependencies();
    vector<float> original_data(param.point_num);
    for (uint32_t file_id=0;file_id<param.original_data_path_vec.size();file_id++)
    {
        param.current_time_step_index=file_id;
        param.current_time_step=param.time_step_vec[file_id];
        string compressed_path=param.original_data_path_vec[file_id]+".unsz";
        string decompressed_path=param.original_data_path_vec[file_id]+".unsz.bin";
        param.current_compressed_bytes=unsz::read_byte_file(compressed_path);

        auto decompression_start_time=chrono::steady_clock::now();
        (param.*decompress_field)();
        if(param.prediction_data_uses_prediction_order)
        {
            auto restore_start_time=chrono::steady_clock::now();
#pragma omp parallel for schedule(static) num_threads(param.thread_num) if(param.prediction_structure_vec.size()>1&&param.thread_num>1)
            for(int64_t original_id=0;original_id<(int64_t)param.point_num;original_id++)
                param.current_decompressed_data[original_id]=param.prediction_order_decompressed_data[param.original_to_prediction_order[original_id]];
            total_prediction_order_restore_time+=chrono::duration<double>(chrono::steady_clock::now()-restore_start_time).count();
        }
        double decompression_time=chrono::duration<double>(chrono::steady_clock::now()-decompression_start_time).count();
        param.validate_decoded_point_set_counts();
        unsz::write_binary_file(decompressed_path,param.current_decompressed_data.data(),param.point_num);

        if(!unsz::try_read_binary_file_exact(param.original_data_path_vec[file_id],original_data.data(),param.point_num))
        {
            printf("Decompressed %s（原始文件不存在，跳过误差验证）\n",decompressed_path.c_str());
            auto time_state_update_start=chrono::steady_clock::now();
            param.update_time_prediction_state();
            double time_state_update_time=chrono::duration<double>(chrono::steady_clock::now()-time_state_update_start).count();
            total_time_prediction_state_update_time+=time_state_update_time;
            decompression_time+=time_state_update_time;
            total_decompression_time+=decompression_time;
            printf("time prediction state update time= %.9f s\n",time_state_update_time);
            printf("file decompression time= %f s\n",decompression_time);
            continue;
        }
        unsz::reconstruction_statistics_c file_statistics=param.calculate_reconstruction_statistics(original_data.data(),param.current_decompressed_data.data(),param.point_num);
        if(file_statistics.max_error_ratio>total_statistics.max_error_ratio)
            max_error_data_path=param.original_data_path_vec[file_id];
        unsz::merge_reconstruction_statistics(total_statistics,file_statistics);
        printf("Decompressed %s, psnr= %f\n",decompressed_path.c_str(),unsz::reconstruction_psnr(file_statistics));
        validated_file_num++;
        auto time_state_update_start=chrono::steady_clock::now();
        param.update_time_prediction_state();
        double time_state_update_time=chrono::duration<double>(chrono::steady_clock::now()-time_state_update_start).count();
        total_time_prediction_state_update_time+=time_state_update_time;
        decompression_time+=time_state_update_time;
        total_decompression_time+=decompression_time;
        printf("time prediction state update time= %.9f s\n",time_state_update_time);
        printf("file decompression time= %f s\n",decompression_time);
    }
    printf("全部文件最大误差/用户误差上限= %.9f\n",total_statistics.max_error_ratio);
    if(total_statistics.max_error_ratio>1.01)
        throw runtime_error("全部文件解压完成后误差检查失败：文件="+max_error_data_path+"，点="+to_string(total_statistics.max_error_point_id)+"，点集="+to_string(total_statistics.max_error_point_set_id)+"，误差="+to_string(total_statistics.max_error)+"，绝对误差上限="+to_string(total_statistics.max_error_abs_eb)+"，误差/上限="+to_string(total_statistics.max_error_ratio));
    printf("total time prediction state update time= %.9f s\n",total_time_prediction_state_update_time);
    if(param.prediction_data_uses_prediction_order)
        printf("total prediction order restore time= %.9f s\n",total_prediction_order_restore_time);
    printf("total decompression time= %f s\n",total_decompression_time);
    if (validated_file_num>0)
        printf("total decompressed psnr= %f\n",unsz::reconstruction_psnr(total_statistics));
}

namespace unsz
{
int decompress_main(int argc,char **argv)
{
    param_c<float,float> param;
    vector<uint32_t> time_step_vec;
    float error_bound=0;
    unordered_set<string> used_option_set;
    for (int arg_id=1;arg_id<argc;arg_id++)
    {
        string option=argv[arg_id];
        if (!used_option_set.insert(option).second)
            throw runtime_error("参数重复："+option);
        if (option=="-data-dir")
        {
            param.data_dir=read_option_value(argc,argv,arg_id,option);
            param.got_data_dir=true;
        }
        else if (option=="-field")
        {
            if(param.got_field)
                throw runtime_error("-field和-field-list不能同时使用");
            param.field_name=read_option_value(argc,argv,arg_id,option);
            param.got_field=true;
        }
        else if(option=="-field-list")
        {
            if(param.got_field)
                throw runtime_error("-field和-field-list不能同时使用");
            param.field_name_vec=parse_field_name_list(read_option_value(argc,argv,arg_id,option));
            param.got_field=true;
        }
        else if (option=="-coordinate-dim")
            param.dim=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-time-step")
        {
            time_step_vec=parse_time_step_list(read_option_value(argc,argv,arg_id,option));
            param.got_time_step=true;
        }
        else if(option=="-time-step-coordinate-file")
            param.time_step_coordinate_path=read_option_value(argc,argv,arg_id,option);
        else if (option=="-M")
        {
            param.error_mode=read_option_value(argc,argv,arg_id,option);
            error_bound=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
            if (param.error_mode!="REL" && param.error_mode!="ABS")
                throw runtime_error("-M的误差模式必须是REL或ABS："+param.error_mode);
            param.got_error_mode=true;
        }
        else if (option=="-partition-size")
            param.partition_size=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-partition-num")
            param.partition_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-thread-num")
            param.thread_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if(option=="-point-set-config")
            param.point_set_config_path=read_option_value(argc,argv,arg_id,option);
        else if (option=="-value-flow")
        {
            param.value_flow=read_option_value(argc,argv,arg_id,option);
            if(param.value_flow=="spatial")param.value_flow="none";
            if(param.value_flow!="none"&&param.value_flow!="auto"&&param.value_flow!="constant_extrapolation"&&param.value_flow!="linear_interpolation"&&param.value_flow!="constant_extrapolation_spatial_residual"&&param.value_flow!="linear_interpolation_spatial_residual"&&param.value_flow!="anchor_interpolation_stride_8"&&param.value_flow!="anchor_interpolation_stride_8_spatial_residual")
                throw runtime_error("未知的value flow："+param.value_flow);
        }
        else if (option=="-label-flow")
        {
            param.label_flow=read_option_value(argc,argv,arg_id,option);
            if(param.label_flow=="spatial")param.label_flow="none";
            if(param.label_flow!="none"&&param.label_flow!="auto"&&param.label_flow!="constant_extrapolation"&&param.label_flow!="linear_interpolation_spatial_residual"&&param.label_flow!="anchor_interpolation_stride_8")
                throw runtime_error("未知的label flow："+param.label_flow);
        }
        else if(option=="-coordinate-transform")
        {
            param.coordinate_transform_mode=read_option_value(argc,argv,arg_id,option);
            if(param.coordinate_transform_mode!="off"&&param.coordinate_transform_mode!="on"&&param.coordinate_transform_mode!="auto")
                throw runtime_error("-coordinate-transform必须是off、on或auto");
        }
        else if(option=="-decompression-stage-time")
            param.print_compression_stage_time=true;
        else
            throw runtime_error("未知参数："+option);
    }
    param.prepare_cli_parameters(time_step_vec,error_bound);
    param.init();
    param.precompute_temporal_dependency_plans();
    for(uint32_t field_id=0;field_id<param.field_name_vec.size();field_id++)
    {
        param.activate_field(field_id);
        printf("BEGIN UNSZ FIELD %s\n",param.field_name.c_str());
        param.read_coordinate_transform_file();
        double field_preprocessing_time=prepare_current_decompression_field(param);
        auto preprocessing_start_time=chrono::steady_clock::now();
        prepare_current_decompression_geometry(param);
        double preprocessing_time=field_preprocessing_time+chrono::duration<double>(chrono::steady_clock::now()-preprocessing_start_time).count();
        printf("preprocessing time excluding coordinate and transform-file input= %f s\n",preprocessing_time);
        run_decompression_field(param);
        param.release_current_spatial_prediction_state();
        printf("END UNSZ FIELD %s\n",param.field_name.c_str());
    }
    return 0;
}
}

int main(int argc,char **argv)
{
    try
    {
        return unsz::decompress_main(argc,argv);
    }
    catch (const exception &error)
    {
        fprintf(stderr,"错误：%s\n",error.what());
        return 1;
    }
}
