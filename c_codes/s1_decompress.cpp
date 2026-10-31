#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <omp.h>
#include "cli_common.hpp"

using namespace std;

template<typename T_coord>
void prepare_current_decompression_geometry(unsz::param_c<T_coord,float>&param)
{
    param.restore_original_coordinates();
    param.apply_coordinate_transform();
    param.prepare_current_spatial_prediction_state(true);
}

template<typename T_coord>
pair<double,double> prepare_current_decompression_field(unsz::param_c<T_coord,float>&param)
{
    auto preprocessing_start_time=chrono::steady_clock::now();
    double metadata_input_time=param.read_compressed_field_metadata();
    param.prepare_time_step_order_from_flows();
    param.prepare_temporal_dependencies();
    double preprocessing_time=unsz::seconds_since(preprocessing_start_time)-metadata_input_time;
    if(param.current_field_uses_tensor)
        printf("compressed tensor flow: %s\n",unsz::tensor_flow_name(param.selected_tensor_flow).c_str());
    else
        printf("compressed flow: label=%s,value=%s\n",param.label_flow.c_str(),param.value_flow.c_str());
    return {preprocessing_time,metadata_input_time};
}

template<typename T_coord>
double run_decompression_field(unsz::param_c<T_coord,float>&param,double preprocessing_time)
{
    uint32_t validated_file_num=0;
    uint64_t total_compressed_bytes=0;
    double total_decompression_time=0,total_time_prediction_state_update_time=0,total_prediction_order_restore_time=0,total_normalized_squared_error=0,total_slice_valid_normalized_squared_error=0,total_slice_normalized_squared_error=0;
    uint64_t total_normalized_value_num=0;
    unsz::reconstruction_statistics_c total_statistics;
    string max_error_data_path;
    if(param.point_set_config_vec.size()!=1&&param.point_set_config_vec.size()!=2)
        throw runtime_error("解压只支持一个或两个点集");
    vector<float> original_data(param.point_num);
    for (uint32_t file_id=0;file_id<param.original_data_path_vec.size();file_id++)
    {
        bool print_file_details=param.original_data_path_vec.size()<=32||param.print_compression_stage_time;
        if(!print_file_details&&file_id%128==0)
            printf("Decompression progress: %u/%u slices\n",file_id,(uint32_t)param.original_data_path_vec.size());
        param.current_time_step_index=file_id;
        param.current_time_step=param.time_step_vec[file_id];
        string compressed_path=param.generated_path(param.original_data_path_vec[file_id],".unsz");
        string decompressed_path=param.generated_path(param.original_data_path_vec[file_id],".unsz.bin");
        param.current_compressed_bytes=unsz::read_byte_file(compressed_path);
        total_compressed_bytes+=param.current_compressed_bytes.size();

        auto decompression_start_time=chrono::steady_clock::now();
        param.decompress_current_file_by_mode();
        if(param.prediction_data_uses_prediction_order)
        {
            auto restore_start_time=chrono::steady_clock::now();
#pragma omp parallel for schedule(static) num_threads(param.thread_num) if(param.prediction_structure_vec.size()>1&&param.thread_num>1)
            for(int64_t original_id=0;original_id<(int64_t)param.point_num;original_id++)
                param.current_decompressed_data[original_id]=param.prediction_order_decompressed_data[param.original_to_prediction_order[original_id]];
            total_prediction_order_restore_time+=unsz::seconds_since(restore_start_time);
        }
        double decompression_time=unsz::seconds_since(decompression_start_time);
        if(print_file_details)
            param.print_current_decompression_diagnostics();
        param.validate_decoded_point_set_counts();
        if(param.write_decompressed_output)
            unsz::write_binary_file(decompressed_path,param.current_decompressed_data.data(),param.point_num);

        if(!unsz::try_read_binary_file_exact(param.original_data_path_vec[file_id],original_data.data(),param.point_num))
        {
            if(print_file_details)
                printf("Decompressed %s（原始文件不存在，跳过误差验证）\n",param.write_decompressed_output?decompressed_path.c_str():compressed_path.c_str());
            auto time_state_update_start=chrono::steady_clock::now();
            param.update_time_prediction_state();
            double time_state_update_time=unsz::seconds_since(time_state_update_start);
            total_time_prediction_state_update_time+=time_state_update_time;
            decompression_time+=time_state_update_time;
            total_decompression_time+=decompression_time;
            if(print_file_details)
            {
                printf("time prediction state update time= %.9f s\n",time_state_update_time);
                printf("file decompression time= %f s\n",decompression_time);
            }
            continue;
        }
        unsz::reconstruction_statistics_c file_statistics=param.calculate_reconstruction_statistics(original_data.data(),param.current_decompressed_data.data(),param.point_num);
        if(file_statistics.max_error_ratio>total_statistics.max_error_ratio)
            max_error_data_path=param.original_data_path_vec[file_id];
        unsz::merge_reconstruction_statistics(total_statistics,file_statistics);
        unsz::add_normalized_reconstruction_error(total_normalized_squared_error,total_normalized_value_num,file_statistics,param.point_set_point_num_vec,param.point_set_min_data_vec,param.point_set_max_data_vec);
        unsz::add_slice_normalized_error(total_slice_valid_normalized_squared_error,total_slice_normalized_squared_error,file_statistics);
        if(print_file_details)
            printf("Decompressed %s, psnr= %f\n",param.write_decompressed_output?decompressed_path.c_str():compressed_path.c_str(),unsz::reconstruction_psnr(file_statistics));
        validated_file_num++;
        auto time_state_update_start=chrono::steady_clock::now();
        param.update_time_prediction_state();
        double time_state_update_time=unsz::seconds_since(time_state_update_start);
        total_time_prediction_state_update_time+=time_state_update_time;
        decompression_time+=time_state_update_time;
        total_decompression_time+=decompression_time;
        if(print_file_details)
        {
            printf("time prediction state update time= %.9f s\n",time_state_update_time);
            printf("file decompression time= %f s\n",decompression_time);
        }
    }
    printf("全部文件最大误差/用户误差上限= %.9f\n",total_statistics.max_error_ratio);
    if(total_statistics.max_error_ratio>1.01)
        throw runtime_error("全部文件解压完成后误差检查失败：文件="+max_error_data_path+"，点="+to_string(total_statistics.max_error_point_id)+"，点集="+to_string(total_statistics.max_error_point_set_id)+"，误差="+to_string(total_statistics.max_error)+"，绝对误差上限="+to_string(total_statistics.max_error_abs_eb)+"，误差/上限="+to_string(total_statistics.max_error_ratio));
    printf("total time prediction state update time= %.9f s\n",total_time_prediction_state_update_time);
    if(param.prediction_data_uses_prediction_order)
        printf("total prediction order restore time= %.9f s\n",total_prediction_order_restore_time);
    printf("total decompression time= %f s\n",total_decompression_time);
    printf("UNSZ FIELD RESULT side=decompression field=%s slices=%u values=%llu valid_values=%llu compressed_bytes=%llu squared_error=%.17g valid_squared_error=%.17g slice_normalized_valid_squared_error=%.17g slice_normalized_squared_error=%.17g field_min=%.17g field_max=%.17g preprocessing_time=%.9f codec_time=%.9f max_error_over_eb=%.17g\n",param.field_name.c_str(),(uint32_t)param.original_data_path_vec.size(),(unsigned long long)total_statistics.point_num,(unsigned long long)total_statistics.valid_point_num,(unsigned long long)total_compressed_bytes,total_statistics.squared_error,total_statistics.valid_squared_error,total_slice_valid_normalized_squared_error,total_slice_normalized_squared_error,total_statistics.valid_minimum,total_statistics.valid_maximum,preprocessing_time,total_decompression_time,total_statistics.max_error_ratio);
    if(validated_file_num>0)
    {
        printf("total decompressed psnr= %f\n",unsz::reconstruction_psnr(total_statistics));
        printf("total squared error= %.17g, total value count= %llu\n",total_statistics.squared_error,(unsigned long long)total_statistics.point_num);
        printf("normalized squared error sum= %.17g, normalized value count= %llu, normalized psnr= %.17g, maximum error over eb= %.17g\n",total_normalized_squared_error,(unsigned long long)total_normalized_value_num,unsz::normalized_reconstruction_psnr(total_normalized_squared_error,total_normalized_value_num),total_statistics.max_error_ratio);
    }
    return total_decompression_time;
}

namespace unsz
{
void decompress_fields(param_c<float,float>&param)
{
    param.reported_decompression_preprocessing_time=0;
    param.reported_decompression_time=0;
    double temporal_dependency_plan_time=0;
    auto temporal_dependency_plan_start_time=chrono::steady_clock::now();
    if(param.tensor_field_config_map.empty())
    {
        if(param.temporal_dependency_plan_cache.empty())
            param.precompute_temporal_dependency_plans();
    }
    else if(param.tensor_dependency_plan_cache.empty())
        param.precompute_all_tensor_dependency_plans();
    temporal_dependency_plan_time=seconds_since(temporal_dependency_plan_start_time);
    printf("temporal dependency plan preprocessing time= %f s\n",temporal_dependency_plan_time);
    double total_preprocessing_time=temporal_dependency_plan_time,total_metadata_input_time=0;
    for(uint32_t field_id=0;field_id<param.field_name_vec.size();field_id++)
    {
        param.activate_field(field_id);
        printf("BEGIN UNSZ FIELD %s\n",param.field_name.c_str());
        param.read_coordinate_transform_file();
        auto [field_preprocessing_time,metadata_input_time]=prepare_current_decompression_field(param);
        total_metadata_input_time+=metadata_input_time;
        auto preprocessing_start_time=chrono::steady_clock::now();
        prepare_current_decompression_geometry(param);
        double preprocessing_time=field_preprocessing_time+seconds_since(preprocessing_start_time);
        total_preprocessing_time+=preprocessing_time;
        printf("preprocessing time excluding coordinate and transform-file input= %f s\n",preprocessing_time);
        param.reported_decompression_time+=run_decompression_field(param,preprocessing_time);
        param.release_current_spatial_prediction_state();
        printf("END UNSZ FIELD %s\n",param.field_name.c_str());
    }
    printf("total decompression preprocessing time excluding coordinate, transform-file and compressed-metadata input= %f s\n",total_preprocessing_time);
    printf("compressed metadata input time excluded from reported time= %f s\n",total_metadata_input_time);
    param.reported_decompression_preprocessing_time=total_preprocessing_time;
}

int decompress_main(int argc,char **argv)
{
    param_c<float,float> param;
    parse_common_args(argc,argv,param,[&](const string&option,auto&)
    {
        if(option=="-decompression-stage-time")
            param.print_compression_stage_time=true;
        else if(option=="-skip-decompressed-output")
            param.write_decompressed_output=false;
        else
            return false;
        return true;
    });
    param.init();
    if(param.field_worker_num==1)
        decompress_fields(param);
    else
        run_field_workers(param,"decompression",[](param_c<float,float>&worker_param){decompress_fields(worker_param);},[](const param_c<float,float>&worker_param){return worker_param.reported_decompression_preprocessing_time+worker_param.reported_decompression_time;});
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
