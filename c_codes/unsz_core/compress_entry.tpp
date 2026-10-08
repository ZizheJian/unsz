#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    pair<double,double> param_c<T_coord,T_data>::preprocess_current_compression_geometry()
    {
        auto coordinate_transform_start_time=chrono::steady_clock::now();
        restore_original_coordinates();
        double coordinate_transform_time=0;
        double coordinate_transform_input_time=0;
        use_coordinate_transform=false;
        coordinate_transform_diagonal.assign(dim,1);
        bool uses_spatial_prediction=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c&config){return config.codec!=point_set_codec_e::constant;});
        if(coordinate_transform_mode!="off"&&uses_spatial_prediction)
        {
            const vector<double>*coordinate_covariance=nullptr;
            double original_flatness=0,transformed_flatness=0,flatness_ratio=0;
            bool train_transform=true;
            if(coordinate_transform_mode=="auto"||coordinate_transform_mode=="on")
            {
                if(original_coordinate_covariance.empty())
                {
                    original_coordinate_covariance=calculate_coordinate_covariance(coord_array,dim);
                    original_coordinate_flatness=calculate_coordinate_flatness(original_coordinate_covariance,dim);
                }
                coordinate_covariance=&original_coordinate_covariance;
                original_flatness=original_coordinate_flatness;
                printf("coordinate transform original flatness=%.9f\n",original_flatness);
                if(coordinate_transform_mode=="auto"&&original_flatness>=coordinate_transform_original_flatness_reject_threshold)
                {
                    train_transform=false;
                    use_coordinate_transform=false;
                    coordinate_transform_diagonal.assign(dim,1);
                    printf("coordinate transform rejected before training: original flatness >= %.3f\n",coordinate_transform_original_flatness_reject_threshold);
                }
            }
            if(train_transform)
            {
                prepare_coordinate_transform_training_data();
                coordinate_transform_input_time=coordinate_transform_training_input_time;
                auto coordinate_transform_optimization_start_time=chrono::steady_clock::now();
                train_coordinate_transform();
                printf("coordinate transform optimization time= %.6f s\n",chrono::duration<double>(chrono::steady_clock::now()-coordinate_transform_optimization_start_time).count());
                transformed_flatness=calculate_coordinate_flatness(*coordinate_covariance,dim,&coordinate_transform_diagonal);
                flatness_ratio=original_flatness>0?transformed_flatness/original_flatness:numeric_limits<double>::infinity();
                if(coordinate_transform_mode=="on")
                    use_coordinate_transform=true;
                if(coordinate_transform_mode=="auto")
                    use_coordinate_transform=original_flatness<=coordinate_transform_original_flatness_enable_threshold||flatness_ratio>=coordinate_transform_relative_flatness_enable_threshold;
                printf("coordinate transform transformed flatness=%.9f,ratio=%.9f,use=%u\n",transformed_flatness,flatness_ratio,(uint32_t)use_coordinate_transform);
            }
            apply_coordinate_transform();
        }
        double coordinate_transform_output_time=0;
        if(coordinate_transform_mode!="off")
        {
            auto coordinate_transform_output_start_time=chrono::steady_clock::now();
            write_coordinate_transform_file();
            coordinate_transform_output_time=chrono::duration<double>(chrono::steady_clock::now()-coordinate_transform_output_start_time).count();
        }
        coordinate_transform_time=chrono::duration<double>(chrono::steady_clock::now()-coordinate_transform_start_time).count()-coordinate_transform_input_time-coordinate_transform_output_time;
        if(coordinate_transform_mode!="off")
            printf("coordinate transform metadata output time excluded from reported time= %f s\n",coordinate_transform_output_time);
        auto prediction_structure_start_time=chrono::steady_clock::now();
        prepare_current_spatial_prediction_state(false);
        double prediction_structure_time=chrono::duration<double>(chrono::steady_clock::now()-prediction_structure_start_time).count();
        return {coordinate_transform_time,prediction_structure_time};
    }

    template<typename T_coord,typename T_data>
    pair<double,double> param_c<T_coord,T_data>::preprocess_current_compression_field()
    {
        double automatic_flow_time=select_automatic_flows();
        auto temporal_dependency_start_time=chrono::steady_clock::now();
        prepare_time_step_order_from_flows();
        prepare_temporal_dependencies();
        double temporal_dependency_time=chrono::duration<double>(chrono::steady_clock::now()-temporal_dependency_start_time).count();
        return {automatic_flow_time,temporal_dependency_time};
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::compress_files(void(param_c<T_coord,T_data>::*compress_file)())
    {
        uint64_t total_original_bytes=0,total_compressed_bytes=0,total_label_stream_bytes=0,total_value_stream_bytes=0,total_shared_metadata_bytes=0;
        double total_compression_time=0,total_time_prediction_state_update_time=0,total_data_input_time=0,total_compressed_output_time=0,total_normalized_squared_error=0;
        uint64_t total_normalized_value_num=0;
        reconstruction_statistics_c total_statistics;
        string max_error_data_path;
        for(uint32_t file_id=0;file_id<original_data_path_vec.size();file_id++)
        {
            current_time_step_index=file_id;
            current_time_step=time_step_vec[file_id];
            auto &original_data_path=original_data_path_vec[file_id];
            bool print_file_details=original_data_path_vec.size()<=32||print_compression_stage_time;
            if(print_file_details)
                printf("Compressing %s\n",original_data_path.c_str());
            else if(file_id%128==0)
                printf("Compression progress: %u/%u slices\n",file_id,(uint32_t)original_data_path_vec.size());
            string compressed_data_path=original_data_path+".unsz";
            current_data.resize(point_num);
            auto input_start_time=chrono::steady_clock::now();
            read_binary_file_exact(original_data_path,current_data.data(),point_num);
            total_data_input_time+=chrono::duration<double>(chrono::steady_clock::now()-input_start_time).count();
            auto compression_start_time=chrono::steady_clock::now();
            (this->*compress_file)();
            double compression_time=chrono::duration<double>(chrono::steady_clock::now()-compression_start_time).count();
            if(print_file_details)
                print_current_compression_diagnostics();
            total_original_bytes+=(uint64_t)point_num*sizeof(T_data);
            total_compressed_bytes+=current_compressed_bytes.size();
            total_label_stream_bytes+=current_label_stream_bytes;
            total_value_stream_bytes+=current_value_stream_bytes;
            total_shared_metadata_bytes+=current_shared_metadata_bytes;
            reconstruction_statistics_c file_statistics=prediction_data_uses_prediction_order?calculate_prediction_order_reconstruction_statistics():calculate_reconstruction_statistics(current_data.data(),current_decompressed_data.data(),point_num);
            if(file_statistics.max_error_ratio>total_statistics.max_error_ratio)
                max_error_data_path=original_data_path;
            merge_reconstruction_statistics(total_statistics,file_statistics);
            add_normalized_reconstruction_error(total_normalized_squared_error,total_normalized_value_num,file_statistics,point_set_point_num_vec,point_set_min_data_vec,point_set_max_data_vec);
            double file_cr=(double)point_num*sizeof(T_data)/current_compressed_bytes.size();
            double file_br=sizeof(T_data)*8/file_cr;
            if(print_file_details)
                printf("cr= %f, br= %f, psnr= %f\n",file_cr,file_br,reconstruction_psnr(file_statistics));
            auto time_state_update_start=chrono::steady_clock::now();
            update_time_prediction_state();
            double time_state_update_time=chrono::duration<double>(chrono::steady_clock::now()-time_state_update_start).count();
            total_time_prediction_state_update_time+=time_state_update_time;
            compression_time+=time_state_update_time;
            total_compression_time+=compression_time;
            if(print_file_details)
            {
                printf("time prediction state update time= %.9f s\n",time_state_update_time);
                printf("file compression time= %f s\n",compression_time);
            }
            auto output_start_time=chrono::steady_clock::now();
            write_binary_file(compressed_data_path,current_compressed_bytes.data(),current_compressed_bytes.size());
            total_compressed_output_time+=chrono::duration<double>(chrono::steady_clock::now()-output_start_time).count();
        }
        printf("全部文件最大误差/用户误差上限= %.9f\n",total_statistics.max_error_ratio);
        if(total_statistics.max_error_ratio>1.01)
            throw runtime_error("全部文件压缩完成后误差检查失败：文件="+max_error_data_path+"，点="+to_string(total_statistics.max_error_point_id)+"，点集="+to_string(total_statistics.max_error_point_set_id)+"，误差="+to_string(total_statistics.max_error)+"，绝对误差上限="+to_string(total_statistics.max_error_abs_eb)+"，误差/上限="+to_string(total_statistics.max_error_ratio));
        double total_cr=(double)total_original_bytes/total_compressed_bytes;
        double total_psnr=reconstruction_psnr(total_statistics);
        double total_br=sizeof(T_data)*8/total_cr;
        printf("total cr= %f, total br= %f, total psnr= %f\n",total_cr,total_br,total_psnr);
        printf("total label stream bytes= %llu, total value stream bytes= %llu, total shared metadata bytes= %llu\n",(unsigned long long)total_label_stream_bytes,(unsigned long long)total_value_stream_bytes,(unsigned long long)total_shared_metadata_bytes);
        printf("total label stream br= %.9f, total value stream br= %.9f, total shared metadata br= %.9f\n",(double)total_label_stream_bytes*8/(point_num*original_data_path_vec.size()),(double)total_value_stream_bytes*8/(point_num*original_data_path_vec.size()),(double)total_shared_metadata_bytes*8/(point_num*original_data_path_vec.size()));
        printf("total time prediction state update time= %.9f s\n",total_time_prediction_state_update_time);
        printf("total compression time= %f s\n",total_compression_time);
        printf("normalized squared error sum= %.17g, normalized value count= %llu, normalized psnr= %.17g, maximum error over eb= %.17g\n",total_normalized_squared_error,(unsigned long long)total_normalized_value_num,normalized_reconstruction_psnr(total_normalized_squared_error,total_normalized_value_num),total_statistics.max_error_ratio);
        printf("compression data input time excluded from reported time= %f s\n",total_data_input_time);
        printf("compressed output time excluded from reported time= %f s\n",total_compressed_output_time);
    }

    template<typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param)
    {
        for(const point_set_config_c&config:param.point_set_config_vec)
        {
            if(config.codec==point_set_codec_e::constant)
                continue;
            if(config.codec!=point_set_codec_e::rbf)
                throw runtime_error("压缩只支持RBF空间预测器");
        }
        if(param.point_set_config_vec.size()!=1&&param.point_set_config_vec.size()!=2)
            throw runtime_error("unsz::compress只支持一个或两个点集");
        double temporal_dependency_plan_time=0;
        auto temporal_dependency_plan_start_time=chrono::steady_clock::now();
        if(param.tensor_field_config_map.empty())
            param.precompute_temporal_dependency_plans();
        else
            param.precompute_all_tensor_dependency_plans();
        temporal_dependency_plan_time=chrono::duration<double>(chrono::steady_clock::now()-temporal_dependency_plan_start_time).count();
        printf("temporal dependency plan preprocessing time= %f s\n",temporal_dependency_plan_time);
        uint32_t field_num=(uint32_t)param.field_name_vec.size();
        param.point_set_statistics_input_time=0;
        double point_set_statistics_time=0,total_preprocessing_time=temporal_dependency_plan_time;
        for(uint32_t field_id=0;field_id<field_num;field_id++)
        {
            param.activate_field(field_id);
            auto point_set_statistics_start_time=chrono::steady_clock::now();
            param.precompute_current_field_point_set_statistics();
            point_set_statistics_time+=chrono::duration<double>(chrono::steady_clock::now()-point_set_statistics_start_time).count();
            if(field_num>1)
                printf("BEGIN UNSZ FIELD %s\n",param.field_name.c_str());
            auto [coordinate_transform_time,prediction_structure_time]=param.preprocess_current_compression_geometry();
            auto [automatic_flow_time,temporal_dependency_time]=param.preprocess_current_compression_field();
            bool has_predicted_value=any_of(param.point_set_config_vec.begin(),param.point_set_config_vec.end(),[](const point_set_config_c&config){return config.codec!=point_set_codec_e::constant;});
            if(param.current_field_uses_tensor)
                printf("selected tensor flow: %s\n",has_predicted_value?tensor_flow_name(param.selected_tensor_flow).c_str():"not_applicable");
            else
                printf("selected flow: label=%s,value=%s\n",param.label_flow.c_str(),has_predicted_value?param.value_flow.c_str():"not_applicable");
            printf("coordinate transform preprocessing time excluding training-data input= %f s\n",coordinate_transform_time);
            printf("prediction DAG and weights preprocessing time= %f s\n",prediction_structure_time);
            printf("automatic flow preprocessing time excluding field-data input= %f s\n",automatic_flow_time);
            printf("temporal dependency preprocessing time= %f s\n",temporal_dependency_time);
            double field_preprocessing_time=coordinate_transform_time+prediction_structure_time+automatic_flow_time+temporal_dependency_time;
            total_preprocessing_time+=field_preprocessing_time;
            printf("preprocessing time excluding coordinate and field-data input= %f s\n",field_preprocessing_time);
            if(param.point_set_config_vec.size()==1)
                one_point_set::compress(param);
            else
                two_point_sets::compress(param);
            param.release_current_spatial_prediction_state();
            if(field_num>1)
                printf("END UNSZ FIELD %s\n",param.field_name.c_str());
        }
        printf("compression file point-set statistics precomputation time excluded from reported time= %f s\n",point_set_statistics_time);
        printf("compression file point-set statistics input time excluded from reported time= %f s\n",param.point_set_statistics_input_time);
        printf("compression file point-set statistics computation time excluded from reported time= %f s\n",point_set_statistics_time-param.point_set_statistics_input_time);
        printf("total compression preprocessing time excluding coordinate and field-data input= %f s\n",total_preprocessing_time);
    }
}
