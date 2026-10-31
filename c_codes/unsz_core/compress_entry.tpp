#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    pair<double,double> param_c<T_coord,T_data>::preprocess_current_compression_geometry()
    {
        auto coordinate_rescaling_start_time=chrono::steady_clock::now();
        restore_original_coordinates();
        double coordinate_rescaling_time=0;
        double coordinate_rescaling_input_time=0;
        use_coordinate_rescaling=false;
        coordinate_rescaling_diagonal.assign(dim,1);
        bool uses_spatial_prediction=any_of(label_config_vec.begin(),label_config_vec.end(),[](const label_config_c&config){return config.codec!=label_codec_e::constant;});
        // 坐标缩放的决策流程，阈值含义见coordinate_rescaling.tpp开头：
        // auto模式先算原始flatness，点云接近各向同性时直接关闭；否则训练缩放系数，再按原始flatness或缩放前后flatness之比决定是否启用。
        // on模式总是训练并启用，off模式不训练。
        if(coordinate_rescaling_mode!="off"&&uses_spatial_prediction)
        {
            const vector<double>*coordinate_covariance=nullptr;
            double original_flatness=0,rescaled_flatness=0,flatness_ratio=0;
            bool train_rescaling=true;
            if(coordinate_rescaling_mode=="auto"||coordinate_rescaling_mode=="on")
            {
                if(original_coordinate_covariance.empty())
                {
                    original_coordinate_covariance=calculate_coordinate_covariance(coord_array,dim);
                    original_coordinate_flatness=calculate_coordinate_flatness(original_coordinate_covariance,dim);
                }
                coordinate_covariance=&original_coordinate_covariance;
                original_flatness=original_coordinate_flatness;
                printf("coordinate rescaling original flatness=%.9f\n",original_flatness);
                if(coordinate_rescaling_mode=="auto"&&original_flatness>=coordinate_rescaling_original_flatness_reject_threshold)
                {
                    train_rescaling=false;
                    use_coordinate_rescaling=false;
                    coordinate_rescaling_diagonal.assign(dim,1);
                    printf("coordinate rescaling rejected before training: original flatness >= %.3f\n",coordinate_rescaling_original_flatness_reject_threshold);
                }
            }
            if(train_rescaling)
            {
                prepare_coordinate_rescaling_training_data();
                coordinate_rescaling_input_time=coordinate_rescaling_training_input_time;
                auto coordinate_rescaling_optimization_start_time=chrono::steady_clock::now();
                train_coordinate_rescaling();
                printf("coordinate rescaling optimization time= %.6f s\n",seconds_since(coordinate_rescaling_optimization_start_time));
                rescaled_flatness=calculate_coordinate_flatness(*coordinate_covariance,dim,&coordinate_rescaling_diagonal);
                flatness_ratio=original_flatness>0?rescaled_flatness/original_flatness:numeric_limits<double>::infinity();
                if(coordinate_rescaling_mode=="on")
                    use_coordinate_rescaling=true;
                if(coordinate_rescaling_mode=="auto")
                    use_coordinate_rescaling=original_flatness<=coordinate_rescaling_original_flatness_enable_threshold||flatness_ratio>=coordinate_rescaling_relative_flatness_enable_threshold;
                printf("coordinate rescaling rescaled flatness=%.9f,ratio=%.9f,use=%u\n",rescaled_flatness,flatness_ratio,(uint32_t)use_coordinate_rescaling);
            }
            apply_coordinate_rescaling();
        }
        double coordinate_rescaling_output_time=0;
        if(coordinate_rescaling_mode!="off")
        {
            auto coordinate_rescaling_output_start_time=chrono::steady_clock::now();
            write_coordinate_rescaling_file();
            coordinate_rescaling_output_time=seconds_since(coordinate_rescaling_output_start_time);
        }
        coordinate_rescaling_time=seconds_since(coordinate_rescaling_start_time)-coordinate_rescaling_input_time-coordinate_rescaling_output_time;
        if(coordinate_rescaling_mode!="off")
            printf("coordinate rescaling metadata output time excluded from reported time= %f s\n",coordinate_rescaling_output_time);
        auto prediction_structure_start_time=chrono::steady_clock::now();
        prepare_current_spatial_prediction_state(false);
        double prediction_structure_time=seconds_since(prediction_structure_start_time);
        return {coordinate_rescaling_time,prediction_structure_time};
    }

    template<typename T_coord,typename T_data>
    pair<double,double> param_c<T_coord,T_data>::preprocess_current_compression_field()
    {
        double automatic_pipeline_time=select_automatic_pipelines();
        auto reference_slices_start_time=chrono::steady_clock::now();
        prepare_slice_order_from_pipelines();
        prepare_reference_slices();
        double reference_slices_time=seconds_since(reference_slices_start_time);
        return {automatic_pipeline_time,reference_slices_time};
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::compress_files()
    {
        uint64_t total_original_bytes=0,total_compressed_bytes=0,total_label_stream_bytes=0,total_value_stream_bytes=0,total_shared_metadata_bytes=0;
        double total_compression_time=0,total_tensor_prediction_state_update_time=0,total_data_input_time=0,total_compressed_output_time=0,total_normalized_squared_error=0,total_slice_valid_normalized_squared_error=0,total_slice_normalized_squared_error=0;
        uint64_t total_normalized_value_num=0;
        reconstruction_statistics_c total_statistics;
        string max_error_data_path;
        for(uint32_t file_id=0;file_id<original_data_path_vec.size();file_id++)
        {
            current_slice_index=file_id;
            current_slice=slice_vec[file_id];
            auto &original_data_path=original_data_path_vec[file_id];
            bool print_file_details=original_data_path_vec.size()<=32||print_compression_stage_time;
            if(print_file_details)
                printf("Compressing %s\n",original_data_path.c_str());
            else if(file_id%128==0)
                printf("Compression progress: %u/%u slices\n",file_id,(uint32_t)original_data_path_vec.size());
            string compressed_data_path=generated_path(original_data_path,".unsz");
            current_data.resize(point_num);
            auto input_start_time=chrono::steady_clock::now();
            read_binary_file_exact(original_data_path,current_data.data(),point_num);
            total_data_input_time+=seconds_since(input_start_time);
            auto compression_start_time=chrono::steady_clock::now();
            compress_current_file_by_mode();
            double compression_time=seconds_since(compression_start_time);
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
            add_normalized_reconstruction_error(total_normalized_squared_error,total_normalized_value_num,file_statistics,label_point_num_vec,label_min_data_vec,label_max_data_vec);
            add_slice_normalized_error(total_slice_valid_normalized_squared_error,total_slice_normalized_squared_error,file_statistics);
            double file_cr=(double)point_num*sizeof(T_data)/current_compressed_bytes.size();
            double file_br=sizeof(T_data)*8/file_cr;
            if(print_file_details)
                printf("cr= %f, br= %f, psnr= %f\n",file_cr,file_br,reconstruction_psnr(file_statistics));
            auto tensor_state_update_start=chrono::steady_clock::now();
            update_tensor_prediction_state();
            double tensor_state_update_time=seconds_since(tensor_state_update_start);
            total_tensor_prediction_state_update_time+=tensor_state_update_time;
            compression_time+=tensor_state_update_time;
            total_compression_time+=compression_time;
            if(print_file_details)
            {
                printf("time prediction state update time= %.9f s\n",tensor_state_update_time);
                printf("file compression time= %f s\n",compression_time);
            }
            auto output_start_time=chrono::steady_clock::now();
            write_binary_file(compressed_data_path,current_compressed_bytes.data(),current_compressed_bytes.size());
            total_compressed_output_time+=seconds_since(output_start_time);
        }
        printf("全部文件最大误差/用户误差上限= %.9f\n",total_statistics.max_error_ratio);
        if(total_statistics.max_error_ratio>1.01)
            throw runtime_error("全部文件压缩完成后误差检查失败：文件="+max_error_data_path+"，点="+to_string(total_statistics.max_error_point_id)+"，标签="+to_string(total_statistics.max_error_label_id)+"，误差="+to_string(total_statistics.max_error)+"，绝对误差上限="+to_string(total_statistics.max_error_abs_eb)+"，误差/上限="+to_string(total_statistics.max_error_ratio));
        double total_cr=(double)total_original_bytes/total_compressed_bytes;
        double total_psnr=reconstruction_psnr(total_statistics);
        double total_br=sizeof(T_data)*8/total_cr;
        printf("total cr= %f, total br= %f, total psnr= %f\n",total_cr,total_br,total_psnr);
        printf("total squared error= %.17g, total value count= %llu\n",total_statistics.squared_error,(unsigned long long)total_statistics.point_num);
        printf("total label stream bytes= %llu, total value stream bytes= %llu, total shared metadata bytes= %llu\n",(unsigned long long)total_label_stream_bytes,(unsigned long long)total_value_stream_bytes,(unsigned long long)total_shared_metadata_bytes);
        printf("total label stream br= %.9f, total value stream br= %.9f, total shared metadata br= %.9f\n",(double)total_label_stream_bytes*8/(point_num*original_data_path_vec.size()),(double)total_value_stream_bytes*8/(point_num*original_data_path_vec.size()),(double)total_shared_metadata_bytes*8/(point_num*original_data_path_vec.size()));
        printf("total time prediction state update time= %.9f s\n",total_tensor_prediction_state_update_time);
        printf("total compression time= %f s\n",total_compression_time);
        printf("normalized squared error sum= %.17g, normalized value count= %llu, normalized psnr= %.17g, maximum error over eb= %.17g\n",total_normalized_squared_error,(unsigned long long)total_normalized_value_num,normalized_reconstruction_psnr(total_normalized_squared_error,total_normalized_value_num),total_statistics.max_error_ratio);
        printf("compression data input time excluded from reported time= %f s\n",total_data_input_time);
        printf("compressed output time excluded from reported time= %f s\n",total_compressed_output_time);
        printf("UNSZ FIELD RESULT side=compression field=%s slices=%u values=%llu valid_values=%llu compressed_bytes=%llu squared_error=%.17g valid_squared_error=%.17g slice_normalized_valid_squared_error=%.17g slice_normalized_squared_error=%.17g field_min=%.17g field_max=%.17g preprocessing_time=%.9f codec_time=%.9f max_error_over_eb=%.17g\n",field_name.c_str(),(uint32_t)original_data_path_vec.size(),(unsigned long long)total_statistics.point_num,(unsigned long long)total_statistics.valid_point_num,(unsigned long long)total_compressed_bytes,total_statistics.squared_error,total_statistics.valid_squared_error,total_slice_valid_normalized_squared_error,total_slice_normalized_squared_error,total_statistics.valid_minimum,total_statistics.valid_maximum,current_field_preprocessing_time,total_compression_time,total_statistics.max_error_ratio);
        reported_compression_time+=total_compression_time;
    }

    template<typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param)
    {
        param.reported_compression_preprocessing_time=0;
        param.reported_compression_time=0;
        for(const label_config_c&config:param.label_config_vec)
        {
            if(config.codec==label_codec_e::constant)
                continue;
            if(config.codec!=label_codec_e::rbf)
                throw runtime_error("压缩只支持RBF空间预测器");
        }
        if(param.label_config_vec.size()!=1&&param.label_config_vec.size()!=2)
            throw runtime_error("unsz::compress只支持一个或两个标签");
        double slice_dag_time=0;
        auto slice_dag_start_time=chrono::steady_clock::now();
        if(param.tensor_field_config_map.empty())
        {
            if(param.slice_dag_cache.empty())
                param.precompute_slice_dags();
        }
        else if(param.tensor_slice_dag_cache.empty())
            param.precompute_all_tensor_slice_dags();
        slice_dag_time=seconds_since(slice_dag_start_time);
        printf("slice DAG preprocessing time= %f s\n",slice_dag_time);
        uint32_t field_num=(uint32_t)param.field_name_vec.size();
        param.label_statistics_input_time=0;
        double label_statistics_time=0,total_preprocessing_time=slice_dag_time;
        for(uint32_t field_id=0;field_id<field_num;field_id++)
        {
            param.activate_field(field_id);
            auto label_statistics_start_time=chrono::steady_clock::now();
            param.precompute_current_field_label_statistics();
            label_statistics_time+=seconds_since(label_statistics_start_time);
            if(field_num>1||param.field_worker_num>1)
                printf("BEGIN UNSZ FIELD %s\n",param.field_name.c_str());
            auto [coordinate_rescaling_time,prediction_structure_time]=param.preprocess_current_compression_geometry();
            auto [automatic_pipeline_time,reference_slices_time]=param.preprocess_current_compression_field();
            bool has_predicted_value=any_of(param.label_config_vec.begin(),param.label_config_vec.end(),[](const label_config_c&config){return config.codec!=label_codec_e::constant;});
            if(param.current_field_uses_tensor)
                printf("selected tensor pipeline: %s\n",has_predicted_value?tensor_pipeline_name(param.selected_tensor_pipeline).c_str():"not_applicable");
            else
                printf("selected pipeline: label=%s,value=%s\n",param.label_pipeline.c_str(),has_predicted_value?param.value_pipeline.c_str():"not_applicable");
            printf("coordinate rescaling preprocessing time excluding training-data input= %f s\n",coordinate_rescaling_time);
            printf("prediction DAG and weights preprocessing time= %f s\n",prediction_structure_time);
            printf("automatic pipeline preprocessing time excluding field-data input= %f s\n",automatic_pipeline_time);
            printf("reference slice preprocessing time= %f s\n",reference_slices_time);
            double field_preprocessing_time=coordinate_rescaling_time+prediction_structure_time+automatic_pipeline_time+reference_slices_time;
            total_preprocessing_time+=field_preprocessing_time;
            printf("preprocessing time excluding coordinate and field-data input= %f s\n",field_preprocessing_time);
            param.current_field_preprocessing_time=field_preprocessing_time;
            param.check_label_value_slice_order();
            param.compress_files();
            param.release_current_spatial_prediction_state();
            if(field_num>1||param.field_worker_num>1)
                printf("END UNSZ FIELD %s\n",param.field_name.c_str());
        }
        printf("compression file label statistics precomputation time excluded from reported time= %f s\n",label_statistics_time);
        printf("compression file label statistics input time excluded from reported time= %f s\n",param.label_statistics_input_time);
        printf("compression file label statistics computation time excluded from reported time= %f s\n",label_statistics_time-param.label_statistics_input_time);
        printf("total compression preprocessing time excluding coordinate and field-data input= %f s\n",total_preprocessing_time);
        param.reported_compression_preprocessing_time=total_preprocessing_time;
    }
}
