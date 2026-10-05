#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::preprocess_compression(chrono::steady_clock::time_point preprocessing_start_time)
    {
        double excluded_input_time=0;
        if(coordinate_transform_mode!="off")
        {
            auto coordinate_transform_start_time=chrono::steady_clock::now();
            vector<double> coordinate_covariance;
            double original_flatness=0,transformed_flatness=0,flatness_ratio=0;
            bool train_transform=true;
            if(coordinate_transform_mode=="auto")
            {
                coordinate_covariance=calculate_coordinate_covariance(coord_array,dim);
                original_flatness=calculate_coordinate_flatness(coordinate_covariance,dim);
                printf("coordinate transform original flatness=%.9f\n",original_flatness);
                if(original_flatness>=coordinate_transform_original_flatness_reject_threshold)
                {
                    train_transform=false;
                    use_coordinate_transform=false;
                    coordinate_transform_diagonal.assign(dim,1);
                    printf("coordinate transform rejected before training: original flatness >= %.3f\n",coordinate_transform_original_flatness_reject_threshold);
                }
            }
            if(train_transform)
            {
                auto training_input_start_time=chrono::steady_clock::now();
                prepare_coordinate_transform_training_data();
                excluded_input_time=chrono::duration<double>(chrono::steady_clock::now()-training_input_start_time).count();
                train_coordinate_transform();
                if(coordinate_transform_mode=="on")
                    use_coordinate_transform=true;
                else
                {
                    transformed_flatness=calculate_coordinate_flatness(coordinate_covariance,dim,&coordinate_transform_diagonal);
                    flatness_ratio=original_flatness>0?transformed_flatness/original_flatness:numeric_limits<double>::infinity();
                    use_coordinate_transform=original_flatness<=coordinate_transform_original_flatness_enable_threshold||flatness_ratio>=coordinate_transform_relative_flatness_enable_threshold;
                }
                if(coordinate_transform_mode=="auto")
                {
                    printf("coordinate transform transformed flatness=%.9f,ratio=%.9f,use=%u\n",transformed_flatness,flatness_ratio,(uint32_t)use_coordinate_transform);
                }
            }
            apply_coordinate_transform();
            write_coordinate_transform_file();
            printf("coordinate transform decision,training and application time excluding training-data input= %f s\n",chrono::duration<double>(chrono::steady_clock::now()-coordinate_transform_start_time).count()-excluded_input_time);
        }
        generate_prediction_structures_and_weights();
        for(uint32_t structure_id=0;structure_id<prediction_structure_vec.size();structure_id++)
        {
            prediction_structure_c<T_calc>&structure=prediction_structure_vec[structure_id];
            if(structure.eb_multiplier.size()!=structure.target_ids.size())
                structure.eb_multiplier=generate_pred_eb_multiplier_vec((uint32_t)structure.target_ids.size(),structure_id==0);
            structure.inverse_eb_multiplier.resize(structure.eb_multiplier.size());
            for(uint32_t prediction_index=0;prediction_index<structure.eb_multiplier.size();prediction_index++)
                structure.inverse_eb_multiplier[prediction_index]=1/structure.eb_multiplier[prediction_index];
        }
        remap_prediction_data_to_prediction_order();
        excluded_input_time+=select_automatic_flows();
        time_step_order_e label_order=get_time_step_order(label_flow),value_order=get_time_step_order(value_flow);
        if(point_set_config_vec.size()==1)
            prepare_time_step_order(value_order);
        else if(label_order==value_order)
            prepare_time_step_order(label_order);
        else if(label_flow=="none")
            prepare_time_step_order(value_order);
        else if(value_flow=="none")
            prepare_time_step_order(label_order);
        else
            prepare_time_step_order(time_step_order_e::chronological);
        double preprocessing_time=chrono::duration<double>(chrono::steady_clock::now()-preprocessing_start_time).count()-excluded_input_time;
        printf("preprocessing time excluding coordinate and field-data input= %f s\n",preprocessing_time);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::compress_files(void(param_c<T_coord,T_data>::*compress_file)())
    {
        prepare_temporal_dependencies();
        uint64_t total_original_bytes=0,total_compressed_bytes=0,total_label_stream_bytes=0,total_value_stream_bytes=0,total_shared_metadata_bytes=0;
        double total_compression_time=0,total_time_prediction_state_update_time=0;
        reconstruction_statistics_c total_statistics;
        string max_error_data_path;
        for(uint32_t file_id=0;file_id<original_data_path_vec.size();file_id++)
        {
            current_time_step_index=file_id;
            current_time_step=time_step_vec[file_id];
            auto &original_data_path=original_data_path_vec[file_id];
            printf("Compressing %s\n",original_data_path.c_str());
            compressed_data_path_vec.push_back(original_data_path+".unsz");
            decompressed_data_path_vec.push_back(original_data_path+".unsz.bin");
            current_data.resize(point_num);
            read_binary_file_exact(original_data_path,current_data.data(),point_num);
            auto compression_start_time=chrono::steady_clock::now();
            (this->*compress_file)();
            double compression_time=chrono::duration<double>(chrono::steady_clock::now()-compression_start_time).count();
            total_original_bytes+=(uint64_t)point_num*sizeof(T_data);
            total_compressed_bytes+=current_compressed_bytes.size();
            total_label_stream_bytes+=current_label_stream_bytes;
            total_value_stream_bytes+=current_value_stream_bytes;
            total_shared_metadata_bytes+=current_shared_metadata_bytes;
            reconstruction_statistics_c file_statistics=prediction_data_uses_prediction_order?calculate_prediction_order_reconstruction_statistics():calculate_reconstruction_statistics(current_data.data(),current_decompressed_data.data(),point_num);
            if(file_statistics.max_error_ratio>total_statistics.max_error_ratio)
                max_error_data_path=original_data_path;
            merge_reconstruction_statistics(total_statistics,file_statistics);
            double file_cr=(double)point_num*sizeof(T_data)/current_compressed_bytes.size();
            double file_br=sizeof(T_data)*8/file_cr;
            printf("cr= %f, br= %f, psnr= %f\n",file_cr,file_br,reconstruction_psnr(file_statistics));
            auto time_state_update_start=chrono::steady_clock::now();
            update_time_prediction_state();
            double time_state_update_time=chrono::duration<double>(chrono::steady_clock::now()-time_state_update_start).count();
            total_time_prediction_state_update_time+=time_state_update_time;
            compression_time+=time_state_update_time;
            total_compression_time+=compression_time;
            printf("time prediction state update time= %.9f s\n",time_state_update_time);
            printf("file compression time= %f s\n",compression_time);
            write_binary_file(compressed_data_path_vec.back(),current_compressed_bytes.data(),current_compressed_bytes.size());
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
    }

    template<point_set_codec_e predictor,typename T_coord,typename T_data>
    void compress_with_predictor(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
    {
        if(param.point_set_config_vec.size()==1)
        {
            param.resolve_single_point_set_time_mode();
            param.preprocess_compression(preprocessing_start_time);
            one_point_set::compress<predictor>(param);
        }
        else if(param.point_set_config_vec.size()==2)
        {
            param.preprocess_compression(preprocessing_start_time);
            two_point_sets::compress<predictor>(param);
        }
        else
            throw runtime_error("unsz::compress只支持一个或两个点集");
    }

    template<typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
    {
        auto automatic_flow_precomputation_start=chrono::steady_clock::now();
        param.precompute_automatic_flow_abs_eb();
        preprocessing_start_time+=chrono::steady_clock::now()-automatic_flow_precomputation_start;
        for(const point_set_config_c&config:param.point_set_config_vec)
        {
            if(config.codec==point_set_codec_e::constant)
                continue;
            if(config.codec!=point_set_codec_e::rbf)
                throw runtime_error("压缩只支持RBF空间预测器");
        }
        return compress_with_predictor<point_set_codec_e::rbf>(param,preprocessing_start_time);
    }
}
