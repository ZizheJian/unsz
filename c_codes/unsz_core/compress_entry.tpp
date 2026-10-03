#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::preprocess_compression(chrono::steady_clock::time_point preprocessing_start_time)
    {
        double adjust_coord_time=0;
        if(if_adjust_coord)
        {
            auto adjust_coord_start_time=chrono::steady_clock::now();
            adjust_coord();
            adjust_coord_time=chrono::duration<double>(chrono::steady_clock::now()-adjust_coord_start_time).count();
        }
        bool load_existing_dag_cache=!dag_cache_path.empty()&&filesystem::exists(dag_cache_path);
        load_or_generate_prediction_structures();
        if(load_existing_dag_cache&&any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::rbf;}))
            generate_rbf_weights_from_loaded_dag();
        for(uint32_t structure_id=0;structure_id<prediction_structure_vec.size();structure_id++)
        {
            prediction_structure_c<T_calc>&structure=prediction_structure_vec[structure_id];
            if(structure.eb_multiplier.size()!=structure.target_ids.size())
                structure.eb_multiplier=generate_pred_eb_multiplier_vec(structure.target_id_groups,(uint32_t)structure.target_ids.size(),structure_id==0);
            structure.inverse_eb_multiplier.resize(structure.eb_multiplier.size());
            for(uint32_t prediction_index=0;prediction_index<structure.eb_multiplier.size();prediction_index++)
                structure.inverse_eb_multiplier[prediction_index]=1/structure.eb_multiplier[prediction_index];
        }
        double validation_time=0;
        if(load_existing_dag_cache)
        {
            auto validation_start_time=chrono::steady_clock::now();
            validate_prediction_structures();
            validation_time=chrono::duration<double>(chrono::steady_clock::now()-validation_start_time).count();
        }
        remap_single_point_set_spatial_references_to_prediction_order();
        double preprocessing_time=chrono::duration<double>(chrono::steady_clock::now()-preprocessing_start_time).count()-adjust_coord_time-validation_time;
        printf("preprocessing time excluding coordinate input and adjust coord= %f s\n",preprocessing_time);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::compress_files(void(param_c<T_coord,T_data>::*compress_file)(),bool write_output,const function<void(uint32_t,const vector<unsigned char>&)>&compressed_stream_consumer,bool validate_reconstruction)
    {
        prepare_temporal_dependencies();
        uint64_t total_original_bytes=0,total_compressed_bytes=0;
        double total_compression_time=0,total_time_prediction_state_update_time=0;
        reconstruction_statistics_c total_statistics;
        string max_error_data_path;
        for(uint32_t file_id=0;file_id<original_data_path_vec.size();file_id++)
        {
            current_time_step_index=file_id;
            current_time_step=time_step_vec[file_id];
            auto &original_data_path=original_data_path_vec[file_id];
            printf("Compressing %s\n",original_data_path.c_str());
            if(write_output)
            {
                compressed_data_path_vec.push_back(original_data_path+".unsz");
                decompressed_data_path_vec.push_back(original_data_path+".unsz.bin");
            }
            current_data.resize(point_num);
            read_binary_file_exact(original_data_path,current_data.data(),point_num);
            auto compression_start_time=chrono::steady_clock::now();
            (this->*compress_file)();
            double compression_time=chrono::duration<double>(chrono::steady_clock::now()-compression_start_time).count();
            if(spatial_references_use_prediction_order)
            {
                const vector<int>&target_ids=prediction_structure_vec.front().target_ids;
                for(uint32_t prediction_index=0;prediction_index<point_num;prediction_index++)
                    current_decompressed_data[target_ids[prediction_index]]=prediction_order_decompressed_data[prediction_index];
            }
            total_original_bytes+=(uint64_t)point_num*sizeof(T_data);
            total_compressed_bytes+=current_compressed_bytes.size();
            if(validate_reconstruction)
            {
                reconstruction_statistics_c file_statistics=calculate_reconstruction_statistics(current_data.data(),current_decompressed_data.data(),point_num);
                if(file_statistics.max_error_ratio>total_statistics.max_error_ratio)
                    max_error_data_path=original_data_path;
                merge_reconstruction_statistics(total_statistics,file_statistics);
                double file_cr=(double)point_num*sizeof(T_data)/current_compressed_bytes.size();
                double file_br=sizeof(T_data)*8/file_cr;
                printf("cr= %f, br= %f, psnr= %f\n",file_cr,file_br,reconstruction_psnr(file_statistics));
            }
            auto time_state_update_start=chrono::steady_clock::now();
            update_time_prediction_state();
            double time_state_update_time=chrono::duration<double>(chrono::steady_clock::now()-time_state_update_start).count();
            total_time_prediction_state_update_time+=time_state_update_time;
            compression_time+=time_state_update_time;
            total_compression_time+=compression_time;
            printf("time prediction state update time= %.9f s\n",time_state_update_time);
            printf("file compression time= %f s\n",compression_time);
            if(compressed_stream_consumer)
                compressed_stream_consumer(file_id,current_compressed_bytes);
            if(write_output)
                write_binary_file(compressed_data_path_vec.back(),current_compressed_bytes.data(),current_compressed_bytes.size());
        }
        last_total_compression_time=total_compression_time;
        if(validate_reconstruction)
        {
            printf("全部文件最大误差/用户误差上限= %.9f\n",total_statistics.max_error_ratio);
            if(total_statistics.max_error_ratio>1.01)
                throw runtime_error("全部文件压缩完成后误差检查失败：文件="+max_error_data_path+"，点="+to_string(total_statistics.max_error_point_id)+"，点集="+to_string(total_statistics.max_error_point_set_id)+"，误差="+to_string(total_statistics.max_error)+"，绝对误差上限="+to_string(total_statistics.max_error_abs_eb)+"，误差/上限="+to_string(total_statistics.max_error_ratio));
            double total_cr=(double)total_original_bytes/total_compressed_bytes;
            double total_psnr=reconstruction_psnr(total_statistics);
            double total_br=sizeof(T_data)*8/total_cr;
            last_total_original_bytes=total_original_bytes;
            last_total_psnr=total_psnr;
            if(!suppress_compression_summary)
            {
                printf("total cr= %f, total br= %f, total psnr= %f\n",total_cr,total_br,total_psnr);
                printf("total time prediction state update time= %.9f s\n",total_time_prediction_state_update_time);
                printf("total compression time= %f s\n",total_compression_time);
            }
        }
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
            if(param.label_value_execution!="staged"&&param.label_value_execution!="fused")
                throw runtime_error("two_point_sets只支持staged或fused执行方式");
            param.preprocess_compression(preprocessing_start_time);
            two_point_sets::compress<predictor>(param);
        }
        else
            throw runtime_error("unsz::compress只支持一个或两个点集");
    }

    template<typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data>&param,chrono::steady_clock::time_point preprocessing_start_time)
    {
        point_set_codec_e predictor=point_set_codec_e::constant;
        for(const point_set_config_c&config:param.point_set_config_vec)
        {
            if(config.codec==point_set_codec_e::constant)
                continue;
            if(predictor!=point_set_codec_e::constant&&predictor!=config.codec)
                throw runtime_error("同一次压缩中的非常量点集必须使用同一种空间预测器");
            predictor=config.codec;
        }
        if(predictor==point_set_codec_e::idw)
            return compress_with_predictor<point_set_codec_e::idw>(param,preprocessing_start_time);
        if(predictor==point_set_codec_e::rbf)
            return compress_with_predictor<point_set_codec_e::rbf>(param,preprocessing_start_time);
        throw runtime_error("压缩至少需要一个使用IDW或RBF的非常量点集");
    }
}
