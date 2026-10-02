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
        }
        if(value_flow=="joint_spatiotemporal")
        {
            if(original_data_path_vec.empty()||point_num%original_data_path_vec.size()!=0)
                throw runtime_error("联合时空点数不能被time step数量整除");
            uint32_t spatial_point_num=point_num/original_data_path_vec.size();
            uint64_t cross_time_ref_num=0,total_ref_num=0,target_with_cross_time_ref_num=0;
            auto accumulate_cross_time_references=[&](const vector<int>&target_ids,const vector<int>&ref_ids,const vector<uint64_t>&ref_begin)
            {
                for(uint32_t prediction_index=0;prediction_index<target_ids.size();prediction_index++)
                {
                    int pred_id=target_ids[prediction_index];
                    bool has_cross_time_ref=false;
                    for(uint64_t ref_offset=ref_begin[prediction_index];ref_offset<ref_begin[prediction_index+1];ref_offset++)
                    {
                        int ref_id=ref_ids[ref_offset];
                        bool cross_time_ref=pred_id!=ref_id&&pred_id%spatial_point_num==ref_id%spatial_point_num;
                        cross_time_ref_num+=cross_time_ref;
                        total_ref_num++;
                        has_cross_time_ref|=cross_time_ref;
                    }
                    target_with_cross_time_ref_num+=has_cross_time_ref;
                }
            };
            for(const prediction_structure_c<T_calc>&structure:prediction_structure_vec)
                accumulate_cross_time_references(structure.target_ids,structure.reference_ids,structure.reference_offsets);
            double cross_time_edge_ratio=0;
            if(total_ref_num>0)
                cross_time_edge_ratio=(double)cross_time_ref_num/total_ref_num;
            printf("cross_time_reference_targets=%llu, target_ratio=%f, reference_edges=%llu, edge_ratio=%f\n",(unsigned long long)target_with_cross_time_ref_num,(double)target_with_cross_time_ref_num/point_num,(unsigned long long)cross_time_ref_num,cross_time_edge_ratio);
        }
        double preprocessing_time=chrono::duration<double>(chrono::steady_clock::now()-preprocessing_start_time).count()-adjust_coord_time;
        printf("preprocessing time excluding adjust coord= %f s\n",preprocessing_time);
        if(load_existing_dag_cache)
            validate_prediction_structures();
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::compress_files(void(param_c<T_coord,T_data>::*compress_file)(),bool write_output,vector<vector<unsigned char>> *compressed_stream_vec,bool validate_reconstruction)
    {
        if(compressed_stream_vec)
            compressed_stream_vec->clear();
        uint64_t total_original_bytes=0,total_compressed_bytes=0,total_point_num=0;
        double total_squared_error=0,total_compression_time=0,total_time_prediction_state_update_time=0,max_error_ratio=0,max_error=0,max_error_abs_eb=0;
        uint32_t max_error_point_id=0,max_error_point_set_id=0;
        string max_error_data_path;
        T_data total_min_data=numeric_limits<T_data>::infinity(),total_max_data=-numeric_limits<T_data>::infinity();
        for(uint32_t file_id=0;file_id<original_data_path_vec.size();file_id++)
        {
            current_time_step=time_step_vec[file_id];
            auto &original_data_path=original_data_path_vec[file_id];
            printf("Compressing %s\n",original_data_path.c_str());
            if(write_output)
            {
                compressed_data_path_vec.push_back(original_data_path+".unsz");
                decompressed_data_path_vec.push_back(original_data_path+".unsz.bin");
            }
            current_data.resize(point_num);
            FILE *f_data=fopen(original_data_path.c_str(),"rb");
            if(!f_data)
                throw runtime_error("找不到原始数据文件："+original_data_path);
            if(fread(current_data.data(),sizeof(T_data),point_num,f_data)!=point_num)
            {
                fclose(f_data);
                throw runtime_error("读取原始数据文件失败："+original_data_path);
            }
            if(fclose(f_data)!=0)
                throw runtime_error("关闭原始数据文件失败："+original_data_path);
            auto compression_start_time=chrono::steady_clock::now();
            (this->*compress_file)();
            double compression_time=chrono::duration<double>(chrono::steady_clock::now()-compression_start_time).count();
            total_original_bytes+=(uint64_t)point_num*sizeof(T_data);
            total_compressed_bytes+=current_compressed_bytes.size();
            total_point_num+=point_num;
            if(validate_reconstruction)
            {
                uint32_t primary_point_set_id=0;
                while(point_set_point_num_vec[primary_point_set_id]==0)
                    primary_point_set_id++;
                double file_squared_error=0;
                T_data file_min_data=numeric_limits<T_data>::infinity(),file_max_data=-numeric_limits<T_data>::infinity();
                for(uint32_t i=0;i<point_num;i++)
                {
                    double error=(double)current_data[i]-current_decompressed_data[i];
                    file_squared_error+=error*error;
                    total_squared_error+=error*error;
                    uint32_t point_set_id=point_set_id_vec.empty()?0:point_set_id_vec[i];
                    double validation_abs_eb=point_set_abs_eb_vec[point_set_id];
                    double error_ratio;
                    if(validation_abs_eb>0)
                        error_ratio=abs(error)/validation_abs_eb;
                    else if(error==0)
                        error_ratio=0;
                    else
                        error_ratio=numeric_limits<double>::infinity();
                    if(error_ratio>max_error_ratio)
                    {
                        max_error_ratio=error_ratio;
                        max_error=error;
                        max_error_abs_eb=validation_abs_eb;
                        max_error_point_id=i;
                        max_error_point_set_id=point_set_id;
                        max_error_data_path=original_data_path;
                    }
                    if(point_set_id==primary_point_set_id)
                    {
                        file_min_data=min(file_min_data,current_data[i]);
                        file_max_data=max(file_max_data,current_data[i]);
                        total_min_data=min(total_min_data,current_data[i]);
                        total_max_data=max(total_max_data,current_data[i]);
                    }
                }
                double file_cr=(double)point_num*sizeof(T_data)/current_compressed_bytes.size();
                double file_br=sizeof(T_data)*8/file_cr;
                double file_mse=file_squared_error/point_num;
                double file_psnr=file_mse==0?numeric_limits<double>::infinity():10*log10(pow((double)file_max_data-file_min_data,2)/file_mse);
                printf("cr= %f, br= %f, psnr= %f\n",file_cr,file_br,file_psnr);
            }
            auto time_state_update_start=chrono::steady_clock::now();
            update_time_prediction_state();
            double time_state_update_time=chrono::duration<double>(chrono::steady_clock::now()-time_state_update_start).count();
            total_time_prediction_state_update_time+=time_state_update_time;
            compression_time+=time_state_update_time;
            total_compression_time+=compression_time;
            printf("time prediction state update time= %.9f s\n",time_state_update_time);
            printf("file compression time= %f s\n",compression_time);
            if(compressed_stream_vec)
                compressed_stream_vec->push_back(current_compressed_bytes);
            if(write_output)
            {
                FILE *f_compressed=fopen(compressed_data_path_vec.back().c_str(),"wb");
                if(!f_compressed)
                    throw runtime_error("无法创建压缩文件："+compressed_data_path_vec.back());
                if(fwrite(current_compressed_bytes.data(),1,current_compressed_bytes.size(),f_compressed)!=current_compressed_bytes.size())
                {
                    fclose(f_compressed);
                    throw runtime_error("写入压缩文件失败："+compressed_data_path_vec.back());
                }
                if(fclose(f_compressed)!=0)
                    throw runtime_error("关闭压缩文件失败："+compressed_data_path_vec.back());
            }
        }
        last_total_compression_time=total_compression_time;
        if(validate_reconstruction)
        {
            printf("全部文件最大误差/用户误差上限= %.9f\n",max_error_ratio);
            if(max_error_ratio>1.01)
                throw runtime_error("全部文件压缩完成后误差检查失败：文件="+max_error_data_path+"，点="+to_string(max_error_point_id)+"，点集="+to_string(max_error_point_set_id)+"，误差="+to_string(max_error)+"，绝对误差上限="+to_string(max_error_abs_eb)+"，误差/上限="+to_string(max_error_ratio));
            double total_cr=(double)total_original_bytes/total_compressed_bytes;
            double total_mse=total_squared_error/total_point_num;
            double total_psnr=total_mse==0?numeric_limits<double>::infinity():10*log10(pow((double)total_max_data-total_min_data,2)/total_mse);
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
            if(param.value_flow=="joint_spatiotemporal")
                param.prepare_joint_spatiotemporal_coordinates();
            param.preprocess_compression(preprocessing_start_time);
            one_point_set::compress<predictor>(param);
        }
        else if(param.point_set_config_vec.size()==2)
        {
            if(param.label_value_execution!="staged"&&param.label_value_execution!="fused")
                throw runtime_error("two_point_sets只支持staged或fused执行方式");
            if(param.label_value_execution=="fused"&&param.value_flow=="joint_spatiotemporal")
                throw runtime_error("融合两点集尚未实现联合时空数值流程");
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
