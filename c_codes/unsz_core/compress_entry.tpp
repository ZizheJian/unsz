#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_compression_structure(chrono::steady_clock::time_point preprocessing_start_time)
    {
        double adjust_coord_time=0;
        if(if_adjust_coord)
        {
            auto adjust_coord_start_time=chrono::steady_clock::now();
            adjust_coord();
            adjust_coord_time=chrono::duration<double>(chrono::steady_clock::now()-adjust_coord_start_time).count();
        }
        load_or_generate_pred_refs_structure();
        if(any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::rbf;}))
            prepare_rbf_prediction_weights();
        if(cached_pred_eb_multiplier_vec.size()!=point_num)
            cached_pred_eb_multiplier_vec=generate_pred_eb_multiplier_vec(true);
        if(value_flow=="joint_spatiotemporal")
        {
            if(original_data_path_vec.empty()||point_num%original_data_path_vec.size()!=0)
                throw runtime_error("联合时空点数不能被time step数量整除");
            uint32_t spatial_point_num=point_num/original_data_path_vec.size();
            uint64_t cross_time_ref_num=0,total_ref_num=0,target_with_cross_time_ref_num=0;
            for(uint32_t pred_ref_id=0;pred_ref_id<pred_target_ids.size();pred_ref_id++)
            {
                int pred_id=pred_target_ids[pred_ref_id];
                uint32_t pred_original_id=pred_id;
                bool has_cross_time_ref=false;
                for(int ref_id:get_pred_ref_ids(pred_ref_id))
                {
                    uint32_t ref_original_id=ref_id;
                    bool cross_time_ref=pred_original_id!=ref_original_id&&pred_original_id%spatial_point_num==ref_original_id%spatial_point_num;
                    cross_time_ref_num+=cross_time_ref;
                    total_ref_num++;
                    has_cross_time_ref|=cross_time_ref;
                }
                target_with_cross_time_ref_num+=has_cross_time_ref;
            }
            double cross_time_edge_ratio=0;
            if(total_ref_num>0)
                cross_time_edge_ratio=(double)cross_time_ref_num/total_ref_num;
            printf("cross_time_reference_targets=%llu, target_ratio=%f, reference_edges=%llu, edge_ratio=%f\n",(unsigned long long)target_with_cross_time_ref_num,(double)target_with_cross_time_ref_num/point_num,(unsigned long long)cross_time_ref_num,cross_time_edge_ratio);
        }
        if(print_prediction_dag_statistics)
            print_prediction_dag_statistics_result();
        double preprocessing_time=chrono::duration<double>(chrono::steady_clock::now()-preprocessing_start_time).count()-adjust_coord_time;
        printf("preprocessing time excluding adjust coord= %f s\n",preprocessing_time);
        validate_pred_refs_structure();
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::compress_files(void(param_c<T_coord,T_data>::*compress_current_file)())
    {
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
            compressed_data_path_vec.push_back(original_data_path+".unsz");
            decompressed_data_path_vec.push_back(original_data_path+".unsz.bin");
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
            (this->*compress_current_file)();
            double compression_time=chrono::duration<double>(chrono::steady_clock::now()-compression_start_time).count();
            total_original_bytes+=(uint64_t)point_num*sizeof(T_data);
            total_compressed_bytes+=current_zstd_bs_vec.size();
            total_point_num+=point_num;
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
                uint32_t point_set_id=point_set_id_vec[i];
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
                if(point_set_id_vec[i]==primary_point_set_id)
                {
                    file_min_data=min(file_min_data,current_data[i]);
                    file_max_data=max(file_max_data,current_data[i]);
                    total_min_data=min(total_min_data,current_data[i]);
                    total_max_data=max(total_max_data,current_data[i]);
                }
            }
            double file_cr=(double)point_num*sizeof(T_data)/current_zstd_bs_vec.size();
            double file_br=sizeof(T_data)*8/file_cr;
            double file_mse=file_squared_error/point_num;
            double file_psnr=file_mse==0?numeric_limits<double>::infinity():10*log10(pow((double)file_max_data-file_min_data,2)/file_mse);
            printf("cr= %f, br= %f, psnr= %f\n",file_cr,file_br,file_psnr);
            auto time_state_update_start=chrono::steady_clock::now();
            update_time_prediction_state();
            double time_state_update_time=chrono::duration<double>(chrono::steady_clock::now()-time_state_update_start).count();
            total_time_prediction_state_update_time+=time_state_update_time;
            compression_time+=time_state_update_time;
            total_compression_time+=compression_time;
            printf("time prediction state update time= %.9f s\n",time_state_update_time);
            printf("file compression time= %f s\n",compression_time);
            FILE *f_compressed=fopen(compressed_data_path_vec.back().c_str(),"wb");
            if(!f_compressed)
                throw runtime_error("无法创建压缩文件："+compressed_data_path_vec.back());
            if(fwrite(current_zstd_bs_vec.data(),1,current_zstd_bs_vec.size(),f_compressed)!=current_zstd_bs_vec.size())
            {
                fclose(f_compressed);
                throw runtime_error("写入压缩文件失败："+compressed_data_path_vec.back());
            }
            if(fclose(f_compressed)!=0)
                throw runtime_error("关闭压缩文件失败："+compressed_data_path_vec.back());
        }
        printf("全部文件最大误差/用户误差上限= %.9f\n",max_error_ratio);
        if(max_error_ratio>1.01)
            throw runtime_error("全部文件压缩完成后误差检查失败：文件="+max_error_data_path+"，点="+to_string(max_error_point_id)+"，点集="+to_string(max_error_point_set_id)+"，误差="+to_string(max_error)+"，绝对误差上限="+to_string(max_error_abs_eb)+"，误差/上限="+to_string(max_error_ratio));
        double total_cr=(double)total_original_bytes/total_compressed_bytes;
        double total_mse=total_squared_error/total_point_num;
        double total_psnr=total_mse==0?numeric_limits<double>::infinity():10*log10(pow((double)total_max_data-total_min_data,2)/total_mse);
        double total_br=sizeof(T_data)*8/total_cr;
        last_total_original_bytes=total_original_bytes;
        last_total_psnr=total_psnr;
        last_total_compression_time=total_compression_time;
        if(!suppress_compression_summary)
        {
            printf("total cr= %f, total br= %f, total psnr= %f\n",total_cr,total_br,total_psnr);
            printf("total time prediction state update time= %.9f s\n",total_time_prediction_state_update_time);
            printf("total compression time= %f s\n",total_compression_time);
        }
    }

    template<typename T_coord,typename T_data>
    void compress(param_c<T_coord,T_data> &param,chrono::steady_clock::time_point preprocessing_start_time)
    {
        if(param.point_set_config_vec.size()==1)
            one_point_set::compress(param,preprocessing_start_time);
        else if(param.point_set_config_vec.size()==2)
            two_point_sets::compress(param,preprocessing_start_time);
        else
            throw runtime_error("unsz::compress只支持一个或两个点集");
    }
}
