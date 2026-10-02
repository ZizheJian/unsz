#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_joint_spatiotemporal_coordinates()
    {
        uint32_t spatial_point_num=point_num;
        uint64_t time_step_num=time_step_vec.size();
        uint64_t joint_point_num=(uint64_t)spatial_point_num*time_step_num;
        if(time_step_num<2)
            throw runtime_error("联合时空压缩至少需要两个time step");
        if(joint_point_num>numeric_limits<uint32_t>::max())
            throw runtime_error("联合时空点数超过uint32_t的表示范围");
        uint32_t spatial_dim=dim;
        dim++;
        Array2D<T_coord> joint_coord_array((uint32_t)joint_point_num,dim);
        for(uint32_t time_id=0;time_id<time_step_num;time_id++)
        {
            T_coord time_coord=time_step_coordinate_vec.empty()?(T_coord)time_step_vec[time_id]:(T_coord)time_step_coordinate_vec[time_step_vec[time_id]];
            uint64_t point_offset=(uint64_t)time_id*spatial_point_num;
            for(uint32_t spatial_point_id=0;spatial_point_id<spatial_point_num;spatial_point_id++)
            {
                T_coord*joint_coord=joint_coord_array[(uint32_t)(point_offset+spatial_point_id)];
                memcpy(joint_coord,coord_array[spatial_point_id],(uint64_t)spatial_dim*sizeof(T_coord));
                joint_coord[spatial_dim]=time_coord;
            }
        }
        coord_array=move(joint_coord_array);
        point_num=(uint32_t)joint_point_num;
        min_coord.assign(dim,numeric_limits<T_coord>::infinity());
        max_coord.assign(dim,-numeric_limits<T_coord>::infinity());
        for(uint32_t point_id=0;point_id<point_num;point_id++)
            for(uint32_t d=0;d<dim;d++)
            {
                min_coord[d]=min(min_coord[d],coord_array[point_id][d]);
                max_coord[d]=max(max_coord[d],coord_array[point_id][d]);
            }
        printf("Prepared joint spatiotemporal coordinates: spatial_points=%u,time_steps=%llu,joint_points=%u,dim=%u\n",spatial_point_num,(unsigned long long)time_step_num,point_num,dim);
    }

    template<typename T_coord,typename T_data>
    template<point_set_codec_e predictor>
    void param_c<T_coord,T_data>::compress_joint_spatiotemporal_files()
    {
        if(original_data_path_vec.empty()||point_num%original_data_path_vec.size()!=0)
            throw runtime_error("联合时空点数与time step文件数量不一致");
        uint32_t spatial_point_num=point_num/original_data_path_vec.size();
        current_data.resize(point_num);
        for(uint32_t file_id=0;file_id<original_data_path_vec.size();file_id++)
            read_binary_file_exact(original_data_path_vec[file_id],current_data.data()+(uint64_t)file_id*spatial_point_num,spatial_point_num);

        string compressed_path=data_dir+field_name+".joint.unsz";
        printf("Compressing %llu files as one joint spatiotemporal field: %s\n",(unsigned long long)original_data_path_vec.size(),compressed_path.c_str());
        auto compression_start_time=chrono::steady_clock::now();
        uint64_t compressed_byte_num=0;
        bool write_output=true;
        if(point_set_config_vec.size()==1)
            one_point_set::spatial::compress<predictor>(*this);
        else if(point_set_config_vec.size()==2&&label_value_execution=="fused")
            two_point_sets::fused::label_spatial_value_spatial::compress<predictor>(*this);
        else if(point_set_config_vec.size()==2&&label_value_execution=="staged")
        {
            if(prediction_structure_vec.size()!=1)
                throw runtime_error("staged调试路径尚不支持多个预测结构");
            two_point_sets::staged::label_spatial::compress<predictor>(*this);
            uint64_t label_byte_num=get_point_set_value_stream_offset(current_compressed_bytes,use_partition_adaptive_predictor_experimental);
            two_point_sets::staged::value_spatial::compress<predictor>(*this);
            uint64_t value_offset=get_point_set_value_stream_offset(current_compressed_bytes,use_partition_adaptive_predictor_experimental);
            compressed_byte_num=label_byte_num+current_compressed_bytes.size()-value_offset;
            write_output=false;
        }
        else
            throw runtime_error("联合时空压缩只支持一个点集，或staged/fused两个点集");
        double compression_time=chrono::duration<double>(chrono::steady_clock::now()-compression_start_time).count();
        if(write_output)
        {
            compressed_byte_num=current_compressed_bytes.size();
            compressed_data_path_vec.push_back(compressed_path);
            write_binary_file(compressed_path,current_compressed_bytes.data(),current_compressed_bytes.size());
        }

        reconstruction_statistics_c statistics=calculate_reconstruction_statistics(current_data.data(),current_decompressed_data.data(),point_num);
        printf("全部文件最大误差/用户误差上限= %.9f\n",statistics.max_error_ratio);
        if(statistics.max_error_ratio>1.01)
        {
            uint32_t time_id=statistics.max_error_point_id/spatial_point_num;
            uint32_t spatial_point_id=statistics.max_error_point_id%spatial_point_num;
            throw runtime_error("联合时空压缩误差检查失败：time step="+to_string(time_step_vec[time_id])+"，空间点="+to_string(spatial_point_id)+"，点集="+to_string(statistics.max_error_point_set_id)+"，误差="+to_string(statistics.max_error)+"，绝对误差上限="+to_string(statistics.max_error_abs_eb)+"，误差/上限="+to_string(statistics.max_error_ratio));
        }
        double cr=(double)point_num*sizeof(T_data)/compressed_byte_num;
        printf("total cr= %f, total br= %f, total psnr= %f\n",cr,sizeof(T_data)*8/cr,reconstruction_psnr(statistics));
        printf("total compression time= %f s\n",compression_time);
        if(!write_output)
            printf("staged mode estimates compression effect in memory and does not write compressed files\n");
    }

    template<typename T_coord,typename T_data>
    template<point_set_codec_e predictor>
    void param_c<T_coord,T_data>::decompress_joint_spatiotemporal_files()
    {
        if(original_data_path_vec.empty()||point_num%original_data_path_vec.size()!=0)
            throw runtime_error("联合时空点数与time step文件数量不一致");
        uint32_t spatial_point_num=point_num/original_data_path_vec.size();
        string compressed_path=data_dir+field_name+".joint.unsz";
        current_compressed_bytes=read_byte_file(compressed_path);
        auto decompression_start_time=chrono::steady_clock::now();
        if(point_set_config_vec.size()==1)
            one_point_set::spatial::decompress<predictor>(*this);
        else if(point_set_config_vec.size()==2&&label_value_execution=="fused")
            two_point_sets::fused::label_spatial_value_spatial::decompress<predictor>(*this);
        else
            throw runtime_error("联合时空解压只支持一个点集或fused两个点集");
        double decompression_time=chrono::duration<double>(chrono::steady_clock::now()-decompression_start_time).count();

        reconstruction_statistics_c total_statistics;
        string max_error_data_path;
        vector<T_data> original_data(spatial_point_num);
        for(uint32_t file_id=0;file_id<original_data_path_vec.size();file_id++)
        {
            uint32_t point_offset=file_id*spatial_point_num;
            string output_path=original_data_path_vec[file_id]+".unsz.bin";
            write_binary_file(output_path,current_decompressed_data.data()+point_offset,spatial_point_num);
            if(!try_read_binary_file_exact(original_data_path_vec[file_id],original_data.data(),spatial_point_num))
            {
                printf("Decompressed %s（原始文件不存在，跳过误差验证）\n",output_path.c_str());
                continue;
            }
            reconstruction_statistics_c file_statistics=calculate_reconstruction_statistics(original_data.data(),current_decompressed_data.data()+point_offset,spatial_point_num,point_offset);
            if(file_statistics.max_error_ratio>total_statistics.max_error_ratio)
                max_error_data_path=original_data_path_vec[file_id];
            merge_reconstruction_statistics(total_statistics,file_statistics);
            printf("Decompressed %s, psnr= %f\n",output_path.c_str(),reconstruction_psnr(file_statistics));
        }
        printf("全部文件最大误差/用户误差上限= %.9f\n",total_statistics.max_error_ratio);
        if(total_statistics.max_error_ratio>1.01)
        {
            uint32_t time_id=total_statistics.max_error_point_id/spatial_point_num;
            uint32_t spatial_point_id=total_statistics.max_error_point_id%spatial_point_num;
            throw runtime_error("联合时空解压误差检查失败：文件="+max_error_data_path+"，time step="+to_string(time_step_vec[time_id])+"，空间点="+to_string(spatial_point_id)+"，点集="+to_string(total_statistics.max_error_point_set_id)+"，误差="+to_string(total_statistics.max_error)+"，绝对误差上限="+to_string(total_statistics.max_error_abs_eb)+"，误差/上限="+to_string(total_statistics.max_error_ratio));
        }
        printf("total decompression time= %f s\n",decompression_time);
        if(total_statistics.point_num>0)
            printf("total decompressed psnr= %f\n",reconstruction_psnr(total_statistics));
    }
}
