#pragma once

namespace unsz
{
template<typename T_coord,typename T_data>
void param_c<T_coord,T_data>::prepare_compression_structure(chrono::steady_clock::time_point preprocessing_start_time)
{
    double adjust_coord_time=0;
    if (if_adjust_coord)
    {
        auto adjust_coord_start_time=chrono::steady_clock::now();
        adjust_coord();
        adjust_coord_time=chrono::duration<double>(chrono::steady_clock::now()-adjust_coord_start_time).count();
    }
    load_or_generate_pred_refs_structure();
    if(any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::rbf;}))
        prepare_rbf_weights();
    bool data_dependent_eb_schedule=eb_schedule=="round_point_ratio_dag" || eb_schedule=="round_point_ratio_dag_direct" || eb_schedule=="round_point_ratio_dag_variance_direct";
    if(!data_dependent_eb_schedule && cached_pred_eb_multiplier_vec.size()!=point_num)
        cached_pred_eb_multiplier_vec=generate_pred_eb_multiplier_vec(true);
    if(value_flow=="joint_spatiotemporal")
    {
        if(original_data_path_vec.empty()||point_num%original_data_path_vec.size()!=0)
            throw runtime_error("联合时空点数不能被time step数量整除");
        uint32_t spatial_point_num=point_num/original_data_path_vec.size();
        uint64_t cross_time_ref_num=0,total_ref_num=0,target_with_cross_time_ref_num=0;
        for (uint32_t pred_ref_id=0;pred_ref_id<pred_target_ids.size();pred_ref_id++)
        {
            int pred_id=pred_target_ids[pred_ref_id];
            uint32_t pred_original_id=pred_id;
            bool has_cross_time_ref=false;
            for (int ref_id:get_pred_ref_ids(pred_ref_id))
            {
                uint32_t ref_original_id=ref_id;
                bool cross_time_ref=pred_original_id!=ref_original_id && pred_original_id%spatial_point_num==ref_original_id%spatial_point_num;
                cross_time_ref_num+=cross_time_ref;
                total_ref_num++;
                has_cross_time_ref|=cross_time_ref;
            }
            target_with_cross_time_ref_num+=has_cross_time_ref;
        }
        printf("cross_time_reference_targets=%llu, target_ratio=%f, reference_edges=%llu, edge_ratio=%f\n",(unsigned long long)target_with_cross_time_ref_num,(double)target_with_cross_time_ref_num/point_num,(unsigned long long)cross_time_ref_num,total_ref_num==0?0:(double)cross_time_ref_num/total_ref_num);
    }
    if (print_prediction_dag_statistics)
        print_prediction_dag_statistics_result();
    double preprocessing_time=chrono::duration<double>(chrono::steady_clock::now()-preprocessing_start_time).count()-adjust_coord_time;
    printf("preprocessing time excluding adjust coord= %f s\n",preprocessing_time);
}

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
    uint32_t spatial_dim=dim,joint_dim=dim+1;
    Array2D<T_coord> joint_coord_array((uint32_t)joint_point_num,joint_dim);
    for(uint32_t time_id=0;time_id<time_step_num;time_id++)
    {
        T_coord time_coord=time_step_coordinate_vec.empty()?(T_coord)time_step_vec[time_id]:(T_coord)time_step_coordinate_vec[time_step_vec[time_id]];
        uint64_t point_offset=(uint64_t)time_id*spatial_point_num;
        for(uint32_t spatial_point_id=0;spatial_point_id<spatial_point_num;spatial_point_id++)
        {
            T_coord *joint_coord=joint_coord_array[(uint32_t)(point_offset+spatial_point_id)];
            memcpy(joint_coord,coord_array[spatial_point_id],(uint64_t)spatial_dim*sizeof(T_coord));
            joint_coord[spatial_dim]=time_coord;
        }
    }
    coord_array=move(joint_coord_array);
    point_num=(uint32_t)joint_point_num;
    dim=joint_dim;
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
void param_c<T_coord,T_data>::compress_files(void(param_c<T_coord,T_data>::*compress_current_file)())
{
    uint64_t total_original_bytes=0,total_compressed_bytes=0,total_point_num=0;
    double total_squared_error=0,total_compression_time=0,max_error_ratio=0,max_error=0,max_error_abs_eb=0;
    uint32_t max_error_point_id=0,max_error_point_set_id=0;
    string max_error_data_path;
    T_data total_min_data=numeric_limits<T_data>::infinity(),total_max_data=-numeric_limits<T_data>::infinity();
    for (uint32_t file_id=0;file_id<original_data_path_vec.size();file_id++)
    {
        current_time_step=time_step_vec[file_id];
        auto &original_data_path=original_data_path_vec[file_id];
        printf("Compressing %s\n",original_data_path.c_str());
        compressed_data_path_vec.push_back(original_data_path+".unsz");
        decompressed_data_path_vec.push_back(original_data_path+".unsz.bin");
        current_data.resize(point_num);
        FILE *f_data=fopen(original_data_path.c_str(),"rb");
        if (!f_data)
            throw runtime_error("找不到原始数据文件："+original_data_path);
        if (fread(current_data.data(),sizeof(T_data),point_num,f_data)!=point_num)
        {
            fclose(f_data);
            throw runtime_error("读取原始数据文件失败："+original_data_path);
        }
        if (fclose(f_data)!=0)
            throw runtime_error("关闭原始数据文件失败："+original_data_path);
        auto compression_start_time=chrono::steady_clock::now();
        (this->*compress_current_file)();
        double compression_time=chrono::duration<double>(chrono::steady_clock::now()-compression_start_time).count();
        total_compression_time+=compression_time;
        printf("file compression time= %f s\n",compression_time);
        save_time_reconstruction();
        total_original_bytes+=(uint64_t)point_num*sizeof(T_data);
        total_compressed_bytes+=current_zstd_bs_vec.size();
        total_point_num+=point_num;
        uint32_t primary_point_set_id=0;
        while (point_set_point_num_vec[primary_point_set_id]==0)
            primary_point_set_id++;
        double file_squared_error=0;
        T_data file_min_data=numeric_limits<T_data>::infinity(),file_max_data=-numeric_limits<T_data>::infinity();
        for (uint32_t i=0;i<point_num;i++)
        {
            double error=(double)current_data[i]-current_decompressed_data[i];
            file_squared_error+=error*error;
            total_squared_error+=error*error;
            uint32_t point_set_id=point_set_id_vec[i];
            double validation_abs_eb=point_set_abs_eb_vec[point_set_id];
            double error_ratio=validation_abs_eb>0?abs(error)/validation_abs_eb:(error==0?0:numeric_limits<double>::infinity());
            if (error_ratio>max_error_ratio)
            {
                max_error_ratio=error_ratio;
                max_error=error;
                max_error_abs_eb=validation_abs_eb;
                max_error_point_id=i;
                max_error_point_set_id=point_set_id;
                max_error_data_path=original_data_path;
            }
            if (point_set_id_vec[i]==primary_point_set_id)
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
        FILE *f_compressed=fopen(compressed_data_path_vec.back().c_str(),"wb");
        if (!f_compressed)
            throw runtime_error("无法创建压缩文件："+compressed_data_path_vec.back());
        if (fwrite(current_zstd_bs_vec.data(),1,current_zstd_bs_vec.size(),f_compressed)!=current_zstd_bs_vec.size())
        {
            fclose(f_compressed);
            throw runtime_error("写入压缩文件失败："+compressed_data_path_vec.back());
        }
        if (fclose(f_compressed)!=0)
            throw runtime_error("关闭压缩文件失败："+compressed_data_path_vec.back());
        advance_time_prediction_buffers();
    }
    printf("全部文件最大误差/用户误差上限= %.9f\n",max_error_ratio);
    if (max_error_ratio>1.01)
        throw runtime_error("全部文件压缩完成后误差检查失败：文件="+max_error_data_path+"，点="+to_string(max_error_point_id)+"，点集="+to_string(max_error_point_set_id)+"，误差="+to_string(max_error)+"，绝对误差上限="+to_string(max_error_abs_eb)+"，误差/上限="+to_string(max_error_ratio));
    double total_cr=(double)total_original_bytes/total_compressed_bytes;
    double total_mse=total_squared_error/total_point_num;
    double total_psnr=total_mse==0?numeric_limits<double>::infinity():10*log10(pow((double)total_max_data-total_min_data,2)/total_mse);
    double total_br=sizeof(T_data)*8/total_cr;
    printf("total cr= %f, total br= %f, total psnr= %f\n",total_cr,total_br,total_psnr);
    printf("total compression time= %f s\n",total_compression_time);

}

template<typename T_coord,typename T_data>
void param_c<T_coord,T_data>::compress_joint_spatiotemporal_files(void(param_c<T_coord,T_data>::*compress_current_field)())
{
    if(original_data_path_vec.empty()||point_num%original_data_path_vec.size()!=0)
        throw runtime_error("联合时空点数与time step文件数量不一致");
    uint32_t spatial_point_num=point_num/original_data_path_vec.size();
    current_data.resize(point_num);
    for(uint32_t file_id=0;file_id<original_data_path_vec.size();file_id++)
    {
        const string &path=original_data_path_vec[file_id];
        FILE *file=fopen(path.c_str(),"rb");
        if(!file)
            throw runtime_error("找不到原始数据文件："+path);
        T_data *output=current_data.data()+(uint64_t)file_id*spatial_point_num;
        if(fread(output,sizeof(T_data),spatial_point_num,file)!=spatial_point_num)
        {
            fclose(file);
            throw runtime_error("读取原始数据文件失败："+path);
        }
        if(fclose(file)!=0)
            throw runtime_error("关闭原始数据文件失败："+path);
    }
    string compressed_path=data_dir+field_name+".joint.unsz";
    compressed_data_path_vec.push_back(compressed_path);
    printf("Compressing %llu files as one joint spatiotemporal field: %s\n",(unsigned long long)original_data_path_vec.size(),compressed_path.c_str());
    auto compression_start_time=chrono::steady_clock::now();
    (this->*compress_current_field)();
    double compression_time=chrono::duration<double>(chrono::steady_clock::now()-compression_start_time).count();
    uint32_t primary_point_set_id=0;
    while(point_set_point_num_vec[primary_point_set_id]==0)
        primary_point_set_id++;
    double squared_error=0,max_error_ratio=0,max_error=0,max_error_abs_eb=0;
    uint32_t max_error_point_id=0,max_error_point_set_id=0;
    T_data minimum=numeric_limits<T_data>::infinity(),maximum=-numeric_limits<T_data>::infinity();
    for(uint32_t point_id=0;point_id<point_num;point_id++)
    {
        double error=(double)current_data[point_id]-current_decompressed_data[point_id];
        squared_error+=error*error;
        uint32_t point_set_id=point_set_id_vec[point_id];
        double validation_abs_eb=point_set_abs_eb_vec[point_set_id];
        double error_ratio=validation_abs_eb>0?abs(error)/validation_abs_eb:(error==0?0:numeric_limits<double>::infinity());
        if(error_ratio>max_error_ratio)
        {
            max_error_ratio=error_ratio;
            max_error=error;
            max_error_abs_eb=validation_abs_eb;
            max_error_point_id=point_id;
            max_error_point_set_id=point_set_id;
        }
        if(point_set_id==primary_point_set_id)
        {
            minimum=min(minimum,current_data[point_id]);
            maximum=max(maximum,current_data[point_id]);
        }
    }
    printf("全部文件最大误差/用户误差上限= %.9f\n",max_error_ratio);
    if(max_error_ratio>1.01)
    {
        uint32_t time_id=max_error_point_id/spatial_point_num,spatial_point_id=max_error_point_id%spatial_point_num;
        throw runtime_error("联合时空压缩误差检查失败：time step="+to_string(time_step_vec[time_id])+"，空间点="+to_string(spatial_point_id)+"，点集="+to_string(max_error_point_set_id)+"，误差="+to_string(max_error)+"，绝对误差上限="+to_string(max_error_abs_eb)+"，误差/上限="+to_string(max_error_ratio));
    }
    FILE *file=fopen(compressed_path.c_str(),"wb");
    if(!file)
        throw runtime_error("无法创建联合时空压缩文件："+compressed_path);
    if(fwrite(current_zstd_bs_vec.data(),1,current_zstd_bs_vec.size(),file)!=current_zstd_bs_vec.size())
    {
        fclose(file);
        throw runtime_error("写入联合时空压缩文件失败："+compressed_path);
    }
    if(fclose(file)!=0)
        throw runtime_error("关闭联合时空压缩文件失败："+compressed_path);
    double cr=(double)point_num*sizeof(T_data)/current_zstd_bs_vec.size();
    double mse=squared_error/point_num;
    double psnr=mse==0?numeric_limits<double>::infinity():10*log10(pow((double)maximum-minimum,2)/mse);
    printf("total cr= %f, total br= %f, total psnr= %f\n",cr,sizeof(T_data)*8/cr,psnr);
    printf("total compression time= %f s\n",compression_time);
}

template<typename T_coord,typename T_data>
void param_c<T_coord,T_data>::decompress_joint_spatiotemporal_files(void(param_c<T_coord,T_data>::*decompress_current_field)())
{
    if(original_data_path_vec.empty()||point_num%original_data_path_vec.size()!=0)
        throw runtime_error("联合时空点数与time step文件数量不一致");
    uint32_t spatial_point_num=point_num/original_data_path_vec.size();
    string compressed_path=data_dir+field_name+".joint.unsz";
    FILE *compressed_file=fopen(compressed_path.c_str(),"rb");
    if(!compressed_file)
        throw runtime_error("找不到联合时空压缩文件："+compressed_path);
    if(fseek(compressed_file,0,SEEK_END)!=0)
    {
        fclose(compressed_file);
        throw runtime_error("无法定位到联合时空压缩文件末尾："+compressed_path);
    }
    long compressed_byte_num_signed=ftell(compressed_file);
    if(compressed_byte_num_signed<0)
    {
        fclose(compressed_file);
        throw runtime_error("无法读取联合时空压缩文件大小："+compressed_path);
    }
    current_zstd_bs_vec.resize((uint64_t)compressed_byte_num_signed);
    if(fseek(compressed_file,0,SEEK_SET)!=0||fread(current_zstd_bs_vec.data(),1,current_zstd_bs_vec.size(),compressed_file)!=current_zstd_bs_vec.size())
    {
        fclose(compressed_file);
        throw runtime_error("读取联合时空压缩文件失败："+compressed_path);
    }
    if(fclose(compressed_file)!=0)
        throw runtime_error("关闭联合时空压缩文件失败："+compressed_path);
    auto decompression_start_time=chrono::steady_clock::now();
    (this->*decompress_current_field)();
    double decompression_time=chrono::duration<double>(chrono::steady_clock::now()-decompression_start_time).count();
    double squared_error=0,max_error_ratio=0,max_error=0,max_error_abs_eb=0;
    uint64_t validated_point_num=0;
    uint32_t max_error_joint_point_id=0,max_error_point_set_id=0;
    string max_error_data_path;
    T_data minimum=numeric_limits<T_data>::infinity(),maximum=-numeric_limits<T_data>::infinity();
    for(uint32_t file_id=0;file_id<original_data_path_vec.size();file_id++)
    {
        uint64_t point_offset=(uint64_t)file_id*spatial_point_num;
        string output_path=original_data_path_vec[file_id]+".unsz.bin";
        FILE *output_file=fopen(output_path.c_str(),"wb");
        if(!output_file)
            throw runtime_error("无法创建解压文件："+output_path);
        if(fwrite(current_decompressed_data.data()+point_offset,sizeof(T_data),spatial_point_num,output_file)!=spatial_point_num)
        {
            fclose(output_file);
            throw runtime_error("写入解压文件失败："+output_path);
        }
        if(fclose(output_file)!=0)
            throw runtime_error("关闭解压文件失败："+output_path);
        FILE *original_file=fopen(original_data_path_vec[file_id].c_str(),"rb");
        if(!original_file)
        {
            printf("Decompressed %s（原始文件不存在，跳过误差验证）\n",output_path.c_str());
            continue;
        }
        vector<T_data> original_data(spatial_point_num);
        if(fread(original_data.data(),sizeof(T_data),spatial_point_num,original_file)!=spatial_point_num)
        {
            fclose(original_file);
            throw runtime_error("读取原始文件失败："+original_data_path_vec[file_id]);
        }
        if(fclose(original_file)!=0)
            throw runtime_error("关闭原始文件失败："+original_data_path_vec[file_id]);
        for(uint32_t spatial_point_id=0;spatial_point_id<spatial_point_num;spatial_point_id++)
        {
            uint32_t joint_point_id=(uint32_t)(point_offset+spatial_point_id);
            double error=(double)original_data[spatial_point_id]-current_decompressed_data[joint_point_id];
            squared_error+=error*error;
            uint32_t point_set_id=point_set_id_vec[joint_point_id];
            double validation_abs_eb=point_set_abs_eb_vec[point_set_id];
            double error_ratio=validation_abs_eb>0?abs(error)/validation_abs_eb:(error==0?0:numeric_limits<double>::infinity());
            if(error_ratio>max_error_ratio)
            {
                max_error_ratio=error_ratio;
                max_error=error;
                max_error_abs_eb=validation_abs_eb;
                max_error_joint_point_id=joint_point_id;
                max_error_point_set_id=point_set_id;
                max_error_data_path=original_data_path_vec[file_id];
            }
            minimum=min(minimum,original_data[spatial_point_id]);
            maximum=max(maximum,original_data[spatial_point_id]);
        }
        validated_point_num+=spatial_point_num;
        printf("Decompressed %s\n",output_path.c_str());
    }
    printf("全部文件最大误差/用户误差上限= %.9f\n",max_error_ratio);
    if(max_error_ratio>1.01)
    {
        uint32_t time_id=max_error_joint_point_id/spatial_point_num,spatial_point_id=max_error_joint_point_id%spatial_point_num;
        throw runtime_error("联合时空解压误差检查失败：文件="+max_error_data_path+"，time step="+to_string(time_step_vec[time_id])+"，空间点="+to_string(spatial_point_id)+"，点集="+to_string(max_error_point_set_id)+"，误差="+to_string(max_error)+"，绝对误差上限="+to_string(max_error_abs_eb)+"，误差/上限="+to_string(max_error_ratio));
    }
    printf("total decompression time= %f s\n",decompression_time);
    if(validated_point_num>0)
    {
        double mse=squared_error/validated_point_num;
        double psnr=mse==0?numeric_limits<double>::infinity():10*log10(pow((double)maximum-minimum,2)/mse);
        printf("total decompressed psnr= %f\n",psnr);
    }
}


namespace one_point_set::spatial::idw
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);}
}

namespace one_point_set::spatial::rbf
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);}
}

namespace one_point_set::constant_extrapolation::idw
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::constant_extrapolation,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);}
}

namespace one_point_set::constant_extrapolation::rbf
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::constant_extrapolation,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);}
}

namespace one_point_set::linear_interpolation::idw
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::linear_interpolation,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);}
}

namespace one_point_set::linear_interpolation::rbf
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::linear_interpolation,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);}
}

namespace one_point_set::constant_extrapolation_then_spatial::idw
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::constant_extrapolation_spatial_residual,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);}
}

namespace one_point_set::constant_extrapolation_then_spatial::rbf
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::constant_extrapolation_spatial_residual,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);}
}

namespace one_point_set::linear_interpolation_then_spatial::idw
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::linear_interpolation_spatial_residual,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);}
}

namespace one_point_set::linear_interpolation_then_spatial::rbf
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param){param.compress_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::linear_interpolation_spatial_residual,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);}
}

namespace one_point_set::joint_spatiotemporal::idw
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param){param.compress_joint_spatiotemporal_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::idw>);}
}

namespace one_point_set::joint_spatiotemporal::rbf
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param){param.compress_joint_spatiotemporal_files(&param_c<T_coord,T_data>::template compress_point_sets<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,point_set_codec_e::rbf>);}
}

namespace one_point_set::spatial
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param)
{
    point_set_codec_e codec=param.point_set_config_vec[0].codec;
    if(codec==point_set_codec_e::idw)
        idw::compress(param);
    else if(codec==point_set_codec_e::rbf)
        rbf::compress(param);
    else
        throw runtime_error("单点集纯空间压缩只支持IDW或RBF预测器");
}
}

namespace one_point_set::constant_extrapolation
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param)
{
    point_set_codec_e codec=param.point_set_config_vec[0].codec;
    if(codec==point_set_codec_e::idw)
        idw::compress(param);
    else if(codec==point_set_codec_e::rbf)
        rbf::compress(param);
    else
        throw runtime_error("单点集零阶外插压缩只支持IDW或RBF空间预测器");
}
}

namespace one_point_set::linear_interpolation
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param)
{
    point_set_codec_e codec=param.point_set_config_vec[0].codec;
    if(codec==point_set_codec_e::idw)
        idw::compress(param);
    else if(codec==point_set_codec_e::rbf)
        rbf::compress(param);
    else
        throw runtime_error("单点集一阶内插压缩只支持IDW或RBF空间预测器");
}
}

namespace one_point_set::constant_extrapolation_then_spatial
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param)
{
    point_set_codec_e codec=param.point_set_config_vec[0].codec;
    if(codec==point_set_codec_e::idw)
        idw::compress(param);
    else if(codec==point_set_codec_e::rbf)
        rbf::compress(param);
    else
        throw runtime_error("单点集零阶外插加空间补偿只支持IDW或RBF空间预测器");
}
}

namespace one_point_set::linear_interpolation_then_spatial
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param)
{
    point_set_codec_e codec=param.point_set_config_vec[0].codec;
    if(codec==point_set_codec_e::idw)
        idw::compress(param);
    else if(codec==point_set_codec_e::rbf)
        rbf::compress(param);
    else
        throw runtime_error("单点集一阶内插加空间补偿只支持IDW或RBF空间预测器");
}
}

namespace one_point_set::joint_spatiotemporal
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param)
{
    point_set_codec_e codec=param.point_set_config_vec[0].codec;
    if(codec==point_set_codec_e::idw)
        idw::compress(param);
    else if(codec==point_set_codec_e::rbf)
        rbf::compress(param);
    else
        throw runtime_error("单点集联合时空压缩只支持IDW或RBF空间预测器");
}
}

namespace one_point_set
{
template<typename T_coord,typename T_data>
void compress(param_c<T_coord,T_data> &param,chrono::steady_clock::time_point preprocessing_start_time)
{
    if(param.point_set_config_vec.size()!=1)
        throw runtime_error("one_point_set::compress要求恰好配置一个点集");
    param.resolve_single_point_set_time_mode();
    if(param.value_flow=="joint_spatiotemporal")
    {
        param.prepare_joint_spatiotemporal_coordinates();
        param.prepare_compression_structure(preprocessing_start_time);
        joint_spatiotemporal::compress(param);
        return;
    }
    param.prepare_compression_structure(preprocessing_start_time);
    if(param.value_flow=="none")
        spatial::compress(param);
    else if(param.value_flow=="constant_extrapolation")
        constant_extrapolation::compress(param);
    else if(param.value_flow=="linear_interpolation")
        linear_interpolation::compress(param);
    else if(param.value_flow=="constant_extrapolation_spatial_residual")
        constant_extrapolation_then_spatial::compress(param);
    else if(param.value_flow=="linear_interpolation_spatial_residual")
        linear_interpolation_then_spatial::compress(param);
    else
        throw runtime_error("未知的单点集数值预测流程："+param.value_flow);
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
