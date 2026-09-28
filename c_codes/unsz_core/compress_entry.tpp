#pragma once

namespace unsz
{
inline vector<unsigned char> read_unsz_file(const string &path)
{
    FILE *file=fopen(path.c_str(),"rb");
    if(!file)
        throw runtime_error("找不到UNSZ文件："+path);
    if(fseek(file,0,SEEK_END)!=0)
    {
        fclose(file);
        throw runtime_error("无法定位UNSZ文件末尾："+path);
    }
    long byte_num_signed=ftell(file);
    if(byte_num_signed<0||fseek(file,0,SEEK_SET)!=0)
    {
        fclose(file);
        throw runtime_error("无法读取UNSZ文件大小："+path);
    }
    vector<unsigned char> bytes((uint64_t)byte_num_signed);
    if(fread(bytes.data(),1,bytes.size(),file)!=bytes.size())
    {
        fclose(file);
        throw runtime_error("读取UNSZ文件失败："+path);
    }
    if(fclose(file)!=0)
        throw runtime_error("关闭UNSZ文件失败："+path);
    return bytes;
}

inline void write_unsz_file(const string &path,const vector<unsigned char> &bytes)
{
    FILE *file=fopen(path.c_str(),"wb");
    if(!file)
        throw runtime_error("无法创建UNSZ文件："+path);
    if(fwrite(bytes.data(),1,bytes.size(),file)!=bytes.size())
    {
        fclose(file);
        throw runtime_error("写入UNSZ文件失败："+path);
    }
    if(fclose(file)!=0)
        throw runtime_error("关闭UNSZ文件失败："+path);
}

inline uint64_t get_point_set_value_stream_offset(const vector<unsigned char> &bytes,bool partition_adaptive)
{
    const unsigned char magic_v7[8]={'U','N','S','Z','P','S','0','7'};
    const unsigned char magic_v8[8]={'U','N','S','Z','P','S','0','8'};
    const unsigned char magic_v9[8]={'U','N','S','Z','P','S','0','9'};
    const unsigned char *begin=bytes.data(),*cursor=begin,*end=begin+bytes.size();
    if((uint64_t)(end-cursor)<8)
        throw runtime_error("UNSZ点集文件头不完整");
    bool legacy=memcmp(cursor,magic_v7,8)==0;
    if(memcmp(cursor,magic_v9,8)==0)
        throw runtime_error("staged调试路径尚不支持partition码流");
    if(!legacy&&memcmp(cursor,magic_v8,8)!=0)
        throw runtime_error("不是受支持的UNSZ点集文件");
    cursor+=8;
    uint32_t version=read_binary<uint32_t>(cursor,end);
    if((legacy&&version!=7)||(!legacy&&version!=8))
        throw runtime_error("不支持的UNSZ点集文件版本");
    read_binary<uint32_t>(cursor,end);
    read_binary<uint32_t>(cursor,end);
    uint32_t point_set_num=read_binary<uint32_t>(cursor,end);
    if(partition_adaptive)
    {
        uint32_t config_num=read_binary<uint32_t>(cursor,end);
        if((uint64_t)(end-cursor)<config_num)
            throw runtime_error("UNSZ partition配置不完整");
        cursor+=config_num;
    }
    if(legacy)
    {
        uint64_t label_byte_num=read_binary<uint64_t>(cursor,end);
        if((uint64_t)(end-cursor)<label_byte_num)
            throw runtime_error("UNSZ标签码流不完整");
        cursor+=label_byte_num;
    }
    else
    {
        uint32_t label_channel_num=read_binary<uint32_t>(cursor,end);
        if(label_channel_num+1!=point_set_num)
            throw runtime_error("UNSZ标签通道数与点集数不一致");
        for(uint32_t label_channel_id=0;label_channel_id<label_channel_num;label_channel_id++)
        {
            read_binary<uint64_t>(cursor,end);
            uint64_t label_byte_num=read_binary<uint64_t>(cursor,end);
            if((uint64_t)(end-cursor)<label_byte_num)
                throw runtime_error("UNSZ标签码流不完整");
            cursor+=label_byte_num;
        }
    }
    return cursor-begin;
}

template<typename T_coord,typename T_data>
void param_c<T_coord,T_data>::compress_staged_two_point_set_files()
{
    if(!partition_pred_ref_ids_vec.empty())
        throw runtime_error("staged调试路径尚不支持partition码流");
    string original_label_flow=label_flow,original_value_flow=value_flow;
    vector<uint32_t> canonical_time_step_vec=time_step_vec;
    vector<string> canonical_data_path_vec=original_data_path_vec;
    vector<uint32_t> sorted_id_vec(canonical_time_step_vec.size());
    iota(sorted_id_vec.begin(),sorted_id_vec.end(),0);
    sort(sorted_id_vec.begin(),sorted_id_vec.end(),[&](uint32_t a,uint32_t b){return canonical_time_step_vec[a]<canonical_time_step_vec[b];});
    vector<uint32_t> sorted_time_step_vec;
    vector<string> sorted_data_path_vec;
    sorted_time_step_vec.reserve(sorted_id_vec.size());
    sorted_data_path_vec.reserve(sorted_id_vec.size());
    for(uint32_t id:sorted_id_vec)
    {
        sorted_time_step_vec.push_back(canonical_time_step_vec[id]);
        sorted_data_path_vec.push_back(canonical_data_path_vec[id]);
    }
    auto reset_time_state=[&]()
    {
        current_data.clear();
        current_decompressed_data.clear();
        previous_time_decompressed_data.clear();
        previous_time_point_set_id_vec.clear();
        reconstructed_time_step_vec.clear();
        reconstructed_time_data_vec.clear();
        reconstructed_time_point_set_id_vec.clear();
        current_time_spatial_residual_data.clear();
        point_set_id_vec.clear();
        compressed_data_path_vec.clear();
        decompressed_data_path_vec.clear();
    };
    auto configure_order=[&](const string &selected_label_flow,const string &selected_value_flow)
    {
        label_flow=selected_label_flow;
        value_flow=selected_value_flow;
        time_step_vec=sorted_time_step_vec;
        original_data_path_vec=sorted_data_path_vec;
        const string &selected_flow=selected_label_flow!="none"?selected_label_flow:selected_value_flow;
        prepare_time_step_order(get_time_step_order(selected_flow));
        reset_time_state();
    };
    map<uint32_t,vector<unsigned char>> label_prefix_map;
    running_staged_file_pass=true;
    suppress_compression_summary=true;
    try
    {
        configure_order(original_label_flow,"none");
        printf("staged label pass: process all labels before values\n");
        compress_files(&param_c<T_coord,T_data>::compress_two_point_sets);
        double label_pass_time=last_total_compression_time;
        for(uint32_t file_id=0;file_id<time_step_vec.size();file_id++)
        {
            vector<unsigned char> bytes=read_unsz_file(original_data_path_vec[file_id]+".unsz");
            uint64_t value_offset=get_point_set_value_stream_offset(bytes,use_partition_adaptive_predictor_experimental);
            label_prefix_map[time_step_vec[file_id]]=vector<unsigned char>(bytes.begin(),bytes.begin()+value_offset);
        }

        configure_order("none",original_value_flow);
        printf("staged value pass: process values after all labels\n");
        compress_files(&param_c<T_coord,T_data>::compress_two_point_sets);
        double value_pass_time=last_total_compression_time;
        uint64_t total_compressed_bytes=0;
        for(uint32_t file_id=0;file_id<time_step_vec.size();file_id++)
        {
            uint32_t time_step=time_step_vec[file_id];
            auto prefix_iterator=label_prefix_map.find(time_step);
            if(prefix_iterator==label_prefix_map.end())
                throw runtime_error("staged标签遍缺少time step "+to_string(time_step));
            string path=original_data_path_vec[file_id]+".unsz";
            vector<unsigned char> value_file=read_unsz_file(path);
            uint64_t value_offset=get_point_set_value_stream_offset(value_file,use_partition_adaptive_predictor_experimental);
            vector<unsigned char> combined=prefix_iterator->second;
            combined.insert(combined.end(),value_file.begin()+value_offset,value_file.end());
            total_compressed_bytes+=combined.size();
            write_unsz_file(path,combined);
        }
        double total_cr=(double)last_total_original_bytes/total_compressed_bytes;
        double total_br=sizeof(T_data)*8/total_cr;
        printf("staged mode is for compression-effect debugging; its time includes separate label and value passes\n");
        printf("total cr= %f, total br= %f, total psnr= %f\n",total_cr,total_br,last_total_psnr);
        printf("total compression time= %f s\n",label_pass_time+value_pass_time);
    }
    catch(...)
    {
        running_staged_file_pass=false;
        suppress_compression_summary=false;
        label_flow=original_label_flow;
        value_flow=original_value_flow;
        throw;
    }
    running_staged_file_pass=false;
    suppress_compression_summary=false;
    label_flow=original_label_flow;
    value_flow=original_value_flow;
}

template<typename T_coord,typename T_data>
void param_c<T_coord,T_data>::prepare_staged_two_point_set_decompression()
{
    if(!partition_pred_ref_ids_vec.empty())
        throw runtime_error("staged调试路径尚不支持partition码流");
    string original_label_flow=label_flow,original_value_flow=value_flow;
    vector<uint32_t> canonical_time_step_vec=time_step_vec;
    vector<string> canonical_data_path_vec=original_data_path_vec;
    vector<uint32_t> sorted_id_vec(canonical_time_step_vec.size());
    iota(sorted_id_vec.begin(),sorted_id_vec.end(),0);
    sort(sorted_id_vec.begin(),sorted_id_vec.end(),[&](uint32_t a,uint32_t b){return canonical_time_step_vec[a]<canonical_time_step_vec[b];});
    vector<uint32_t> sorted_time_step_vec;
    vector<string> sorted_data_path_vec;
    sorted_time_step_vec.reserve(sorted_id_vec.size());
    sorted_data_path_vec.reserve(sorted_id_vec.size());
    for(uint32_t id:sorted_id_vec)
    {
        sorted_time_step_vec.push_back(canonical_time_step_vec[id]);
        sorted_data_path_vec.push_back(canonical_data_path_vec[id]);
    }
    auto reset_time_state=[&]()
    {
        current_decompressed_data.clear();
        previous_time_decompressed_data.clear();
        previous_time_point_set_id_vec.clear();
        reconstructed_time_step_vec.clear();
        reconstructed_time_data_vec.clear();
        reconstructed_time_point_set_id_vec.clear();
        current_time_spatial_residual_data.clear();
        point_set_id_vec.clear();
    };
    auto configure_order=[&](const string &selected_label_flow,const string &selected_value_flow)
    {
        label_flow=selected_label_flow;
        value_flow=selected_value_flow;
        time_step_vec=sorted_time_step_vec;
        original_data_path_vec=sorted_data_path_vec;
        const string &selected_flow=selected_label_flow!="none"?selected_label_flow:selected_value_flow;
        prepare_time_step_order(get_time_step_order(selected_flow));
        reset_time_state();
    };
    staged_point_set_id_map.clear();
    configure_order(original_label_flow,"none");
    decode_labels_only=true;
    use_staged_point_set_ids=false;
    staged_label_decompression_time=0;
    try
    {
        printf("staged label decompression pass: restore all labels before values\n");
        for(uint32_t file_id=0;file_id<time_step_vec.size();file_id++)
        {
            current_time_step=time_step_vec[file_id];
            current_zstd_bs_vec=read_unsz_file(original_data_path_vec[file_id]+".unsz");
            auto label_decompression_start_time=chrono::steady_clock::now();
            decompress_two_point_sets();
            staged_label_decompression_time+=chrono::duration<double>(chrono::steady_clock::now()-label_decompression_start_time).count();
            staged_point_set_id_map[current_time_step]=point_set_id_vec;
            bool anchor_flow=uses_anchor_interpolation_stride_8_flow();
            if(anchor_flow||current_prediction_framework.label_mode==time_dimension_mode_e::endpoint_agreement)
            {
                reconstructed_time_step_vec.push_back(current_time_step);
                reconstructed_time_data_vec.push_back({});
                reconstructed_time_point_set_id_vec.push_back(point_set_id_vec);
            }
            bool first_anchor=anchor_flow&&current_time_step==0;
            if(first_anchor||current_prediction_framework.label_mode==time_dimension_mode_e::constant_extrapolation)
                previous_time_point_set_id_vec=point_set_id_vec;
        }
    }
    catch(...)
    {
        decode_labels_only=false;
        label_flow=original_label_flow;
        value_flow=original_value_flow;
        throw;
    }
    decode_labels_only=false;
    configure_order("none",original_value_flow);
    use_staged_point_set_ids=true;
}

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
    if(point_set_config_vec.size()==2&&label_value_execution=="staged"&&!running_staged_file_pass)
    {
        compress_staged_two_point_set_files();
        return;
    }
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
    last_total_original_bytes=total_original_bytes;
    last_total_psnr=total_psnr;
    last_total_compression_time=total_compression_time;
    if(!suppress_compression_summary)
    {
        printf("total cr= %f, total br= %f, total psnr= %f\n",total_cr,total_br,total_psnr);
        printf("total compression time= %f s\n",total_compression_time);
    }

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
