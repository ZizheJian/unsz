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
    suppress_compression_summary=true;
    try
    {
        configure_order(original_label_flow,"none");
        printf("staged label pass: process all labels before values\n");
        compress_files(&param_c<T_coord,T_data>::dispatch_two_point_set_compression_by_mode);
        double label_pass_time=last_total_compression_time;
        for(uint32_t file_id=0;file_id<time_step_vec.size();file_id++)
        {
            vector<unsigned char> bytes=read_unsz_file(original_data_path_vec[file_id]+".unsz");
            uint64_t value_offset=get_point_set_value_stream_offset(bytes,use_partition_adaptive_predictor_experimental);
            label_prefix_map[time_step_vec[file_id]]=vector<unsigned char>(bytes.begin(),bytes.begin()+value_offset);
        }

        configure_order("none",original_value_flow);
        printf("staged value pass: process values after all labels\n");
        compress_files(&param_c<T_coord,T_data>::dispatch_two_point_set_compression_by_mode);
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
        suppress_compression_summary=false;
        label_flow=original_label_flow;
        value_flow=original_value_flow;
        throw;
    }
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
            dispatch_two_point_set_decompression_by_mode();
            staged_label_decompression_time+=chrono::duration<double>(chrono::steady_clock::now()-label_decompression_start_time).count();
            staged_point_set_id_map[current_time_step]=point_set_id_vec;
            bool anchor_flow=uses_anchor_interpolation_stride_8_flow();
            if(anchor_flow||current_prediction_framework.label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual)
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
}
