#pragma once

namespace unsz
{
inline uint64_t get_point_set_value_stream_offset(const vector<unsigned char> &bytes,bool partition_adaptive)
{
    const unsigned char *begin=bytes.data(),*cursor=begin,*end=begin+bytes.size();
    if((uint64_t)(end-cursor)<8)
        throw runtime_error("UNSZ点集文件头不完整");
    if(memcmp(cursor,compression_file_magic,8)!=0)
        throw runtime_error("staged估算只支持当前UNSZ payload格式");
    cursor+=8;
    uint32_t version=read_binary<uint32_t>(cursor,end);
    if(version!=compression_file_version)
        throw runtime_error("不支持的UNSZ点集文件版本");
    read_binary<uint32_t>(cursor,end);
    uint32_t data_type_bytes=read_binary<uint32_t>(cursor,end);
    uint32_t point_set_num=read_binary<uint32_t>(cursor,end);
    if(partition_adaptive)
    {
        uint32_t config_num=read_binary<uint32_t>(cursor,end);
        if((uint64_t)(end-cursor)<config_num)
            throw runtime_error("UNSZ partition配置不完整");
        cursor+=config_num;
    }
    uint64_t point_set_metadata_bytes=(uint64_t)point_set_num*(sizeof(uint8_t)+sizeof(uint32_t)+2*data_type_bytes);
    if((uint64_t)(end-cursor)<point_set_metadata_bytes)
        throw runtime_error("UNSZ点集元数据不完整");
    cursor+=point_set_metadata_bytes;
    uint32_t payload_num=read_binary<uint32_t>(cursor,end);
    if(payload_num!=1)
        throw runtime_error("staged估算要求单个payload");
    uint64_t payload_begin=read_binary<uint64_t>(cursor,end);
    uint64_t payload_end=read_binary<uint64_t>(cursor,end);
    if(payload_begin!=0||payload_end!=(uint64_t)(end-cursor))
        throw runtime_error("UNSZ payload偏移表与文件长度不一致");
    if(point_set_num==2)
    {
        read_binary<uint64_t>(cursor,end);
        uint64_t label_byte_num=read_binary<uint64_t>(cursor,end);
        if((uint64_t)(end-cursor)<label_byte_num)
            throw runtime_error("UNSZ标签码流不完整");
        cursor+=label_byte_num;
    }
    return cursor-begin;
}

template<typename T_coord,typename T_data>
template<point_set_codec_e predictor>
void param_c<T_coord,T_data>::compress_staged_two_point_sets_files()
{
    if(prediction_structure_vec.size()>1)
        throw runtime_error("staged调试路径尚不支持多个预测结构");
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
        current_time_spatial_residual_valid.clear();
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
    vector<vector<unsigned char>> compressed_stream_vec;
    suppress_compression_summary=true;
    try
    {
        configure_order(original_label_flow,"none");
        printf("staged label pass: process all labels before values\n");
        compress_files(&param_c<T_coord,T_data>::template dispatch_staged_label_compression<predictor>,false,&compressed_stream_vec,false);
        double label_pass_time=last_total_compression_time;
        for(uint32_t file_id=0;file_id<time_step_vec.size();file_id++)
        {
            vector<unsigned char> &bytes=compressed_stream_vec[file_id];
            uint64_t value_offset=get_point_set_value_stream_offset(bytes,use_partition_adaptive_predictor_experimental);
            label_prefix_map[time_step_vec[file_id]]=vector<unsigned char>(bytes.begin(),bytes.begin()+value_offset);
        }

        configure_order("none",original_value_flow);
        printf("staged value pass: process values after all labels\n");
        compress_files(&param_c<T_coord,T_data>::template dispatch_staged_value_compression<predictor>,false,&compressed_stream_vec);
        double value_pass_time=last_total_compression_time;
        uint64_t total_compressed_bytes=0;
        for(uint32_t file_id=0;file_id<time_step_vec.size();file_id++)
        {
            uint32_t time_step=time_step_vec[file_id];
            auto prefix_iterator=label_prefix_map.find(time_step);
            if(prefix_iterator==label_prefix_map.end())
                throw runtime_error("staged标签遍缺少time step "+to_string(time_step));
            vector<unsigned char> &value_file=compressed_stream_vec[file_id];
            uint64_t value_offset=get_point_set_value_stream_offset(value_file,use_partition_adaptive_predictor_experimental);
            total_compressed_bytes+=prefix_iterator->second.size()+value_file.size()-value_offset;
        }
        double total_cr=(double)last_total_original_bytes/total_compressed_bytes;
        double total_br=sizeof(T_data)*8/total_cr;
        printf("staged mode estimates compression effect in memory and does not write compressed files\n");
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

}
