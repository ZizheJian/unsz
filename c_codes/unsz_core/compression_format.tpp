#pragma once

namespace unsz
{
    inline constexpr unsigned char compression_file_magic[8]={'U','N','S','Z','P','S','1','1'};
    inline constexpr uint32_t compression_file_version=17;

    inline uint8_t encode_prediction_flow(const string&flow)
    {
        if(flow=="none")return 0;
        if(flow=="constant_extrapolation")return 1;
        if(flow=="linear_interpolation")return 2;
        if(flow=="constant_extrapolation_spatial_residual")return 3;
        if(flow=="linear_interpolation_spatial_residual")return 4;
        if(flow=="anchor_interpolation_stride_8")return 5;
        if(flow=="anchor_interpolation_stride_8_spatial_residual")return 6;
        throw runtime_error("不能写入未知预测flow："+flow);
    }

    inline string decode_prediction_flow(uint8_t flow)
    {
        if(flow==0)return "none";
        if(flow==1)return "constant_extrapolation";
        if(flow==2)return "linear_interpolation";
        if(flow==3)return "constant_extrapolation_spatial_residual";
        if(flow==4)return "linear_interpolation_spatial_residual";
        if(flow==5)return "anchor_interpolation_stride_8";
        if(flow==6)return "anchor_interpolation_stride_8_spatial_residual";
        throw runtime_error("压缩文件包含未知预测flow编号："+to_string(flow));
    }

    template<typename T>
    void append_binary(vector<unsigned char> &bytes,const T &value)
    {
        uint64_t begin=bytes.size();
        bytes.resize(begin+sizeof(T));
        memcpy(bytes.data()+begin,&value,sizeof(T));
    }

    template<typename T>
    T read_binary(const unsigned char *&cursor,const unsigned char *end)
    {
        if ((uint64_t)(end-cursor)<sizeof(T))
            throw runtime_error("压缩文件头不完整");
        T value;
        memcpy(&value,cursor,sizeof(T));
        cursor+=sizeof(T);
        return value;
    }

    template<typename T>
    vector<unsigned char> huffman_encode(vector<T> &value_vec,T *min_value=nullptr,T *max_value=nullptr,uint64_t *header_byte_num=nullptr)
    {
        if (value_vec.empty())
            return {};
        huffman_encoder_class<T> encoder;
        encoder.preprocess_encode(value_vec.data(),value_vec.size());
        if(min_value)
            *min_value=encoder.get_min_value();
        if(max_value)
            *max_value=encoder.get_max_value();
        uint64_t max_byte_num=(uint64_t)value_vec.size()*(2*sizeof(T)+1)+64;
        vector<unsigned char> bytes(max<uint64_t>(4096,max_byte_num));
        unsigned char *cursor=bytes.data();
        encoder.save(cursor);
        if(header_byte_num)
            *header_byte_num=cursor-bytes.data();
        encoder.encode(value_vec.data(),value_vec.size(),cursor);
        bytes.resize(cursor-bytes.data());
        return bytes;
    }

    template<typename T>
    vector<T> huffman_decode(const unsigned char *bytes,uint64_t byte_num,uint64_t value_num)
    {
        if (value_num==0)
        {
            if (byte_num!=0)
                throw runtime_error("空数据流具有非空Huffman字节流");
            return {};
        }
        if (byte_num==0)
            throw runtime_error("Huffman字节流为空");
        const unsigned char *cursor=bytes;
        huffman_encoder_class<T> decoder;
        decoder.load(cursor);
        vector<T> result(value_num);
        decoder.decode(cursor,result.data(),value_num);
        if ((uint64_t)(cursor-bytes)!=byte_num)
            throw runtime_error("Huffman解码消耗的字节数与数据流长度不一致");
        return result;
    }

    inline vector<unsigned char> zstd_compress(const vector<unsigned char> &source,int compression_level=3)
    {
        vector<unsigned char> result(ZSTD_compressBound(source.size()));
        uint64_t result_size=ZSTD_compress(result.data(),result.size(),source.data(),source.size(),compression_level);
        if (ZSTD_isError(result_size))
            throw runtime_error("Zstd压缩失败："+string(ZSTD_getErrorName(result_size)));
        result.resize(result_size);
        return result;
    }

    inline vector<unsigned char> zstd_decompress(const unsigned char *source,uint64_t source_size)
    {
        uint64_t result_size=ZSTD_getFrameContentSize(source,source_size);
        if (result_size==ZSTD_CONTENTSIZE_ERROR || result_size==ZSTD_CONTENTSIZE_UNKNOWN)
            throw runtime_error("无法从Zstd帧中读取解压大小");
        vector<unsigned char> result(result_size);
        uint64_t actual_size=ZSTD_decompress(result.data(),result.size(),source,source_size);
        if (ZSTD_isError(actual_size))
            throw runtime_error("Zstd解压失败："+string(ZSTD_getErrorName(actual_size)));
        if (actual_size!=result_size)
            throw runtime_error("Zstd解压后的大小与帧记录不一致");
        return result;
    }

    inline void append_varuint(vector<unsigned char> &bytes,uint64_t value)
    {
        while (value>=128)
        {
            bytes.push_back((value&127)|128);
            value>>=7;
        }
        bytes.push_back(value);
    }

    template<typename T>
    inline vector<unsigned char> binary_one_position_delta_vec(const vector<T> &value_vec)
    {
        vector<unsigned char> result;
        uint64_t one_num=0;
        for (uint32_t value:value_vec)
            one_num+=value;
        append_varuint(result,one_num);
        uint64_t previous_position_plus_one=0;
        for (uint64_t i=0;i<value_vec.size();i++)
            if (value_vec[i])
            {
                uint64_t position_plus_one=i+1;
                append_varuint(result,position_plus_one-previous_position_plus_one);
                previous_position_plus_one=position_plus_one;
            }
        return result;
    }

    inline uint64_t read_varuint(const unsigned char *&cursor,const unsigned char *end)
    {
        uint64_t result=0;
        uint32_t shift=0;
        while (cursor<end && shift<=63)
        {
            uint8_t byte=*cursor++;
            result|=(uint64_t)(byte&127)<<shift;
            if ((byte&128)==0)
                return result;
            shift+=7;
        }
        throw runtime_error("变长无符号整数不完整或超出uint64_t范围");
    }

    inline vector<uint8_t> decode_binary_one_position_delta_vec(vector<unsigned char> &bytes,uint64_t value_num)
    {
        const unsigned char *cursor=bytes.data(),*end=bytes.data()+bytes.size();
        uint64_t one_num=read_varuint(cursor,end);
        if (one_num>value_num)
            throw runtime_error("二进制标签中1的数量超过标签总数");
        vector<uint8_t> result(value_num,0);
        uint64_t position_plus_one=0;
        for (uint64_t i=0;i<one_num;i++)
        {
            uint64_t delta=read_varuint(cursor,end);
            if (delta==0 || delta>value_num-position_plus_one)
                throw runtime_error("二进制标签位置差分越界");
            position_plus_one+=delta;
            result[position_plus_one-1]=1;
        }
        if (cursor!=end)
            throw runtime_error("二进制标签位置差分流末尾存在未解析数据");
        return result;
    }

    template<typename T_coord,typename T_data>
    template<bool two_point_sets>
    void param_c<T_coord,T_data>::decode_current_compression_streams(decompression_file_state_c<T_data>&state)
    {
        uint32_t &point_set_num=state.point_set_num;
        bool &has_label=state.has_label;
        uint64_t &label_target_num=state.label_target_num;
        vector<vector<uint8_t>> &structure_label_symbols=state.structure_label_symbols;
        vector<vector<vector<int32_t>>> &structure_point_set_qb_vec=state.structure_point_set_qb_vec;
        vector<const unsigned char*>&structure_unpredictable_value_begin_vec=state.structure_unpredictable_value_begin_vec,&structure_unpredictable_value_end_vec=state.structure_unpredictable_value_end_vec;
        vector<unsigned char> &container=current_compressed_bytes;
        const unsigned char *cursor=container.data(),*end=container.data()+container.size();
        if((uint64_t)(end-cursor)<8||memcmp(cursor,compression_file_magic,8)!=0)
            throw runtime_error("不是受支持的UNSZ点集文件");
        cursor+=8;
        uint32_t version=read_binary<uint32_t>(cursor,end);
        if(version!=compression_file_version)
            throw runtime_error("不支持的UNSZ文件版本："+to_string(version));
        uint32_t file_point_num=read_binary<uint32_t>(cursor,end);
        uint32_t data_type_bytes=read_binary<uint32_t>(cursor,end);
        point_set_num=read_binary<uint32_t>(cursor,end);
        bool file_uses_tensor=read_binary<uint8_t>(cursor,end)!=0;
        if(file_uses_tensor!=current_field_uses_tensor)
            throw runtime_error("压缩文件的tensor模式与解压参数不一致");
        if(file_uses_tensor)
        {
            tensor_flow_c file_tensor_flow=decode_tensor_flow(read_binary<uint8_t>(cursor,end));
            array<uint32_t,2> file_tensor_shape{{read_binary<uint32_t>(cursor,end),read_binary<uint32_t>(cursor,end)}};
            if(!tensor_flow_selected||encode_tensor_flow(file_tensor_flow)!=encode_tensor_flow(selected_tensor_flow)||file_tensor_shape!=current_tensor_shape)
                throw runtime_error("压缩文件记录的tensor flow或形状与解压参数不一致");
        }
        else
        {
            string file_label_flow=decode_prediction_flow(read_binary<uint8_t>(cursor,end));
            string file_value_flow=decode_prediction_flow(read_binary<uint8_t>(cursor,end));
            if(file_label_flow!=label_flow||file_value_flow!=value_flow)
                throw runtime_error("压缩文件记录的预测flow与解压参数不一致");
        }
        if (file_point_num!=point_num)
            throw runtime_error("压缩文件点数与坐标点数不一致");
        if (data_type_bytes!=sizeof(T_data))
            throw runtime_error("压缩文件的数据类型大小与解压程序不一致");
        if constexpr(two_point_sets)
        {
            if(point_set_num!=2)
                throw runtime_error("两点集field配置要求码流包含两个点集");
        }
        else if(point_set_num!=1)
            throw runtime_error("单点集field配置要求码流包含一个点集");
        if(point_set_num>2)
            throw runtime_error("UNSZ只支持一个或两个点集");
        has_label=point_set_num==2;
        label_target_num=0;
        point_set_point_num_vec.resize(point_set_num);
        point_set_min_data_vec.resize(point_set_num);
        point_set_max_data_vec.resize(point_set_num);
        point_set_abs_eb_vec.resize(point_set_num);
        point_set_codec_vec.resize(point_set_num);
        if(prediction_structure_vec.empty())
            throw runtime_error("UNSZ payload码流缺少预测结构");
        uint64_t metadata_point_num=0;
        for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            uint8_t source_point_set_id=read_binary<uint8_t>(cursor,end);
            if(point_set_id>=point_set_source_id_vec.size()||source_point_set_id!=point_set_source_id_vec[point_set_id])
                throw runtime_error("压缩文件的原始点集编号与当前field配置不一致");
            uint8_t effective_codec=read_binary<uint8_t>(cursor,end);
            if(effective_codec!=(uint8_t)point_set_config_vec[point_set_id].codec)
                throw runtime_error("压缩文件的effective codec与当前field配置不一致");
            uint8_t codec=read_binary<uint8_t>(cursor,end);
            if(codec!=(uint8_t)point_set_codec_e::constant&&codec!=(uint8_t)point_set_codec_e::rbf)
                throw runtime_error("压缩文件包含未知的点集编码方式");
            point_set_codec_vec[point_set_id]=(point_set_codec_e)codec;
            point_set_point_num_vec[point_set_id]=read_binary<uint32_t>(cursor,end);
            metadata_point_num+=point_set_point_num_vec[point_set_id];
            point_set_min_data_vec[point_set_id]=read_binary<T_data>(cursor,end);
            point_set_max_data_vec[point_set_id]=read_binary<T_data>(cursor,end);
        }
        if(metadata_point_num!=point_num)
            throw runtime_error("压缩文件中的各点集点数之和与总点数不一致");
        uint32_t payload_num=read_binary<uint32_t>(cursor,end);
        if(payload_num!=prediction_structure_vec.size())
            throw runtime_error("压缩文件的payload数量与当前预测结构数量不一致");
        vector<uint64_t> payload_begin_vec(payload_num+1);
        for(uint32_t payload_id=0;payload_id<=payload_num;payload_id++)
            payload_begin_vec[payload_id]=read_binary<uint64_t>(cursor,end);
        if(payload_begin_vec.front()!=0||payload_begin_vec.back()!=(uint64_t)(end-cursor))
            throw runtime_error("payload偏移表与文件长度不一致");
        for(uint32_t payload_id=0;payload_id<payload_num;payload_id++)
            if(payload_begin_vec[payload_id]>payload_begin_vec[payload_id+1])
                throw runtime_error("payload偏移表不是递增的");
        structure_label_symbols.resize(payload_num);
        structure_point_set_qb_vec.resize(payload_num,vector<vector<int32_t>>(point_set_num));
        structure_unpredictable_value_begin_vec.resize(payload_num);
        structure_unpredictable_value_end_vec.resize(payload_num);
        const unsigned char *payload_base=cursor;
        exception_ptr lossless_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(payload_num>1&&thread_num>1)
        for(int64_t payload_index=0;payload_index<(int64_t)payload_num;payload_index++)
        {
            try
            {
                uint32_t payload_id=(uint32_t)payload_index;
                const unsigned char *payload_cursor=payload_base+payload_begin_vec[payload_id];
                const unsigned char *payload_end=payload_base+payload_begin_vec[payload_id+1];
                if(has_label)
                {
                    uint64_t target_num=read_binary<uint64_t>(payload_cursor,payload_end);
                    uint64_t zstd_byte_num=read_binary<uint64_t>(payload_cursor,payload_end);
                    if(target_num!=prediction_structure_vec[payload_id].target_ids.size())
                        throw runtime_error("payload标签符号数量与预测目标数量不一致");
                    if((uint64_t)(payload_end-payload_cursor)<zstd_byte_num)
                        throw runtime_error("payload标签流不完整");
                    if(target_num==0&&zstd_byte_num!=0)
                        throw runtime_error("空payload标签流具有非空Zstd字节流");
                    if(target_num>0)
                    {
                        vector<unsigned char> delta_bytes=zstd_decompress(payload_cursor,zstd_byte_num);
                        structure_label_symbols[payload_id]=decode_binary_one_position_delta_vec(delta_bytes,target_num);
                    }
                    payload_cursor+=zstd_byte_num;
                }
                for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
                {
                    uint64_t qb_num=read_binary<uint64_t>(payload_cursor,payload_end);
                    uint64_t zstd_byte_num=read_binary<uint64_t>(payload_cursor,payload_end);
                    if(qb_num>prediction_structure_vec[payload_id].target_ids.size()||(uint64_t)(payload_end-payload_cursor)<zstd_byte_num)
                        throw runtime_error("payload数值流不完整");
                    if(qb_num==0&&zstd_byte_num!=0)
                        throw runtime_error("空payload数值流具有非空Zstd字节流");
                    if(qb_num>0)
                    {
                        vector<unsigned char> huffman_bytes=zstd_decompress(payload_cursor,zstd_byte_num);
                        structure_point_set_qb_vec[payload_id][point_set_id]=huffman_decode<int32_t>(huffman_bytes.data(),huffman_bytes.size(),qb_num);
                    }
                    payload_cursor+=zstd_byte_num;
                }
                if((payload_end-payload_cursor)%sizeof(T_data)!=0)
                    throw runtime_error("payload不可预测值流的字节数不完整");
                structure_unpredictable_value_begin_vec[payload_id]=payload_cursor;
                structure_unpredictable_value_end_vec[payload_id]=payload_end;
            }
            catch(...)
            {
#pragma omp critical
                if(!lossless_exception)
                    lossless_exception=current_exception();
            }
        }
        if(lossless_exception)
            rethrow_exception(lossless_exception);
        for(const vector<uint8_t>&label_symbols:structure_label_symbols)
            label_target_num+=label_symbols.size();
        cursor=end;
    }

    template<typename T_coord,typename T_data>
    encoded_compression_payload_c param_c<T_coord,T_data>::encode_compression_payload(vector<uint8_t>&label_symbols,bool has_label,vector<vector<int32_t>>&point_set_qb_vec,vector<T_data>&unpredictable_values)
    {
        encoded_compression_payload_c result;
        if(has_label)
        {
            vector<unsigned char> delta_bytes;
            if(!label_symbols.empty())
                delta_bytes=binary_one_position_delta_vec(label_symbols);
            uint64_t delta_byte_num=delta_bytes.size();
            vector<unsigned char> zstd_bytes;
            if(!delta_bytes.empty())
                zstd_bytes=zstd_compress(delta_bytes,zstd_compression_level);
            vector<unsigned char>().swap(delta_bytes);
            append_binary<uint64_t>(result.bytes,label_symbols.size());
            append_binary<uint64_t>(result.bytes,zstd_bytes.size());
            result.bytes.insert(result.bytes.end(),zstd_bytes.begin(),zstd_bytes.end());
            result.label_position_delta_byte_num+=delta_byte_num;
            result.label_zstd_byte_num+=zstd_bytes.size();
        }
        for(uint32_t point_set_id=0;point_set_id<point_set_qb_vec.size();point_set_id++)
        {
            vector<int32_t> &qb_vec=point_set_qb_vec[point_set_id];
            vector<unsigned char> huffman_bytes=huffman_encode(qb_vec);
            uint64_t huffman_byte_num=huffman_bytes.size();
            vector<unsigned char> zstd_bytes;
            if(!huffman_bytes.empty())
                zstd_bytes=zstd_compress(huffman_bytes,zstd_compression_level);
            vector<unsigned char>().swap(huffman_bytes);
            append_binary<uint64_t>(result.bytes,qb_vec.size());
            append_binary<uint64_t>(result.bytes,zstd_bytes.size());
            result.bytes.insert(result.bytes.end(),zstd_bytes.begin(),zstd_bytes.end());
            result.value_huffman_byte_num+=huffman_byte_num;
            result.value_zstd_byte_num+=zstd_bytes.size();
        }
        if(!unpredictable_values.empty())
        {
            const unsigned char *unpredictable_bytes=(const unsigned char*)unpredictable_values.data();
            result.bytes.insert(result.bytes.end(),unpredictable_bytes,unpredictable_bytes+unpredictable_values.size()*sizeof(T_data));
        }
        result.unpredictable_value_byte_num=unpredictable_values.size()*sizeof(T_data);
        return result;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::assemble_current_compression_file(compression_file_state_c&state)
    {
        auto stage_start_time=chrono::steady_clock::now();
        uint32_t point_set_num=state.point_set_num;
        uint32_t payload_num=prediction_structure_vec.size();
        if(payload_num==0||state.encoded_payload_vec.size()!=payload_num)
            throw runtime_error("压缩payload数量与预测结构数量不一致");
        vector<encoded_compression_payload_c> &payload_vec=state.encoded_payload_vec;
        uint64_t label_position_delta_byte_num=0,label_zstd_byte_num=0,value_huffman_byte_num=0,value_zstd_byte_num=0,unpredictable_value_byte_num=0;
        for(const encoded_compression_payload_c &payload:payload_vec)
        {
            label_position_delta_byte_num+=payload.label_position_delta_byte_num;
            label_zstd_byte_num+=payload.label_zstd_byte_num;
            value_huffman_byte_num+=payload.value_huffman_byte_num;
            value_zstd_byte_num+=payload.value_zstd_byte_num;
            unpredictable_value_byte_num+=payload.unpredictable_value_byte_num;
        }
        vector<unsigned char> container;
        uint64_t temporal_metadata_byte_num=current_field_uses_tensor?2*sizeof(uint8_t)+2*sizeof(uint32_t):3*sizeof(uint8_t);
        uint64_t container_byte_num=8+4*sizeof(uint32_t)+temporal_metadata_byte_num+(uint64_t)point_set_num*(3*sizeof(uint8_t)+sizeof(uint32_t)+2*sizeof(T_data))+sizeof(uint32_t)+(uint64_t)(payload_num+1)*sizeof(uint64_t);
        for(const encoded_compression_payload_c&payload:payload_vec)
            container_byte_num+=payload.bytes.size();
        container.reserve((size_t)container_byte_num);
        container.insert(container.end(),compression_file_magic,compression_file_magic+8);
        append_binary<uint32_t>(container,compression_file_version);
        append_binary<uint32_t>(container,point_num);
        append_binary<uint32_t>(container,sizeof(T_data));
        append_binary<uint32_t>(container,point_set_num);
        append_binary<uint8_t>(container,current_field_uses_tensor?1:0);
        if(current_field_uses_tensor)
        {
            if(!tensor_flow_selected)
                throw runtime_error("写入压缩文件前尚未选择tensor flow");
            append_binary<uint8_t>(container,encode_tensor_flow(selected_tensor_flow));
            append_binary<uint32_t>(container,current_tensor_shape[0]);
            append_binary<uint32_t>(container,current_tensor_shape[1]);
        }
        else
        {
            append_binary<uint8_t>(container,encode_prediction_flow(label_flow));
            append_binary<uint8_t>(container,encode_prediction_flow(value_flow));
        }
        for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            append_binary<uint8_t>(container,point_set_source_id_vec[point_set_id]);
            append_binary<uint8_t>(container,(uint8_t)point_set_config_vec[point_set_id].codec);
            append_binary<uint8_t>(container,(uint8_t)point_set_codec_vec[point_set_id]);
            append_binary<uint32_t>(container,point_set_point_num_vec[point_set_id]);
            append_binary<T_data>(container,point_set_min_data_vec[point_set_id]);
            append_binary<T_data>(container,point_set_max_data_vec[point_set_id]);
        }
        append_binary<uint32_t>(container,payload_num);
        uint64_t payload_begin=0;
        append_binary<uint64_t>(container,payload_begin);
        for(const encoded_compression_payload_c &payload:payload_vec)
        {
            payload_begin+=payload.bytes.size();
            append_binary<uint64_t>(container,payload_begin);
        }
        for(encoded_compression_payload_c &payload:payload_vec)
        {
            container.insert(container.end(),payload.bytes.begin(),payload.bytes.end());
            vector<unsigned char>().swap(payload.bytes);
        }
        current_compressed_bytes=move(container);
        current_label_stream_bytes=state.has_label?(uint64_t)payload_num*2*sizeof(uint64_t)+label_zstd_byte_num:0;
        current_value_stream_bytes=(uint64_t)payload_num*point_set_num*2*sizeof(uint64_t)+value_zstd_byte_num+unpredictable_value_byte_num;
        current_shared_metadata_bytes=current_compressed_bytes.size()-current_label_stream_bytes-current_value_stream_bytes;
        state.lossless_encoding_time+=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        current_compression_diagnostics.point_set_classification_time=state.point_set_classification_time;
        current_compression_diagnostics.point_set_setup_time=state.point_set_setup_time;
        current_compression_diagnostics.label_prediction_time=state.label_prediction_time;
        current_compression_diagnostics.value_prediction_time=state.value_prediction_time;
        current_compression_diagnostics.lossless_encoding_time=state.lossless_encoding_time;
        current_compression_diagnostics.label_target_num=state.label_target_num;
        current_compression_diagnostics.label_error_num=state.label_error_num;
        current_compression_diagnostics.cross_time_prediction_num=state.cross_time_prediction_num;
        current_compression_diagnostics.label_position_delta_byte_num=label_position_delta_byte_num;
        current_compression_diagnostics.label_zstd_byte_num=label_zstd_byte_num;
        current_compression_diagnostics.value_huffman_byte_num=value_huffman_byte_num;
        current_compression_diagnostics.value_zstd_byte_num=value_zstd_byte_num;
        current_compression_diagnostics.unpredictable_value_byte_num=unpredictable_value_byte_num;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::print_current_compression_diagnostics() const
    {
        const compression_diagnostics_c&diagnostics=current_compression_diagnostics;
        double label_prediction_error_ratio=diagnostics.label_target_num?(double)diagnostics.label_error_num/diagnostics.label_target_num:0;
        if(current_prediction_framework.value_mode!=time_dimension_mode_e::spatial)
            printf("cross_time_prediction_num=%llu, ratio=%f\n",(unsigned long long)diagnostics.cross_time_prediction_num,(double)diagnostics.cross_time_prediction_num/point_num);
        printf("label_prediction_error_ratio=%f, label_position_delta_bytes=%llu, label_zstd_bytes=%llu, zstd_level=%d\n",label_prediction_error_ratio,(unsigned long long)diagnostics.label_position_delta_byte_num,(unsigned long long)diagnostics.label_zstd_byte_num,zstd_compression_level);
        printf("value_huffman_bytes=%llu, value_zstd_bytes=%llu, unpredictable_value_bytes=%llu, file_bytes=%llu\n",(unsigned long long)diagnostics.value_huffman_byte_num,(unsigned long long)diagnostics.value_zstd_byte_num,(unsigned long long)diagnostics.unpredictable_value_byte_num,(unsigned long long)current_compressed_bytes.size());
        printf("label_stream_bytes=%llu, value_stream_bytes=%llu, shared_metadata_bytes=%llu\n",(unsigned long long)current_label_stream_bytes,(unsigned long long)current_value_stream_bytes,(unsigned long long)current_shared_metadata_bytes);
        printf("point_set_num=%u",(uint32_t)point_set_codec_vec.size());
        for(uint32_t point_set_id=0;point_set_id<point_set_codec_vec.size();point_set_id++)
        {
            const char *codec_name;
            if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                codec_name="constant";
            else if(point_set_codec_vec[point_set_id]==point_set_codec_e::rbf)
                codec_name="rbf";
            else
                throw runtime_error("未知的点集编码方式");
            printf(", set%u_num=%u, set%u_codec=%s",point_set_id,point_set_point_num_vec[point_set_id],point_set_id,codec_name);
        }
        printf("\n");
        if(print_compression_stage_time)
        {
            double point_set_time=diagnostics.point_set_classification_time+diagnostics.point_set_setup_time;
            printf("compression stage time: point_set=%f s, label_prediction=%f s, value_prediction=%f s, lossless_encoding=%f s\n",point_set_time,diagnostics.label_prediction_time,diagnostics.value_prediction_time,diagnostics.lossless_encoding_time);
            printf("point set stage time: classification=%f s, setup=%f s\n",diagnostics.point_set_classification_time,diagnostics.point_set_setup_time);
        }
    }

    template<typename T_coord,typename T_data>
    double param_c<T_coord,T_data>::read_compressed_field_metadata()
    {
        if(original_data_path_vec.empty())
            throw runtime_error("自动flow解压缺少数据文件路径");
        constexpr uint64_t maximum_header_byte_num=8+6*sizeof(uint32_t)+3*sizeof(uint8_t)+2*(3*sizeof(uint8_t)+sizeof(uint32_t)+2*sizeof(T_data));
        vector<unsigned char> bytes(maximum_header_byte_num);
        string compressed_path=generated_path(original_data_path_vec.front(),".unsz");
        auto input_start_time=chrono::steady_clock::now();
        FILE*file=fopen(compressed_path.c_str(),"rb");
        if(!file)
            throw runtime_error("找不到文件："+compressed_path);
        uint64_t read_byte_num=fread(bytes.data(),1,bytes.size(),file);
        fclose(file);
        double input_time=chrono::duration<double>(chrono::steady_clock::now()-input_start_time).count();
        constexpr uint64_t fixed_header_byte_num=8+4*sizeof(uint32_t)+2*sizeof(uint8_t);
        if(read_byte_num<fixed_header_byte_num)
            throw runtime_error("UNSZ压缩文件头不完整："+compressed_path);
        bytes.resize(read_byte_num);
        const unsigned char*cursor=bytes.data(),*end=bytes.data()+bytes.size();
        if((uint64_t)(end-cursor)<8||memcmp(cursor,compression_file_magic,8)!=0)
            throw runtime_error("不是受支持的UNSZ点集文件");
        cursor+=8;
        uint32_t version=read_binary<uint32_t>(cursor,end);
        if(version!=compression_file_version)
            throw runtime_error("不支持的UNSZ文件版本："+to_string(version));
        uint32_t file_point_num=read_binary<uint32_t>(cursor,end);
        if(file_point_num!=point_num)
            throw runtime_error("压缩文件的点数与坐标文件不一致");
        uint32_t file_data_type_size=read_binary<uint32_t>(cursor,end);
        if(file_data_type_size!=sizeof(T_data))
            throw runtime_error("压缩文件的数据类型大小与解压程序不一致");
        uint32_t point_set_num=read_binary<uint32_t>(cursor,end);
        if(point_set_num==0||point_set_num>2)
            throw runtime_error("压缩文件包含无效点集数量");
        bool file_uses_tensor=read_binary<uint8_t>(cursor,end)!=0;
        if(file_uses_tensor!=current_field_uses_tensor)
            throw runtime_error("压缩文件的tensor模式与解压参数不一致");
        if(file_uses_tensor)
        {
            selected_tensor_flow=decode_tensor_flow(read_binary<uint8_t>(cursor,end));
            tensor_flow_selected=true;
            array<uint32_t,2> file_tensor_shape{{read_binary<uint32_t>(cursor,end),read_binary<uint32_t>(cursor,end)}};
            if(file_tensor_shape!=current_tensor_shape)
                throw runtime_error("压缩文件记录的tensor形状与metadata不一致");
        }
        else
        {
            string file_label_flow=decode_prediction_flow(read_binary<uint8_t>(cursor,end));
            string file_value_flow=decode_prediction_flow(read_binary<uint8_t>(cursor,end));
            if(label_flow=="auto")
                label_flow=file_label_flow;
            else if(label_flow!=file_label_flow)
                throw runtime_error("压缩文件记录的label flow与解压参数不一致");
            if(value_flow=="auto")
                value_flow=file_value_flow;
            else if(value_flow!=file_value_flow)
                throw runtime_error("压缩文件记录的value flow与解压参数不一致");
        }
        vector<uint8_t> source_ids(point_set_num),effective_codecs(point_set_num);
        for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            source_ids[point_set_id]=read_binary<uint8_t>(cursor,end);
            effective_codecs[point_set_id]=read_binary<uint8_t>(cursor,end);
            read_binary<uint8_t>(cursor,end);
            read_binary<uint32_t>(cursor,end);
            read_binary<T_data>(cursor,end);
            read_binary<T_data>(cursor,end);
            if(source_ids[point_set_id]>=configured_point_set_config_vec.size())
                throw runtime_error("压缩文件的原始点集编号越界");
            if(effective_codecs[point_set_id]!=(uint8_t)point_set_codec_e::constant&&effective_codecs[point_set_id]!=(uint8_t)point_set_codec_e::rbf)
                throw runtime_error("压缩文件包含未知的effective codec");
        }
        point_set_config_vec.clear();
        for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            point_set_config_vec.push_back(configured_point_set_config_vec[source_ids[point_set_id]]);
            point_set_config_vec.back().codec=(point_set_codec_e)effective_codecs[point_set_id];
        }
        point_set_source_id_vec=move(source_ids);
        if(point_set_num==1)
        {
            point_set_rule_vec.clear();
            label_flow="none";
        }
        else
            point_set_rule_vec=configured_point_set_rule_vec;
        return input_time;
    }

}
