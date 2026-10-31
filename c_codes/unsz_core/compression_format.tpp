#pragma once

namespace unsz
{
    inline constexpr unsigned char compression_file_magic[8]={'U','N','S','Z','P','S','1','1'};
    inline constexpr uint32_t compression_file_version=17;

    inline uint8_t encode_prediction_pipeline(const string&pipeline)
    {
        if(pipeline=="independent")return 0;
        if(pipeline=="extrapolation")return 1;
        if(pipeline=="interpolation")return 2;
        if(pipeline=="extrapolation_spatial_residual")return 3;
        if(pipeline=="interpolation_spatial_residual")return 4;
        if(pipeline=="anchor_interpolation")return 5;
        if(pipeline=="anchor_interpolation_spatial_residual")return 6;
        throw runtime_error("不能写入未知预测pipeline："+pipeline);
    }

    inline string decode_prediction_pipeline(uint8_t pipeline)
    {
        if(pipeline==0)return "independent";
        if(pipeline==1)return "extrapolation";
        if(pipeline==2)return "interpolation";
        if(pipeline==3)return "extrapolation_spatial_residual";
        if(pipeline==4)return "interpolation_spatial_residual";
        if(pipeline==5)return "anchor_interpolation";
        if(pipeline==6)return "anchor_interpolation_spatial_residual";
        throw runtime_error("压缩文件包含未知预测pipeline编号："+to_string(pipeline));
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
    template<bool two_labels>
    void param_c<T_coord,T_data>::decode_current_compression_streams(decompression_file_state_c<T_data>&state)
    {
        uint32_t &label_num=state.label_num;
        bool &has_label=state.has_label;
        uint64_t &label_target_num=state.label_target_num;
        vector<vector<uint8_t>> &structure_label_symbols=state.structure_label_symbols;
        vector<vector<vector<int32_t>>> &structure_label_quantized_error_vec=state.structure_label_quantized_error_vec;
        vector<const unsigned char*>&structure_unpredictable_value_begin_vec=state.structure_unpredictable_value_begin_vec,&structure_unpredictable_value_end_vec=state.structure_unpredictable_value_end_vec;
        vector<unsigned char> &container=current_compressed_bytes;
        const unsigned char *cursor=container.data(),*end=container.data()+container.size();
        if((uint64_t)(end-cursor)<8||memcmp(cursor,compression_file_magic,8)!=0)
            throw runtime_error("不是受支持的UNSZ标签文件");
        cursor+=8;
        uint32_t version=read_binary<uint32_t>(cursor,end);
        if(version!=compression_file_version)
            throw runtime_error("不支持的UNSZ文件版本："+to_string(version));
        uint32_t file_point_num=read_binary<uint32_t>(cursor,end);
        uint32_t data_type_bytes=read_binary<uint32_t>(cursor,end);
        label_num=read_binary<uint32_t>(cursor,end);
        bool file_uses_tensor=read_binary<uint8_t>(cursor,end)!=0;
        if(file_uses_tensor!=current_field_uses_tensor)
            throw runtime_error("压缩文件的tensor模式与解压参数不一致");
        if(file_uses_tensor)
        {
            tensor_pipeline_c file_tensor_pipeline=decode_tensor_pipeline(read_binary<uint8_t>(cursor,end));
            array<uint32_t,2> file_tensor_shape{{read_binary<uint32_t>(cursor,end),read_binary<uint32_t>(cursor,end)}};
            if(!tensor_pipeline_selected||encode_tensor_pipeline(file_tensor_pipeline)!=encode_tensor_pipeline(selected_tensor_pipeline)||file_tensor_shape!=current_tensor_shape)
                throw runtime_error("压缩文件记录的tensor pipeline或形状与解压参数不一致");
        }
        else
        {
            string file_label_pipeline=decode_prediction_pipeline(read_binary<uint8_t>(cursor,end));
            string file_value_pipeline=decode_prediction_pipeline(read_binary<uint8_t>(cursor,end));
            if(file_label_pipeline!=label_pipeline||file_value_pipeline!=value_pipeline)
                throw runtime_error("压缩文件记录的预测pipeline与解压参数不一致");
        }
        if (file_point_num!=point_num)
            throw runtime_error("压缩文件点数与坐标点数不一致");
        if (data_type_bytes!=sizeof(T_data))
            throw runtime_error("压缩文件的数据类型大小与解压程序不一致");
        if constexpr(two_labels)
        {
            if(label_num!=2)
                throw runtime_error("两种标签field配置要求码流包含两个标签");
        }
        else if(label_num!=1)
            throw runtime_error("单标签field配置要求码流包含一个标签");
        if(label_num>2)
            throw runtime_error("UNSZ只支持一个或两个标签");
        has_label=label_num==2;
        label_target_num=0;
        label_point_num_vec.resize(label_num);
        label_min_data_vec.resize(label_num);
        label_max_data_vec.resize(label_num);
        label_abs_eb_vec.resize(label_num);
        label_codec_vec.resize(label_num);
        if(prediction_structure_vec.empty())
            throw runtime_error("UNSZ payload码流缺少预测结构");
        uint64_t metadata_point_num=0;
        for(uint32_t label_id=0;label_id<label_num;label_id++)
        {
            uint8_t source_label_id=read_binary<uint8_t>(cursor,end);
            if(label_id>=label_source_id_vec.size()||source_label_id!=label_source_id_vec[label_id])
                throw runtime_error("压缩文件的原始标签编号与当前field配置不一致");
            uint8_t effective_codec=read_binary<uint8_t>(cursor,end);
            if(effective_codec!=(uint8_t)label_config_vec[label_id].codec)
                throw runtime_error("压缩文件的effective codec与当前field配置不一致");
            uint8_t codec=read_binary<uint8_t>(cursor,end);
            if(codec!=(uint8_t)label_codec_e::constant&&codec!=(uint8_t)label_codec_e::rbf)
                throw runtime_error("压缩文件包含未知的标签编码方式");
            label_codec_vec[label_id]=(label_codec_e)codec;
            label_point_num_vec[label_id]=read_binary<uint32_t>(cursor,end);
            metadata_point_num+=label_point_num_vec[label_id];
            label_min_data_vec[label_id]=read_binary<T_data>(cursor,end);
            label_max_data_vec[label_id]=read_binary<T_data>(cursor,end);
        }
        if(metadata_point_num!=point_num)
            throw runtime_error("压缩文件中的各标签点数之和与总点数不一致");
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
        structure_label_quantized_error_vec.resize(payload_num,vector<vector<int32_t>>(label_num));
        structure_unpredictable_value_begin_vec.resize(payload_num);
        structure_unpredictable_value_end_vec.resize(payload_num);
        const unsigned char *payload_base=cursor;
        parallel_for_tasks(payload_num,thread_num,[&](uint32_t payload_id)
        {
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
            for(uint32_t label_id=0;label_id<label_num;label_id++)
            {
                uint64_t quantized_error_num=read_binary<uint64_t>(payload_cursor,payload_end);
                uint64_t zstd_byte_num=read_binary<uint64_t>(payload_cursor,payload_end);
                if(quantized_error_num>prediction_structure_vec[payload_id].target_ids.size()||(uint64_t)(payload_end-payload_cursor)<zstd_byte_num)
                    throw runtime_error("payload数值流不完整");
                if(quantized_error_num==0&&zstd_byte_num!=0)
                    throw runtime_error("空payload数值流具有非空Zstd字节流");
                if(quantized_error_num>0)
                {
                    vector<unsigned char> huffman_bytes=zstd_decompress(payload_cursor,zstd_byte_num);
                    structure_label_quantized_error_vec[payload_id][label_id]=huffman_decode<int32_t>(huffman_bytes.data(),huffman_bytes.size(),quantized_error_num);
                }
                payload_cursor+=zstd_byte_num;
            }
            if((payload_end-payload_cursor)%sizeof(T_data)!=0)
                throw runtime_error("payload不可预测值流的字节数不完整");
            structure_unpredictable_value_begin_vec[payload_id]=payload_cursor;
            structure_unpredictable_value_end_vec[payload_id]=payload_end;
        });
        for(const vector<uint8_t>&label_symbols:structure_label_symbols)
            label_target_num+=label_symbols.size();
        cursor=end;
    }

    template<typename T_coord,typename T_data>
    encoded_compression_payload_c param_c<T_coord,T_data>::encode_compression_payload(vector<uint8_t>&label_symbols,bool has_label,vector<vector<int32_t>>&label_quantized_error_vec,vector<T_data>&unpredictable_values)
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
        for(uint32_t label_id=0;label_id<label_quantized_error_vec.size();label_id++)
        {
            vector<int32_t> &quantized_error_vec=label_quantized_error_vec[label_id];
            vector<unsigned char> huffman_bytes=huffman_encode(quantized_error_vec);
            uint64_t huffman_byte_num=huffman_bytes.size();
            vector<unsigned char> zstd_bytes;
            if(!huffman_bytes.empty())
                zstd_bytes=zstd_compress(huffman_bytes,zstd_compression_level);
            vector<unsigned char>().swap(huffman_bytes);
            append_binary<uint64_t>(result.bytes,quantized_error_vec.size());
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
        uint32_t label_num=state.label_num;
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
        uint64_t pipeline_metadata_byte_num=current_field_uses_tensor?2*sizeof(uint8_t)+2*sizeof(uint32_t):3*sizeof(uint8_t);
        uint64_t container_byte_num=8+4*sizeof(uint32_t)+pipeline_metadata_byte_num+(uint64_t)label_num*(3*sizeof(uint8_t)+sizeof(uint32_t)+2*sizeof(T_data))+sizeof(uint32_t)+(uint64_t)(payload_num+1)*sizeof(uint64_t);
        for(const encoded_compression_payload_c&payload:payload_vec)
            container_byte_num+=payload.bytes.size();
        container.reserve((size_t)container_byte_num);
        container.insert(container.end(),compression_file_magic,compression_file_magic+8);
        append_binary<uint32_t>(container,compression_file_version);
        append_binary<uint32_t>(container,point_num);
        append_binary<uint32_t>(container,sizeof(T_data));
        append_binary<uint32_t>(container,label_num);
        append_binary<uint8_t>(container,current_field_uses_tensor?1:0);
        if(current_field_uses_tensor)
        {
            if(!tensor_pipeline_selected)
                throw runtime_error("写入压缩文件前尚未选择tensor pipeline");
            append_binary<uint8_t>(container,encode_tensor_pipeline(selected_tensor_pipeline));
            append_binary<uint32_t>(container,current_tensor_shape[0]);
            append_binary<uint32_t>(container,current_tensor_shape[1]);
        }
        else
        {
            append_binary<uint8_t>(container,encode_prediction_pipeline(label_pipeline));
            append_binary<uint8_t>(container,encode_prediction_pipeline(value_pipeline));
        }
        for(uint32_t label_id=0;label_id<label_num;label_id++)
        {
            append_binary<uint8_t>(container,label_source_id_vec[label_id]);
            append_binary<uint8_t>(container,(uint8_t)label_config_vec[label_id].codec);
            append_binary<uint8_t>(container,(uint8_t)label_codec_vec[label_id]);
            append_binary<uint32_t>(container,label_point_num_vec[label_id]);
            append_binary<T_data>(container,label_min_data_vec[label_id]);
            append_binary<T_data>(container,label_max_data_vec[label_id]);
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
        current_value_stream_bytes=(uint64_t)payload_num*label_num*2*sizeof(uint64_t)+value_zstd_byte_num+unpredictable_value_byte_num;
        current_shared_metadata_bytes=current_compressed_bytes.size()-current_label_stream_bytes-current_value_stream_bytes;
        state.lossless_encoding_time+=seconds_since(stage_start_time);
        current_compression_diagnostics.label_classification_time=state.label_classification_time;
        current_compression_diagnostics.label_setup_time=state.label_setup_time;
        current_compression_diagnostics.label_prediction_time=state.label_prediction_time;
        current_compression_diagnostics.value_prediction_time=state.value_prediction_time;
        current_compression_diagnostics.lossless_encoding_time=state.lossless_encoding_time;
        current_compression_diagnostics.label_target_num=state.label_target_num;
        current_compression_diagnostics.label_error_num=state.label_error_num;
        current_compression_diagnostics.tensor_prediction_num=state.tensor_prediction_num;
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
        if(current_prediction_framework.value_mode!=slice_prediction_mode_e::independent)
            printf("cross_time_prediction_num=%llu, ratio=%f\n",(unsigned long long)diagnostics.tensor_prediction_num,(double)diagnostics.tensor_prediction_num/point_num);
        printf("label_prediction_error_ratio=%f, label_position_delta_bytes=%llu, label_zstd_bytes=%llu, zstd_level=%d\n",label_prediction_error_ratio,(unsigned long long)diagnostics.label_position_delta_byte_num,(unsigned long long)diagnostics.label_zstd_byte_num,zstd_compression_level);
        printf("value_huffman_bytes=%llu, value_zstd_bytes=%llu, unpredictable_value_bytes=%llu, file_bytes=%llu\n",(unsigned long long)diagnostics.value_huffman_byte_num,(unsigned long long)diagnostics.value_zstd_byte_num,(unsigned long long)diagnostics.unpredictable_value_byte_num,(unsigned long long)current_compressed_bytes.size());
        printf("label_stream_bytes=%llu, value_stream_bytes=%llu, shared_metadata_bytes=%llu\n",(unsigned long long)current_label_stream_bytes,(unsigned long long)current_value_stream_bytes,(unsigned long long)current_shared_metadata_bytes);
        printf("label_num=%u",(uint32_t)label_codec_vec.size());
        for(uint32_t label_id=0;label_id<label_codec_vec.size();label_id++)
        {
            const char *codec_name;
            if(label_codec_vec[label_id]==label_codec_e::constant)
                codec_name="constant";
            else if(label_codec_vec[label_id]==label_codec_e::rbf)
                codec_name="rbf";
            else
                throw runtime_error("未知的标签编码方式");
            printf(", set%u_num=%u, set%u_codec=%s",label_id,label_point_num_vec[label_id],label_id,codec_name);
        }
        printf("\n");
        if(print_compression_stage_time)
        {
            double label_time=diagnostics.label_classification_time+diagnostics.label_setup_time;
            printf("compression stage time: label=%f s, label_prediction=%f s, value_prediction=%f s, lossless_encoding=%f s\n",label_time,diagnostics.label_prediction_time,diagnostics.value_prediction_time,diagnostics.lossless_encoding_time);
            printf("label stage time: classification=%f s, setup=%f s\n",diagnostics.label_classification_time,diagnostics.label_setup_time);
        }
    }

    template<typename T_coord,typename T_data>
    double param_c<T_coord,T_data>::read_compressed_field_metadata()
    {
        if(original_data_path_vec.empty())
            throw runtime_error("自动pipeline解压缺少数据文件路径");
        constexpr uint64_t maximum_header_byte_num=8+6*sizeof(uint32_t)+3*sizeof(uint8_t)+2*(3*sizeof(uint8_t)+sizeof(uint32_t)+2*sizeof(T_data));
        vector<unsigned char> bytes(maximum_header_byte_num);
        string compressed_path=generated_path(original_data_path_vec.front(),".unsz");
        auto input_start_time=chrono::steady_clock::now();
        FILE*file=fopen(compressed_path.c_str(),"rb");
        if(!file)
            throw runtime_error("找不到文件："+compressed_path);
        uint64_t read_byte_num=fread(bytes.data(),1,bytes.size(),file);
        fclose(file);
        double input_time=seconds_since(input_start_time);
        constexpr uint64_t fixed_header_byte_num=8+4*sizeof(uint32_t)+2*sizeof(uint8_t);
        if(read_byte_num<fixed_header_byte_num)
            throw runtime_error("UNSZ压缩文件头不完整："+compressed_path);
        bytes.resize(read_byte_num);
        const unsigned char*cursor=bytes.data(),*end=bytes.data()+bytes.size();
        if((uint64_t)(end-cursor)<8||memcmp(cursor,compression_file_magic,8)!=0)
            throw runtime_error("不是受支持的UNSZ标签文件");
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
        uint32_t label_num=read_binary<uint32_t>(cursor,end);
        if(label_num==0||label_num>2)
            throw runtime_error("压缩文件包含无效标签数量");
        bool file_uses_tensor=read_binary<uint8_t>(cursor,end)!=0;
        if(file_uses_tensor!=current_field_uses_tensor)
            throw runtime_error("压缩文件的tensor模式与解压参数不一致");
        if(file_uses_tensor)
        {
            selected_tensor_pipeline=decode_tensor_pipeline(read_binary<uint8_t>(cursor,end));
            tensor_pipeline_selected=true;
            array<uint32_t,2> file_tensor_shape{{read_binary<uint32_t>(cursor,end),read_binary<uint32_t>(cursor,end)}};
            if(file_tensor_shape!=current_tensor_shape)
                throw runtime_error("压缩文件记录的tensor形状与metadata不一致");
        }
        else
        {
            string file_label_pipeline=decode_prediction_pipeline(read_binary<uint8_t>(cursor,end));
            string file_value_pipeline=decode_prediction_pipeline(read_binary<uint8_t>(cursor,end));
            if(label_pipeline=="auto")
                label_pipeline=file_label_pipeline;
            else if(label_pipeline!=file_label_pipeline)
                throw runtime_error("压缩文件记录的label pipeline与解压参数不一致");
            if(value_pipeline=="auto")
                value_pipeline=file_value_pipeline;
            else if(value_pipeline!=file_value_pipeline)
                throw runtime_error("压缩文件记录的value pipeline与解压参数不一致");
        }
        vector<uint8_t> source_ids(label_num),effective_codecs(label_num);
        for(uint32_t label_id=0;label_id<label_num;label_id++)
        {
            source_ids[label_id]=read_binary<uint8_t>(cursor,end);
            effective_codecs[label_id]=read_binary<uint8_t>(cursor,end);
            read_binary<uint8_t>(cursor,end);
            read_binary<uint32_t>(cursor,end);
            read_binary<T_data>(cursor,end);
            read_binary<T_data>(cursor,end);
            if(source_ids[label_id]>=configured_label_config_vec.size())
                throw runtime_error("压缩文件的原始标签编号越界");
            if(effective_codecs[label_id]!=(uint8_t)label_codec_e::constant&&effective_codecs[label_id]!=(uint8_t)label_codec_e::rbf)
                throw runtime_error("压缩文件包含未知的effective codec");
        }
        label_config_vec.clear();
        for(uint32_t label_id=0;label_id<label_num;label_id++)
        {
            label_config_vec.push_back(configured_label_config_vec[source_ids[label_id]]);
            label_config_vec.back().codec=(label_codec_e)effective_codecs[label_id];
        }
        label_source_id_vec=move(source_ids);
        if(label_num==1)
        {
            label_rule_vec.clear();
            label_pipeline="independent";
        }
        else
            label_rule_vec=configured_label_rule_vec;
        field_label_min_data={{numeric_limits<double>::infinity(),numeric_limits<double>::infinity()}};
        field_label_max_data={{-numeric_limits<double>::infinity(),-numeric_limits<double>::infinity()}};
        if(any_of(label_config_vec.begin(),label_config_vec.end(),[](const label_config_c&config){return config.error_mode==label_error_mode_e::field_relative;}))
        {
            // FIELD_REL的ABS EB依赖整个field的range，因此读取全部slice文件头中的各标签min/max；文件读取不计时
            for(const string&data_path:original_data_path_vec)
            {
                string slice_path=generated_path(data_path,".unsz");
                vector<unsigned char> slice_bytes(maximum_header_byte_num);
                auto slice_input_start_time=chrono::steady_clock::now();
                FILE*slice_file=fopen(slice_path.c_str(),"rb");
                if(!slice_file)
                    throw runtime_error("找不到文件："+slice_path);
                slice_bytes.resize(fread(slice_bytes.data(),1,slice_bytes.size(),slice_file));
                fclose(slice_file);
                input_time+=seconds_since(slice_input_start_time);
                const unsigned char*slice_cursor=slice_bytes.data(),*slice_end=slice_bytes.data()+slice_bytes.size();
                if((uint64_t)(slice_end-slice_cursor)<8||memcmp(slice_cursor,compression_file_magic,8)!=0)
                    throw runtime_error("不是受支持的UNSZ标签文件："+slice_path);
                slice_cursor+=8;
                for(uint32_t header_value_id=0;header_value_id<3;header_value_id++)
                    read_binary<uint32_t>(slice_cursor,slice_end);
                if(read_binary<uint32_t>(slice_cursor,slice_end)!=label_num)
                    throw runtime_error("同一field的slice标签数量不一致："+slice_path);
                if(read_binary<uint8_t>(slice_cursor,slice_end))
                {
                    read_binary<uint8_t>(slice_cursor,slice_end);
                    read_binary<uint32_t>(slice_cursor,slice_end);
                    read_binary<uint32_t>(slice_cursor,slice_end);
                }
                else
                {
                    read_binary<uint8_t>(slice_cursor,slice_end);
                    read_binary<uint8_t>(slice_cursor,slice_end);
                }
                for(uint32_t label_id=0;label_id<label_num;label_id++)
                {
                    for(uint32_t codec_byte_id=0;codec_byte_id<3;codec_byte_id++)
                        read_binary<uint8_t>(slice_cursor,slice_end);
                    uint32_t slice_point_num=read_binary<uint32_t>(slice_cursor,slice_end);
                    T_data slice_min=read_binary<T_data>(slice_cursor,slice_end),slice_max=read_binary<T_data>(slice_cursor,slice_end);
                    if(slice_point_num==0)
                        continue;
                    field_label_min_data[label_id]=min(field_label_min_data[label_id],(double)slice_min);
                    field_label_max_data[label_id]=max(field_label_max_data[label_id],(double)slice_max);
                }
            }
        }
        return input_time;
    }

}
