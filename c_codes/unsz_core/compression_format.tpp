#pragma once

namespace unsz
{
    inline constexpr unsigned char compression_file_magic[8]={'U','N','S','Z','P','S','1','1'};
    inline constexpr uint32_t compression_file_version=11;

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
    vector<unsigned char> huffman_encode(vector<T> &value_vec,T *min_value=nullptr,T *max_value=nullptr)
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
        if (use_partition_adaptive_predictor_experimental)
        {
            uint32_t block_config_num=read_binary<uint32_t>(cursor,end);
            if (block_config_num!=partition_point_ids_vec.size() || (uint64_t)(end-cursor)<block_config_num)
                throw runtime_error("partition预测配置数量与partition数量不一致");
            partition_prediction_config_id_vec.assign(cursor,cursor+block_config_num);
            cursor+=block_config_num;
            uint32_t candidate_num=get_partition_prediction_config_vec().size();
            for (uint8_t config_id:partition_prediction_config_id_vec)
                if (config_id>=candidate_num)
                    throw runtime_error("压缩文件中的partition预测配置编号越界");
            prepare_partition_adaptive_sample_mask();
        }

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
            uint8_t codec=read_binary<uint8_t>(cursor,end);
            if(codec!=(uint8_t)point_set_codec_e::idw&&codec!=(uint8_t)point_set_codec_e::constant&&codec!=(uint8_t)point_set_codec_e::rbf)
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
            vector<unsigned char> zstd_bytes;
            if(!delta_bytes.empty())
                zstd_bytes=zstd_compress(delta_bytes,zstd_compression_level);
            append_binary<uint64_t>(result.bytes,label_symbols.size());
            append_binary<uint64_t>(result.bytes,zstd_bytes.size());
            result.bytes.insert(result.bytes.end(),zstd_bytes.begin(),zstd_bytes.end());
            result.label_position_delta_byte_num+=delta_bytes.size();
            result.label_zstd_byte_num+=zstd_bytes.size();
        }
        for(uint32_t point_set_id=0;point_set_id<point_set_qb_vec.size();point_set_id++)
        {
            vector<int32_t> &qb_vec=point_set_qb_vec[point_set_id];
            vector<unsigned char> huffman_bytes=huffman_encode(qb_vec);
            vector<unsigned char> zstd_bytes;
            if(!huffman_bytes.empty())
                zstd_bytes=zstd_compress(huffman_bytes,zstd_compression_level);
            append_binary<uint64_t>(result.bytes,qb_vec.size());
            append_binary<uint64_t>(result.bytes,zstd_bytes.size());
            result.bytes.insert(result.bytes.end(),zstd_bytes.begin(),zstd_bytes.end());
            result.value_huffman_byte_num+=huffman_bytes.size();
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
        uint64_t container_byte_num=8+4*sizeof(uint32_t)+(uint64_t)point_set_num*(sizeof(uint8_t)+sizeof(uint32_t)+2*sizeof(T_data))+sizeof(uint32_t)+(uint64_t)(payload_num+1)*sizeof(uint64_t);
        if(use_partition_adaptive_predictor_experimental)
            container_byte_num+=sizeof(uint32_t)+partition_prediction_config_id_vec.size();
        for(const encoded_compression_payload_c&payload:payload_vec)
            container_byte_num+=payload.bytes.size();
        container.reserve((size_t)container_byte_num);
        container.insert(container.end(),compression_file_magic,compression_file_magic+8);
        append_binary<uint32_t>(container,compression_file_version);
        append_binary<uint32_t>(container,point_num);
        append_binary<uint32_t>(container,sizeof(T_data));
        append_binary<uint32_t>(container,point_set_num);
        if(use_partition_adaptive_predictor_experimental)
        {
            append_binary<uint32_t>(container,partition_prediction_config_id_vec.size());
            container.insert(container.end(),partition_prediction_config_id_vec.begin(),partition_prediction_config_id_vec.end());
        }
        for(uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
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
            container.insert(container.end(),payload.bytes.begin(),payload.bytes.end());
        current_compressed_bytes=move(container);
        state.lossless_encoding_time+=chrono::duration<double>(chrono::steady_clock::now()-stage_start_time).count();
        double label_prediction_error_ratio=0;
        if(state.label_target_num>0)
            label_prediction_error_ratio=(double)state.label_error_num/state.label_target_num;
        printf("label_prediction_error_ratio=%f, label_position_delta_bytes=%llu, label_zstd_bytes=%llu, zstd_level=%d\n",label_prediction_error_ratio,(unsigned long long)label_position_delta_byte_num,(unsigned long long)label_zstd_byte_num,zstd_compression_level);
        printf("value_huffman_bytes=%llu, value_zstd_bytes=%llu, unpredictable_value_bytes=%llu, file_bytes=%llu\n",(unsigned long long)value_huffman_byte_num,(unsigned long long)value_zstd_byte_num,(unsigned long long)unpredictable_value_byte_num,(unsigned long long)current_compressed_bytes.size());

        printf("point_set_num=%u",point_set_num);
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            const char *codec_name;
            if(point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                codec_name="constant";
            else if(point_set_codec_vec[point_set_id]==point_set_codec_e::rbf)
                codec_name="rbf";
            else
                codec_name="idw";
            printf(", set%u_num=%u, set%u_codec=%s",point_set_id,point_set_point_num_vec[point_set_id],point_set_id,codec_name);
        }
        printf("\n");
        if (print_compression_stage_time)
        {
            double point_set_time=state.point_set_classification_time+state.point_set_setup_time;
            printf("compression stage time: point_set=%f s, label_prediction=%f s, value_prediction=%f s, lossless_encoding=%f s\n",point_set_time,state.label_prediction_time,state.value_prediction_time,state.lossless_encoding_time);
            printf("point set stage time: classification=%f s, setup=%f s\n",state.point_set_classification_time,state.point_set_setup_time);
        }
    }

}
