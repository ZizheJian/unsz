#pragma once

namespace unsz
{
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
    vector<unsigned char> huffman_encode(vector<T> &value_vec)
    {
        if (value_vec.empty())
            return {};
        huffman_encoder_class<T> encoder;
        encoder.preprocess_encode(value_vec.data(),value_vec.size());
        vector<unsigned char> bytes(max<uint64_t>(4096,(uint64_t)value_vec.size()*64));
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

    inline vector<unsigned char> zstd_compress(vector<unsigned char> &source,int compression_level=3)
    {
        vector<unsigned char> result(ZSTD_compressBound(source.size()));
        uint64_t result_size=ZSTD_compress(result.data(),result.size(),source.data(),source.size(),compression_level);
        if (ZSTD_isError(result_size))
            throw runtime_error("Zstd压缩失败："+string(ZSTD_getErrorName(result_size)));
        result.resize(result_size);
        return result;
    }

    inline vector<unsigned char> zstd_decompress(vector<unsigned char> &source)
    {
        uint64_t result_size=ZSTD_getFrameContentSize(source.data(),source.size());
        if (result_size==ZSTD_CONTENTSIZE_ERROR || result_size==ZSTD_CONTENTSIZE_UNKNOWN)
            throw runtime_error("无法从Zstd帧中读取解压大小");
        vector<unsigned char> result(result_size);
        uint64_t actual_size=ZSTD_decompress(result.data(),result.size(),source.data(),source.size());
        if (ZSTD_isError(actual_size))
            throw runtime_error("Zstd解压失败："+string(ZSTD_getErrorName(actual_size)));
        if (actual_size!=result_size)
            throw runtime_error("Zstd解压后的大小与帧记录不一致");
        return result;
    }

    template<typename T>
    inline vector<unsigned char> binary_to_byte_vec(vector<T> &value_vec)
    {
        vector<unsigned char> result(value_vec.size());
        for (uint64_t i=0;i<value_vec.size();i++)
        {
            if (value_vec[i]>1)
                throw runtime_error("二进制标签只能是0或1");
            result[i]=value_vec[i];
        }
        return result;
    }

    template<typename T>
    inline vector<unsigned char> binary_to_bit_vec(vector<T> &value_vec)
    {
        vector<unsigned char> result((value_vec.size()+7)/8,0);
        for (uint64_t i=0;i<value_vec.size();i++)
        {
            if (value_vec[i]>1)
                throw runtime_error("二进制标签只能是0或1");
            result[i/8]|=value_vec[i]<<(i%8);
        }
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
    inline vector<unsigned char> binary_one_position_delta_vec(vector<T> &value_vec)
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

    inline vector<uint32_t> decode_binary_one_position_delta_vec(vector<unsigned char> &bytes,uint64_t value_num)
    {
        const unsigned char *cursor=bytes.data(),*end=bytes.data()+bytes.size();
        uint64_t one_num=read_varuint(cursor,end);
        if (one_num>value_num)
            throw runtime_error("二进制标签中1的数量超过标签总数");
        vector<uint32_t> result(value_num,0);
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

    template<typename T>
    inline vector<unsigned char> binary_run_length_vec(vector<T> &value_vec)
    {
        vector<unsigned char> result;
        if (value_vec.empty())
            return result;
        result.push_back(value_vec[0]);
        uint64_t run_length=1;
        for (uint64_t i=1;i<value_vec.size();i++)
            if (value_vec[i]==value_vec[i-1])
                run_length++;
            else
            {
                append_varuint(result,run_length);
                run_length=1;
            }
        append_varuint(result,run_length);
        return result;
    }

    inline uint64_t zstd_size(vector<unsigned char> &source)
    {
        return source.empty()?0:zstd_compress(source).size();
    }
}
