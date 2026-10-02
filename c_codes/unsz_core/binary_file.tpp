#pragma once

namespace unsz
{
    template<typename T>
    void read_binary_file_exact(const string&path,T*output,uint64_t value_num)
    {
        FILE*file=fopen(path.c_str(),"rb");
        if(!file)
            throw runtime_error("找不到文件："+path);
        if(fseek(file,0,SEEK_END)!=0)
        {
            fclose(file);
            throw runtime_error("无法定位到文件末尾："+path);
        }
        long byte_num_signed=ftell(file);
        uint64_t expected_byte_num=value_num*sizeof(T);
        if(byte_num_signed<0||(uint64_t)byte_num_signed!=expected_byte_num)
        {
            fclose(file);
            throw runtime_error("文件大小与预期不一致："+path);
        }
        if(fseek(file,0,SEEK_SET)!=0||fread(output,sizeof(T),value_num,file)!=value_num)
        {
            fclose(file);
            throw runtime_error("读取文件失败："+path);
        }
        if(fclose(file)!=0)
            throw runtime_error("关闭文件失败："+path);
    }

    template<typename T>
    bool try_read_binary_file_exact(const string&path,T*output,uint64_t value_num)
    {
        if(!filesystem::exists(path))
            return false;
        read_binary_file_exact(path,output,value_num);
        return true;
    }

    template<typename T>
    void write_binary_file(const string&path,const T*data,uint64_t value_num)
    {
        FILE*file=fopen(path.c_str(),"wb");
        if(!file)
            throw runtime_error("无法创建文件："+path);
        if(fwrite(data,sizeof(T),value_num,file)!=value_num)
        {
            fclose(file);
            throw runtime_error("写入文件失败："+path);
        }
        if(fclose(file)!=0)
            throw runtime_error("关闭文件失败："+path);
    }

    inline vector<unsigned char> read_byte_file(const string&path)
    {
        FILE*file=fopen(path.c_str(),"rb");
        if(!file)
            throw runtime_error("找不到文件："+path);
        if(fseek(file,0,SEEK_END)!=0)
        {
            fclose(file);
            throw runtime_error("无法定位到文件末尾："+path);
        }
        long byte_num_signed=ftell(file);
        if(byte_num_signed<0)
        {
            fclose(file);
            throw runtime_error("无法读取文件大小："+path);
        }
        vector<unsigned char>bytes((uint64_t)byte_num_signed);
        if(fseek(file,0,SEEK_SET)!=0||fread(bytes.data(),1,bytes.size(),file)!=bytes.size())
        {
            fclose(file);
            throw runtime_error("读取文件失败："+path);
        }
        if(fclose(file)!=0)
            throw runtime_error("关闭文件失败："+path);
        return bytes;
    }
}
