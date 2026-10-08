#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include "unsz_core/unsz.hpp"

using namespace unsz;

struct sample_field_c
{
    string name,path,height_path;
    uint32_t time_num=0,height_num=0,sample_num=0;
};

struct estimate_result_c
{
    uint64_t possible_target_num=0,target_num=0,unpredictable_num=0,huffman_bytes=0,zstd_bytes=0,raw_unpredictable_bytes=0;
    double entropy_bits_per_value=0,encoded_bits_per_value=0,normalized_prediction_rmse=0;
};

template<typename T>
vector<T> read_exact(const string&path,uint64_t value_num)
{
    ifstream input(path,ios::binary|ios::ate);
    if(!input)
        throw runtime_error("无法读取文件："+path);
    uint64_t byte_num=(uint64_t)input.tellg();
    if(byte_num!=value_num*sizeof(T))
        throw runtime_error("文件大小不正确："+path);
    input.seekg(0);
    vector<T> values(value_num);
    input.read((char*)values.data(),byte_num);
    if(!input)
        throw runtime_error("文件读取失败："+path);
    return values;
}

vector<sample_field_c> read_manifest(const string&path)
{
    ifstream input(path);
    if(!input)
        throw runtime_error("无法读取manifest："+path);
    string line;
    getline(input,line);
    vector<sample_field_c> fields;
    while(getline(input,line))
    {
        if(line.empty())
            continue;
        string token;
        stringstream parser(line);
        sample_field_c field;
        getline(parser,field.name,'\t');
        getline(parser,field.path,'\t');
        getline(parser,field.height_path,'\t');
        getline(parser,token,'\t');field.time_num=stoul(token);
        getline(parser,token,'\t');field.height_num=stoul(token);
        getline(parser,token,'\t');field.sample_num=stoul(token);
        fields.push_back(move(field));
    }
    return fields;
}

inline uint64_t data_index(uint32_t time_index,uint32_t height_index,uint32_t sample_index,uint32_t height_num,uint32_t sample_num)
{
    return ((uint64_t)time_index*height_num+height_index)*sample_num+sample_index;
}

estimate_result_c estimate_axis(const vector<float>&data,const vector<double>&height_coordinates,uint32_t time_num,uint32_t height_num,uint32_t sample_num,bool time_axis,uint32_t predictor_order,double rel_eb)
{
    vector<float> slice_abs_eb((uint64_t)time_num*height_num);
    for(uint32_t time_index=0;time_index<time_num;time_index++)
        for(uint32_t height_index=0;height_index<height_num;height_index++)
        {
            auto begin=data.begin()+data_index(time_index,height_index,0,height_num,sample_num);
            auto bounds=minmax_element(begin,begin+sample_num);
            slice_abs_eb[(uint64_t)time_index*height_num+height_index]=(*bounds.second-*bounds.first)*(float)rel_eb;
        }
    vector<float> reconstructed=data,unpredictable_values;
    vector<int32_t> qbs;
    uint64_t maximum_target_num=(uint64_t)time_num*height_num*sample_num;
    qbs.reserve(maximum_target_num);
    double normalized_squared_prediction_error=0;
    uint64_t normalized_error_num=0;
    uint32_t root_num=predictor_order+1;
    auto predict=[&](uint32_t time_index,uint32_t height_index,uint32_t sample_index)
    {
        if(time_axis)
        {
            float previous=reconstructed[data_index(time_index-1,height_index,sample_index,height_num,sample_num)];
            if(predictor_order==0)
                return previous;
            float previous2=reconstructed[data_index(time_index-2,height_index,sample_index,height_num,sample_num)];
            return previous+(previous-previous2);
        }
        float previous=reconstructed[data_index(time_index,height_index-1,sample_index,height_num,sample_num)];
        if(predictor_order==0)
            return previous;
        float previous2=reconstructed[data_index(time_index,height_index-2,sample_index,height_num,sample_num)];
        double ratio=(height_coordinates[height_index]-height_coordinates[height_index-1])/(height_coordinates[height_index-1]-height_coordinates[height_index-2]);
        return previous+(float)ratio*(previous-previous2);
    };
    if(time_axis)
    {
        for(uint32_t time_index=root_num;time_index<time_num;time_index++)
            for(uint32_t height_index=0;height_index<height_num;height_index++)
            {
                float abs_eb=slice_abs_eb[(uint64_t)time_index*height_num+height_index];
                if(!(abs_eb>0))
                    continue;
                float inverse_double_abs_eb=(float).5/abs_eb;
                for(uint32_t sample_index=0;sample_index<sample_num;sample_index++)
                {
                    uint64_t index=data_index(time_index,height_index,sample_index,height_num,sample_num);
                    float prediction=predict(time_index,height_index,sample_index),range=abs_eb/(float)rel_eb;
                    double normalized_error=(data[index]-prediction)/range;
                    normalized_squared_prediction_error+=normalized_error*normalized_error;
                    normalized_error_num++;
                    qbs.push_back(quantize_or_store_unpredictable(data[index],prediction,abs_eb,inverse_double_abs_eb,unpredictable_values,reconstructed[index]));
                }
            }
    }
    else
    {
        for(uint32_t height_index=root_num;height_index<height_num;height_index++)
            for(uint32_t time_index=0;time_index<time_num;time_index++)
            {
                float abs_eb=slice_abs_eb[(uint64_t)time_index*height_num+height_index];
                if(!(abs_eb>0))
                    continue;
                float inverse_double_abs_eb=(float).5/abs_eb;
                for(uint32_t sample_index=0;sample_index<sample_num;sample_index++)
                {
                    uint64_t index=data_index(time_index,height_index,sample_index,height_num,sample_num);
                    float prediction=predict(time_index,height_index,sample_index),range=abs_eb/(float)rel_eb;
                    double normalized_error=(data[index]-prediction)/range;
                    normalized_squared_prediction_error+=normalized_error*normalized_error;
                    normalized_error_num++;
                    qbs.push_back(quantize_or_store_unpredictable(data[index],prediction,abs_eb,inverse_double_abs_eb,unpredictable_values,reconstructed[index]));
                }
            }
    }
    estimate_result_c result;
    result.possible_target_num=(uint64_t)(time_axis?time_num-root_num:time_num)*(time_axis?height_num:height_num-root_num)*sample_num;
    result.target_num=qbs.size();
    result.unpredictable_num=unpredictable_values.size();
    if(qbs.empty())
        return result;
    unordered_map<int32_t,uint64_t> frequencies;
    for(int32_t qb:qbs)
        frequencies[qb]++;
    double entropy=0;
    for(const auto&entry:frequencies)
    {
        double probability=(double)entry.second/qbs.size();
        entropy-=probability*log2(probability);
    }
    vector<unsigned char> huffman_bytes=huffman_encode(qbs),zstd_bytes=zstd_compress(huffman_bytes);
    result.huffman_bytes=huffman_bytes.size();
    result.zstd_bytes=zstd_bytes.size();
    result.raw_unpredictable_bytes=unpredictable_values.size()*sizeof(float);
    result.entropy_bits_per_value=entropy;
    result.encoded_bits_per_value=(double)(result.zstd_bytes+result.raw_unpredictable_bytes)*8/result.target_num;
    result.normalized_prediction_rmse=sqrt(normalized_squared_prediction_error/normalized_error_num);
    return result;
}

int main(int argc,char**argv)
{
    try
    {
        if(argc!=4)
            throw runtime_error("用法：e1_compare_time_height manifest.tsv output.csv rel_eb");
        vector<sample_field_c> fields=read_manifest(argv[1]);
        ofstream output(argv[2]);
        if(!output)
            throw runtime_error("无法写入结果CSV");
        double rel_eb=stod(argv[3]);
        output<<"field,height_num,axis,predictor,applicable,possible_target_num,target_num,target_coverage,unpredictable_num,huffman_bytes,zstd_bytes,raw_unpredictable_bytes,entropy_bits_per_value,encoded_bits_per_value,normalized_prediction_rmse\n";
        for(const sample_field_c&field:fields)
        {
            printf("读取field：%s\n",field.name.c_str());
            vector<float> data=read_exact<float>(field.path,(uint64_t)field.time_num*field.height_num*field.sample_num);
            vector<double> heights=read_exact<double>(field.height_path,field.height_num);
            for(uint32_t predictor_order=0;predictor_order<=1;predictor_order++)
                for(bool time_axis:{true,false})
                {
                    estimate_result_c result=estimate_axis(data,heights,field.time_num,field.height_num,field.sample_num,time_axis,predictor_order,rel_eb);
                    double coverage=result.possible_target_num?(double)result.target_num/result.possible_target_num:0;
                    output<<field.name<<','<<field.height_num<<','<<(time_axis?"time":"height")<<','<<(predictor_order==0?"previous":"linear_extrapolation")<<','<<(result.target_num?1:0)<<','<<result.possible_target_num<<','<<result.target_num<<','<<setprecision(12)<<coverage<<','<<result.unpredictable_num<<','<<result.huffman_bytes<<','<<result.zstd_bytes<<','<<result.raw_unpredictable_bytes<<','<<result.entropy_bits_per_value<<','<<result.encoded_bits_per_value<<','<<result.normalized_prediction_rmse<<'\n';
                    printf("  axis=%s,predictor=%s,coverage=%.6f,bits/value=%.6f,normalized_rmse=%.6g\n",time_axis?"time":"height",predictor_order==0?"previous":"linear_extrapolation",coverage,result.encoded_bits_per_value,result.normalized_prediction_rmse);
                }
        }
        return 0;
    }
    catch(const exception&error)
    {
        fprintf(stderr,"错误：%s\n",error.what());
        return 1;
    }
}
