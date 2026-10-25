#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

#if defined(FIELD_RELATIVE_SZ3)
#include <SZ3/api/sz.hpp>
using codec_config=SZ3::Config;
constexpr const char*codec_name="sz3-nd";
#elif defined(FIELD_RELATIVE_QOZ)
#include <QoZ/api/sz.hpp>
using codec_config=QoZ::Config;
constexpr const char*codec_name="qoz2-q4-nd";
#else
#error "Select exactly one field-relative codec"
#endif

using namespace std;
using clock_type=chrono::steady_clock;

uint64_t fnv1a(const vector<float>&data)
{
    const unsigned char*bytes=reinterpret_cast<const unsigned char*>(data.data());
    uint64_t hash=1469598103934665603ULL;
    for(size_t index=0;index<data.size()*sizeof(float);index++)
    {
        hash^=bytes[index];
        hash*=1099511628211ULL;
    }
    return hash;
}

void write_json_number(ostream&output,double value)
{
    if(isfinite(value))output<<value;
    else output<<"null";
}

string shape_text(const vector<size_t>&dims)
{
    string text="[";
    for(size_t index=0;index<dims.size();index++)text+=(index?",":"")+to_string(dims[index]);
    return text+"]";
}

int main(int argc,char**argv)
{
    try
    {
        if(argc!=7&&argc!=8)
            throw runtime_error("Usage: benchmark_<codec> <data-dir> <field> <time-num> <height-num> <rel-eb> <output-json> [point-limit]");
        string data_dir=argv[1],field=argv[2],output_path=argv[6];
        size_t time_num=stoull(argv[3]),height_num=stoull(argv[4]),slice_num=time_num*height_num,point_limit=argc==8?stoull(argv[7]):0;
        double rel_eb=stod(argv[5]);
        if(!time_num||!height_num||slice_num/height_num!=time_num||!(rel_eb>0)||!isfinite(rel_eb))
            throw runtime_error("Invalid dimensions or relative error bound");
        size_t point_num=0;
        vector<float>data;
        double field_min=numeric_limits<double>::infinity(),field_max=-numeric_limits<double>::infinity();
        uint64_t valid_value_num=0;
        for(size_t slice_id=0;slice_id<slice_num;slice_id++)
        {
            string path=data_dir+"/"+field+".dat."+to_string(slice_id);
            ifstream input(path,ios::binary|ios::ate);
            if(!input)throw runtime_error("Cannot open input: "+path);
            streamoff byte_num=input.tellg();
            if(byte_num<=0||byte_num%sizeof(float))throw runtime_error("Invalid float32 input size: "+path);
            size_t file_point_num=(size_t)byte_num/sizeof(float);
            if(!slice_id)
            {
                point_num=point_limit?min(point_limit,file_point_num):file_point_num;
                if(!point_num||slice_num>numeric_limits<size_t>::max()/point_num)throw runtime_error("Invalid total element count");
                data.resize(slice_num*point_num);
            }
            else if(file_point_num<point_num||(!point_limit&&file_point_num!=point_num))throw runtime_error("Inconsistent input size: "+path);
            input.seekg(0);
            input.read(reinterpret_cast<char*>(data.data()+slice_id*point_num),point_num*sizeof(float));
            if(!input)throw runtime_error("Input read failed: "+path);
            for(size_t point_id=0;point_id<point_num;point_id++)
            {
                float value=data[slice_id*point_num+point_id];
                if(!isfinite(value))throw runtime_error("Non-finite input value: "+path);
                if(value!=-99999.0f)
                {
                    field_min=min(field_min,(double)value);
                    field_max=max(field_max,(double)value);
                    valid_value_num++;
                }
            }
        }
        if(!valid_value_num)throw runtime_error("Input has no ordinary values");
        double field_range=field_max-field_min,abs_eb=rel_eb*field_range;
        if(!(abs_eb>=0)||!isfinite(abs_eb))throw runtime_error("Invalid absolute error bound");
        uint64_t input_hash=fnv1a(data);
        bool use_2d=height_num==1;
        codec_config config=use_2d?codec_config(time_num,point_num):codec_config(time_num,height_num,point_num);
#if defined(FIELD_RELATIVE_SZ3)
        config.errorBoundMode=SZ3::EB_ABS;
        config.absErrorBound=abs_eb;
        config.openmp=false;
#else
        config.errorBoundMode=QoZ::EB_ABS;
        config.absErrorBound=abs_eb;
        config.QoZ=4;
        config.tuningTarget=QoZ::TUNING_TARGET_RD;
        config.openmp=false;
#endif
        size_t compressed_bytes=0;
        auto start=clock_type::now();
        unique_ptr<char[]>encoded(SZ_compress<float>(config,data.data(),compressed_bytes));
        double compression_time=chrono::duration<double>(clock_type::now()-start).count();
        if(!encoded||!compressed_bytes)throw runtime_error("Compression returned an empty stream");
        bool input_unchanged=fnv1a(data)==input_hash;
        if(!input_unchanged)throw runtime_error("Compression modified the input array");
        vector<float>reconstructed(data.size());
        codec_config decoded_config;
        float*reconstructed_pointer=reconstructed.data();
        start=clock_type::now();
        SZ_decompress<float>(decoded_config,encoded.get(),compressed_bytes,reconstructed_pointer);
        double decompression_time=chrono::duration<double>(clock_type::now()-start).count();
#if defined(FIELD_RELATIVE_SZ3)
        if(decoded_config.dims!=config.dims||decoded_config.num!=data.size())throw runtime_error("Decompressed shape mismatch: expected="+shape_text(config.dims)+",decoded="+shape_text(decoded_config.dims)+",expected_num="+to_string(data.size())+",decoded_num="+to_string(decoded_config.num));
#endif
        double squared_error=0,max_error=0;
        bool finite=true;
        for(size_t index=0;index<data.size();index++)
        {
            finite&=isfinite(reconstructed[index]);
            double error=abs((double)data[index]-reconstructed[index]);
            squared_error+=error*error;
            max_error=max(max_error,error);
        }
        double max_error_over_eb=abs_eb>0?max_error/abs_eb:max_error==0?0:numeric_limits<double>::infinity();
        bool valid=finite&&max_error_over_eb<=1.01;
        double mse=squared_error/data.size(),psnr=mse?10*log10(field_range*field_range/mse):numeric_limits<double>::infinity();
        rusage usage{};
        getrusage(RUSAGE_SELF,&usage);
        ofstream output(output_path);
        if(!output)throw runtime_error("Cannot open output JSON: "+output_path);
        output<<setprecision(17)<<"{\n"
            <<"  \"field\": \""<<field<<"\",\n"
            <<"  \"compressor\": \""<<codec_name<<"\",\n"
            <<"  \"layout\": \""<<(use_2d?"time-point":"time-height-point")<<"\",\n"
            <<"  \"time_num\": "<<time_num<<",\n"
            <<"  \"height_num\": "<<height_num<<",\n"
            <<"  \"point_num\": "<<point_num<<",\n"
            <<"  \"value_num\": "<<data.size()<<",\n"
            <<"  \"valid_value_num\": "<<valid_value_num<<",\n"
            <<"  \"input_bytes\": "<<data.size()*sizeof(float)<<",\n"
            <<"  \"rel_eb\": "<<rel_eb<<",\n"
            <<"  \"field_min\": "<<field_min<<",\n"
            <<"  \"field_max\": "<<field_max<<",\n"
            <<"  \"field_range\": "<<field_range<<",\n"
            <<"  \"codec_error_bound_mode\": \"ABS\",\n"
            <<"  \"codec_abs_eb\": "<<abs_eb<<",\n"
            <<"  \"compressed_bytes\": "<<compressed_bytes<<",\n"
            <<"  \"br\": "<<8.0*compressed_bytes/data.size()<<",\n"
            <<"  \"squared_error\": "<<squared_error<<",\n"
            <<"  \"psnr\": ";
        write_json_number(output,psnr);
        output<<",\n"
            <<"  \"compression_time\": "<<compression_time<<",\n"
            <<"  \"decompression_time\": "<<decompression_time<<",\n"
            <<"  \"compression_MB_s\": "<<data.size()*sizeof(float)/1e6/compression_time<<",\n"
            <<"  \"decompression_MB_s\": "<<data.size()*sizeof(float)/1e6/decompression_time<<",\n"
            <<"  \"max_error\": "<<max_error<<",\n"
            <<"  \"max_error_over_eb\": ";
        write_json_number(output,max_error_over_eb);
        output<<",\n"
            <<"  \"input_unchanged\": "<<(input_unchanged?"true":"false")<<",\n"
            <<"  \"valid\": "<<(valid?"true":"false")<<",\n"
            <<"  \"max_rss_kib\": "<<usage.ru_maxrss<<"\n"
            <<"}\n";
        cout<<"Completed codec="<<codec_name<<",field="<<field<<",shape=["<<time_num<<','<<height_num<<','<<point_num<<"],abs_eb="<<abs_eb<<",br="<<8.0*compressed_bytes/data.size()<<",valid="<<valid<<",max_rss_kib="<<usage.ru_maxrss<<endl;
        return valid?0:2;
    }
    catch(const exception&error)
    {
        cerr<<error.what()<<endl;
        return 1;
    }
}
