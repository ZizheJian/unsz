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
#include <omp.h>

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
    auto program_start=clock_type::now();
    try
    {
        if(argc!=7&&argc!=8)
            throw runtime_error("Usage: compress_one_field_with_<codec> <data-dir> <field> <time-num> <height-num> <rel-eb> <output-json> [point-limit]");
        string data_dir=argv[1],field=argv[2],output_path=argv[6];
        size_t time_num=stoull(argv[3]),height_num=stoull(argv[4]),slice_num=time_num*height_num,point_limit=argc==8?stoull(argv[7]):0;
        double rel_eb=stod(argv[5]);
        if(!time_num||!height_num||slice_num/height_num!=time_num||!(rel_eb>0)||!isfinite(rel_eb))
            throw runtime_error("Invalid dimensions or relative error bound");
        auto start=clock_type::now();
        string first_path=data_dir+"/"+field+".dat.0";
        ifstream first_input(first_path,ios::binary|ios::ate);
        if(!first_input)throw runtime_error("Cannot open input: "+first_path);
        streamoff first_byte_num=first_input.tellg();
        if(first_byte_num<=0||first_byte_num%sizeof(float))throw runtime_error("Invalid float32 input size: "+first_path);
        first_input.close();
        size_t first_file_point_num=(size_t)first_byte_num/sizeof(float),point_num=point_limit?min(point_limit,first_file_point_num):first_file_point_num;
        if(!point_num||slice_num>numeric_limits<size_t>::max()/point_num)throw runtime_error("Invalid total element count");
        size_t value_num=slice_num*point_num;
        unique_ptr<float[]>data(new float[value_num]);
        double field_min=numeric_limits<double>::infinity(),field_max=-numeric_limits<double>::infinity();
        uint64_t valid_value_num=0;
        vector<int>input_errors(slice_num);
        int helper_thread_num=max(1,omp_get_max_threads()),io_thread_num=(int)min<size_t>(helper_thread_num,slice_num);
#pragma omp parallel for num_threads(io_thread_num) schedule(static) reduction(min:field_min) reduction(max:field_max) reduction(+:valid_value_num)
        for(int64_t slice_id=0;slice_id<(int64_t)slice_num;slice_id++)
        {
            string path=data_dir+"/"+field+".dat."+to_string(slice_id);
            ifstream input(path,ios::binary|ios::ate);
            if(!input){input_errors[slice_id]=1;continue;}
            streamoff byte_num=input.tellg();
            if(byte_num<=0||byte_num%sizeof(float)){input_errors[slice_id]=2;continue;}
            size_t file_point_num=(size_t)byte_num/sizeof(float);
            if(file_point_num<point_num||(!point_limit&&file_point_num!=point_num)){input_errors[slice_id]=3;continue;}
            input.seekg(0);
            input.read(reinterpret_cast<char*>(data.get()+slice_id*point_num),point_num*sizeof(float));
            if(!input){input_errors[slice_id]=4;continue;}
            for(size_t point_id=0;point_id<point_num;point_id++)
            {
                float value=data[slice_id*point_num+point_id];
                if(!isfinite(value)){input_errors[slice_id]=5;continue;}
                if(value!=-99999.0f)
                {
                    field_min=min(field_min,(double)value);
                    field_max=max(field_max,(double)value);
                    valid_value_num++;
                }
            }
        }
        auto failed_input=find_if(input_errors.begin(),input_errors.end(),[](int error){return error;});
        if(failed_input!=input_errors.end())throw runtime_error("Input loading failed: "+data_dir+"/"+field+".dat."+to_string(failed_input-input_errors.begin())+",error="+to_string(*failed_input));
        double input_and_range_time=chrono::duration<double>(clock_type::now()-start).count();
        if(!valid_value_num)throw runtime_error("Input has no ordinary values");
        double field_range=field_max-field_min,abs_eb=rel_eb*field_range;
        if(!(abs_eb>=0)||!isfinite(abs_eb))throw runtime_error("Invalid absolute error bound");
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
        start=clock_type::now();
        unique_ptr<char[]>encoded(SZ_compress<float>(config,data.get(),compressed_bytes));
        double compression_time=chrono::duration<double>(clock_type::now()-start).count();
        if(!encoded||!compressed_bytes)throw runtime_error("Compression returned an empty stream");
        unique_ptr<float[]>reconstructed(new float[value_num]);
        codec_config decoded_config;
        float*reconstructed_pointer=reconstructed.get();
        start=clock_type::now();
        SZ_decompress<float>(decoded_config,encoded.get(),compressed_bytes,reconstructed_pointer);
        double decompression_time=chrono::duration<double>(clock_type::now()-start).count();
#if defined(FIELD_RELATIVE_SZ3)
        if(decoded_config.dims!=config.dims||decoded_config.num!=value_num)throw runtime_error("Decompressed shape mismatch: expected="+shape_text(config.dims)+",decoded="+shape_text(decoded_config.dims)+",expected_num="+to_string(value_num)+",decoded_num="+to_string(decoded_config.num));
#endif
        start=clock_type::now();
        double squared_error=0,max_error=0;
        int finite=1;
#pragma omp parallel for num_threads(helper_thread_num) schedule(static) reduction(+:squared_error) reduction(max:max_error) reduction(&:finite)
        for(int64_t index=0;index<(int64_t)value_num;index++)
        {
            finite&=isfinite(reconstructed[index]);
            double error=abs((double)data[index]-reconstructed[index]);
            squared_error+=error*error;
            max_error=max(max_error,error);
        }
        double verification_time=chrono::duration<double>(clock_type::now()-start).count();
        double max_error_over_eb=abs_eb>0?max_error/abs_eb:max_error==0?0:numeric_limits<double>::infinity();
        bool valid=finite&&max_error_over_eb<=1.01;
        double mse=squared_error/value_num,psnr=mse?10*log10(field_range*field_range/mse):numeric_limits<double>::infinity();
        start=clock_type::now();
#pragma omp parallel sections num_threads(3)
        {
#pragma omp section
            data.reset();
#pragma omp section
            encoded.reset();
#pragma omp section
            reconstructed.reset();
        }
        double final_buffer_release_time=chrono::duration<double>(clock_type::now()-start).count();
        double program_time_before_output=chrono::duration<double>(clock_type::now()-program_start).count();
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
            <<"  \"value_num\": "<<value_num<<",\n"
            <<"  \"valid_value_num\": "<<valid_value_num<<",\n"
            <<"  \"input_bytes\": "<<value_num*sizeof(float)<<",\n"
            <<"  \"rel_eb\": "<<rel_eb<<",\n"
            <<"  \"field_min\": "<<field_min<<",\n"
            <<"  \"field_max\": "<<field_max<<",\n"
            <<"  \"field_range\": "<<field_range<<",\n"
            <<"  \"codec_error_bound_mode\": \"ABS\",\n"
            <<"  \"codec_abs_eb\": "<<abs_eb<<",\n"
            <<"  \"compressed_bytes\": "<<compressed_bytes<<",\n"
            <<"  \"br\": "<<8.0*compressed_bytes/value_num<<",\n"
            <<"  \"squared_error\": "<<squared_error<<",\n"
            <<"  \"psnr\": ";
        write_json_number(output,psnr);
        output<<",\n"
            <<"  \"input_and_range_time\": "<<input_and_range_time<<",\n"
            <<"  \"compression_time\": "<<compression_time<<",\n"
            <<"  \"decompression_time\": "<<decompression_time<<",\n"
            <<"  \"verification_time\": "<<verification_time<<",\n"
            <<"  \"final_buffer_release_time\": "<<final_buffer_release_time<<",\n"
            <<"  \"program_time_before_output\": "<<program_time_before_output<<",\n"
            <<"  \"compression_MB_s\": "<<value_num*sizeof(float)/1e6/compression_time<<",\n"
            <<"  \"decompression_MB_s\": "<<value_num*sizeof(float)/1e6/decompression_time<<",\n"
            <<"  \"max_error\": "<<max_error<<",\n"
            <<"  \"max_error_over_eb\": ";
        write_json_number(output,max_error_over_eb);
        output<<",\n"
            <<"  \"valid\": "<<(valid?"true":"false")<<",\n"
            <<"  \"max_rss_kib\": "<<usage.ru_maxrss<<"\n"
            <<"}\n";
        cout<<"Completed codec="<<codec_name<<",field="<<field<<",shape=["<<time_num<<','<<height_num<<','<<point_num<<"],abs_eb="<<abs_eb<<",br="<<8.0*compressed_bytes/value_num<<",valid="<<valid<<",max_rss_kib="<<usage.ru_maxrss<<endl;
        return valid?0:2;
    }
    catch(const exception&error)
    {
        cerr<<error.what()<<endl;
        return 1;
    }
}
