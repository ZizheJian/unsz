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
#include <new>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <omp.h>
#if defined(FIELD_RELATIVE_SZ3)
#include <SZ3/api/sz.hpp>
using codec_config=SZ3::Config;
constexpr const char*codec_name="sz3-nd-point-partitioned";
#elif defined(FIELD_RELATIVE_QOZ)
#include <QoZ/api/sz.hpp>
using codec_config=QoZ::Config;
constexpr const char*codec_name="qoz2-q4-nd-point-partitioned";
#else
#error "Select exactly one field-relative codec"
#endif

using namespace std;
using clock_type=chrono::steady_clock;

struct eb_run
{
    double rel_eb;
    string output_path;
};

void write_json_number(ostream&output,double value)
{
    if(isfinite(value))output<<value;
    else output<<"null";
}

int main(int argc,char**argv)
{
    auto program_start=clock_type::now();
    try
    {
        if(argc!=8&&(argc<10||argc%2))throw runtime_error("Usage: compress_one_field_with_<codec>_partitions <data-dir> <field> <time-num> <height-num> <rel-eb> <partition-num> <output-json> OR <data-dir> <field> <time-num> <height-num> <partition-num> <rel-eb> <output-json> [<rel-eb> <output-json> ...]");
        string data_dir=argv[1],field=argv[2];
        size_t time_num=stoull(argv[3]),height_num=stoull(argv[4]),slice_num=time_num*height_num,partition_num=stoull(argv[argc==8?6:5]);
        vector<eb_run>eb_runs;
        if(argc==8)eb_runs.push_back({stod(argv[5]),argv[7]});
        else for(int argument=6;argument<argc;argument+=2)eb_runs.push_back({stod(argv[argument]),argv[argument+1]});
        if(!time_num||!height_num||slice_num/height_num!=time_num||!partition_num||(partition_num&(partition_num-1))||partition_num>(size_t)numeric_limits<int>::max())throw runtime_error("Invalid dimensions or partition count");
        for(const auto&run:eb_runs)if(!(run.rel_eb>0)||!isfinite(run.rel_eb))throw runtime_error("Invalid relative error bound");
        auto start=clock_type::now();
        string first_path=data_dir+"/"+field+".dat.0";
        ifstream first_input(first_path,ios::binary|ios::ate);
        if(!first_input)throw runtime_error("Cannot open input: "+first_path);
        streamoff first_byte_num=first_input.tellg();
        if(first_byte_num<=0||first_byte_num%sizeof(float))throw runtime_error("Invalid float32 input size: "+first_path);
        first_input.close();
        size_t point_num=(size_t)first_byte_num/sizeof(float);
        if(partition_num>point_num||slice_num>numeric_limits<size_t>::max()/point_num)throw runtime_error("Invalid point or value count");
        size_t value_num=slice_num*point_num;
        unique_ptr<float[]>data(new float[value_num]);
        double field_min=numeric_limits<double>::infinity(),field_max=-numeric_limits<double>::infinity();
        uint64_t valid_value_num=0;
        vector<int>input_errors(slice_num);
        int io_thread_num=(int)min(partition_num,slice_num);
#pragma omp parallel for num_threads(io_thread_num) schedule(static) reduction(min:field_min) reduction(max:field_max) reduction(+:valid_value_num)
        for(int64_t slice_id=0;slice_id<(int64_t)slice_num;slice_id++)
        {
            string path=data_dir+"/"+field+".dat."+to_string(slice_id);
            ifstream input(path,ios::binary|ios::ate);
            if(!input){input_errors[slice_id]=1;continue;}
            streamoff byte_num=input.tellg();
            if(byte_num<=0||byte_num%sizeof(float)){input_errors[slice_id]=2;continue;}
            size_t file_point_num=(size_t)byte_num/sizeof(float);
            if(file_point_num!=point_num){input_errors[slice_id]=3;continue;}
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
        double field_range=field_max-field_min;
        const float*original_data=data.get();
        vector<size_t>point_begins(partition_num),partition_point_nums(partition_num);
        for(size_t partition_id=0;partition_id<partition_num;partition_id++)
        {
            size_t point_begin=partition_id*point_num/partition_num,point_end=(partition_id+1)*point_num/partition_num;
            point_begins[partition_id]=point_begin;
            partition_point_nums[partition_id]=point_end-point_begin;
        }
        bool all_valid=true;
        for(size_t eb_index=0;eb_index<eb_runs.size();eb_index++)
        {
            auto eb_start=clock_type::now();
            double rel_eb=eb_runs[eb_index].rel_eb,abs_eb=rel_eb*field_range;
            const string&output_path=eb_runs[eb_index].output_path;
            if(!(abs_eb>=0)||!isfinite(abs_eb))throw runtime_error("Invalid absolute error bound");
            vector<unique_ptr<float[]>>partitioned(partition_num);
            start=clock_type::now();
#pragma omp parallel for num_threads(partition_num) schedule(static)
            for(int64_t partition_id=0;partition_id<(int64_t)partition_num;partition_id++)partitioned[partition_id].reset(new(nothrow) float[slice_num*partition_point_nums[partition_id]]);
            if(any_of(partitioned.begin(),partitioned.end(),[](const auto&buffer){return !buffer;}))throw runtime_error("Partition buffer allocation failed");
#pragma omp parallel for num_threads(partition_num) schedule(static)
            for(int64_t partition_id=0;partition_id<(int64_t)partition_num;partition_id++)
            {
                size_t partition_point_num=partition_point_nums[partition_id],point_begin=point_begins[partition_id];
                for(size_t slice_id=0;slice_id<slice_num;slice_id++)memcpy(partitioned[partition_id].get()+slice_id*partition_point_num,original_data+slice_id*point_num+point_begin,partition_point_num*sizeof(float));
            }
            double compression_partition_time=chrono::duration<double>(clock_type::now()-start).count();
            vector<unique_ptr<char[]>>encoded(partition_num);
            vector<size_t>compressed_sizes(partition_num);
            vector<codec_config>configs(partition_num);
            int compression_failed=0;
            start=clock_type::now();
#pragma omp parallel for num_threads(partition_num) schedule(static) reduction(|:compression_failed)
            for(int64_t partition_id=0;partition_id<(int64_t)partition_num;partition_id++)
            {
                size_t partition_point_num=partition_point_nums[partition_id];
                codec_config config=height_num==1?codec_config(time_num,partition_point_num):codec_config(time_num,height_num,partition_point_num);
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
                configs[partition_id]=config;
                encoded[partition_id].reset(SZ_compress<float>(configs[partition_id],partitioned[partition_id].get(),compressed_sizes[partition_id]));
                compression_failed|=!encoded[partition_id]||!compressed_sizes[partition_id];
            }
            double codec_compression_time=chrono::duration<double>(clock_type::now()-start).count();
            if(compression_failed)throw runtime_error("At least one partition compression failed");
            start=clock_type::now();
#pragma omp parallel for num_threads(partition_num) schedule(static)
            for(int64_t partition_id=0;partition_id<(int64_t)partition_num;partition_id++)partitioned[partition_id].reset();
            vector<unique_ptr<float[]>>().swap(partitioned);
            double compression_buffer_release_time=chrono::duration<double>(clock_type::now()-start).count();
            vector<unique_ptr<float[]>>partition_reconstructed(partition_num);
            vector<codec_config>decoded_configs(partition_num);
            start=clock_type::now();
#pragma omp parallel for num_threads(partition_num) schedule(static)
            for(int64_t partition_id=0;partition_id<(int64_t)partition_num;partition_id++)partition_reconstructed[partition_id].reset(new(nothrow) float[slice_num*partition_point_nums[partition_id]]);
            if(any_of(partition_reconstructed.begin(),partition_reconstructed.end(),[](const auto&buffer){return !buffer;}))throw runtime_error("Decompression buffer allocation failed");
            double decompression_buffer_allocation_time=chrono::duration<double>(clock_type::now()-start).count();
            start=clock_type::now();
#pragma omp parallel for num_threads(partition_num) schedule(static)
            for(int64_t partition_id=0;partition_id<(int64_t)partition_num;partition_id++)
            {
                codec_config decoded_config;
#if defined(FIELD_RELATIVE_QOZ)
                decoded_config.openmp=false;
#endif
                float*reconstructed_pointer=partition_reconstructed[partition_id].get();
                SZ_decompress<float>(decoded_config,encoded[partition_id].get(),compressed_sizes[partition_id],reconstructed_pointer);
                decoded_configs[partition_id]=decoded_config;
            }
            double codec_decompression_time=chrono::duration<double>(clock_type::now()-start).count();
#if defined(FIELD_RELATIVE_SZ3)
            for(size_t partition_id=0;partition_id<partition_num;partition_id++)if(decoded_configs[partition_id].dims!=configs[partition_id].dims||decoded_configs[partition_id].num!=slice_num*partition_point_nums[partition_id])throw runtime_error("Decompressed partition shape mismatch");
#endif
            start=clock_type::now();
            unique_ptr<float[]>reconstructed(new float[value_num]);
#pragma omp parallel for num_threads(partition_num) schedule(static)
            for(int64_t partition_id=0;partition_id<(int64_t)partition_num;partition_id++)
            {
                size_t partition_point_num=partition_point_nums[partition_id],point_begin=point_begins[partition_id];
                for(size_t slice_id=0;slice_id<slice_num;slice_id++)memcpy(reconstructed.get()+slice_id*point_num+point_begin,partition_reconstructed[partition_id].get()+slice_id*partition_point_num,partition_point_num*sizeof(float));
            }
            double decompression_merge_time=chrono::duration<double>(clock_type::now()-start).count();
            start=clock_type::now();
            double squared_error=0,valid_squared_error=0,max_error=0;
            int finite=1;
#pragma omp parallel for num_threads(partition_num) schedule(static) reduction(+:squared_error,valid_squared_error) reduction(max:max_error) reduction(&:finite)
            for(int64_t index=0;index<(int64_t)value_num;index++)
            {
                finite&=isfinite(reconstructed[index]);
                double error=abs((double)original_data[index]-reconstructed[index]);
                squared_error+=error*error;
                if(original_data[index]!=-99999.0f)valid_squared_error+=error*error;
                max_error=max(max_error,error);
            }
            double verification_time=chrono::duration<double>(clock_type::now()-start).count();
            double max_error_over_eb=abs_eb>0?max_error/abs_eb:max_error==0?0:numeric_limits<double>::infinity();
            bool valid=finite&&max_error_over_eb<=1.01;
            all_valid&=valid;
            double mse=squared_error/value_num,psnr=mse?10*log10(field_range*field_range/mse):numeric_limits<double>::infinity();
            size_t partition_metadata_bytes=sizeof(uint64_t)*(partition_num+2),compressed_bytes=accumulate(compressed_sizes.begin(),compressed_sizes.end(),partition_metadata_bytes);
            double compression_time=compression_partition_time+codec_compression_time,decompression_time=codec_decompression_time+decompression_merge_time;
            start=clock_type::now();
#pragma omp parallel for num_threads(partition_num) schedule(static)
            for(int64_t partition_id=0;partition_id<(int64_t)partition_num;partition_id++)
            {
                encoded[partition_id].reset();
                partition_reconstructed[partition_id].reset();
            }
            reconstructed.reset();
            double final_buffer_release_time=chrono::duration<double>(clock_type::now()-start).count();
            double eb_processing_time_before_output=chrono::duration<double>(clock_type::now()-eb_start).count();
            double program_time_before_output=chrono::duration<double>(clock_type::now()-program_start).count();
            rusage usage{};
            getrusage(RUSAGE_SELF,&usage);
            ofstream output(output_path);
            if(!output)throw runtime_error("Cannot open output JSON: "+output_path);
            output<<setprecision(17)<<"{\n"
                <<"  \"field\": \""<<field<<"\",\n"
                <<"  \"compressor\": \""<<codec_name<<"\",\n"
                <<"  \"layout\": \""<<(height_num==1?"time-point":"time-height-point")<<"\",\n"
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
                <<"  \"partition_axis\": \"point\",\n"
                <<"  \"partition_num\": "<<partition_num<<",\n"
                <<"  \"partition_codec_threads\": 1,\n"
                <<"  \"partition_metadata_bytes\": "<<partition_metadata_bytes<<",\n"
                <<"  \"compressed_bytes\": "<<compressed_bytes<<",\n"
                <<"  \"br\": "<<8.0*compressed_bytes/value_num<<",\n"
                <<"  \"squared_error\": "<<squared_error<<",\n"
                <<"  \"valid_squared_error\": "<<valid_squared_error<<",\n"
                <<"  \"psnr\": ";
            write_json_number(output,psnr);
            output<<",\n"
                <<"  \"input_and_range_time\": "<<input_and_range_time<<",\n"
                <<"  \"input_shared_across_eb_num\": "<<eb_runs.size()<<",\n"
                <<"  \"eb_index_in_process\": "<<eb_index<<",\n"
                <<"  \"compression_partition_time\": "<<compression_partition_time<<",\n"
                <<"  \"codec_compression_time\": "<<codec_compression_time<<",\n"
                <<"  \"compression_time\": "<<compression_time<<",\n"
                <<"  \"compression_buffer_release_time\": "<<compression_buffer_release_time<<",\n"
                <<"  \"decompression_buffer_allocation_time\": "<<decompression_buffer_allocation_time<<",\n"
                <<"  \"codec_decompression_time\": "<<codec_decompression_time<<",\n"
                <<"  \"decompression_merge_time\": "<<decompression_merge_time<<",\n"
                <<"  \"decompression_time\": "<<decompression_time<<",\n"
                <<"  \"verification_time\": "<<verification_time<<",\n"
                <<"  \"final_buffer_release_time\": "<<final_buffer_release_time<<",\n"
                <<"  \"eb_processing_time_before_output\": "<<eb_processing_time_before_output<<",\n"
                <<"  \"program_time_before_output\": "<<program_time_before_output<<",\n"
                <<"  \"max_error\": "<<max_error<<",\n"
                <<"  \"max_error_over_eb\": ";
            write_json_number(output,max_error_over_eb);
            output<<",\n"
                <<"  \"valid\": "<<(valid?"true":"false")<<",\n"
                <<"  \"max_rss_kib\": "<<usage.ru_maxrss<<"\n"
                <<"}\n";
            cout<<"Completed codec="<<codec_name<<",field="<<field<<",eb="<<eb_index+1<<'/'<<eb_runs.size()<<",partitions="<<partition_num<<",abs_eb="<<abs_eb<<",br="<<8.0*compressed_bytes/value_num<<",compression_time="<<compression_time<<",decompression_time="<<decompression_time<<",valid="<<valid<<",max_rss_kib="<<usage.ru_maxrss<<endl;
        }
        data.reset();
        return all_valid?0:2;
    }
    catch(const exception&error)
    {
        cerr<<error.what()<<endl;
        return 1;
    }
}
