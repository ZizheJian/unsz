#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <SZ3/api/sz.hpp>

using namespace std;
using clock_type=chrono::steady_clock;

int main(int argc,char**argv)
{
    try
    {
        if(argc!=8&&argc!=9)
            throw runtime_error("Usage: benchmark_sz3 <data-dir> <field> <time-num> <height-num> <rel-eb> <global-relative|strict-slice-relative> <output-json> [point-limit]");
        string data_dir=argv[1],field=argv[2],mode=argv[6],output_path=argv[7];
        size_t time_num=stoull(argv[3]),height_num=stoull(argv[4]),slice_num=time_num*height_num,point_limit=argc==9?stoull(argv[8]):0;
        double rel_eb=stod(argv[5]);
        if(!time_num||!height_num||!(rel_eb>0)||!isfinite(rel_eb)||(mode!="global-relative"&&mode!="strict-slice-relative"))
            throw runtime_error("Invalid dimensions, relative error bound, or error-bound mode");
        size_t point_num=0;
        vector<float>data;
        vector<double>slice_min(slice_num,numeric_limits<double>::infinity()),slice_max(slice_num,-numeric_limits<double>::infinity());
        vector<uint64_t>slice_valid_num(slice_num);
        double global_min=numeric_limits<double>::infinity(),global_max=-numeric_limits<double>::infinity();
        for(size_t slice_id=0;slice_id<slice_num;slice_id++)
        {
            string path=data_dir+"/"+field+".dat."+to_string(slice_id);
            ifstream input(path,ios::binary|ios::ate);
            if(!input)
                throw runtime_error("Cannot open input: "+path);
            streamoff byte_num=input.tellg();
            if(byte_num<=0||byte_num%sizeof(float))
                throw runtime_error("Invalid float32 input size: "+path);
            size_t file_point_num=(size_t)byte_num/sizeof(float);
            if(!slice_id)
            {
                point_num=point_limit?min(point_limit,file_point_num):file_point_num;
                if(!point_num||slice_num>numeric_limits<size_t>::max()/point_num)
                    throw runtime_error("Invalid total element count");
                data.resize(slice_num*point_num);
            }
            else if(file_point_num<(point_limit?point_num:file_point_num)||(!point_limit&&file_point_num!=point_num))
                throw runtime_error("Inconsistent input size: "+path);
            input.seekg(0);
            input.read(reinterpret_cast<char*>(data.data()+slice_id*point_num),point_num*sizeof(float));
            if(!input)
                throw runtime_error("Input read failed: "+path);
            for(size_t point_id=0;point_id<point_num;point_id++)
            {
                float value=data[slice_id*point_num+point_id];
                if(!isfinite(value))
                    throw runtime_error("Non-finite input value: "+path);
                if(value!=-99999.0f)
                {
                    slice_min[slice_id]=min(slice_min[slice_id],(double)value);
                    slice_max[slice_id]=max(slice_max[slice_id],(double)value);
                    slice_valid_num[slice_id]++;
                }
            }
            if(slice_valid_num[slice_id])
            {
                global_min=min(global_min,slice_min[slice_id]);
                global_max=max(global_max,slice_max[slice_id]);
            }
        }
        if(!isfinite(global_min)||!isfinite(global_max))
            throw runtime_error("Input has no valid values");
        double global_abs_eb=rel_eb*(global_max-global_min),strict_abs_eb=numeric_limits<double>::infinity();
        size_t constant_slice_num=0;
        for(size_t slice_id=0;slice_id<slice_num;slice_id++)
        {
            double range=slice_valid_num[slice_id]?slice_max[slice_id]-slice_min[slice_id]:0;
            if(range>0)
                strict_abs_eb=min(strict_abs_eb,rel_eb*range);
            else
                constant_slice_num++;
        }
        if(mode=="strict-slice-relative"&&constant_slice_num)
            throw runtime_error("A single lossy tensor stream cannot exactly preserve constant slices under the slice-relative protocol");
        double abs_eb=mode=="global-relative"?global_abs_eb:strict_abs_eb;
        if(!(abs_eb>0)||!isfinite(abs_eb))
            throw runtime_error("Invalid absolute error bound");
        SZ3::Config config(time_num,height_num,point_num);
        config.errorBoundMode=SZ3::EB_ABS;
        config.absErrorBound=abs_eb;
        config.openmp=false;
        size_t compressed_bytes=0;
        auto start=clock_type::now();
        unique_ptr<char[]>encoded(SZ_compress<float>(config,data.data(),compressed_bytes));
        double compression_time=chrono::duration<double>(clock_type::now()-start).count();
        vector<float>reconstructed(data.size());
        SZ3::Config decoded_config;
        float*reconstructed_pointer=reconstructed.data();
        start=clock_type::now();
        SZ_decompress<float>(decoded_config,encoded.get(),compressed_bytes,reconstructed_pointer);
        double decompression_time=chrono::duration<double>(clock_type::now()-start).count();
        if(decoded_config.dims!=config.dims||decoded_config.num!=data.size())
            throw runtime_error("SZ3 decompressed shape mismatch");
        double squared_normalized_error=0,max_error=0,max_error_over_codec_eb=0,max_error_over_slice_eb=0;
        bool finite=true,slice_protocol_valid=true;
        for(size_t slice_id=0;slice_id<slice_num;slice_id++)
        {
            double range=slice_valid_num[slice_id]?slice_max[slice_id]-slice_min[slice_id]:0,slice_abs_eb=rel_eb*range;
            for(size_t point_id=0;point_id<point_num;point_id++)
            {
                size_t index=slice_id*point_num+point_id;
                finite&=isfinite(reconstructed[index]);
                double error=abs((double)data[index]-reconstructed[index]);
                max_error=max(max_error,error);
                max_error_over_codec_eb=max(max_error_over_codec_eb,error/abs_eb);
                if(slice_abs_eb>0)
                {
                    max_error_over_slice_eb=max(max_error_over_slice_eb,error/slice_abs_eb);
                    squared_normalized_error+=error*error/(range*range);
                    slice_protocol_valid&=error<=1.01*slice_abs_eb;
                }
                else
                    slice_protocol_valid&=error==0;
            }
        }
        bool codec_valid=finite&&max_error_over_codec_eb<=1.01;
        double nmse=squared_normalized_error/data.size(),psnr=nmse? -10*log10(nmse):numeric_limits<double>::infinity();
        rusage usage{};
        getrusage(RUSAGE_SELF,&usage);
        ofstream output(output_path);
        if(!output)
            throw runtime_error("Cannot open output JSON: "+output_path);
        output<<setprecision(17)<<"{\n"
            <<"  \"field\": \""<<field<<"\",\n"
            <<"  \"compressor\": \"sz3-3d\",\n"
            <<"  \"layout\": \"time-height-point\",\n"
            <<"  \"error_bound_mode\": \""<<mode<<"\",\n"
            <<"  \"time_num\": "<<time_num<<",\n"
            <<"  \"height_num\": "<<height_num<<",\n"
            <<"  \"point_num\": "<<point_num<<",\n"
            <<"  \"input_bytes\": "<<data.size()*sizeof(float)<<",\n"
            <<"  \"rel_eb\": "<<rel_eb<<",\n"
            <<"  \"global_abs_eb\": "<<global_abs_eb<<",\n"
            <<"  \"strict_slice_abs_eb\": "<<strict_abs_eb<<",\n"
            <<"  \"codec_abs_eb\": "<<abs_eb<<",\n"
            <<"  \"constant_slice_num\": "<<constant_slice_num<<",\n"
            <<"  \"compressed_bytes\": "<<compressed_bytes<<",\n"
            <<"  \"br\": "<<8.0*compressed_bytes/data.size()<<",\n"
            <<"  \"psnr\": "<<psnr<<",\n"
            <<"  \"compression_time\": "<<compression_time<<",\n"
            <<"  \"decompression_time\": "<<decompression_time<<",\n"
            <<"  \"compression_MB_s\": "<<data.size()*sizeof(float)/1e6/compression_time<<",\n"
            <<"  \"decompression_MB_s\": "<<data.size()*sizeof(float)/1e6/decompression_time<<",\n"
            <<"  \"max_error\": "<<max_error<<",\n"
            <<"  \"max_error_over_codec_eb\": "<<max_error_over_codec_eb<<",\n"
            <<"  \"max_error_over_slice_eb\": "<<max_error_over_slice_eb<<",\n"
            <<"  \"codec_valid\": "<<(codec_valid?"true":"false")<<",\n"
            <<"  \"slice_protocol_valid\": "<<(finite&&slice_protocol_valid?"true":"false")<<",\n"
            <<"  \"max_rss_kib\": "<<usage.ru_maxrss<<"\n"
            <<"}\n";
        cout<<"Completed field="<<field<<",shape=["<<time_num<<','<<height_num<<','<<point_num<<"],mode="<<mode<<",br="<<8.0*compressed_bytes/data.size()<<",codec_valid="<<codec_valid<<",slice_protocol_valid="<<slice_protocol_valid<<",max_rss_kib="<<usage.ru_maxrss<<endl;
        return codec_valid?0:2;
    }
    catch(const exception&error)
    {
        cerr<<error.what()<<endl;
        return 1;
    }
}
