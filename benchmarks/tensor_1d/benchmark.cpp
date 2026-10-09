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
#include <omp.h>
#include <sys/resource.h>
#if BASELINE_sz3
#include <SZ3/api/sz.hpp>
#elif BASELINE_zfp
#include <zfp.h>
#else
std::vector<unsigned char> pfpl_compress(const float*,size_t,float,bool);
void pfpl_decompress(const unsigned char*,float*,size_t);
#endif

using namespace std;
using clock_type=chrono::steady_clock;

struct codec_c
{
    vector<unsigned char>bytes;
#if BASELINE_sz3
    unique_ptr<char[]>encoded;
    size_t encoded_size=0;
#endif
    size_t byte_num() const
    {
#if BASELINE_sz3
        return encoded_size;
#else
        return bytes.size();
#endif
    }
    void compress(const vector<float>&data,float abs_eb,int threads)
    {
#if BASELINE_sz3
        SZ3::Config config(data.size());
        config.errorBoundMode=SZ3::EB_ABS;
        config.absErrorBound=abs_eb;
        config.openmp=threads>1;
        if(abs_eb==0)
            config.cmprAlgo=SZ3::ALGO_LOSSLESS;
        encoded.reset(SZ_compress<float>(config,data.data(),encoded_size));
#elif BASELINE_zfp
        zfp_field*field=zfp_field_1d(const_cast<float*>(data.data()),zfp_type_float,data.size());
        zfp_stream*codec=zfp_stream_open(nullptr);
        if(abs_eb==0)
            zfp_stream_set_reversible(codec);
        else
            zfp_stream_set_accuracy(codec,abs_eb);
        if(threads>1&&(!zfp_stream_set_execution(codec,zfp_exec_omp)||!zfp_stream_set_omp_threads(codec,threads)))
            throw runtime_error("ZFP OpenMP execution is unavailable");
        bytes.resize(zfp_stream_maximum_size(codec,field));
        bitstream*stream=stream_open(bytes.data(),bytes.size());
        zfp_stream_set_bit_stream(codec,stream);
        zfp_stream_rewind(codec);
        if(!zfp_write_header(codec,field,ZFP_HEADER_FULL))
            throw runtime_error("ZFP header write failed");
        size_t size=zfp_compress(codec,field);
        if(!size)
            throw runtime_error("ZFP compression failed");
        bytes.resize(size);
        stream_close(stream);
        zfp_stream_close(codec);
        zfp_field_free(field);
#else
        bytes=pfpl_compress(data.data(),data.size(),abs_eb,abs_eb==0);
#endif
    }
    void decompress(vector<float>&data)
    {
#if BASELINE_sz3
        SZ3::Config config;
        float*output=data.data();
        SZ_decompress<float>(config,encoded.get(),encoded_size,output);
        if(config.N!=1||config.num!=data.size())
            throw runtime_error("SZ3 decompressed shape mismatch");
#elif BASELINE_zfp
        zfp_field*field=zfp_field_alloc();
        zfp_stream*codec=zfp_stream_open(nullptr);
        bitstream*stream=stream_open(bytes.data(),bytes.size());
        zfp_stream_set_bit_stream(codec,stream);
        zfp_stream_rewind(codec);
        if(!zfp_read_header(codec,field,ZFP_HEADER_FULL)||zfp_field_dimensionality(field)!=1||zfp_field_size(field,nullptr)!=data.size())
            throw runtime_error("ZFP decompressed shape mismatch");
        zfp_field_set_pointer(field,data.data());
        if(!zfp_decompress(codec,field))
            throw runtime_error("ZFP decompression failed");
        stream_close(stream);
        zfp_stream_close(codec);
        zfp_field_free(field);
#else
        pfpl_decompress(bytes.data(),data.data(),data.size());
#endif
    }
};

int main(int argc,char**argv)
{
    try
    {
        if(argc!=5)
            throw runtime_error("Usage: benchmark <input-list> <rel-eb> <threads> <slice-csv>");
        double rel_eb=stod(argv[2]);
        int threads=stoi(argv[3]);
        if(!(rel_eb>0)||!isfinite(rel_eb)||threads<1)
            throw runtime_error("Invalid relative error bound or thread count");
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        ifstream paths(argv[1]);
        ofstream output(argv[4]);
        if(!paths||!output)
            throw runtime_error("Cannot open input list or slice CSV");
        output<<setprecision(17)<<"slice_id,point_num,valid_point_num,valid_min,valid_max,abs_eb,compressed_bytes,compression_time,decompression_time,squared_error,max_error,max_error_over_eb,valid,zero_eb\n";
        string path;
        uint64_t slice_id=0,failed=0;
        while(getline(paths,path))
        {
            ifstream input(path,ios::binary|ios::ate);
            if(!input)
                throw runtime_error("Cannot open input: "+path);
            streamoff size=input.tellg();
            if(size<=0||size%sizeof(float))
                throw runtime_error("Invalid float32 input size: "+path);
            vector<float>data((size_t)size/sizeof(float));
            input.seekg(0);
            input.read(reinterpret_cast<char*>(data.data()),size);
            if(!input)
                throw runtime_error("Input read failed: "+path);
            float minimum=numeric_limits<float>::infinity(),maximum=-minimum;
            uint64_t valid_num=0;
            bool subnormal=false;
            for(float value:data)
            {
                if(!isfinite(value))
                    throw runtime_error("Non-finite input value: "+path);
                subnormal|=fpclassify(value)==FP_SUBNORMAL;
                if(value!=-99999.0f)
                {
                    minimum=min(minimum,value);
                    maximum=max(maximum,value);
                    valid_num++;
                }
            }
            if(!valid_num)
                minimum=maximum=0;
            float abs_eb=(float)(rel_eb*((double)maximum-minimum));
#if !BASELINE_sz3 && !BASELINE_zfp
            if(abs_eb==0&&subnormal)
                throw runtime_error("PFPL protected-values exact mode does not cover subnormal inputs");
#endif
            codec_c codec;
            auto start=clock_type::now();
            codec.compress(data,abs_eb,threads);
            double compression_time=chrono::duration<double>(clock_type::now()-start).count();
            vector<float>reconstructed(data.size());
            start=clock_type::now();
            codec.decompress(reconstructed);
            double decompression_time=chrono::duration<double>(clock_type::now()-start).count();
            double squared_error=0,max_error=0;
            bool finite=true;
            for(size_t point_id=0;point_id<data.size();point_id++)
            {
                finite&=isfinite(reconstructed[point_id]);
                double difference=(double)data[point_id]-reconstructed[point_id];
                squared_error+=difference*difference;
                max_error=max(max_error,abs(difference));
            }
            double ratio=abs_eb>0?max_error/abs_eb:max_error==0?0:numeric_limits<double>::infinity();
            bool valid=finite&&ratio<=1.01;
            failed+=!valid;
            output<<slice_id<<','<<data.size()<<','<<valid_num<<','<<minimum<<','<<maximum<<','<<abs_eb<<','<<codec.byte_num()<<','<<compression_time<<','<<decompression_time<<','<<squared_error<<','<<max_error<<','<<ratio<<','<<valid<<','<<(abs_eb==0)<<'\n';
            if(slice_id%128==0||!valid)
            {
                output.flush();
                cout<<"slice="<<slice_id<<",max_error_over_eb="<<ratio<<",valid="<<valid<<endl;
            }
            slice_id++;
        }
        if(!slice_id)
            throw runtime_error("Empty input list");
        rusage usage{};
        getrusage(RUSAGE_SELF,&usage);
        cout<<"Completed slices="<<slice_id<<",failed="<<failed<<",max_rss_kib="<<usage.ru_maxrss<<endl;
        return failed?2:0;
    }
    catch(const exception&error)
    {
        cerr<<error.what()<<endl;
        return 1;
    }
}
