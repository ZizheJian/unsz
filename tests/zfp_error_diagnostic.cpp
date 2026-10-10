#include <zfp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

template<class T>std::vector<T>roundtrip(const std::vector<T>&input,double tolerance,bool reversible,bool header)
{
    zfp_type type=sizeof(T)==4?zfp_type_float:zfp_type_double;
    zfp_field*field=zfp_field_1d(const_cast<T*>(input.data()),type,input.size());
    zfp_stream*encoder=zfp_stream_open(nullptr);
    double effective=0;
    if(reversible)
        zfp_stream_set_reversible(encoder);
    else
        effective=zfp_stream_set_accuracy(encoder,tolerance);
    std::vector<unsigned char>bytes(zfp_stream_maximum_size(encoder,field));
    bitstream*bits=stream_open(bytes.data(),bytes.size());
    zfp_stream_set_bit_stream(encoder,bits);
    if(header&&!zfp_write_header(encoder,field,ZFP_HEADER_FULL))
        throw std::runtime_error("Header write failed");
    if(!zfp_compress(encoder,field))
        throw std::runtime_error("Compression failed");
    zfp_stream_close(encoder);
    stream_close(bits);
    std::vector<T>output(input.size());
    zfp_field_set_pointer(field,output.data());
    zfp_stream*decoder=zfp_stream_open(nullptr);
    bits=stream_open(bytes.data(),bytes.size());
    zfp_stream_set_bit_stream(decoder,bits);
    if(header)
    {
        if(!zfp_read_header(decoder,field,ZFP_HEADER_FULL))
            throw std::runtime_error("Header read failed");
    }
    else if(reversible)
        zfp_stream_set_reversible(decoder);
    else
        zfp_stream_set_accuracy(decoder,tolerance);
    if(!reversible&&zfp_stream_accuracy(decoder)!=effective)
        throw std::runtime_error("Decoder tolerance differs from encoder");
    if(!zfp_decompress(decoder,field))
        throw std::runtime_error("Decompression failed");
    stream_close(bits);
    zfp_stream_close(decoder);
    zfp_field_free(field);
    return output;
}

template<class T>void block_test(const std::vector<T>&values,double tolerance,bool reversible=false,bool header=true)
{
    auto output=roundtrip(values,tolerance,reversible,header);
    double maximum=0;
    for(size_t i=0;i<values.size();i++)
        maximum=std::max(maximum,std::abs((double)output[i]-values[i]));
    std::cout<<"type="<<(sizeof(T)==4?"float":"double")<<",reversible="<<reversible<<",header="<<header<<",requested="<<tolerance<<",maximum="<<maximum<<",ratio="<<maximum/tolerance<<"\ninput:";
    for(T value:values)
        std::cout<<' '<<value;
    std::cout<<"\noutput:";
    for(T value:output)
        std::cout<<' '<<value;
    std::cout<<'\n';
}

void compare_block(const std::vector<float>&values,double tolerance)
{
    float largest=0;
    for(float value:values)
        largest=std::max(largest,std::abs(value));
    int exponent=0;
    std::frexp(largest,&exponent);
    float scale=std::ldexp(1.0f,30-std::max(exponent,-126));
    std::cout<<"float_block_exponent="<<exponent<<",forward_scale="<<scale<<",forward_scale_finite="<<std::isfinite(scale)<<'\n';
    block_test(values,tolerance);
    block_test(values,tolerance,false,false);
    block_test(std::vector<double>(values.begin(),values.end()),tolerance);
    block_test(values,tolerance,true);
}

int main(int argc,char**argv)
{
    std::cout<<std::setprecision(17);
    if(argc==1)
    {
        std::cout<<"SYNTHETIC normal\n";
        compare_block({0.0001f,0.0002f,0.0003f,0.0004f},1e-7);
        std::cout<<"SYNTHETIC mixed sentinel\n";
        compare_block({-99999.0f,0.0001f,0.0002f,0.0003f},1e-7);
        std::cout<<"SYNTHETIC tiny normal values\n";
        compare_block({0.0f,1e-30f,1.5e-30f,2e-30f},2e-33);
        return 0;
    }
    std::ifstream file(argv[1],std::ios::binary|std::ios::ate);
    if(!file)
        throw std::runtime_error("Input open failed");
    size_t size=file.tellg();
    std::vector<float>values(size/sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()),size);
    double minimum=std::numeric_limits<double>::infinity(),maximum=-minimum;
    for(float value:values)
        if(value!=-99999.0f)
        {
            minimum=std::min(minimum,(double)value);
            maximum=std::max(maximum,(double)value);
        }
    double tolerance=(float)(0.001*(maximum-minimum));
    if(argc>=3&&(std::string(argv[2])=="--full-double"||std::string(argv[2])=="--investigate-double"))
    {
        if(argc>=4)
            tolerance=std::stod(argv[3]);
        if(!std::isfinite(tolerance)||tolerance<=0)
            throw std::runtime_error("Expected a positive finite ABS EB");
        auto restored=roundtrip(std::vector<double>(values.begin(),values.end()),tolerance,false,true);
        double double_error=0,float_error=0;
        size_t double_violations=0,float_violations=0,nonfinite=0,sentinel_violations=0,worst=0;
        for(size_t i=0;i<values.size();i++)
        {
            float rounded=static_cast<float>(restored[i]);
            if(!std::isfinite(restored[i])||!std::isfinite(rounded))
            {
                nonfinite++;
                continue;
            }
            double de=std::abs(restored[i]-values[i]),fe=std::abs((double)rounded-values[i]);
            double_error=std::max(double_error,de);
            float_error=std::max(float_error,fe);
            double_violations+=de>tolerance;
            float_violations+=fe>tolerance;
            sentinel_violations+=fe>tolerance&&values[i]==-99999.0f;
            if(fe>=float_error)
                worst=i;
        }
        std::cout<<"{\"point_num\":"<<values.size()<<",\"abs_eb\":"<<tolerance<<",\"double_max_ratio\":"<<double_error/tolerance<<",\"float_max_ratio\":"<<float_error/tolerance<<",\"double_violations\":"<<double_violations<<",\"float_violations\":"<<float_violations<<",\"sentinel_violations\":"<<sentinel_violations<<",\"worst_point\":"<<worst<<",\"worst_input\":"<<values[worst]<<",\"worst_output_double\":"<<restored[worst]<<",\"nonfinite\":"<<nonfinite;
        if(std::string(argv[2])=="--investigate-double")
        {
            size_t begin=worst/4*4,end=std::min(begin+4,values.size());
            std::vector<double>block(values.begin()+begin,values.begin()+end);
            std::cout<<",\"block_input\":[";
            for(size_t i=0;i<block.size();i++)
                std::cout<<(i?",":"")<<block[i];
            std::cout<<"],\"block_trials\":[";
            for(int trial=0;trial<7;trial++)
            {
                double bound=trial<4?tolerance/std::ldexp(1.0,trial):tolerance;
                auto test_block=block;
                if(trial==5)
                    for(double&value:test_block)
                        if(value==-99999)
                            value=0;
                if(trial==6)
                    bound=1e-300;
                auto result=roundtrip(test_block,bound,trial==4,true);
                double error=0;
                for(size_t i=0;i<test_block.size();i++)
                    error=std::max(error,std::abs((double)(float)result[i]-test_block[i]));
                std::cout<<(trial?",":"")<<"{\"trial\":"<<trial<<",\"ratio\":"<<error/tolerance<<",\"output\":[";
                for(size_t i=0;i<result.size();i++)
                    std::cout<<(i?",":"")<<result[i];
                std::cout<<"]}";
            }
            auto half=roundtrip(std::vector<double>(values.begin(),values.end()),tolerance/2,false,true);
            double half_error=0;
            size_t half_bad=0;
            for(size_t i=0;i<values.size();i++)
            {
                double error=std::abs((double)(float)half[i]-values[i]);
                half_error=std::max(half_error,error);
                half_bad+=error>tolerance;
            }
            std::cout<<"],\"full_half_ratio\":"<<half_error/tolerance<<",\"full_half_violations\":"<<half_bad;
        }
        std::cout<<"}\n";
        return nonfinite||double_violations||float_violations?2:0;
    }
    auto output=roundtrip(values,tolerance,false,true);
    size_t worst=0;
    for(size_t i=1;i<values.size();i++)
        if(std::abs((double)output[i]-values[i])>std::abs((double)output[worst]-values[worst]))
            worst=i;
    size_t begin=worst/4*4,end=std::min(begin+4,values.size());
    std::cout<<"REAL worst_point="<<worst<<",block_begin="<<begin<<",range="<<maximum-minimum<<'\n';
    compare_block(std::vector<float>(values.begin()+begin,values.begin()+end),tolerance);
}
