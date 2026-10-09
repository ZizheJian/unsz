#define main pfpl_original_compress_main
#include "../../third_party/PFPL/src/f32_abs_comp_cpu.cpp"
#undef main
#include <vector>

std::vector<unsigned char> pfpl_compress(const float*data,size_t point_num,float error_bound,bool exact)
{
    size_t input_bytes=point_num*sizeof(float);
    if(input_bytes>=2147221529ULL)
        throw std::runtime_error("PFPL input exceeds the upstream size limit");
    size_t chunks=(input_bytes+CS-1)/CS;
    std::vector<unsigned char>output(16+chunks*sizeof(unsigned short)+chunks*CS);
    int output_bytes=0;
    // 正常浮点数和零在零误差任务中通过上游protected-values参数精确保留。
    h_encode(reinterpret_cast<const byte*>(data),(int)input_bytes,output.data(),output_bytes,exact?1.0f:error_bound,exact?std::numeric_limits<float>::min():std::numeric_limits<float>::infinity());
    output.resize(output_bytes);
    return output;
}
