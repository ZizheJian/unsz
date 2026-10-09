#define main pfpl_original_decompress_main
#include "../../third_party/PFPL/src/f32_abs_decomp_cpu.cpp"
#undef main

void pfpl_decompress(const unsigned char*input,float*output,size_t expected_point_num)
{
    int output_bytes=0;
    h_decode(input,reinterpret_cast<byte*>(output),output_bytes);
    if(output_bytes!=expected_point_num*sizeof(float))
        throw std::runtime_error("PFPL decompressed size mismatch");
}
