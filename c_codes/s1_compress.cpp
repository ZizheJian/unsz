#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <queue>
#include <numeric>
#include <random>
#include <cstdio>
#include "huffman.hpp"
#include "third_party/zstd/lib/zstd.h"
#include "unsz_core/unsz.hpp"

using namespace std;

template<typename T_coord>
void run_compression(unsz::param_c<T_coord,float> &param)
{
    if (param.if_adjust_coord)
        param.adjust_coord();
    param.generate_pred_refs_structure();

    uint64_t total_original_bytes=0,total_compressed_bytes=0,total_point_num=0;
    double total_squared_error=0;
    float total_min_data=numeric_limits<float>::infinity(),total_max_data=-numeric_limits<float>::infinity();
    for (auto &original_data_path:param.original_data_path_vec)
    {
        printf("Compressing %s\n",original_data_path.c_str());
        param.compressed_data_path_vec.push_back(original_data_path+".unsz");
        param.decompressed_data_path_vec.push_back(original_data_path+".unsz.bin");
        param.current_data.resize(param.point_num);
        FILE *f_data=fopen(original_data_path.c_str(),"rb");
        fread(param.current_data.data(),sizeof(float),param.point_num,f_data);
        fclose(f_data);
        param.compress();
        total_original_bytes+=(uint64_t)param.point_num*sizeof(float);
        total_compressed_bytes+=param.current_zstd_bs_vec.size();
        total_point_num+=param.point_num;
        for (uint32_t i=0;i<param.point_num;i++)
            if (!param.is_abnormal_vec[i])
            {
                double error=(double)param.current_data[i]-param.current_decompressed_data[i];
                total_squared_error+=error*error;
                total_min_data=min(total_min_data,param.current_data[i]);
                total_max_data=max(total_max_data,param.current_data[i]);
            }
        FILE *f_compressed=fopen(param.compressed_data_path_vec.back().c_str(),"wb");
        fwrite(param.current_zstd_bs_vec.data(),1,param.current_zstd_bs_vec.size(),f_compressed);
        fclose(f_compressed);
    }
    double total_cr=(double)total_original_bytes/total_compressed_bytes;
    double total_mse=total_squared_error/total_point_num;
    double total_psnr=10*log10(pow((double)total_max_data-total_min_data,2)/total_mse);
    double total_br=sizeof(float)*8/total_cr;
    printf("total cr= %f, total br= %f, total psnr= %f\n",total_cr,total_br,total_psnr);

}

template<typename T_coord>
void general_run_compression(unsz::param_c<T_coord,float> &param)
{
    if constexpr (is_same<T_coord,double>::value)
    {
        printf("使用double处理坐标和距离\n");
        run_compression(param);
    }
    else if (!param.need_double_coord())
    {
        printf("使用float处理坐标和距离\n");
        run_compression(param);
    }
    else
    {
        printf("坐标跨度较大，将坐标和距离计算从float升级为double\n");
        unsz::param_c<double,float> double_param(param);
        general_run_compression(double_param);
    }
}

int main(int argc,char **argv)
{
    if (argc!=5 && argc!=6)
        throw runtime_error("用法：s1_compress <data_dir> <variable_name> <time_step_num> <rel_eb> [round_stats]");
    unsz::param_c<float,float> param;
    string data_dir=argv[1],variable_name=argv[2];
    uint64_t time_step_num64=stoull(argv[3]);
    if (time_step_num64>numeric_limits<uint32_t>::max())
        throw runtime_error("time_step_num超过uint32_t的表示范围");
    uint32_t time_step_num=(uint32_t)time_step_num64;
    if (!data_dir.empty() && data_dir.back()!='/')
        data_dir+="/";
    param.coord_path=data_dir+"coords.dat";
    for (uint32_t i=0;i<time_step_num;i++)
        param.original_data_path_vec.push_back(data_dir+variable_name+".dat."+to_string(i));
    param.dim=3;
    param.rel_eb=stod(argv[4]);
    param.got_rel_eb=true;
    param.cross_time_compress=false;
    param.if_adjust_coord=true;
    if (argc==6)
    {
        if (string(argv[5])!="round_stats")
            throw runtime_error("未知的诊断参数："+string(argv[5]));
        param.print_round_statistics=true;
    }

    param.init();
    general_run_compression(param);

    
    
    

    

    

    // //////// 预测、量化 ////////
    // vector<float> compressed_data_vec=data_vec;
    // vector<int64_t> qb_vec;
    // qb_vec.reserve(node_num);
    // clear_valid_num(block_vec);
    // for (int i=0;i<round_num;i++)
    // {
    //     printf("Compressing round %d\n",i);
    //     if (i==0)
    //     {
    //         for (int j=0;j<round_ids[i].size();j++)
    //         {
    //             int64_t qb=round(compressed_data_vec[round_ids[i][j]]/(2*abs_eb));
    //             qb_vec.push_back(qb);
    //             compressed_data_vec[round_ids[i][j]]=qb*2*abs_eb;
    //             block_vec[id_vec2[round_ids[i][j]]].valid_num=1;
    //         }
    //         update_valid_num(block_vec,0);
    //     }
    //     else
    //     {
    //         for (int j=0;j<round_ids[i].size();j++)
    //         {
    //             int tgt_id=round_ids[i][j];
    //             priority_queue<pair<float,int>> nearest_vec;
    //             find_nearest(nearest_vec,4,block_vec,0,coord_vec,coord_vec.data()+tgt_id*3,true);
    //             vector<float> ref_data,ref_dist;
    //             while (!nearest_vec.empty())
    //             {
    //                 int ref_id=nearest_vec.top().second;
    //                 nearest_vec.pop();
    //                 ref_data.push_back(compressed_data_vec[ref_id]);
    //                 ref_dist.push_back(sqrt(get_dist_to_node(coord_vec.data()+tgt_id*3,coord_vec.data()+ref_id*3)));
    //             }
    //             float total_weight=0;
    //             for (auto dist:ref_dist)
    //                 total_weight+=pow(dist,-3.61);
    //             float pred=0;
    //             for (int k=0;k<ref_data.size();k++)
    //                 pred+=ref_data[k]*pow(ref_dist[k],-3.61)/total_weight;
    //             int64_t qb=round((compressed_data_vec[tgt_id]-pred)/(2*abs_eb));
    //             qb_vec.push_back(qb);
    //             compressed_data_vec[tgt_id]=pred+qb*2*abs_eb;
    //             block_vec[id_vec2[tgt_id]].valid_num=1;
    //             update_valid_num(block_vec,0);
    //         }
    //     }
    // }
    // printf("qb_vec.size()=%zu\n",qb_vec.size());
    // printf("node_num=%d\n",node_num);
    // for (int i=0;i<node_num;i++)
    //     if (abs(data_vec[i]-compressed_data_vec[i])>abs_eb*1.001)
    //         throw runtime_error("压缩后误差超过绝对误差阈值");
}
