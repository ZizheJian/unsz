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

int main()
{
    unsz::param_c<float,float> param;
    param.coord_path="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/coords.dat";
    for (int i=0;i<=161;i++)
        param.original_data_path_vec.push_back(
            "/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina/attr0.dat."+to_string(i));
    param.dim=3;
    param.rel_eb=1e-2;
    param.got_rel_eb=true;
    
    param.cross_time_compress=false;
    param.if_adjust_coord=true;

    param.init();
    if (param.if_adjust_coord)
        param.adjust_coord();
    exit(0);
    param.generate_pred_refs_structure();

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
        FILE *f_compressed=fopen(param.compressed_data_path_vec.back().c_str(),"wb");
        fwrite(param.current_zstd_bs_vec.data(),1,param.current_zstd_bs_vec.size(),f_compressed);
        fclose(f_compressed);
    }

    exit(0);

    
    
    

    

    

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