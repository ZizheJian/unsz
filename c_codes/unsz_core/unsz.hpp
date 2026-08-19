#pragma once

#include <vector>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace std;

namespace unsz
{
    template<typename T_coord>
    T_coord get_dist_to_coord(T_coord *coord1,T_coord *coord2,uint32_t dim);

    template<typename T_coord>
    T_coord get_dist_to_block(T_coord *coord,T_coord *min_coord,T_coord *max_coord,uint32_t dim);

    template<typename T>
    class Array2D{
        public:
            uint32_t n0=0,n1=0;
            vector<T> data;

            Array2D()=default;
            Array2D(uint32_t n0_,uint32_t n1_):n0(n0_),n1(n1_),data((uint64_t)n0_*n1_){}
            void resize(uint32_t n0_,uint32_t n1_)
            {
                n0=n0_;
                n1=n1_;
                data.resize((uint64_t)n0*n1);
            }
            inline T*operator[](uint32_t i){return data.data()+(uint64_t)i*n1;}
    };

    template<typename T_coord>
    class block_c
    {
        public:
            int parent_id=-1;
            int left_child_id=-1,right_child_id=-1;
            vector<T_coord> min_coord,max_coord;
            uint32_t index_begin=0,index_end=0;
            int valid_num=0;//当前实际有效的点，根据情况可能表示已压缩的点或者正在预测的点
    };

    template<typename T_coord>
    class block_tree_c
    {
        public:
            uint32_t dim=3;

            vector<block_c<T_coord>> block_vec;
            vector<int> original_id_to_tree_id;
            vector<int> tree_id_to_original_id;
            vector<int> original_id_to_block_vec_id;
            vector<int> tree_id_to_block_vec_id;//没有block_vec_id_to_tree_id，因为叶节点有index_begin和index_end

            void split_block(Array2D<T_coord> &coord_array,int block_id);
            block_tree_c(Array2D<T_coord> &coord_array,vector<T_coord> &min_coord,vector<T_coord> &max_coord);
            void clear_all_valid_num();
            void update_all_valid_num(int block_id);
            void update_valid_num_to_root(int block_id,int change_value);
            void find_original_nearest(priority_queue<pair<T_coord,int>> &nearest_queue,Array2D<T_coord> &coord_array,
                uint32_t nearest_num,int block_id,T_coord *query_coord,bool valid_only=false);
                //寻找最近的点。如果query_coord是一个点的坐标，那么会找到自身
    };

    template<typename T_coord,typename T_data>
    class param_c
    {
        public:
            param_c()=default;
            template<typename T_coord2>
            param_c(const param_c<T_coord2,T_data> &other);

            string coord_path;
            vector<string> original_data_path_vec,compressed_data_path_vec,decompressed_data_path_vec;
            uint32_t dim=3;
            uint32_t point_num=0;
            T_data rel_eb=0,abs_eb=0;
            bool got_rel_eb=false,got_abs_eb=false;
            bool cross_time_compress=false;
            bool if_adjust_coord=false;
            bool print_round_statistics=false;
            uint32_t predict_round_num=10;
            uint32_t max_ref_num=10;//在不知道哪些点有效的情况下，需要找到最近的max_ref_num个参考点
            uint32_t actual_ref_num=10;//实际使用的参考点个数
            bool predict_abnormal_points=false;
            string predictor="inverse_proportion";//fitting / inverse_proportion

            Array2D<T_coord> coord_array,coord_array_backup;
            vector<T_coord> min_coord,max_coord;
            vector<block_tree_c<T_coord>> block_vec;
            vector<int> id_map1;//从原始id映射到block_tree上node的专用id（没有被使用）
            vector<int> point_id_on_tree_to_original_id;//从block_tree上node的专用id映射到原始id
            vector<int> id_map2;//从block_tree上node的专用id映射到leaf block在block_vec中的id
            vector<int> id_map_reverse2;//从leaf block在block_vec中的id映射到block_tree上node的专用id（没有被使用）
            vector<vector<int>> round_target_ids;//每一轮的预测目标
            vector<pair<int,vector<int>>> pred_refs_vec;//每个预测目标及其可用的参考点，编号都是原始id

            vector<T_data> current_data,current_decompressed_data;
            T_data current_min_data=0,current_max_data=0;
            T_data current_abs_eb=0;
            vector<int64_t> qb_vec;
            vector<unsigned char> current_zstd_bs_vec;
            vector<bool> is_abnormal_vec;
            uint32_t current_valid_node_num=0;

            void init();
            bool need_double_coord();
            void adjust_coord();
            void generate_pred_refs_structure();
            void split_block(int block_id);
            void generate_id_map2();
            void clear_all_valid_num();
            void update_all_valid_num(int block_id);
            void update_valid_num_to_root(int block_id,int change_value);
            void find_nearest(priority_queue<pair<T_coord,int>> &nearest_queue,int nearest_num,int block_id,T_coord *query_coord,bool valid_only=false);

            bool is_abnormal(T_data val);
            void compress();
            void predict_and_quantize_inverse_proportion_with_abnormal(int pred_id,vector<int> &ref_ids);
            void predict_and_quantize_inverse_proportion_without_abnormal(int pred_id,vector<int> &ref_ids);
            void predict_and_quantize_fitting_with_abnormal(int pred_id,vector<int> &ref_ids);
            void predict_and_quantize_fitting_without_abnormal(int pred_id,vector<int> &ref_ids);
            
            bool solve_4x4(double A_in[4][4],double b_in[4],double x[4]);
    };
}

#include "distance.tpp"
#include "param.tpp"
#include "tree.tpp"
#include "init.tpp"
#include "adjust_coord.tpp"
#include "generate_pred_refs_structure.tpp"
#include "is_abnormal.tpp"
#include "compress.tpp"
