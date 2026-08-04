#pragma once

#include <vector>
#include <chrono>
#include <cstring>

using namespace std;

namespace unsz
{
    template<typename T>
    class Array2D{
        public:
            int n0,n1;
            vector<T> data;

            Array2D(){}
            Array2D(int n0_,int n1_):n0(n0_),n1(n1_),data(n0_*n1_){}
            void resize(int n0_,int n1_)
            {
                n0=n0_;
                n1=n1_;
                data.resize(n0*n1);
            }
            inline T*operator[](int i){return data.data()+i*n1;}
    };

    template<typename T_coord>
    class block_c
    {
        public:
            int parent_id=-1;
            int left_child_id=-1,right_child_id=-1;
            vector<T_coord> min_coord,max_coord;
            int index_begin=0,index_end=0;
            int valid_num=0;//当前实际有效的点，根据情况可能表示已压缩的点或者正在预测的点
    };

    template<typename T_coord>
    class block_tree_c
    {
        public:
            vector<block_c<T_coord>> block_vec;
            vector<int> original_id_to_point_id_on_tree;
            vector<int> point_id_on_tree_to_original_id;
            vector<int> point_id_on_tree_to_block_vec_id;

            block_tree_c(Array2D<T_coord> &coord_array,vector<T_coord> &min_coord,vector<T_coord> &max_coord);
            void split_block(Array2D<T_coord> &coord_array,int block_id);
    };

    template<typename T_coord,typename T_data>
    class param_c
    {
        public:
            string coord_path;
            vector<string> original_data_path_vec,compressed_data_path_vec,decompressed_data_path_vec;
            int dim=3;
            int node_num;
            T_data rel_eb,abs_eb;
            bool got_rel_eb=false,got_abs_eb=false;
            bool cross_time_compress=false;
            bool if_adjust_coord=false;
            int predict_round_num=10;
            int max_ref_num=10;//在不知道哪些点有效的情况下，需要找到最近的max_ref_num个参考点
            int actual_ref_num=10;//实际使用的参考点个数
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
            vector<pair<int,vector<int>>> pred_refs_vec;//每个参考点可用的预测目标，编号都是原始id

            vector<T_data> current_data,current_decompressed_data;
            T_data current_min_data,current_max_data;
            T_data current_abs_eb;
            vector<int64_t> qb_vec;
            vector<unsigned char> current_zstd_bs_vec;
            vector<bool> is_abnormal_vec;
            int current_valid_node_num=0;

            void init();
            void adjust_coord();
            void generate_pred_refs_structure();
            void split_block(int block_id);
            void generate_id_map2();
            void clear_all_valid_num();
            void update_all_valid_num(int block_id);
            void update_valid_num_to_root(int block_id,int change_value);
            T_coord get_dist_to_node(T_coord *coord1,T_coord *coord2);
            T_coord get_dist_to_block(T_coord *coord,T_coord *min_coord,T_coord *max_coord);
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

#include "tree.tpp"
#include "init.tpp"
#include "adjust_coord.tpp"
#include "generate_pred_refs_structure.tpp"
#include "is_abnormal.tpp"
#include "compress.tpp"