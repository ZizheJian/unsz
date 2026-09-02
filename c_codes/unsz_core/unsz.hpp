#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <map>
#include <memory>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <fstream>

using namespace std;

namespace unsz
{
    enum class point_set_codec_e:uint8_t{inverse_proportion=0,fitting=1,constant=2,hybrid_loo=3,use_default=255};
    enum class point_set_error_mode_e:uint8_t{use_global=0,relative=1,absolute=2};
    enum class partition_fitting_method_e:uint8_t{direct=0,continuous_mix=1,idw_residual=2};

    struct point_set_config_c
    {
        point_set_codec_e codec=point_set_codec_e::use_default;
        point_set_error_mode_e error_mode=point_set_error_mode_e::use_global;
        double error_bound=0;
    };

    template<typename T_data>
    using point_set_classifier_t=bool(*)(T_data);

#ifndef UNSZ_POINT_SET_CONFIG_HEADER
#define UNSZ_POINT_SET_CONFIG_HEADER user_point_set.hpp
#endif
#define UNSZ_STRINGIFY_POINT_SET_CONFIG_IMPL(value) #value
#define UNSZ_STRINGIFY_POINT_SET_CONFIG(value) UNSZ_STRINGIFY_POINT_SET_CONFIG_IMPL(value)
#include UNSZ_STRINGIFY_POINT_SET_CONFIG(UNSZ_POINT_SET_CONFIG_HEADER)
#undef UNSZ_STRINGIFY_POINT_SET_CONFIG
#undef UNSZ_STRINGIFY_POINT_SET_CONFIG_IMPL

    template<typename T>
    class array_view_c
    {
        private:
            const T *data_ptr=nullptr;
            uint32_t element_num=0;

        public:
            array_view_c()=default;
            array_view_c(const vector<T> &data):data_ptr(data.data()),element_num((uint32_t)data.size()){}
            array_view_c(const T *data,uint32_t size):data_ptr(data),element_num(size){}
            inline bool empty() const{return element_num==0;}
            inline uint32_t size() const{return element_num;}
            inline const T&operator[](uint32_t id) const{return data_ptr[id];}
            inline const T*begin() const{return data_ptr;}
            inline const T*end() const{return data_ptr?data_ptr+element_num:data_ptr;}
    };

    struct partition_prediction_config_c
    {
        uint32_t ref_num;
        double inverse_distance_power;
        bool use_fitting;
        double fitting_weight_power=0;
        double fitting_ridge=0;
        bool normalize_fitting_coord=false;
        partition_fitting_method_e fitting_method=partition_fitting_method_e::direct;
    };

    template<typename T_coord>
    class nearest_queue_c
    {
        private:
            static constexpr uint32_t small_capacity=11;//默认max_ref_num=10，反向查询额外包含目标自身
            array<pair<T_coord,int>,small_capacity> small_data;
            uint32_t small_size=0;
            priority_queue<pair<T_coord,int>> large_queue;
            bool use_large_queue=false;

        public:
            inline bool empty() const{return use_large_queue?large_queue.empty():small_size==0;}
            inline uint32_t size() const{return use_large_queue?(uint32_t)large_queue.size():small_size;}
            inline const pair<T_coord,int>&top() const{return use_large_queue?large_queue.top():small_data[small_size-1];}
            inline void push(const pair<T_coord,int> &value)
            {
                if (use_large_queue)
                {
                    large_queue.push(value);
                    return;
                }
                if (small_size==small_capacity)
                {
                    for (uint32_t i=0;i<small_size;i++)
                        large_queue.push(small_data[i]);
                    large_queue.push(value);
                    use_large_queue=true;
                    return;
                }
                uint32_t insert_id=small_size;
                while (insert_id>0 && value<small_data[insert_id-1])
                {
                    small_data[insert_id]=small_data[insert_id-1];
                    insert_id--;
                }
                small_data[insert_id]=value;
                small_size++;
            }
            inline void pop()
            {
                if (use_large_queue)
                {
                    large_queue.pop();
                    if (large_queue.empty())
                    {
                        use_large_queue=false;
                        small_size=0;
                    }
                }
                else
                    small_size--;
            }
    };

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
            uint32_t index_begin=0,index_end=0;
    };

    template<typename T_coord>
    class block_tree_structure_c
    {
        public:
            uint32_t dim=3;
            vector<block_c<T_coord>> block_vec;
            vector<T_coord> block_coord_bounds;//每个block连续保存dim个最小坐标和dim个最大坐标
            vector<int> original_id_to_tree_id;
            vector<int> tree_id_to_original_id;
            vector<int> original_id_to_block_vec_id;
            vector<int> tree_id_to_block_vec_id;
            Array2D<T_coord> tree_order_coord_array;
            uint32_t leaf_point_num=1;
    };

    template<typename T_coord>
    class block_tree_c
    {
        public:
            uint32_t dim=3;
            shared_ptr<block_tree_structure_c<T_coord>> structure;
            vector<int> valid_num;//当前实际有效的点数，根据情况可能表示已压缩的点或者正在预测的点
            vector<uint8_t> point_valid;//按tree_id保存逐点有效性，多点叶节点查询时使用

            inline T_coord*get_block_min_coord(int block_id){return structure->block_coord_bounds.data()+(uint64_t)block_id*dim*2;}
            inline T_coord*get_block_max_coord(int block_id){return get_block_min_coord(block_id)+dim;}
            inline uint64_t get_block_num() const{return structure->block_vec.size();}
            inline int get_original_leaf_block_id(int original_id) const{return structure->original_id_to_block_vec_id[original_id];}

            void split_block(Array2D<T_coord> &coord_array,int block_id);
            block_tree_c(Array2D<T_coord> &coord_array,vector<T_coord> &min_coord,vector<T_coord> &max_coord,uint32_t leaf_point_num=1);
            block_tree_c(const block_tree_c<T_coord> &other)=default;
            void clear_all_valid_num();
            void set_point_valid(int original_id,bool value);
            void update_all_valid_num(int block_id);
            void update_point_valid_num_to_root(int original_id,int change_value);
            void update_point_valid_num_to_partition_root(int original_id,int change_value,int partition_root_block_id);
            void find_original_nearest(nearest_queue_c<T_coord> &nearest_queue,Array2D<T_coord> &coord_array,
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

            string data_dir,variable_name,error_mode,coord_path,prediction_structure_path,hybrid_selector_path,adjust_coord_matrix_path,adjust_coord_training_type;
            vector<string> original_data_path_vec,compressed_data_path_vec,decompressed_data_path_vec;
            uint32_t dim=3;
            uint32_t point_num=0;
            T_data rel_eb=0,abs_eb=0;
            bool got_rel_eb=false,got_abs_eb=false;
            bool got_data_dir=false,got_variable=false,got_time_step=false,got_error_mode=false;
            vector<point_set_classifier_t<T_data>> point_set_classifier_vec=get_user_point_set_classifier_vec<T_data>();
            vector<point_set_config_c> point_set_config_vec=get_user_point_set_config_vec();
            bool check_original_data_files=true;
            bool cross_time_compress=false;
            string cross_time_predictor_experimental="none";//none / previous / interpolation
            bool if_adjust_coord=false;
            vector<double> adjust_coord_matrix_parameter_vec;
            bool print_round_statistics=false;
            bool print_prediction_geometry_statistics=false;
            bool print_label_compression_statistics=false;
            bool print_compression_stage_time=false;
            bool print_prediction_dag_statistics=false;
            bool print_pointwise_oracle_statistics_experimental=false;
            bool use_prediction_order_layout_experimental=false;
            bool use_normalized_weight_experimental=false;
            bool use_prediction_order_label_experimental=false;
            bool use_predictability_order_experimental=false;
            bool use_forward_mean_distance_only_order_experimental=false;
            bool use_forward_s2_only_order_experimental=false;
            bool use_forward_direction_only_order_experimental=false;
            bool use_forward_mean_distance_s2_order_experimental=false;
            bool use_forward_mean_distance_direction_order_experimental=false;
            bool use_reverse_predictability_order_experimental=false;
            bool use_reverse_distribution_order_experimental=false;
            bool use_reverse_mean_distance_only_order_experimental=false;
            bool use_reverse_reference_reuse_order_experimental=false;
            bool use_reverse_s2_only_order_experimental=false;
            bool use_reverse_direction_only_order_experimental=false;
            bool use_reverse_direction_order_experimental=false;
            bool use_reverse_direction_distribution_order_experimental=false;
            vector<uint32_t> order_training_time_step_vec_experimental;//非空时使用多个训练time step的预测误差生成公共顺序
            bool use_global_order_before_round_split_experimental=false;
            uint32_t d_only_cache_refill_threshold_experimental=8;//0表示任一参考点失效就重新查询，即精确模式
            uint32_t order_candidate_num_experimental=0;//0表示使用max_ref_num
            uint32_t cross_time_spatial_point_num_experimental=0;//仅用于统计展平时空点集中的跨时间参考
            uint32_t kd_tree_leaf_size=32;
            uint32_t partition_size_experimental=0;//0表示不分区；非0时所有参考点必须来自同一partition
            uint32_t partition_num_experimental=0;//直接指定2、4、8等2的幂个partition
            uint32_t thread_num=1;
            bool use_partition_adaptive_predictor_experimental=false;
            uint32_t partition_adaptive_sample_num_experimental=256;
            string partition_fitting_candidates_experimental="original";//original / weighted / ridge / mixed / residual / extended / all
            string multi_huffman_mode_experimental="none";//none / round / partition / partition_round
            string eb_schedule_experimental="round_point_ratio";//none / round / point / round_point / round_ratio / round_point_ratio / round_point_ratio_dag / round_point_ratio_dag_direct / round_point_ratio_dag_variance_direct
            double eb_schedule_start_ratio=0.8;
            uint32_t predict_round_num=8;
            uint32_t max_ref_num=10;//不知道点集时先保留候选点，再筛选同点集参考点
            uint32_t actual_ref_num=10;//标签预测使用的候选点数，也是数值预测最多使用的同点集参考点数
            double inverse_distance_power=3.6;//欧氏距离的负指数；代码中的dist是平方距离，因此实际使用-power/2
            int zstd_compression_level=3;
            string point_set_predictor="inverse_proportion";//inverse_proportion / majority / continuous_actual / continuous_reflect
            string predictor="inverse_proportion";//没有逐点集配置时使用的默认预测器

            Array2D<T_coord> coord_array,coord_array_backup;
            vector<T_coord> min_coord,max_coord;
            vector<block_tree_c<T_coord>> block_vec;
            vector<int> id_map1;//从原始id映射到block_tree上node的专用id（没有被使用）
            vector<int> point_id_on_tree_to_original_id;//从block_tree上node的专用id映射到原始id
            vector<int> id_map2;//从block_tree上node的专用id映射到leaf block在block_vec中的id
            vector<int> id_map_reverse2;//从leaf block在block_vec中的id映射到block_tree上node的专用id（没有被使用）
            vector<vector<int>> round_target_ids;//每一轮的预测目标
            vector<int> pred_target_ids,pred_ref_ids;//预测目标与连续参考点数组；实验性内部重排关闭时为原始id，开启时为预测id
            vector<uint64_t> pred_ref_begin;//第i个目标的参考点区间为[pred_ref_begin[i],pred_ref_begin[i+1])
            vector<double> pred_ref_weights,pred_ref_label_normalized_weights,pred_ref_value_normalized_weights;
            vector<uint32_t> prediction_id_to_original_id,original_id_to_prediction_id;
            vector<vector<int>> partition_point_ids_vec;
            vector<uint32_t> point_partition_id_vec;
            vector<int> partition_root_block_ids;
            vector<vector<uint32_t>> partition_pred_ref_ids_vec;
            vector<uint8_t> partition_prediction_config_id_vec,partition_adaptive_sample_mask;

            vector<T_data> current_data,current_decompressed_data;
            vector<T_data> previous_time_decompressed_data;
            vector<uint32_t> previous_time_point_set_id_vec;
            vector<uint32_t> time_step_vec,reconstructed_time_step_vec;
            vector<vector<T_data>> reconstructed_time_data_vec;
            vector<vector<uint32_t>> reconstructed_time_point_set_id_vec;
            uint32_t current_time_step=0;
            int32_t interpolation_left_history_id=-1,interpolation_right_history_id=-1;
            double interpolation_right_weight=0;
            T_data current_min_data=0,current_max_data=0;
            T_data current_abs_eb=0;
            vector<int32_t> qb_vec;
            vector<unsigned char> current_zstd_bs_vec;
            vector<uint32_t> point_set_id_vec;
            vector<uint32_t> point_set_point_num_vec;
            vector<T_data> point_set_min_data_vec,point_set_max_data_vec,point_set_abs_eb_vec;
            vector<point_set_codec_e> point_set_codec_vec;
            uint32_t current_valid_node_num=0;
            uint64_t hybrid_prediction_num=0,hybrid_fitting_prediction_num=0;
            vector<uint8_t> hybrid_selector_vec;

            void init();
            void prepare_time_interpolation();
            bool predict_time_interpolation(uint32_t original_id,uint32_t point_set_id,T_data &prediction);
            void save_time_reconstruction();
            bool need_double_coord();
            void load_adjust_coord_matrix_file();
            void adjust_coord();
            void load_or_generate_pred_refs_structure();
            void build_partitions();
            void build_partitions(block_tree_c<T_coord> &tree);
            void generate_partition_pred_refs_structure();
            vector<partition_prediction_config_c> get_partition_prediction_config_vec();
            vector<double> generate_pred_eb_multiplier_vec(bool print_statistics=false);
            void select_partition_prediction_configs();
            void prepare_partition_adaptive_sample_mask();
            array_view_c<int> get_pred_ref_ids(uint32_t pred_ref_id) const;
            array_view_c<double> get_pred_ref_weights(uint32_t pred_ref_id,bool label_normalized=false,bool value_normalized=false) const;
            T_data predict_partition_adaptive_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,uint8_t config_id);
            T_data predict_partition_adaptive_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,const partition_prediction_config_c &config);
            uint32_t get_multi_huffman_group_num();
            uint32_t get_multi_huffman_group_id(uint32_t round_id,int pred_id);
            void generate_pred_refs_structure();
            void prepare_pred_refs_structure();
            void print_prediction_geometry_statistics_result();
            void save_pred_refs_structure();
            void load_pred_refs_structure();
            void print_prediction_dag_statistics_result();
            void split_block(int block_id);
            void generate_id_map2();
            void clear_all_valid_num();
            void update_all_valid_num(int block_id);
            void update_valid_num_to_root(int block_id,int change_value);
            void find_nearest(priority_queue<pair<T_coord,int>> &nearest_queue,int nearest_num,int block_id,T_coord *query_coord,bool valid_only=false);

            uint32_t get_point_set_id(T_data val);
            uint32_t predict_binary_label(array_view_c<int> ref_ids,array_view_c<double> ref_weights,const vector<uint8_t> &label_vec,bool use_majority);
            double predict_binary_label_value(array_view_c<int> ref_ids,array_view_c<double> ref_weights,const vector<double> &label_value_vec);
            double predict_binary_label_value_normalized(array_view_c<int> ref_ids,array_view_c<double> ref_weights,const vector<double> &label_value_vec);
            T_data predict_inverse_value_from_coords(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double power=-1);
            bool predict_inverse_value_and_loo_error(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double power,T_data &prediction,double &loo_error);
            bool predict_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction,double weight_power=0,double ridge=0,bool normalize_coord=false);
            bool predict_fitting_value_and_loo_error(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction,double &loo_error,double weight_power=0,double ridge=0,bool normalize_coord=true);
            bool predict_continuous_mix_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double idw_power,double fitting_weight_power,T_data &prediction);
            bool predict_idw_residual_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double idw_power,double residual_weight_power,T_data &prediction);
            T_data predict_value(int pred_id,array_view_c<int> ref_ids,array_view_c<double> ref_weights,uint32_t point_set_id,vector<T_data> &data);
            T_data predict_value_normalized(array_view_c<int> ref_ids,array_view_c<double> ref_weights,vector<T_data> &data);
            void compress();
            void restore_compression_result_to_original_order();
            void decompress();
            
            bool solve_4x4(double A_in[4][4],double b_in[4],double x[4]);
            bool solve_linear_system(vector<vector<double>> A,vector<double> b,vector<double> &x);
    };
}

#include "distance.tpp"
#include "param.tpp"
#include "tree.tpp"
#include "init.tpp"
#include "adjust_coord_matrix_file.tpp"
#include "adjust_coord.tpp"
#include "partition.tpp"
#include "partition_adaptive_prediction.tpp"
#include "eb_schedule.tpp"
#include "generate_pred_refs_structure.tpp"
#include "prediction_dag.tpp"
#include "point_set.tpp"
#include "time_prediction.tpp"
#include "format.tpp"
#include "multi_huffman.tpp"
#include "prediction_structure.tpp"
#include "compress.tpp"
#include "decompress.tpp"
