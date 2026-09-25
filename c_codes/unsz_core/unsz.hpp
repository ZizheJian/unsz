#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <cstdint>
#include <cstdio>
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
#include <sstream>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include "../huffman.hpp"
#include "../third_party/zstd/lib/zstd.h"

using namespace std;

#include "simple_json.hpp"

namespace unsz
{
    enum class point_set_codec_e:uint8_t{inverse_proportion=0,constant=2,rbf_fitting=5};
    enum class point_set_error_mode_e:uint8_t{relative=0,absolute=1};
    enum class point_set_rule_operator_e:uint8_t{less=0,less_equal=1,greater=2,greater_equal=3,equal=4,not_equal=5};
    enum class partition_fitting_method_e:uint8_t{direct=0,continuous_mix=1,idw_residual=2};
    enum class time_dimension_mode_e:uint8_t{not_applicable=0,spatial=1,constant_extrapolation=2,endpoint_agreement=5,linear_interpolation=6,joint_spatiotemporal=7,linear_interpolation_spatial_residual=8,constant_extrapolation_spatial_residual=9};
    enum class label_value_execution_e:uint8_t{no_label=0,staged=1,fused=2};

    struct prediction_framework_c
    {
        bool has_label=false;
        time_dimension_mode_e label_mode=time_dimension_mode_e::not_applicable;
        time_dimension_mode_e value_mode=time_dimension_mode_e::spatial;
        label_value_execution_e execution=label_value_execution_e::no_label;
    };

    struct point_set_config_c
    {
        point_set_codec_e codec=point_set_codec_e::inverse_proportion;
        point_set_error_mode_e error_mode=point_set_error_mode_e::relative;
        double eb=0;
    };

    struct point_set_rule_c
    {
        point_set_rule_operator_e operator_type=point_set_rule_operator_e::equal;
        double value=0;
    };

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
            static constexpr uint32_t small_capacity=11;//常用max_ref_num不超过10，反向查询额外包含目标自身
            struct distance_less_c
            {
                inline bool operator()(const pair<T_coord,int> &a,const pair<T_coord,int> &b) const{return a.first<b.first;}
            };
            array<pair<T_coord,int>,small_capacity> small_data;
            uint32_t small_size=0;
            priority_queue<pair<T_coord,int>,vector<pair<T_coord,int>>,distance_less_c> large_queue;
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
                while (insert_id>0 && value.first<small_data[insert_id-1].first)
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

    struct nearest_query_stats_c
    {
        uint64_t query_num=0,block_visit_num=0,empty_block_prune_num=0,leaf_visit_num=0,point_check_num=0,invalid_point_skip_num=0,point_distance_num=0,block_distance_num=0,queue_push_num=0,queue_replace_num=0,far_child_prune_num=0;
    };

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
            vector<uint64_t> leaf_valid_mask;//按block保存叶节点中有效点的位图；只在实验路径启用
            bool use_leaf_valid_mask=false;
            bool use_iterative_query=false;

            inline T_coord*get_block_min_coord(int block_id){return structure->block_coord_bounds.data()+(uint64_t)block_id*dim*2;}
            inline T_coord*get_block_max_coord(int block_id){return get_block_min_coord(block_id)+dim;}
            inline uint64_t get_block_num() const{return structure->block_vec.size();}
            inline int get_original_leaf_block_id(int original_id) const{return structure->original_id_to_block_vec_id[original_id];}

            void split_block(Array2D<T_coord> &coord_array,int block_id);
            block_tree_c(Array2D<T_coord> &coord_array,vector<T_coord> &min_coord,vector<T_coord> &max_coord,uint32_t leaf_point_num=1);
            block_tree_c(const block_tree_c<T_coord> &other)=default;
            void enable_leaf_valid_mask();
            void clear_all_valid_num();
            void set_point_valid(int original_id,bool value);
            void update_all_valid_num(int block_id);
            void update_point_valid_num_to_root(int original_id,int change_value);
            void update_point_valid_num_to_partition_root(int original_id,int change_value,int partition_root_block_id);
            void find_original_nearest(nearest_queue_c<T_coord> &nearest_queue,Array2D<T_coord> &coord_array,
                uint32_t nearest_num,int block_id,T_coord *query_coord,bool valid_only=false,nearest_query_stats_c *stats=nullptr);
            void find_original_nearest_iterative(nearest_queue_c<T_coord> &nearest_queue,Array2D<T_coord> &coord_array,
                uint32_t nearest_num,int block_id,T_coord *query_coord,bool valid_only=false,nearest_query_stats_c *stats=nullptr);
            bool find_original_nearest_other_iterative(T_coord *query_coord,int excluded_original_id,int block_id,T_coord &nearest_distance,int &nearest_original_id,bool valid_only=false,nearest_query_stats_c *stats=nullptr);
                //寻找最近的点。如果query_coord是一个点的坐标，那么会找到自身
    };

    template<typename T_coord,typename T_data>
    class param_c
    {
        public:
            param_c()=default;
            template<typename T_coord2>
            param_c(const param_c<T_coord2,T_data> &other);

            string data_dir,variable_name,error_mode,coord_path,time_step_coordinate_path,prediction_structure_path,adjust_coord_matrix_path,adjust_coord_training_type,point_set_config_path;
            vector<string> original_data_path_vec,compressed_data_path_vec,decompressed_data_path_vec;
            uint32_t dim=3;
            uint32_t point_num=0;
            T_data rel_eb=0,abs_eb=0;
            bool got_rel_eb=false,got_abs_eb=false;
            bool got_data_dir=false,got_variable=false,got_time_step=false,got_error_mode=false,got_predictor=false,no_prediction_dag_cache=false;
            vector<point_set_rule_c> point_set_rule_vec;
            vector<point_set_config_c> point_set_config_vec;
            bool check_original_data_files=true;
            bool cross_time_compress=false;
            string cross_time_predictor_experimental="none";//none / constant_extrapolation / linear_interpolation / constant_extrapolation_spatial_residual / linear_interpolation_spatial_residual
            string cross_time_label_predictor_experimental="none";//none / previous / endpoint_agreement
            bool if_adjust_coord=false;
            vector<double> adjust_coord_matrix_parameter_vec;
            bool print_round_statistics=false;
            bool print_prediction_geometry_statistics=false;
            bool print_label_compression_statistics=false;
            bool print_prediction_residual_statistics=false;
            bool print_compression_stage_time=false;
            bool print_prediction_dag_statistics=false;
            bool print_pointwise_oracle_statistics_experimental=false;
            bool reorder_data_by_prediction_order_experimental=false;
            string qb_storage_order="prediction";//prediction / original
            double last_reorder_data_by_prediction_order_time=0;
            bool use_normalized_weight_experimental=false;
            bool use_prediction_order_label_experimental=false;
            bool use_fused_two_point_set_prediction_experimental=false;
            bool use_linear_previous_label_experimental=false;
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
            vector<uint32_t> order_reference_time_step_vec_experimental;//非空时使用多个参考time step的预测误差搜索公共顺序
            bool use_reference_data_rbf_score_experimental=false;//参考数据辅助顺序使用RBF而非IDW预测误差评分
            bool use_global_order_before_round_split_experimental=false;
            uint32_t cache_subpartition_num_experimental=0;//后3轮按kd-tree上层区域连续处理；0表示关闭
            uint32_t d_only_cache_refill_threshold_experimental=3;//0表示任一参考点失效就重新查询，即精确模式
            uint32_t order_candidate_num_experimental=0;//0表示使用max_ref_num
            uint32_t ordering_msd_ref_num_experimental=0;//0表示使用max_ref_num；只控制反向MSD排序评分的参考点数量
            uint32_t ordering_msd_neighbor_prefetch_num_experimental=0;//非0时为1点MSD评分预取近邻，并尝试复用为排序候选
            bool use_msd_candidate_score_geometry_reuse_experimental=false;//用排序候选云的距离界证明最近评分参考点时不查kd-tree
            bool use_kd_query_diagnostics_experimental=false;//分别统计排序候选和MSD评分查询的kd-tree工作量
            bool use_kd_leaf_valid_mask_experimental=false;//叶节点用位图枚举有效点，避免逐点检查失效状态
            bool use_kd_iterative_query_experimental=false;//固定栈迭代遍历kd-tree，避免递归函数调用
            bool use_kd_scalar_one_reference_query_experimental=false;//1点MSD用标量最近邻查询并排除目标自身
            bool use_rbf_ldlt_experimental=false;//常数趋势RBF用正定核矩阵的LDLT分解求权重
            uint32_t precomputed_neighbor_num_experimental=0;//0保持原DAG生成；非0只在初始阶段用kd-tree预存近邻
            uint32_t multihead_execution_head_num_experimental=0;//非0时按空间头轮转执行DAG，并按头连续存储qb
            uint32_t time_anchor_stride_experimental=0;//非0时使用固定步长锚点和区间内插的两点集时间流程
            uint32_t time_anchor_first_step=0,time_anchor_last_step=0,time_anchor_last_regular_step=0;
            uint32_t cross_time_spatial_point_num_experimental=0;//非0时启用联合时空压缩，并记录每个time step的空间点数
            uint32_t kd_tree_leaf_size=32;
            uint32_t partition_size_experimental=0;//0表示不分区；非0时所有参考点必须来自同一partition
            uint32_t partition_num_experimental=0;//直接指定2、4、8等2的幂个partition
            uint32_t thread_num=1;
            bool use_partition_adaptive_predictor_experimental=false;
            uint32_t partition_adaptive_sample_num_experimental=256;
            double partition_adaptive_min_improvement_experimental=0;
            string partition_fitting_candidates_experimental="original";//original / weighted / ridge / mixed / residual / extended / all
            string multi_huffman_mode_experimental="none";//none / round / partition / partition_round
            string eb_schedule_experimental="round_point_ratio";//none / round / point / point_log2_ratio / round_point / round_ratio / round_point_ratio / round_point_ratio_dag / round_point_ratio_dag_direct / round_point_ratio_dag_variance_direct
            double eb_schedule_start_ratio=0.8;
            double eb_schedule_log2_shift_experimental=0;//point_log2_ratio的指数位移
            uint32_t predict_round_num=8;
            double round_size_growth_ratio_experimental=2;
            uint32_t max_ref_num=4;//不知道点集时先保留候选点，再筛选同点集参考点
            uint32_t actual_ref_num=4;//标签预测使用的候选点数，也是数值预测最多使用的同点集参考点数
            double inverse_distance_power=2;//欧氏距离的负指数；代码中的dist是平方距离，因此实际使用-power/2
            string rbf_kernel="gaussian";//gaussian / exponential / matern32 / matern52 / inverse_multiquadric
            double rbf_shape=1;
            double rbf_nugget=1e-8;
            uint32_t rbf_polynomial_degree=0;
            bool rbf_parameter_specified=false;
            bool use_rbf_filtered_fallback_nearest_experimental=false;
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
            vector<uint32_t> multihead_execution_pred_ref_ids,multihead_qb_index_by_pred_ref_id;
            vector<uint64_t> pred_ref_begin;//第i个目标的参考点区间为[pred_ref_begin[i],pred_ref_begin[i+1])
            vector<double> pred_ref_weights,pred_ref_label_normalized_weights,pred_ref_value_normalized_weights,pred_ref_rbf_weights;
            uint32_t single_point_set_rbf10_fast_begin=0;
            vector<double> cached_pred_eb_multiplier_vec;//不依赖当前field数据的EB调度只生成一次，由全部time step复用
            vector<double> cached_pred_eb_inverse_multiplier_vec;//对应EB倍率的倒数，避免每个time step重复执行逐点除法
            vector<uint32_t> prediction_id_to_original_id,original_id_to_prediction_id;
            vector<vector<int>> partition_point_ids_vec;
            vector<uint32_t> point_partition_id_vec;
            vector<int> partition_root_block_ids;
            vector<vector<uint32_t>> partition_pred_ref_ids_vec;
            vector<uint8_t> partition_prediction_config_id_vec,partition_adaptive_sample_mask;

            vector<T_data> current_data,current_decompressed_data;
            vector<T_data> previous_time_decompressed_data;
            vector<uint8_t> previous_time_point_set_id_vec;
            vector<uint32_t> time_step_vec,reconstructed_time_step_vec;
            vector<double> time_step_coordinate_vec;
            vector<vector<T_data>> reconstructed_time_data_vec;
            vector<vector<uint8_t>> reconstructed_time_point_set_id_vec;
            vector<T_data> current_time_spatial_residual_data;
            uint32_t current_time_step=0;
            prediction_framework_c current_prediction_framework;
            int32_t linear_interpolation_left_history_id=-1,linear_interpolation_right_history_id=-1;
            double linear_interpolation_right_weight=0;
            T_data current_min_data=0,current_max_data=0;
            T_data current_abs_eb=0;
            vector<int32_t> qb_vec;
            vector<unsigned char> current_zstd_bs_vec;
            vector<uint8_t> point_set_id_vec;
            vector<uint32_t> point_set_point_num_vec;
            vector<T_data> point_set_min_data_vec,point_set_max_data_vec,point_set_abs_eb_vec;
            vector<point_set_codec_e> point_set_codec_vec;
            uint32_t current_valid_node_num=0;

            void init();
            void load_time_step_coordinates();
            void prepare_linear_interpolation_order();
            void prepare_time_anchor_order();
            void configure_point_sets();
            void prepare_linear_interpolation();
            bool predict_linear_interpolation(uint32_t original_id,uint32_t point_set_id,T_data &prediction);
            template<time_dimension_mode_e value_mode>
            bool predict_time_spatial_residual_baseline_fixed(uint32_t original_id,uint32_t point_set_id,T_data &prediction);
            template<time_dimension_mode_e value_mode>
            void update_time_spatial_residual_fixed(int pred_id,uint32_t original_id,uint32_t point_set_id);
            template<time_dimension_mode_e label_mode>
            bool predict_time_label_fixed(uint32_t original_id,uint32_t label_channel_id,uint32_t &prediction);
            void save_time_reconstruction();
            prediction_framework_c get_prediction_framework(uint32_t point_set_num) const;
            uint32_t get_last_regular_time_anchor() const;
            bool current_time_step_is_regular_anchor() const;
            void resolve_single_point_set_time_mode();
            void advance_time_prediction_buffers();
            bool need_double_coord();
            void load_adjust_coord_matrix_file();
            void adjust_coord();
            void load_or_generate_pred_refs_structure();
            void build_partitions();
            void build_partitions(block_tree_c<T_coord> &tree);
            vector<uint32_t> get_round_target_num_vec(uint32_t local_point_num) const;
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
            void generate_precomputed_neighbor_pred_refs_structure();
            void prepare_pred_refs_structure();
            void prepare_multihead_execution_order();
            void prepare_rbf_fitting_weights();
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
            bool predict_binary_label_and_inverse_value(array_view_c<int> ref_ids,array_view_c<double> ref_weights,array_view_c<double> normalized_ref_weights,const vector<T_data> &data,uint32_t value_point_set_id,bool predict_value,uint32_t &predicted_label,T_data &predicted_value);
            bool predict_known_point_set_inverse_value(array_view_c<int> ref_ids,array_view_c<double> ref_weights,array_view_c<double> normalized_ref_weights,const vector<T_data> &data,uint32_t value_point_set_id,bool predict_value,T_data &predicted_value);
            T_data predict_inverse_value_from_coords(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double power=-1);
            bool predict_inverse_value_and_loo_error(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double power,T_data &prediction,double &loo_error);
            bool predict_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction,double weight_power=0,double ridge=0,bool normalize_coord=false);
            bool calculate_rbf_fitting_weights(int pred_id,array_view_c<int> ref_ids,vector<double> &weights);
            bool predict_rbf_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction);
            bool predict_fitting_value_and_loo_error(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction,double &loo_error,double weight_power=0,double ridge=0,bool normalize_coord=true);
            bool predict_continuous_mix_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double idw_power,double fitting_weight_power,T_data &prediction);
            bool predict_idw_residual_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double idw_power,double residual_weight_power,T_data &prediction);
            template<point_set_codec_e codec>
            T_data predict_value_fixed(uint32_t pred_ref_id,int pred_id,array_view_c<int> ref_ids,array_view_c<double> ref_weights,uint32_t point_set_id,vector<T_data> &data);
            T_data predict_single_point_set_rbf_value(uint32_t pred_ref_id,vector<T_data> &data);
            T_data predict_single_point_set_rbf10_value(uint32_t pred_ref_id,uint64_t ref_begin,vector<T_data> &data);
            T_data predict_value_normalized(array_view_c<int> ref_ids,array_view_c<double> ref_weights,vector<T_data> &data);
            void prepare_compression_structure(chrono::steady_clock::time_point preprocessing_start_time);
            void prepare_joint_spatiotemporal_coordinates();
            void compress_files(void(param_c<T_coord,T_data>::*compress_current_file)());
            void compress_joint_spatiotemporal_files(void(param_c<T_coord,T_data>::*compress_current_field)());
            void decompress_joint_spatiotemporal_files(void(param_c<T_coord,T_data>::*decompress_current_field)());
            void compress_two_point_sets();
            template<time_dimension_mode_e label_mode,time_dimension_mode_e value_mode,label_value_execution_e execution>
            void compress_two_point_sets_fixed();
            template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
            void compress_point_sets();
            void restore_compression_result_to_original_order();
            void decompress_single_point_set();
            void decompress_two_point_sets();
            template<time_dimension_mode_e label_mode,time_dimension_mode_e value_mode,label_value_execution_e execution>
            void decompress_two_point_sets_fixed();
            template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
            void decompress_point_sets();
            
            bool solve_4x4(double A_in[4][4],double b_in[4],double x[4]);
            bool solve_linear_system(vector<vector<double>> &A,vector<double> &b,vector<double> &x);
    };
}

#include "distance.tpp"
#include "param.tpp"
#include "tree.tpp"
#include "init.tpp"
#include "adjust_coord_matrix_file.tpp"
#include "adjust_coord.tpp"
#include "prediction_round.tpp"
#include "partition.tpp"
#include "partition_adaptive_prediction.tpp"
#include "eb_schedule.tpp"
#include "generate_pred_refs_structure.tpp"
#include "precomputed_neighbor_pred_refs_structure.tpp"
#include "prediction_dag.tpp"
#include "point_set_config.tpp"
#include "point_set.tpp"
#include "time_prediction.tpp"
#include "prediction_framework.tpp"
#include "format.tpp"
#include "multi_huffman.tpp"
#include "prediction_structure.tpp"
#include "multihead_execution.tpp"
#include "two_point_sets_compress_entry.tpp"
#include "compress_entry.tpp"
#include "compress.tpp"
#include "decompress.tpp"
