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
#include <functional>
#include "../huffman.hpp"
#include "../third_party/zstd/lib/zstd.h"

using namespace std;

#include "simple_json.hpp"

namespace unsz
{
    enum class point_set_codec_e:uint8_t{idw=0,constant=2,rbf=5};
    enum class point_set_error_mode_e:uint8_t{relative=0,absolute=1};
    enum class point_set_rule_operator_e:uint8_t{less=0,less_equal=1,greater=2,greater_equal=3,equal=4,not_equal=5};
    enum class partition_fitting_method_e:uint8_t{direct=0,continuous_mix=1,idw_residual=2};
    enum class time_dimension_mode_e:uint8_t{not_applicable=0,spatial=1,constant_extrapolation=2,label_linear_interpolation_spatial_residual=5,linear_interpolation=6,joint_spatiotemporal=7,linear_interpolation_spatial_residual=8,constant_extrapolation_spatial_residual=9};
    enum class label_value_execution_e:uint8_t{no_label=0,staged=1,fused=2};
    enum class time_step_order_e:uint8_t{chronological=0,interpolation=1,anchor_stride_8=2};

    struct prediction_framework_c
    {
        bool has_label=false;
        time_dimension_mode_e label_mode=time_dimension_mode_e::not_applicable;
        time_dimension_mode_e value_mode=time_dimension_mode_e::spatial;
        label_value_execution_e execution=label_value_execution_e::no_label;
    };

    struct temporal_dependency_c
    {
        array<int32_t,2> source_indices{{-1,-1}};
        uint8_t source_num=0;
    };

    struct encoded_compression_payload_c
    {
        vector<unsigned char> bytes;
        uint64_t label_position_delta_byte_num=0,label_zstd_byte_num=0,value_huffman_byte_num=0,value_zstd_byte_num=0,unpredictable_value_byte_num=0;
    };

    struct compression_file_state_c
    {
        double point_set_classification_time=0,point_set_setup_time=0,label_prediction_time=0,value_prediction_time=0,lossless_encoding_time=0;
        uint32_t point_set_num=0;
        bool has_label=false;
        vector<encoded_compression_payload_c> encoded_payload_vec;
        uint64_t label_target_num=0,label_error_num=0,cross_time_prediction_num=0;
    };

    struct reconstruction_statistics_c
    {
        uint64_t point_num=0;
        double squared_error=0,minimum=numeric_limits<double>::infinity(),maximum=-numeric_limits<double>::infinity();
        double max_error_ratio=0,max_error=0,max_error_abs_eb=0;
        uint32_t max_error_point_id=0,max_error_point_set_id=0;
    };

    template<typename T_data>
    struct value_compression_result_c
    {
        uint64_t cross_time_prediction_num=0;
        vector<vector<int32_t>> point_set_qb_vec;
        vector<T_data> unpredictable_value_vec;
    };

    template<typename T_data>
    struct spatial_value_source_c
    {
        const vector<T_data>*data=nullptr;
        const vector<uint8_t>*valid_ref_mask=nullptr;
    };

    template<typename T_data>
    struct fused_spatial_prediction_c
    {
        uint32_t label=0;
        array<T_data,2> values{{0,0}};
    };

    template<typename T_data>
    struct decompression_file_state_c
    {
        double stream_decode_time=0,label_decode_time=0,value_decode_time=0;
        uint32_t point_set_num=0;
        bool has_label=false;
        uint64_t label_target_num=0;
        vector<vector<uint8_t>> structure_label_symbols;
        vector<vector<vector<int32_t>>> structure_point_set_qb_vec;
        vector<const unsigned char*> structure_unpredictable_value_begin_vec,structure_unpredictable_value_end_vec;
    };

    struct point_set_config_c
    {
        point_set_codec_e codec=point_set_codec_e::idw;
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

    template<typename T_calc>
    struct prediction_structure_c
    {
        struct reference_range_c
        {
            uint64_t begin;
            uint32_t num;
        };

        vector<vector<int>> target_id_groups;
        vector<int> target_ids,reference_ids;
        vector<uint64_t> reference_offsets;
        vector<T_calc> prediction_weights;
        vector<uint8_t> rbf_fallback_flags;
        vector<double> eb_multiplier;

        array_view_c<int> get_reference_ids(uint32_t prediction_id) const
        {
            uint64_t begin=reference_offsets[prediction_id];
            uint32_t reference_num=(uint32_t)(reference_offsets[prediction_id+1]-begin);
            if(reference_num==0)
                return {};
            return {reference_ids.data()+begin,reference_num};
        }

        bool uses_standard_reference_layout(uint32_t max_reference_num) const
        {
            uint32_t initial_target_num=min<uint32_t>(max_reference_num,target_ids.size());
            uint64_t initial_reference_num=initial_target_num<2?0:(uint64_t)initial_target_num*(initial_target_num-1)/2;
            return reference_offsets.size()==target_ids.size()+1&&reference_offsets[initial_target_num]==initial_reference_num&&reference_offsets.back()==initial_reference_num+(uint64_t)(target_ids.size()-initial_target_num)*max_reference_num;
        }

        reference_range_c get_reference_range(uint32_t prediction_index,uint32_t max_reference_num,bool standard_reference_layout) const
        {
            if(standard_reference_layout&&prediction_index>=max_reference_num)
            {
                uint64_t initial_reference_num=(uint64_t)max_reference_num*(max_reference_num-1)/2;
                return {initial_reference_num+(uint64_t)(prediction_index-max_reference_num)*max_reference_num,max_reference_num};
            }
            uint64_t begin=reference_offsets[prediction_index];
            return {begin,(uint32_t)(reference_offsets[prediction_index+1]-begin)};
        }

    };

    struct partition_prediction_config_c
    {
        uint32_t ref_num;
        double idw_power;
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
            static constexpr uint32_t small_capacity=11;//常用prediction_max_ref_num不超过10，反向查询额外包含目标自身
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
            vector<uint64_t> leaf_valid_mask;//按block保存叶节点中有效点的位图
            bool use_leaf_valid_mask=false;

            inline T_coord*get_block_min_coord(int block_id){return structure->block_coord_bounds.data()+(uint64_t)block_id*dim*2;}
            inline T_coord*get_block_max_coord(int block_id){return get_block_min_coord(block_id)+dim;}
            inline uint64_t get_block_num() const{return structure->block_vec.size();}
            void split_block(Array2D<T_coord> &coord_array,int block_id);
            void initialize(Array2D<T_coord> &coord_array,vector<T_coord> &min_coord,vector<T_coord> &max_coord,const vector<int>&original_ids,uint32_t leaf_point_num);
            block_tree_c(Array2D<T_coord> &coord_array,vector<T_coord> &min_coord,vector<T_coord> &max_coord,uint32_t leaf_point_num=1);
            block_tree_c(Array2D<T_coord> &coord_array,vector<T_coord> &min_coord,vector<T_coord> &max_coord,const vector<int>&original_ids,uint32_t leaf_point_num=1);
            block_tree_c(const block_tree_c<T_coord> &other)=default;
            void enable_leaf_valid_mask();
            void clear_all_valid_num();
            void set_point_valid(int original_id,bool value);
            void update_all_valid_num(int block_id);
            void update_point_valid_num_to_root(int original_id,int change_value);
            void find_original_nearest(nearest_queue_c<T_coord> &nearest_queue,Array2D<T_coord> &coord_array,
                uint32_t nearest_num,int block_id,T_coord *query_coord,bool valid_only=false);
                //寻找最近的点。如果query_coord是一个点的坐标，那么会找到自身
    };

    template<typename T_coord,typename T_data>
    class param_c
    {
        public:
            using T_calc=common_type_t<T_coord,T_data>;

            param_c()=default;

            string data_dir,field_name,error_mode,coord_path,time_step_coordinate_path,dag_cache_path,adjust_coord_matrix_file,adjust_coord_training_type,point_set_config_path;
            vector<string> original_data_path_vec,compressed_data_path_vec,decompressed_data_path_vec;
            uint32_t dim=3;
            uint32_t point_num=0;
            T_data rel_eb=0,abs_eb=0;
            bool got_rel_eb=false,got_abs_eb=false;
            bool got_data_dir=false,got_field=false,got_time_step=false,got_error_mode=false,got_predictor=false,got_prediction_round_num=false;
            vector<point_set_rule_c> point_set_rule_vec;
            vector<point_set_config_c> point_set_config_vec;
            string value_flow="none";
            string label_flow="none";
            bool if_adjust_coord=false;
            vector<double> adjust_coord_matrix_parameter_vec;
            bool print_compression_stage_time=false,spatial_references_use_prediction_order=false;
            string label_value_execution="staged";//staged / fused
            string ordering_scope="local";//local / global
            uint32_t ordering_score_min_ref_num=0;//必须显式设置；缓存中的有效评分参考点少于该值时重新查询
            uint32_t ordering_candidate_num=0;//必须显式设置
            uint32_t ordering_score_max_ref_num=0;//必须显式设置；只控制反向MSD排序评分的参考点数量
            uint32_t kd_tree_leaf_size=32;
            uint32_t partition_size=0;//0表示不分区；非0时所有参考点必须来自同一partition
            uint32_t partition_num=0;//直接指定2、4、8等2的幂个partition
            uint32_t thread_num=1;
            bool use_partition_adaptive_predictor_experimental=false;
            uint32_t partition_adaptive_sample_num_experimental=256;
            double partition_adaptive_min_improvement_experimental=0;
            string partition_fitting_candidates_experimental="original";//original / weighted / ridge / mixed / residual / extended / all
            string eb_schedule="round_point_ratio";//none / round / point / inverse_rank_log2 / round_point / round_ratio / round_point_ratio
            double eb_schedule_start_ratio=0.8;
            double eb_schedule_log2_shift=0;//inverse_rank_log2的指数位移
            double eb_schedule_max_log2_exponent=7;//inverse_rank_log2的最小EB倍率由该指数限制，不依赖预测轮数
            uint32_t prediction_round_num=8;
            double round_size_growth_ratio_experimental=2;
            uint32_t prediction_min_ref_num=1;//同点集参考点少于该值时使用fallback
            uint32_t prediction_max_ref_num=4;//DAG保存的参考点数，也是主预测器最多使用的同点集参考点数
            double idw_power=2;//欧氏距离的负指数；代码中的dist是平方距离，因此实际使用-power/2
            string rbf_kernel="gaussian";//gaussian / exponential / matern32 / matern52 / inverse_multiquadric
            double rbf_shape=1;
            double rbf_nugget=1e-8;
            bool rbf_parameter_specified=false;
            int zstd_compression_level=3;
            string predictor;

            Array2D<T_coord> coord_array;
            vector<T_coord> min_coord,max_coord;
            vector<prediction_structure_c<T_calc>> prediction_structure_vec;//未切分时只有一个结构；切分后每个结构独立完整
            vector<vector<int>> partition_point_ids_vec;
            vector<uint32_t> point_partition_id_vec;
            vector<uint8_t> partition_prediction_config_id_vec,partition_adaptive_sample_mask;

            vector<T_data> current_data,current_decompressed_data,prediction_order_decompressed_data;
            vector<uint32_t> time_step_vec;
            vector<T_coord> time_step_coordinate_vec;
            vector<vector<T_data>> reconstructed_time_data_vec;
            vector<vector<uint8_t>> reconstructed_time_point_set_id_vec;
            vector<temporal_dependency_c> temporal_value_dependency_vec,temporal_label_dependency_vec;
            vector<uint32_t> temporal_value_remaining_outdegree_vec,temporal_label_remaining_outdegree_vec;
            const vector<T_data>*previous_time_value_source=nullptr;
            const vector<uint8_t>*previous_time_value_point_set_source=nullptr,*previous_time_label_source=nullptr;
            vector<T_data> current_time_spatial_residual_data;
            vector<uint8_t> current_time_spatial_residual_valid;
            uint32_t current_time_step=0,current_time_step_index=0;
            prediction_framework_c current_prediction_framework;
            array<T_calc,2> temporal_weights{{0,0}};
            bool previous_time_value_available=false,previous_time_label_available=false,temporal_value_interpolation_available=false,temporal_label_interpolation_available=false;
            vector<unsigned char> current_compressed_bytes;
            vector<uint8_t> point_set_id_vec;
            vector<uint32_t> point_set_point_num_vec;
            vector<T_data> point_set_min_data_vec,point_set_max_data_vec,point_set_abs_eb_vec;
            vector<point_set_codec_e> point_set_codec_vec;
            bool suppress_compression_summary=false;
            double last_total_psnr=0,last_total_compression_time=0;
            uint64_t last_total_original_bytes=0;

            void init();
            void load_time_step_coordinates();
            time_step_order_e get_time_step_order(const string &flow) const;
            void prepare_time_step_order(time_step_order_e order);
            void configure_point_sets();
            void prepare_cli_parameters(const vector<uint32_t>&selected_time_step_vec,T_data error_bound);
            void prepare_temporal_dependencies();
            void prepare_temporal_prediction();
            template<time_dimension_mode_e value_mode>
            bool predict_temporal_value(uint32_t original_id,uint32_t point_set_id,T_data &prediction);
            template<time_dimension_mode_e value_mode>
            void store_temporal_residual(int pred_id,bool used_temporal_prediction,T_data temporal_prediction);
            template<time_dimension_mode_e label_mode>
            bool predict_temporal_label(uint32_t original_id,uint32_t label_channel_id,uint32_t &prediction);
            void update_time_prediction_state();
            prediction_framework_c get_prediction_framework(uint32_t point_set_num) const;
            bool uses_anchor_interpolation_stride_8_flow() const;
            bool current_time_step_is_regular_anchor() const;
            void resolve_single_point_set_time_mode();
            template<point_set_codec_e predictor>
            void dispatch_single_point_set_compression();
            void load_adjust_coord_matrix_file();
            void adjust_coord();
            void load_or_generate_prediction_structures();
            void remap_single_point_set_spatial_references_to_prediction_order();
            void build_partitions();
            vector<uint32_t> get_round_target_num_vec(uint32_t local_point_num) const;
            void generate_partition_prediction_structures(bool uses_idw,bool uses_rbf);
            vector<partition_prediction_config_c> get_partition_prediction_config_vec();
            vector<double> generate_pred_eb_multiplier_vec(const vector<vector<int>>&target_id_groups,uint32_t target_num,bool print_statistics=false);
            void select_partition_prediction_configs();
            void prepare_partition_adaptive_sample_mask();
            void write_normalized_idw_prediction_weights(int pred_id,array_view_c<int> ref_ids,T_calc *weights,array_view_c<T_coord> target_ref_squared_distances={});
            void append_normalized_idw_prediction_weights(int pred_id,array_view_c<int> ref_ids,vector<T_calc> &weights,array_view_c<T_coord> target_ref_squared_distances={});
            bool should_use_selected_adaptive_prediction(int pred_id) const;
            T_data predict_selected_adaptive_value(int pred_id,array_view_c<int> ref_ids,vector<T_data>&data);
            T_data predict_partition_adaptive_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,uint8_t config_id);
            T_data predict_partition_adaptive_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,const partition_prediction_config_c &config);
            T_data predict_partition_idw_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double power);
            bool predict_partition_idw_value_and_loo_error(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double power,T_data &prediction,double &loo_error);
            bool predict_partition_linear_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction,double weight_power=0,double ridge=0,bool normalize_coord=false);
            bool predict_partition_linear_fitting_value_and_loo_error(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction,double &loo_error,double weight_power=0,double ridge=0,bool normalize_coord=true);
            bool predict_partition_continuous_mix_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double idw_power,double fitting_weight_power,T_data &prediction);
            bool predict_partition_idw_residual_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double idw_power,double residual_weight_power,T_data &prediction);
            void generate_prediction_structures_and_weights();
            void generate_prediction_structure_and_weights(block_tree_c<T_coord>&tree,const vector<int>&point_ids,const vector<T_coord>&active_min_coord,const vector<T_coord>&active_max_coord,prediction_structure_c<T_calc>&structure,chrono::steady_clock::time_point generation_start_time,bool uses_idw,bool uses_rbf);
            void generate_global_prediction_structure_and_weights(block_tree_c<T_coord>&tree,const vector<int>&point_ids,const vector<T_coord>&active_min_coord,const vector<T_coord>&active_max_coord,prediction_structure_c<T_calc>&structure,chrono::steady_clock::time_point generation_start_time,bool uses_idw,bool uses_rbf);
            void generate_local_prediction_structure_and_weights(block_tree_c<T_coord>&tree,const vector<int>&point_ids,const vector<T_coord>&active_min_coord,const vector<T_coord>&active_max_coord,prediction_structure_c<T_calc>&structure,chrono::steady_clock::time_point generation_start_time,bool uses_idw,bool uses_rbf);
            template<bool global_order>
            void sort_targets_by_reverse_msd(block_tree_c<T_coord>&target_tree,block_tree_c<T_coord>&reference_tree,const vector<T_coord>&active_min_coord,const vector<T_coord>&active_max_coord,prediction_structure_c<T_calc>&structure,bool uses_idw,bool uses_rbf,vector<T_coord>&prediction_ref_coords,vector<T_coord>&prediction_ref_squared_distances);
            void validate_prediction_structures() const;
            void generate_rbf_weights_from_loaded_dag();
            void save_prediction_structure_cache();
            void load_prediction_structure_cache();
            T_data predict_value_one_point_set(uint64_t ref_begin,uint32_t ref_num,const vector<int>&ref_ids,const vector<T_calc>&weights,const vector<T_data>&data);
            T_data predict_value_one_point_set_4(uint64_t ref_begin,const vector<int>&ref_ids,const vector<T_calc>&weights,const vector<T_data>&data);
            T_data predict_value_one_point_set_10(uint64_t ref_begin,const vector<int>&ref_ids,const vector<T_calc>&weights,const vector<T_data>&data);
            uint32_t predict_label(uint64_t ref_begin,uint32_t ref_num,const vector<int>&ref_ids,const vector<T_calc>&weights,const vector<uint8_t>&label_vec);
            uint32_t predict_label_4(uint64_t ref_begin,const vector<int>&ref_ids,const vector<T_calc>&weights,const vector<uint8_t>&label_vec);
            uint32_t predict_label_10(uint64_t ref_begin,const vector<int>&ref_ids,const vector<T_calc>&weights,const vector<uint8_t>&label_vec);
            T_data predict_value_two_point_sets_idw(array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,const vector<T_data> &data,uint32_t value_point_set_id,const vector<uint8_t> *valid_ref_mask=nullptr);
            T_data predict_value_two_point_sets_rbf(bool has_rbf_weights,array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,const vector<T_data> &data,uint32_t value_point_set_id,const vector<uint8_t> *valid_ref_mask=nullptr);
            fused_spatial_prediction_c<T_data> predict_fused_spatial_idw(array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,bool predict_label,const array<spatial_value_source_c<T_data>,2>&spatial_value_sources);
            fused_spatial_prediction_c<T_data> predict_fused_spatial_rbf(bool has_rbf_weights,array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,bool predict_label,const array<spatial_value_source_c<T_data>,2>&spatial_value_sources);
            bool solve_scalar_rbf_weights(int pred_id,array_view_c<int> ref_ids,vector<T_calc> &weights,array_view_c<T_coord> target_ref_squared_distances={});
            void generate_batched_rbf_weights(const vector<T_coord>&reference_coords,const vector<T_coord>&target_ref_squared_distances,const vector<int>&target_ids,const vector<int>&ref_ids,const vector<uint64_t>&ref_begin,vector<T_calc>&weights,vector<uint8_t>&rbf_fallback);
            void preprocess_compression(chrono::steady_clock::time_point preprocessing_start_time);
            void prepare_joint_spatiotemporal_coordinates();
            reconstruction_statistics_c calculate_reconstruction_statistics(const T_data*original,const T_data*reconstructed,uint32_t value_num,uint32_t point_offset=0) const;
            void compress_files(void(param_c<T_coord,T_data>::*compress_file)(),bool write_output=true,const function<void(uint32_t,const vector<unsigned char>&)>&compressed_stream_consumer={},bool validate_reconstruction=true);
            template<point_set_codec_e predictor>
            void compress_staged_two_point_sets_files();
            template<point_set_codec_e predictor>
            void compress_joint_spatiotemporal_files();
            template<point_set_codec_e predictor>
            void decompress_joint_spatiotemporal_files();
            template<point_set_codec_e predictor>
            void dispatch_staged_label_compression();
            template<point_set_codec_e predictor>
            void dispatch_staged_value_compression();
            template<point_set_codec_e predictor>
            void dispatch_fused_compression();
            template<bool two_point_sets>
            void classify_current_point_sets(compression_file_state_c&state);
            void prepare_current_point_set_codecs_and_eb(compression_file_state_c&state);
            template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution>
            void prepare_current_prediction_state(compression_file_state_c&state);
            template<bool two_point_sets,label_value_execution_e fixed_execution>
            void initialize_current_compression_buffers(compression_file_state_c&state);
            template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution>
            void prepare_current_compression(compression_file_state_c&state);
            template<time_dimension_mode_e fixed_label_mode,point_set_codec_e fixed_predictor>
            void compress_labels(compression_file_state_c&state);
            template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,point_set_codec_e fixed_predictor>
            value_compression_result_c<T_data> compress_structure_values(const prediction_structure_c<T_calc>&prediction_structure);
            template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,point_set_codec_e fixed_predictor>
            void compress_values(compression_file_state_c&state);
            template<time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,point_set_codec_e fixed_predictor>
            void compress_labels_and_values(compression_file_state_c&state);
            encoded_compression_payload_c encode_compression_payload(vector<uint8_t>&label_symbols,bool has_label,vector<vector<int32_t>>&point_set_qb_vec,vector<T_data>&unpredictable_value_vec);
            void assemble_current_compression_file(compression_file_state_c&state);
            template<bool two_point_sets>
            void decode_current_compression_streams(decompression_file_state_c<T_data>&state);
            void dispatch_single_point_set_decompression();
            void dispatch_two_point_sets_decompression_by_mode();
            void validate_decoded_point_set_counts() const;
            template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
            void decompress_current_file();
            template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
            void decompress_values(decompression_file_state_c<T_data>&state);
            template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode,label_value_execution_e fixed_execution,point_set_codec_e fixed_predictor>
            void decompress_labels_and_values(decompression_file_state_c<T_data>&state);
            
            template<typename T_system>
            bool solve_linear_system(vector<vector<T_system>> &A,vector<T_system> &b,vector<T_system> &x);
    };
}

#include "distance.tpp"
#include "binary_file.tpp"
#include "tree.tpp"
#include "init.tpp"
#include "adjust_coord_matrix_file.tpp"
#include "adjust_coord.tpp"
#include "prediction_round.tpp"
#include "partition.tpp"
#include "partition_adaptive_prediction.tpp"
#include "eb_schedule.tpp"
#include "generate_prediction_structure_and_weights.tpp"
#include "point_set_config.tpp"
#include "cli_parameter.tpp"
#include "spatial_prediction.tpp"
#include "temporal_prediction.tpp"
#include "prediction_framework.tpp"
#include "compression_format.tpp"
#include "reconstruction_statistics.tpp"
#include "prediction_weights.tpp"
#include "prediction_structure_cache.tpp"
#include "compression_flows.tpp"
#include "decompression_flows.tpp"
#include "one_point_set_compress_entry.tpp"
#include "two_point_sets_compress_entry.tpp"
#include "staged_two_point_sets_entry.tpp"
#include "joint_spatiotemporal_entry.tpp"
#include "compress_entry.tpp"
#include "compress.tpp"
#include "decompress.tpp"
