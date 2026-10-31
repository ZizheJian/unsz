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
#include <functional>
#include <memory>
#include <mutex>
#include <exception>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include "../huffman.hpp"
#include "../third_party/zstd/lib/zstd.h"

using namespace std;

#include "simple_json.hpp"

// 代码术语与论文术语对照：
// point set（点集）：按数值规则把点分成两类，点集0是普通值，点集1是异常值（如-99999）。只有一类时称单点集。
// label（标签）：每个点属于哪个点集。label stream即论文中的异常值标签流。
// time step / time / temporal：slice和跨slice预测。名字沿用最早按时间步组织数据的写法，二维tensor（如月份×层）的两个轴也用这套名字。
// flow：跨slice预测方案，value_flow用于数值，label_flow用于标签。prediction framework/mode是某个slice按flow解析出的具体预测方式。
// tensor flow：二维tensor的跨slice预测方案，指定每个轴的预测方式和两个轴的处理顺序。
// anchor8（anchor_interpolation_stride_8）：每8个slice取一个锚点，锚点之间按间距4、2、1分层插值。
// spatial residual：跨slice预测之后，再对邻近点上的跨slice预测残差做空间预测，加到跨slice预测上。
// prediction structure：一个partition内的DAG，包括预测顺序、参考点和RBF权重。不分区时只有一个。
// qb：量化bin编号。unpredictable value：量化后仍超出误差上限而原样保存的值。
// MSD：mean squared distance，见generate_prediction_structure_and_weights.tpp。
namespace unsz
{
    inline double seconds_since(chrono::steady_clock::time_point start){return chrono::duration<double>(chrono::steady_clock::now()-start).count();}

    // 用OpenMP并行执行function(0)..function(task_num-1)。异常不能抛出OpenMP并行区，因此记录第一个异常，全部任务结束后再抛出。
    template<typename Function>
    void parallel_for_tasks(uint32_t task_num,uint32_t thread_num,Function function)
    {
        exception_ptr first_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(task_num>1&&thread_num>1)
        for(int64_t task_id=0;task_id<(int64_t)task_num;task_id++)
        {
            try
            {
                function((uint32_t)task_id);
            }
            catch(...)
            {
#pragma omp critical
                if(!first_exception)
                    first_exception=current_exception();
            }
        }
        if(first_exception)
            rethrow_exception(first_exception);
    }

    template<typename T>
    class shared_vector_c
    {
        private:
            shared_ptr<vector<T>>values=make_shared<vector<T>>();

        public:
            using iterator=typename vector<T>::iterator;
            using const_iterator=typename vector<T>::const_iterator;

            void reset(){values=make_shared<vector<T>>();}
            void clear(){values->clear();}
            void resize(uint64_t size){values->resize(size);}
            void assign(uint64_t size,const T&value){values->assign(size,value);}
            void reserve(uint64_t size){values->reserve(size);}
            void push_back(const T&value){values->push_back(value);}
            bool empty() const{return values->empty();}
            uint64_t size() const{return values->size();}
            T&front(){return values->front();}
            const T&front() const{return values->front();}
            T&back(){return values->back();}
            const T&back() const{return values->back();}
            T&operator[](uint64_t index){return (*values)[index];}
            const T&operator[](uint64_t index) const{return (*values)[index];}
            iterator begin(){return values->begin();}
            const_iterator begin() const{return values->begin();}
            iterator end(){return values->end();}
            const_iterator end() const{return values->end();}
    };

    enum class point_set_codec_e:uint8_t{constant=2,rbf=5};
    enum class point_set_error_mode_e:uint8_t{relative=0,absolute=1,field_relative=2};
    enum class point_set_rule_operator_e:uint8_t{less=0,less_equal=1,greater=2,greater_equal=3,equal=4,not_equal=5};
    // label_linear_interpolation_spatial_residual：对标签做两侧slice的线性插值，两侧标签一致时插值结果就是该标签。
    // 不一致时插值得到0.5，无法判定，改用空间预测补充。流程与数值的“插值＋空间残差”相似，因此沿用这个名字。
    enum class time_dimension_mode_e:uint8_t{not_applicable=0,spatial=1,constant_extrapolation=2,label_linear_interpolation_spatial_residual=5,linear_interpolation=6,linear_interpolation_spatial_residual=8,constant_extrapolation_spatial_residual=9};
    enum class time_step_order_e:uint8_t{chronological=0,interpolation=1,anchor_stride_8=2};
    enum class tensor_axis_mode_e:uint8_t{none=0,constant_extrapolation=1,anchor_interpolation_stride_8=2};
    enum class tensor_schedule_e:uint8_t{separable=0,interleaved_anchor=1};

    struct tensor_flow_c
    {
        array<tensor_axis_mode_e,2> axis_modes{{tensor_axis_mode_e::none,tensor_axis_mode_e::none}};
        array<uint8_t,2> axis_order{{0,1}};
        array<uint8_t,2> interpolation_axis_order{{0,1}};
        tensor_schedule_e schedule=tensor_schedule_e::separable;
        bool use_spatial_residual=false;
    };

    struct tensor_field_config_c
    {
        array<uint32_t,2> shape{{0,0}};
        array<vector<double>,2> axis_coordinates;
        string dependency_plan_key;
        string active_level_count_path;
        uint32_t active_level_count_offset=0;
    };

    struct active_level_count_data_c
    {
        vector<uint32_t> counts;
        uint32_t maximum=0;
    };

    struct prediction_framework_c
    {
        bool has_label=false;
        time_dimension_mode_e label_mode=time_dimension_mode_e::not_applicable;
        time_dimension_mode_e value_mode=time_dimension_mode_e::spatial;
    };

    struct temporal_dependency_c
    {
        array<int32_t,2> source_indices{{-1,-1}};
        array<double,2> source_weights{{0,0}};
        uint8_t source_num=0;
    };

    struct temporal_dependency_plan_c
    {
        vector<uint32_t> processing_order;
        vector<temporal_dependency_c> dependencies;
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

    struct compression_diagnostics_c
    {
        double point_set_classification_time=0,point_set_setup_time=0,label_prediction_time=0,value_prediction_time=0,lossless_encoding_time=0;
        uint64_t label_target_num=0,label_error_num=0,cross_time_prediction_num=0,label_position_delta_byte_num=0,label_zstd_byte_num=0,value_huffman_byte_num=0,value_zstd_byte_num=0,unpredictable_value_byte_num=0;
    };

    struct decompression_diagnostics_c
    {
        double stream_decode_time=0,label_decode_time=0,value_decode_time=0;
    };

    struct reconstruction_statistics_c
    {
        uint64_t point_num=0;
        double squared_error=0,minimum=numeric_limits<double>::infinity(),maximum=-numeric_limits<double>::infinity();
        // valid统计只包含配置中第0个点集的点，即排除-99999等异常值后的普通值
        uint64_t valid_point_num=0;
        double valid_squared_error=0,valid_minimum=numeric_limits<double>::infinity(),valid_maximum=-numeric_limits<double>::infinity();
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
        point_set_codec_e codec=point_set_codec_e::rbf;
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

        // target_ids按预测顺序列出目标点编号。第i个目标的参考点是reference_ids[reference_offsets[i]..reference_offsets[i+1])，按距离从近到远。
        // remap_prediction_data_to_prediction_order之后，reference_ids原地改存参考点的预测顺序下标而不是点编号，重建值也按预测顺序存放。
        // 原地复写省掉了第二个参考数组，get_storage_id据此给出重建值的存放位置。
        vector<int> target_ids,reference_ids;
        vector<uint64_t> reference_offsets;
        vector<T_calc> prediction_weights;
        vector<uint8_t> rbf_fallback_flags;
        vector<T_calc> eb_multiplier,inverse_eb_multiplier;
        uint32_t prediction_order_offset=0;

        inline uint32_t get_storage_id(uint32_t prediction_index,int target_id,bool uses_prediction_order) const
        {
            return uses_prediction_order?prediction_order_offset+prediction_index:(uint32_t)target_id;
        }

        // 标准参考布局：global DAG中预测序号为k（从0开始）的目标前面只有k个点，k<10时只有k个参考点，前10个目标共有0+1+...+9=45个参考点。
        // k>=10的目标恰好有10个参考点，从45+(k-10)*10开始，不必读reference_offsets。
        // 这个函数检查当前结构是否满足该布局（local排序或点数很少的partition可能不满足），满足时单点集逐点循环的第二段直接按上式递推参考区间，并固定使用10点预测。
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

    template<typename T_calc>
    struct spatial_prediction_state_c
    {
        shared_vector_c<prediction_structure_c<T_calc>>structures;
        shared_vector_c<int>original_to_prediction_order;
        bool uses_prediction_order=false;
    };

    template<typename T_calc>
    struct shared_spatial_prediction_cache_c
    {
        once_flag initialization_flag;
        shared_ptr<const spatial_prediction_state_c<T_calc>>state;
    };

    template<typename T_coord>
    class nearest_queue_c
    {
        private:
            static constexpr uint32_t small_capacity=11;//固定10个RBF参考点，反向查询额外包含目标自身
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
            static constexpr uint32_t rbf_reference_num=10;
            static constexpr double rbf_shape=0.01;
            static constexpr double rbf_nugget=1e-6;

            param_c()=default;

            string data_dir,output_dir,field_name,error_mode,coord_path,time_step_coordinate_path,point_set_config_path,tensor_metadata_path;
            vector<string> field_name_vec;
            vector<string> original_data_path_vec;
            string generated_path(const string&input_path,const string&suffix) const{return output_dir+filesystem::path(input_path).filename().string()+suffix;}
            string field_generated_path(const string&suffix) const{return output_dir+field_name+suffix;}
            map<string,tensor_field_config_c> tensor_field_config_map;
            bool current_field_uses_tensor=false;
            array<uint32_t,2> current_tensor_shape{{0,0}};
            array<vector<double>,2> current_tensor_axis_coordinates;
            string current_tensor_dependency_plan_key;
            string current_active_level_count_path;
            uint32_t current_active_level_count_offset=0;
            map<string,active_level_count_data_c> active_level_count_cache;
            const vector<uint32_t>*current_active_level_counts=nullptr;
            uint32_t dim=3;
            uint32_t point_num=0;
            T_data rel_eb=0,abs_eb=0;
            bool got_rel_eb=false,got_abs_eb=false;
            bool got_data_dir=false,got_field=false,got_time_step=false,got_error_mode=false;
            vector<point_set_rule_c> point_set_rule_vec;
            vector<point_set_config_c> point_set_config_vec;
            vector<point_set_rule_c> configured_point_set_rule_vec;
            vector<point_set_config_c> configured_point_set_config_vec;
            vector<uint8_t> point_set_source_id_vec;
            string value_flow="auto";
            string label_flow="auto";
            string configured_value_flow="auto",configured_label_flow="auto";
            uint32_t automatic_flow_sample_num=2048,automatic_flow_start_num=4;
            vector<array<T_data,2>> compression_file_point_set_min_data_vec,compression_file_point_set_max_data_vec,compression_file_point_set_abs_eb_vec;
            vector<array<point_set_codec_e,2>> compression_file_point_set_codec_vec;
            array<double,2> field_point_set_min_data{{0,0}},field_point_set_max_data{{0,0}};//FIELD_REL按当前field全部slice计算的各点集range
            string coordinate_transform_mode="auto";//off / on / auto；off和on仅用于实验
            double coordinate_transform_first_sample_ratio=0.3;
            bool use_coordinate_transform=false;
            vector<double> coordinate_transform_diagonal;
            vector<double> original_coordinate_covariance;
            double original_coordinate_flatness=0;
            vector<double> coordinate_transform_training_coordinate_differences,coordinate_transform_training_value_differences;
            vector<uint32_t> coordinate_transform_training_pair_group_ids;
            uint32_t coordinate_transform_training_group_num=0;
            double coordinate_transform_training_input_time=0,point_set_statistics_input_time=0;
            vector<uint32_t> coordinate_transform_sample_target_offsets;
            vector<int> coordinate_transform_sample_target_ids,coordinate_transform_sample_reference_ids;
            bool print_compression_stage_time=false,prediction_data_uses_prediction_order=false,coordinates_are_original=true,write_decompressed_output=true;
            string ordering_scope="global";//正式压缩固定使用global
            uint32_t ordering_score_min_ref_num=1;
            uint32_t ordering_candidate_num=16;
            uint32_t ordering_score_max_ref_num=1;
            uint32_t kd_tree_leaf_size=32;
            uint32_t partition_num=0;//0表示不分区；否则为2、4、8等2的幂，所有参考点必须来自同一partition
            uint32_t thread_num=1;
            uint32_t field_worker_num=1;
            vector<uint32_t>field_worker_id_vec;
            string eb_schedule="inverse_rank_log2";//none / inverse_rank_log2
            double eb_schedule_start_ratio=0.8;
            double eb_schedule_log2_shift=0.5;//inverse_rank_log2的指数位移
            double eb_schedule_max_log2_exponent=7;//inverse_rank_log2的最小EB倍率由该指数限制，不依赖预测轮数
            uint32_t prediction_round_num=8;
            double round_size_growth_ratio_experimental=2;
            int zstd_compression_level=3;

            Array2D<T_coord> coord_array,original_coord_array;
            vector<T_coord> min_coord,max_coord,original_min_coord,original_max_coord;
            shared_vector_c<prediction_structure_c<T_calc>>prediction_structure_vec;//未切分时只有一个结构；切分后每个结构独立完整
            shared_ptr<shared_spatial_prediction_cache_c<T_calc>>shared_spatial_prediction_cache=make_shared<shared_spatial_prediction_cache_c<T_calc>>();
            vector<T_data> current_data,current_decompressed_data,prediction_order_decompressed_data;
            shared_vector_c<int>original_to_prediction_order;
            vector<uint32_t> time_step_vec;
            vector<uint32_t> configured_time_step_vec;
            vector<T_coord> time_step_coordinate_vec;
            vector<vector<T_data>> reconstructed_time_data_vec;
            vector<vector<uint8_t>> reconstructed_time_point_set_id_vec;
            vector<temporal_dependency_c> temporal_value_dependency_vec,temporal_label_dependency_vec;
            map<string,temporal_dependency_plan_c> temporal_dependency_plan_cache;
            map<string,map<uint8_t,temporal_dependency_plan_c>> tensor_dependency_plan_cache;
            tensor_flow_c selected_tensor_flow;
            bool tensor_flow_selected=false;
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
            vector<uint8_t> point_set_id_vec,prediction_order_point_set_id_vec;
            vector<uint32_t> point_set_point_num_vec;
            vector<T_data> point_set_min_data_vec,point_set_max_data_vec,point_set_abs_eb_vec;
            vector<point_set_codec_e> point_set_codec_vec;
            uint64_t current_label_stream_bytes=0,current_value_stream_bytes=0,current_shared_metadata_bytes=0;
            compression_diagnostics_c current_compression_diagnostics;
            decompression_diagnostics_c current_decompression_diagnostics;
            double reported_compression_preprocessing_time=0,reported_compression_time=0,reported_decompression_preprocessing_time=0,reported_decompression_time=0;
            double current_field_preprocessing_time=0;

            void init();
            void load_time_step_coordinates();
            time_step_order_e get_time_step_order(const string &flow) const;
            vector<uint32_t> generate_time_step_processing_order(time_step_order_e order) const;
            void prepare_time_step_order(time_step_order_e order);
            void prepare_tensor_slice_order();
            void prepare_time_step_order_from_flows();
            void configure_point_sets();
            uint8_t classify_point_set(T_data value,uint32_t point_id,uint32_t slice_id) const;
            T_data point_set_abs_eb(uint32_t point_set_id,bool field_range_known) const;
            void load_tensor_metadata();
            void activate_tensor_field(const tensor_field_config_c&config);
            void load_current_active_level_counts();
            void prepare_cli_parameters(const vector<uint32_t>&selected_time_step_vec,T_data error_bound);
            void activate_field(uint32_t field_id);
            void precompute_all_tensor_dependency_plans();
            void precompute_temporal_dependency_plans();
            void prepare_temporal_dependencies();
            temporal_dependency_plan_c generate_temporal_dependency_plan(const string&flow) const;
            const temporal_dependency_plan_c&get_temporal_dependency_plan(const string&flow) const;
            const temporal_dependency_plan_c&get_tensor_dependency_plan(const tensor_flow_c&flow) const;
            void prepare_temporal_prediction();
            template<time_dimension_mode_e value_mode>
            bool predict_temporal_value(uint32_t storage_id,uint32_t point_set_id,T_data &prediction);
            template<time_dimension_mode_e value_mode>
            void store_temporal_residual(uint32_t storage_id,bool used_temporal_prediction,T_data temporal_prediction);
            template<time_dimension_mode_e value_mode>
            auto get_single_point_set_cross_slice_predictor() const;
            template<time_dimension_mode_e label_mode>
            bool predict_temporal_label(uint32_t storage_id,uint32_t label_channel_id,uint32_t &prediction);
            void update_time_prediction_state();
            prediction_framework_c get_prediction_framework(uint32_t point_set_num) const;
            bool uses_anchor_interpolation_stride_8_flow() const;
            bool current_time_step_is_regular_anchor() const;
            void resolve_single_point_set_time_mode();
            void check_label_value_time_step_order() const;
            bool coordinate_transform_value_is_trainable(T_data value) const;
            void prepare_coordinate_transform_sample_geometry(uint32_t sample_file_num);
            void prepare_coordinate_transform_training_data();
            void train_coordinate_transform();
            void restore_original_coordinates();
            void apply_coordinate_transform();
            void write_coordinate_transform_file() const;
            void read_coordinate_transform_file();
            void remap_prediction_data_to_prediction_order(bool for_decompression=false);
            void finalize_current_spatial_prediction_state(bool for_decompression);
            void prepare_current_spatial_prediction_state(bool for_decompression);
            bool activate_shared_spatial_prediction_state();
            void release_current_spatial_prediction_state();
            vector<vector<int>> build_partitions();
            vector<uint32_t> get_round_target_num_vec(uint32_t local_point_num) const;
            void generate_partition_prediction_structures(bool uses_rbf);
            vector<T_calc> generate_pred_eb_multiplier_vec(uint32_t target_num,bool print_statistics=false);
            void generate_prediction_structures_and_weights();
            void generate_prediction_structure_and_weights(block_tree_c<T_coord>&tree,const vector<int>&point_ids,const vector<T_coord>&active_min_coord,const vector<T_coord>&active_max_coord,prediction_structure_c<T_calc>&structure,chrono::steady_clock::time_point generation_start_time,bool uses_rbf);
            void generate_global_prediction_structure_and_weights(block_tree_c<T_coord>&tree,const vector<int>&point_ids,const vector<T_coord>&active_min_coord,const vector<T_coord>&active_max_coord,prediction_structure_c<T_calc>&structure,chrono::steady_clock::time_point generation_start_time,bool uses_rbf);
            void generate_local_prediction_structure_and_weights(block_tree_c<T_coord>&tree,const vector<int>&point_ids,const vector<T_coord>&active_min_coord,const vector<T_coord>&active_max_coord,prediction_structure_c<T_calc>&structure,chrono::steady_clock::time_point generation_start_time,bool uses_rbf);
            template<bool global_order>
            void sort_targets_by_reverse_msd(block_tree_c<T_coord>&target_tree,block_tree_c<T_coord>&reference_tree,const vector<T_coord>&active_min_coord,const vector<T_coord>&active_max_coord,vector<vector<int>>&target_id_groups,prediction_structure_c<T_calc>&structure);
            T_data predict_value_one_point_set(uint64_t ref_begin,uint32_t ref_num,const vector<int>&ref_ids,const vector<T_calc>&weights,const vector<T_data>&data);
            T_data predict_value_one_point_set_10(uint64_t ref_begin,const vector<int>&ref_ids,const vector<T_calc>&weights,const vector<T_data>&data);
            uint32_t predict_label(uint64_t ref_begin,uint32_t ref_num,const vector<int>&ref_ids,const vector<T_calc>&weights,const vector<uint8_t>&label_vec);
            uint32_t predict_label_10(uint64_t ref_begin,const vector<int>&ref_ids,const vector<T_calc>&weights,const vector<uint8_t>&label_vec);
            T_data predict_value_two_point_sets_rbf(bool has_rbf_weights,array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,const vector<T_data> &data,uint32_t value_point_set_id,const vector<uint8_t> *valid_ref_mask=nullptr);
            fused_spatial_prediction_c<T_data> predict_fused_spatial_rbf(bool has_rbf_weights,array_view_c<int> ref_ids,array_view_c<T_calc> ref_weights,bool predict_label,const array<spatial_value_source_c<T_data>,2>&spatial_value_sources);
            void generate_batched_rbf_weights(const vector<int>&target_ids,const vector<int>&ref_ids,const vector<uint64_t>&ref_begin,vector<T_calc>&weights,vector<uint8_t>&rbf_fallback);
            pair<double,double> preprocess_current_compression_geometry();
            pair<double,double> preprocess_current_compression_field();
            double select_automatic_flows();
            void validate_automatic_flow_selection() const;
            void precompute_current_field_point_set_statistics();
            double read_compressed_field_metadata();
            uint32_t valid_point_set_id() const;
            reconstruction_statistics_c calculate_reconstruction_statistics(const T_data*original,const T_data*reconstructed,uint32_t value_num,uint32_t point_offset=0) const;
            reconstruction_statistics_c calculate_prediction_order_reconstruction_statistics() const;
            void compress_files();
            void compress_current_file_by_mode();
            template<bool two_point_sets,time_dimension_mode_e value_mode,time_dimension_mode_e label_mode>
            void compress_current_file();
            template<bool two_point_sets,bool collect_statistics>
            void classify_current_point_sets(compression_file_state_c&state);
            void prepare_current_point_set_codecs_and_eb(compression_file_state_c&state);
            void load_current_point_set_statistics(compression_file_state_c&state);
            template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode>
            void prepare_current_prediction_state(compression_file_state_c&state);
            template<bool two_point_sets>
            void initialize_current_compression_buffers(compression_file_state_c&state);
            template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode>
            void prepare_current_compression(compression_file_state_c&state);
            template<time_dimension_mode_e value_mode>
            value_compression_result_c<T_data> compress_structure_values(const prediction_structure_c<T_calc>&prediction_structure);
            template<time_dimension_mode_e value_mode>
            void compress_values(compression_file_state_c&state);
            template<time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode>
            void compress_labels_and_values(compression_file_state_c&state);
            encoded_compression_payload_c encode_compression_payload(vector<uint8_t>&label_symbols,bool has_label,vector<vector<int32_t>>&point_set_qb_vec,vector<T_data>&unpredictable_value_vec);
            void assemble_current_compression_file(compression_file_state_c&state);
            void print_current_compression_diagnostics() const;
            template<bool two_point_sets>
            void decode_current_compression_streams(decompression_file_state_c<T_data>&state);
            void decompress_current_file_by_mode();
            void print_current_decompression_diagnostics() const;
            void validate_decoded_point_set_counts() const;
            template<bool two_point_sets,time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode>
            void decompress_current_file();
            template<time_dimension_mode_e value_mode>
            void decompress_values(decompression_file_state_c<T_data>&state);
            template<time_dimension_mode_e fixed_value_mode,time_dimension_mode_e fixed_label_mode>
            void decompress_labels_and_values(decompression_file_state_c<T_data>&state);
    };
}

#include "distance.tpp"
#include "binary_file.tpp"
#include "tree.tpp"
#include "tensor_dependency.tpp"
#include "init.tpp"
#include "coordinate_transform.tpp"
#include "prediction_round.tpp"
#include "partition.tpp"
#include "eb_schedule.tpp"
#include "generate_prediction_structure_and_weights.tpp"
#include "point_set_config.tpp"
#include "tensor_config.tpp"
#include "cli_parameter.tpp"
#include "spatial_prediction.tpp"
#include "temporal_prediction.tpp"
#include "prediction_framework.tpp"
#include "compression_format.tpp"
#include "reconstruction_statistics.tpp"
#include "prediction_weights.tpp"
#include "prediction_order.tpp"
#include "spatial_prediction_state.tpp"
#include "compress_entry.tpp"
#include "compress.tpp"
#include "automatic_flow_selection.tpp"
#include "decompress.tpp"
