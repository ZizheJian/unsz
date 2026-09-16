#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include "parsers.hpp"
#include "unsz_core/unsz.hpp"
#include "compress_cli.hpp"

using namespace std;

vector<uint32_t> parse_time_step_vec(string expression)
{
    vector<uint32_t> result;
    if (expression.size()>=5 && expression.front()=='[' && (expression.back()==']' || expression.back()==')'))
    {
        uint64_t comma_position=expression.find(',');
        if (comma_position==string::npos || expression.find(',',comma_position+1)!=string::npos)
            throw runtime_error("时间步区间必须包含且只包含一个逗号："+expression);
        uint32_t left=parse_uint32(expression.substr(1,comma_position-1),"时间步左端点");
        uint32_t right=parse_uint32(expression.substr(comma_position+1,expression.size()-comma_position-2),"时间步右端点");
        uint64_t end=(uint64_t)right+(expression.back()==']');
        if ((uint64_t)left>=end)
            throw runtime_error("时间步区间为空或端点顺序错误："+expression);
        result.reserve(end-left);
        for (uint64_t time_step=left;time_step<end;time_step++)
            result.push_back((uint32_t)time_step);
    }
    else if (expression.size()>=3 && expression.front()=='{' && expression.back()=='}')
    {
        if (expression[expression.size()-2]==',')
            throw runtime_error("时间步列表末尾不能有逗号："+expression);
        unordered_set<uint32_t> used_time_step_set;
        uint64_t begin=1,end=0;
        while (begin<expression.size()-1)
        {
            end=expression.find(',',begin);
            if (end==string::npos)
                end=expression.size()-1;
            uint32_t time_step=parse_uint32(expression.substr(begin,end-begin),"时间步");
            if (!used_time_step_set.insert(time_step).second)
                throw runtime_error("时间步列表中存在重复值："+to_string(time_step));
            result.push_back(time_step);
            begin=end+1;
        }
        if (result.empty())
            throw runtime_error("时间步列表不能为空");
    }
    else
        throw runtime_error("时间步必须写成区间'[a,b]'、'[a,b)'或列表'{a1,a2,...}'："+expression);
    return result;
}

namespace unsz
{
    void parse_args(int argc,char**argv,param_c<float,float>&param)
    {
        vector<uint32_t> time_step_vec;
        float error_bound=0;
        unordered_set<string> used_option_set;
        for (int arg_id=1;arg_id<argc;arg_id++)
        {
            if (!used_option_set.insert(argv[arg_id]).second)
                throw runtime_error("参数重复："+string(argv[arg_id]));
            if (strcmp(argv[arg_id],"-data-dir")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-data-dir后面缺少参数值");
                param.data_dir=argv[++arg_id];
                param.got_data_dir=true;
            }
            else if (strcmp(argv[arg_id],"-variable")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-variable后面缺少参数值");
                param.variable_name=argv[++arg_id];
                param.got_variable=true;
            }
            else if (strcmp(argv[arg_id],"-dim")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-dim后面缺少参数值");
                param.dim=parse_positive_uint32(argv[++arg_id],"-dim");
            }
            else if (strcmp(argv[arg_id],"-time-step")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-time-step后面缺少参数值");
                time_step_vec=parse_time_step_vec(argv[++arg_id]);
                param.got_time_step=true;
            }
            else if(strcmp(argv[arg_id],"-time-step-coordinate-file")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-time-step-coordinate-file后面缺少文件路径");
                param.time_step_coordinate_path=argv[++arg_id];
            }
            else if (strcmp(argv[arg_id],"-M")==0)
            {
                if (arg_id+2>=argc)
                    throw runtime_error("-M后面需要误差模式和误差值");
                param.error_mode=argv[++arg_id];
                error_bound=parse_positive_float(argv[++arg_id],"-M");
                if (param.error_mode!="REL" && param.error_mode!="ABS")
                    throw runtime_error("-M的误差模式必须是REL或ABS："+param.error_mode);
                param.got_error_mode=true;
            }
            else if (strcmp(argv[arg_id],"-max-ref-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-max-ref-num后面缺少参数值");
                param.max_ref_num=parse_positive_uint32(argv[++arg_id],"-max-ref-num");
            }
            else if (strcmp(argv[arg_id],"-actual-ref-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-actual-ref-num后面缺少参数值");
                param.actual_ref_num=parse_positive_uint32(argv[++arg_id],"-actual-ref-num");
            }
            else if(strcmp(argv[arg_id],"-ordering-msd-ref-num-experimental")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-ordering-msd-ref-num-experimental后面缺少参数值");
                param.ordering_msd_ref_num_experimental=parse_positive_uint32(argv[++arg_id],"-ordering-msd-ref-num-experimental");
            }
            else if(strcmp(argv[arg_id],"-ordering-msd-neighbor-prefetch-num-experimental")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-ordering-msd-neighbor-prefetch-num-experimental后面缺少参数值");
                param.ordering_msd_neighbor_prefetch_num_experimental=parse_positive_uint32(argv[++arg_id],"-ordering-msd-neighbor-prefetch-num-experimental");
            }
            else if(strcmp(argv[arg_id],"-msd-candidate-score-geometry-reuse-experimental")==0)
                param.use_msd_candidate_score_geometry_reuse_experimental=true;
            else if(strcmp(argv[arg_id],"-kd-query-diagnostics-experimental")==0)
                param.use_kd_query_diagnostics_experimental=true;
            else if(strcmp(argv[arg_id],"-kd-leaf-valid-mask-experimental")==0)
                param.use_kd_leaf_valid_mask_experimental=true;
            else if(strcmp(argv[arg_id],"-kd-iterative-query-experimental")==0)
                param.use_kd_iterative_query_experimental=true;
            else if(strcmp(argv[arg_id],"-kd-scalar-one-reference-query-experimental")==0)
                param.use_kd_scalar_one_reference_query_experimental=true;
            else if(strcmp(argv[arg_id],"-rbf-ldlt-experimental")==0)
                param.use_rbf_ldlt_experimental=true;
            else if(strcmp(argv[arg_id],"-precomputed-neighbor-num-experimental")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-precomputed-neighbor-num-experimental后面缺少参数值");
                param.precomputed_neighbor_num_experimental=parse_positive_uint32(argv[++arg_id],"-precomputed-neighbor-num-experimental");
            }
            else if(strcmp(argv[arg_id],"-multihead-execution-head-num-experimental")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-multihead-execution-head-num-experimental后面缺少参数值");
                param.multihead_execution_head_num_experimental=parse_positive_uint32(argv[++arg_id],"-multihead-execution-head-num-experimental");
            }
            else if(strcmp(argv[arg_id],"-predict-round-num")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-predict-round-num后面缺少参数值");
                param.predict_round_num=parse_positive_uint32(argv[++arg_id],"-predict-round-num");
                if(param.predict_round_num>63)
                    throw runtime_error("-predict-round-num必须在[1,63]内");
            }
            else if(strcmp(argv[arg_id],"-round-size-growth-ratio-experimental")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-round-size-growth-ratio-experimental后面缺少参数值");
                param.round_size_growth_ratio_experimental=parse_positive_float(argv[++arg_id],"-round-size-growth-ratio-experimental");
                if(param.round_size_growth_ratio_experimental<1 || param.round_size_growth_ratio_experimental>16)
                    throw runtime_error("-round-size-growth-ratio-experimental必须在[1,16]内");
            }
            else if (strcmp(argv[arg_id],"-inverse-distance-power")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-inverse-distance-power后面缺少参数值");
                param.inverse_distance_power=parse_positive_float(argv[++arg_id],"-inverse-distance-power");
            }
            else if(strcmp(argv[arg_id],"-zstd-level")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-zstd-level后面缺少参数值");
                param.zstd_compression_level=(int)parse_positive_uint32(argv[++arg_id],"-zstd-level");
                if(param.zstd_compression_level>22)
                    throw runtime_error("-zstd-level必须在[1,22]内");
            }
            else if(strcmp(argv[arg_id],"-rbf-kernel")==0)
            {
                param.rbf_parameter_specified=true;
                if(arg_id+1>=argc)
                    throw runtime_error("-rbf-kernel后面缺少核函数名称");
                param.rbf_kernel=argv[++arg_id];
                if(param.rbf_kernel!="gaussian" && param.rbf_kernel!="exponential" && param.rbf_kernel!="matern32" && param.rbf_kernel!="matern52" && param.rbf_kernel!="inverse_multiquadric")
                    throw runtime_error("-rbf-kernel必须是gaussian、exponential、matern32、matern52或inverse_multiquadric");
            }
            else if(strcmp(argv[arg_id],"-rbf-shape")==0)
            {
                param.rbf_parameter_specified=true;
                if(arg_id+1>=argc)
                    throw runtime_error("-rbf-shape后面缺少参数值");
                param.rbf_shape=parse_positive_float(argv[++arg_id],"-rbf-shape");
            }
            else if(strcmp(argv[arg_id],"-rbf-nugget")==0)
            {
                param.rbf_parameter_specified=true;
                if(arg_id+1>=argc)
                    throw runtime_error("-rbf-nugget后面缺少参数值");
                param.rbf_nugget=parse_positive_float(argv[++arg_id],"-rbf-nugget");
            }
            else if(strcmp(argv[arg_id],"-rbf-polynomial-degree")==0)
            {
                param.rbf_parameter_specified=true;
                if(arg_id+1>=argc)
                    throw runtime_error("-rbf-polynomial-degree后面缺少参数值");
                param.rbf_polynomial_degree=parse_uint32(argv[++arg_id],"-rbf-polynomial-degree");
                if(param.rbf_polynomial_degree>2)
                    throw runtime_error("-rbf-polynomial-degree必须是0、1或2");
            }
            else if (strcmp(argv[arg_id],"-d-only-cache-refill-threshold-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-d-only-cache-refill-threshold-experimental后面缺少参数值");
                param.d_only_cache_refill_threshold_experimental=parse_uint32(argv[++arg_id],"-d-only-cache-refill-threshold-experimental");
            }
            else if (strcmp(argv[arg_id],"-cross-time-spatial-point-num-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-cross-time-spatial-point-num-experimental后面缺少空间点数");
                param.cross_time_spatial_point_num_experimental=parse_positive_uint32(argv[++arg_id],"-cross-time-spatial-point-num-experimental");
            }
            else if(strcmp(argv[arg_id],"-time-anchor-stride-experimental")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-time-anchor-stride-experimental后面缺少步长");
                param.time_anchor_stride_experimental=parse_positive_uint32(argv[++arg_id],"-time-anchor-stride-experimental");
            }
            else if (strcmp(argv[arg_id],"-order-candidate-num-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-order-candidate-num-experimental后面缺少参数值");
                param.order_candidate_num_experimental=parse_positive_uint32(argv[++arg_id],"-order-candidate-num-experimental");
            }
            else if (strcmp(argv[arg_id],"-dag-cache")==0)
            {
                if(param.no_prediction_dag_cache)
                    throw runtime_error("-dag-cache不能和-no-dag-cache同时使用");
                if (arg_id+1>=argc)
                    throw runtime_error("-dag-cache后面缺少参数值");
                param.prediction_structure_path=argv[++arg_id];
            }
            else if(strcmp(argv[arg_id],"-no-dag-cache")==0)
            {
                if(!param.prediction_structure_path.empty())
                    throw runtime_error("-no-dag-cache不能和-dag-cache同时使用");
                param.no_prediction_dag_cache=true;
            }
            else if (strcmp(argv[arg_id],"-kd-leaf-size")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-kd-leaf-size后面缺少参数值");
                param.kd_tree_leaf_size=parse_positive_uint32(argv[++arg_id],"-kd-leaf-size");
            }
            else if (strcmp(argv[arg_id],"-partition-size")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-partition-size后面缺少参数值");
                param.partition_size_experimental=parse_positive_uint32(argv[++arg_id],"-partition-size");
            }
            else if (strcmp(argv[arg_id],"-partition-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-partition-num后面缺少参数值");
                param.partition_num_experimental=parse_positive_uint32(argv[++arg_id],"-partition-num");
            }
            else if (strcmp(argv[arg_id],"-thread-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-thread-num后面缺少参数值");
                param.thread_num=parse_positive_uint32(argv[++arg_id],"-thread-num");
            }
            else if (strcmp(argv[arg_id],"-partition-adaptive-predictor-experimental")==0)
                param.use_partition_adaptive_predictor_experimental=true;
            else if (strcmp(argv[arg_id],"-pointwise-oracle-statistics-experimental")==0)
                param.print_pointwise_oracle_statistics_experimental=true;
            else if (strcmp(argv[arg_id],"-partition-adaptive-sample-num-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-partition-adaptive-sample-num-experimental后面缺少参数值");
                param.partition_adaptive_sample_num_experimental=parse_positive_uint32(argv[++arg_id],"-partition-adaptive-sample-num-experimental");
            }
            else if (strcmp(argv[arg_id],"-partition-adaptive-min-improvement-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-partition-adaptive-min-improvement-experimental后面缺少参数值");
                param.partition_adaptive_min_improvement_experimental=parse_positive_float(argv[++arg_id],"-partition-adaptive-min-improvement-experimental");
                if (param.partition_adaptive_min_improvement_experimental>1)
                    throw runtime_error("-partition-adaptive-min-improvement-experimental不能大于1");
            }
            else if (strcmp(argv[arg_id],"-partition-fitting-candidates-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-partition-fitting-candidates-experimental后面缺少模式");
                param.partition_fitting_candidates_experimental=argv[++arg_id];
                if (param.partition_fitting_candidates_experimental!="original" && param.partition_fitting_candidates_experimental!="weighted" && param.partition_fitting_candidates_experimental!="ridge" && param.partition_fitting_candidates_experimental!="mixed" && param.partition_fitting_candidates_experimental!="residual" && param.partition_fitting_candidates_experimental!="extended" && param.partition_fitting_candidates_experimental!="all")
                    throw runtime_error("-partition-fitting-candidates-experimental必须是original、weighted、ridge、mixed、residual、extended或all");
            }
            else if (strcmp(argv[arg_id],"-point-set-predictor")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-point-set-predictor后面缺少参数值");
                param.point_set_predictor=argv[++arg_id];
            }
            else if(strcmp(argv[arg_id],"-point-set-config")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-point-set-config后面缺少文件路径");
                param.point_set_config_path=argv[++arg_id];
            }
            else if (strcmp(argv[arg_id],"-cross-time-predictor-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-cross-time-predictor-experimental后面缺少模式");
                param.cross_time_predictor_experimental=argv[++arg_id];
                if (param.cross_time_predictor_experimental!="constant_extrapolation" && param.cross_time_predictor_experimental!="linear_interpolation" && param.cross_time_predictor_experimental!="constant_extrapolation_spatial_residual" && param.cross_time_predictor_experimental!="linear_interpolation_spatial_residual")
                    throw runtime_error("-cross-time-predictor-experimental目前只支持constant_extrapolation、linear_interpolation、constant_extrapolation_spatial_residual或linear_interpolation_spatial_residual");
            }
            else if (strcmp(argv[arg_id],"-cross-time-label-predictor-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-cross-time-label-predictor-experimental后面缺少模式");
                param.cross_time_label_predictor_experimental=argv[++arg_id];
                if(param.cross_time_label_predictor_experimental!="previous"&&param.cross_time_label_predictor_experimental!="endpoint_agreement")
                    throw runtime_error("-cross-time-label-predictor-experimental必须是previous或endpoint_agreement");
            }
            else if (strcmp(argv[arg_id],"-predictor")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-predictor后面缺少参数值");
                param.predictor=argv[++arg_id];
                if(param.predictor!="inverse_proportion" && param.predictor!="rbf_fitting")
                    throw runtime_error("-predictor必须是inverse_proportion或rbf_fitting");
                param.got_predictor=true;
            }
            else if (strcmp(argv[arg_id],"-round-stats")==0)
                param.print_round_statistics=true;
            else if (strcmp(argv[arg_id],"-prediction-geometry-stats")==0)
                param.print_prediction_geometry_statistics=true;
            else if (strcmp(argv[arg_id],"-label-stats")==0)
                param.print_label_compression_statistics=true;
            else if (strcmp(argv[arg_id],"-prediction-residual-stats")==0)
                param.print_prediction_residual_statistics=true;
            else if (strcmp(argv[arg_id],"-adjust-coord-matrix-file")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-adjust-coord-matrix-file后面缺少文件路径");
                param.adjust_coord_matrix_path=argv[++arg_id];
            }
            else if (strcmp(argv[arg_id],"-compression-stage-time")==0)
                param.print_compression_stage_time=true;
            else if (strcmp(argv[arg_id],"-parallelism-stats")==0)
                param.print_prediction_dag_statistics=true;
            else if(strcmp(argv[arg_id],"-reorder-data-by-prediction-order-experimental")==0)
                param.reorder_data_by_prediction_order_experimental=true;
            else if(strcmp(argv[arg_id],"-qb-storage-order")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-qb-storage-order后面缺少参数值");
                param.qb_storage_order=argv[++arg_id];
                if(param.qb_storage_order!="prediction"&&param.qb_storage_order!="original")
                    throw runtime_error("-qb-storage-order必须是prediction或original");
            }
            else if (strcmp(argv[arg_id],"-normalized-weight-experimental")==0)
                param.use_normalized_weight_experimental=true;
            else if (strcmp(argv[arg_id],"-fused-two-point-set-prediction-experimental")==0)
                param.use_fused_two_point_set_prediction_experimental=true;
            else if (strcmp(argv[arg_id],"-linear-previous-label-experimental")==0)
                param.use_linear_previous_label_experimental=true;
            else if(strcmp(argv[arg_id],"-rbf-filtered-fallback-nearest-experimental")==0)
                param.use_rbf_filtered_fallback_nearest_experimental=true;
            else if (strcmp(argv[arg_id],"-predictability-order-experimental")==0)
                param.use_predictability_order_experimental=true;
            else if (strcmp(argv[arg_id],"-forward-mean-distance-only-order-experimental")==0)
                param.use_forward_mean_distance_only_order_experimental=true;
            else if (strcmp(argv[arg_id],"-forward-direction-only-order-experimental")==0)
                param.use_forward_direction_only_order_experimental=true;
            else if (strcmp(argv[arg_id],"-forward-mean-distance-s2-order-experimental")==0)
                param.use_forward_mean_distance_s2_order_experimental=true;
            else if (strcmp(argv[arg_id],"-forward-mean-distance-direction-order-experimental")==0)
                param.use_forward_mean_distance_direction_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reverse-predictability-order-experimental")==0)
                param.use_reverse_predictability_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reverse-distribution-order-experimental")==0)
                param.use_reverse_distribution_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reverse-mean-distance-only-order-experimental")==0)
                param.use_reverse_mean_distance_only_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reverse-direction-only-order-experimental")==0)
                param.use_reverse_direction_only_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reverse-direction-order-experimental")==0)
                param.use_reverse_direction_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reverse-direction-distribution-order-experimental")==0)
                param.use_reverse_direction_distribution_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reference-data-order-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-reference-data-order-experimental后面缺少参考time step列表");
                param.order_reference_time_step_vec_experimental=parse_time_step_vec(argv[++arg_id]);
            }
            else if(strcmp(argv[arg_id],"-reference-data-rbf-score-experimental")==0)
                param.use_reference_data_rbf_score_experimental=true;
            else if (strcmp(argv[arg_id],"-global-order-before-round-split-experimental")==0)
                param.use_global_order_before_round_split_experimental=true;
            else if (strcmp(argv[arg_id],"-eb-schedule-experimental")==0)
            {
                if (arg_id+2>=argc)
                    throw runtime_error("-eb-schedule-experimental后面需要模式和起始倍率");
                param.eb_schedule_experimental=argv[++arg_id];
                param.eb_schedule_start_ratio=parse_positive_float(argv[++arg_id],"-eb-schedule-experimental");
                if(param.eb_schedule_experimental!="none" && param.eb_schedule_experimental!="round" && param.eb_schedule_experimental!="point" && param.eb_schedule_experimental!="point_log2_ratio" && param.eb_schedule_experimental!="round_point" && param.eb_schedule_experimental!="round_ratio" && param.eb_schedule_experimental!="round_point_ratio" && param.eb_schedule_experimental!="round_point_ratio_dag" && param.eb_schedule_experimental!="round_point_ratio_dag_direct" && param.eb_schedule_experimental!="round_point_ratio_dag_variance_direct")
                    throw runtime_error("未知的实验性EB序列模式："+param.eb_schedule_experimental);
                if (param.eb_schedule_start_ratio>1)
                    throw runtime_error("-eb-schedule-experimental的起始倍率不能大于1");
            }
            else if(strcmp(argv[arg_id],"-eb-schedule-log2-shift-experimental")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-eb-schedule-log2-shift-experimental后面缺少参数值");
                param.eb_schedule_log2_shift_experimental=parse_positive_float(argv[++arg_id],"-eb-schedule-log2-shift-experimental");
            }
            else
                throw runtime_error("未知参数："+string(argv[arg_id]));
        }
        if(!param.got_data_dir || !param.got_variable || !param.got_time_step || (!param.got_error_mode && param.point_set_config_path.empty()))
            throw runtime_error("用法：s1_compress -data-dir <目录> -variable <变量名> -time-step <区间或列表> (-M <REL|ABS> <误差> | -point-set-config <JSON路径>) [-dim <维数>] [-time-anchor-stride-experimental <步长>] [-adjust-coord-matrix-file <路径>] [-cross-time-predictor-experimental <constant_extrapolation|linear_interpolation|constant_extrapolation_spatial_residual|linear_interpolation_spatial_residual>] [-cross-time-label-predictor-experimental <previous|endpoint_agreement>] [-predictor <inverse_proportion|rbf_fitting>] [-rbf-kernel <gaussian|exponential|matern32|matern52|inverse_multiquadric>] [-rbf-shape <正数>] [-rbf-nugget <正数>] [-rbf-polynomial-degree <0|1|2>] [-rbf-filtered-fallback-nearest-experimental] [-max-ref-num <数量>] [-actual-ref-num <数量>] [-predict-round-num <数量>] [-round-size-growth-ratio-experimental <比例>] [-inverse-distance-power <指数>] [-zstd-level <1到22>] [-point-set-predictor <名称>] [-dag-cache <路径> | -no-dag-cache] [-kd-leaf-size <数量>] [-partition-size <数量>] [-partition-num <数量>] [-thread-num <数量>] [-partition-adaptive-predictor-experimental] [-pointwise-oracle-statistics-experimental] [-partition-adaptive-sample-num-experimental <数量>] [-partition-adaptive-min-improvement-experimental <比例>] [-partition-fitting-candidates-experimental <original|weighted|ridge|mixed|residual|extended|all>] [-round-stats] [-prediction-geometry-stats] [-label-stats] [-prediction-residual-stats] [-compression-stage-time] [-parallelism-stats] [-reorder-data-by-prediction-order-experimental] [-qb-storage-order <prediction|original>] [-normalized-weight-experimental] [-fused-two-point-set-prediction-experimental] [-linear-previous-label-experimental] [-predictability-order-experimental] [-forward-mean-distance-only-order-experimental] [-forward-direction-only-order-experimental] [-forward-mean-distance-s2-order-experimental] [-forward-mean-distance-direction-order-experimental] [-reverse-predictability-order-experimental] [-reverse-distribution-order-experimental] [-reverse-mean-distance-only-order-experimental] [-reverse-direction-only-order-experimental] [-reverse-direction-order-experimental] [-reverse-direction-distribution-order-experimental] [-reference-data-order-experimental <time step列表>] [-reference-data-rbf-score-experimental] [-global-order-before-round-split-experimental] [-d-only-cache-refill-threshold-experimental <数量>] [-order-candidate-num-experimental <数量>] [-ordering-msd-ref-num-experimental <数量>] [-ordering-msd-neighbor-prefetch-num-experimental <数量>] [-msd-candidate-score-geometry-reuse-experimental] [-kd-query-diagnostics-experimental] [-kd-leaf-valid-mask-experimental] [-kd-iterative-query-experimental] [-kd-scalar-one-reference-query-experimental] [-rbf-ldlt-experimental] [-precomputed-neighbor-num-experimental <数量>] [-multihead-execution-head-num-experimental <数量>] [-eb-schedule-experimental <模式> <倍率>] [-eb-schedule-log2-shift-experimental <正数>]");
        if (param.point_set_predictor!="inverse_proportion" && param.point_set_predictor!="majority" && param.point_set_predictor!="continuous_actual" && param.point_set_predictor!="continuous_reflect")
            throw runtime_error("未知的点集标签预测器："+param.point_set_predictor);
        if (param.use_partition_adaptive_predictor_experimental && param.partition_size_experimental==0 && param.partition_num_experimental==0)
            throw runtime_error("partition自适应预测器需要指定空间partition");
        if (param.thread_num>1 && param.partition_size_experimental==0 && param.partition_num_experimental==0)
            throw runtime_error("多线程模式需要指定partition数量或大小");
        if (param.print_pointwise_oracle_statistics_experimental && !param.use_partition_adaptive_predictor_experimental)
            throw runtime_error("逐点oracle统计需要启用partition自适应预测器");
        if (!param.data_dir.empty() && param.data_dir.back()!='/')
            param.data_dir+="/";
        if(!param.no_prediction_dag_cache&&param.prediction_structure_path.empty())
        {
            string dag_file_name=param.variable_name+".dag";
            if(param.cross_time_spatial_point_num_experimental>0)
                dag_file_name=param.variable_name+".joint.dag";
            else if(param.precomputed_neighbor_num_experimental>0)
                dag_file_name=param.variable_name+".precomputed"+to_string(param.precomputed_neighbor_num_experimental)+".dag";
            param.prediction_structure_path=param.data_dir+".unsz_cache/"+dag_file_name;
        }
        param.coord_path=param.data_dir+"coords.dat";
        for (uint32_t time_step:time_step_vec)
            param.original_data_path_vec.push_back(param.data_dir+param.variable_name+".dat."+to_string(time_step));
        param.time_step_vec=time_step_vec;
        if (param.got_error_mode && param.error_mode=="REL")
        {
            param.rel_eb=error_bound;
            param.got_rel_eb=true;
        }
        else if (param.got_error_mode)
        {
            param.abs_eb=error_bound;
            param.got_abs_eb=true;
        }
        param.configure_point_sets();
        if(param.point_set_config_vec.size()==1)
            param.resolve_single_point_set_time_mode();
        if(param.use_reference_data_rbf_score_experimental && param.order_reference_time_step_vec_experimental.empty())
            throw runtime_error("-reference-data-rbf-score-experimental必须和-reference-data-order-experimental一起使用");
        if(param.use_reference_data_rbf_score_experimental && none_of(param.point_set_config_vec.begin(),param.point_set_config_vec.end(),[](const unsz::point_set_config_c &config){return config.codec==unsz::point_set_codec_e::rbf_fitting;}))
            throw runtime_error("RBF参考数据评分要求至少一个点集使用rbf_fitting");
        if(param.point_set_classification==point_set_classification_e::label_file && !param.order_reference_time_step_vec_experimental.empty())
            throw runtime_error("label_file分类暂时不能和-reference-data-order-experimental一起使用");
        param.cross_time_compress=false;
        param.load_adjust_coord_matrix_file();
        if (param.actual_ref_num>param.max_ref_num)
            throw runtime_error("actual_ref_num不能大于max_ref_num");

    }

    int parse_args_and_select_types(int argc,char**argv)
    {
        param_c<float,float> param;
        parse_args(argc,argv,param);
        auto preprocessing_start_time=chrono::steady_clock::now();
        param.init();
        if(param.need_double_coord())
        {
            printf("坐标跨度较大，将坐标和距离计算从float升级为double\n");
            param_c<double,float> typed_param(param);
            unsz::compress(typed_param,preprocessing_start_time);
        }
        else
        {
            printf("使用float处理坐标和距离\n");
            unsz::compress(param,preprocessing_start_time);
        }
        return 0;
    }
}
