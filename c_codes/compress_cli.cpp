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
            else if (strcmp(argv[arg_id],"-field")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-field后面缺少参数值");
                param.field_name=argv[++arg_id];
                param.got_field=true;
            }
            else if (strcmp(argv[arg_id],"-coordinate-dim")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-coordinate-dim后面缺少参数值");
                param.dim=parse_positive_uint32(argv[++arg_id],"-coordinate-dim");
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
            else if (strcmp(argv[arg_id],"-prediction-min-ref-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-prediction-min-ref-num后面缺少参数值");
                param.prediction_min_ref_num=parse_positive_uint32(argv[++arg_id],"-prediction-min-ref-num");
            }
            else if (strcmp(argv[arg_id],"-prediction-max-ref-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-prediction-max-ref-num后面缺少参数值");
                param.prediction_max_ref_num=parse_positive_uint32(argv[++arg_id],"-prediction-max-ref-num");
            }
            else if(strcmp(argv[arg_id],"-ordering-score-max-ref-num")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-ordering-score-max-ref-num后面缺少参数值");
                param.ordering_score_max_ref_num=parse_positive_uint32(argv[++arg_id],"-ordering-score-max-ref-num");
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
            else if(strcmp(argv[arg_id],"-rbf-solver")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-rbf-solver后面缺少求解器名称");
                string solver=argv[++arg_id];
                if(solver!="general"&&solver!="ldlt")
                    throw runtime_error("-rbf-solver必须是general或ldlt");
                param.use_rbf_ldlt=solver=="ldlt";
            }
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
            else if(strcmp(argv[arg_id],"-prediction-round-num")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-prediction-round-num后面缺少参数值");
                param.prediction_round_num=parse_positive_uint32(argv[++arg_id],"-prediction-round-num");
                if(param.prediction_round_num>63)
                    throw runtime_error("-prediction-round-num必须在[1,63]内");
            }
            else if(strcmp(argv[arg_id],"-round-size-growth-ratio-experimental")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-round-size-growth-ratio-experimental后面缺少参数值");
                param.round_size_growth_ratio_experimental=parse_positive_float(argv[++arg_id],"-round-size-growth-ratio-experimental");
                if(param.round_size_growth_ratio_experimental<1 || param.round_size_growth_ratio_experimental>16)
                    throw runtime_error("-round-size-growth-ratio-experimental必须在[1,16]内");
            }
            else if (strcmp(argv[arg_id],"-idw-power")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-idw-power后面缺少参数值");
                param.idw_power=parse_positive_float(argv[++arg_id],"-idw-power");
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
            else if (strcmp(argv[arg_id],"-ordering-score-min-ref-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-ordering-score-min-ref-num后面缺少参数值");
                param.ordering_score_min_ref_num=parse_positive_uint32(argv[++arg_id],"-ordering-score-min-ref-num");
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
            else if (strcmp(argv[arg_id],"-ordering-candidate-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-ordering-candidate-num后面缺少参数值");
                param.ordering_candidate_num=parse_positive_uint32(argv[++arg_id],"-ordering-candidate-num");
            }
            else if (strcmp(argv[arg_id],"-dag-cache")==0)
            {
                if(param.no_prediction_dag_cache)
                    throw runtime_error("-dag-cache不能和-no-dag-cache同时使用");
                if (arg_id+1>=argc)
                    throw runtime_error("-dag-cache后面缺少参数值");
                param.dag_cache_path=argv[++arg_id];
            }
            else if(strcmp(argv[arg_id],"-no-dag-cache")==0)
            {
                if(!param.dag_cache_path.empty())
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
            else if (strcmp(argv[arg_id],"-value-flow")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-value-flow后面缺少模式");
                param.value_flow=argv[++arg_id];
                if (param.value_flow!="constant_extrapolation" && param.value_flow!="linear_interpolation" && param.value_flow!="constant_extrapolation_spatial_residual" && param.value_flow!="linear_interpolation_spatial_residual")
                    throw runtime_error("-value-flow目前只支持constant_extrapolation、linear_interpolation、constant_extrapolation_spatial_residual或linear_interpolation_spatial_residual");
            }
            else if (strcmp(argv[arg_id],"-label-flow")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-label-flow后面缺少模式");
                param.label_flow=argv[++arg_id];
                if(param.label_flow!="constant_extrapolation"&&param.label_flow!="endpoint_agreement")
                    throw runtime_error("-label-flow必须是constant_extrapolation或endpoint_agreement");
            }
            else if (strcmp(argv[arg_id],"-predictor")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-predictor后面缺少参数值");
                param.predictor=argv[++arg_id];
                if(param.predictor!="idw" && param.predictor!="rbf")
                    throw runtime_error("-predictor必须是idw或rbf");
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
                param.adjust_coord_matrix_file=argv[++arg_id];
            }
            else if (strcmp(argv[arg_id],"-compression-stage-time")==0)
                param.print_compression_stage_time=true;
            else if (strcmp(argv[arg_id],"-parallelism-stats")==0)
                param.print_prediction_dag_statistics=true;
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
            else if(strcmp(argv[arg_id],"-label-value-execution")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-label-value-execution后面缺少执行方式");
                param.label_value_execution=argv[++arg_id];
                if(param.label_value_execution!="staged"&&param.label_value_execution!="fused")
                    throw runtime_error("-label-value-execution必须是staged或fused");
            }
            else if (strcmp(argv[arg_id],"-linear-previous-label-experimental")==0)
                param.use_linear_previous_label_experimental=true;
            else if(strcmp(argv[arg_id],"-rbf-fallback")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-rbf-fallback后面缺少预测器名称");
                string fallback=argv[++arg_id];
                if(fallback!="idw"&&fallback!="nearest")
                    throw runtime_error("-rbf-fallback必须是idw或nearest");
                param.rbf_fallback_nearest=fallback=="nearest";
            }
            else if (strcmp(argv[arg_id],"-predictability-order-experimental")==0)
                param.use_predictability_order_experimental=true;
            else if (strcmp(argv[arg_id],"-forward-msd-order")==0)
                param.use_forward_msd_order=true;
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
            else if (strcmp(argv[arg_id],"-reverse-msd-order")==0)
                param.use_reverse_msd_order=true;
            else if (strcmp(argv[arg_id],"-reverse-direction-only-order-experimental")==0)
                param.use_reverse_direction_only_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reverse-direction-order-experimental")==0)
                param.use_reverse_direction_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reverse-direction-distribution-order-experimental")==0)
                param.use_reverse_direction_distribution_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reference-data-order")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-reference-data-order后面缺少参考time step列表");
                param.reference_time_step_vec=parse_time_step_vec(argv[++arg_id]);
            }
            else if(strcmp(argv[arg_id],"-reference-data-rbf-score")==0)
                param.use_reference_data_rbf_score=true;
            else if(strcmp(argv[arg_id],"-ordering-scope")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-ordering-scope后面缺少排序规模");
                param.ordering_scope=argv[++arg_id];
                if(param.ordering_scope!="local"&&param.ordering_scope!="global")
                    throw runtime_error("-ordering-scope必须是local或global");
            }
            else if (strcmp(argv[arg_id],"-eb-schedule")==0)
            {
                if (arg_id+2>=argc)
                    throw runtime_error("-eb-schedule后面需要模式和起始倍率");
                param.eb_schedule=argv[++arg_id];
                param.eb_schedule_start_ratio=parse_positive_float(argv[++arg_id],"-eb-schedule");
                if(param.eb_schedule!="none" && param.eb_schedule!="round" && param.eb_schedule!="point" && param.eb_schedule!="inverse_rank_log2" && param.eb_schedule!="round_point" && param.eb_schedule!="round_ratio" && param.eb_schedule!="round_point_ratio" && param.eb_schedule!="round_point_ratio_dag" && param.eb_schedule!="round_point_ratio_dag_direct" && param.eb_schedule!="round_point_ratio_dag_variance_direct")
                    throw runtime_error("未知的EB序列模式："+param.eb_schedule);
                if (param.eb_schedule_start_ratio>1)
                    throw runtime_error("-eb-schedule的起始倍率不能大于1");
            }
            else if(strcmp(argv[arg_id],"-eb-schedule-log2-shift")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-eb-schedule-log2-shift后面缺少参数值");
                param.eb_schedule_log2_shift=parse_positive_float(argv[++arg_id],"-eb-schedule-log2-shift");
            }
            else if(strcmp(argv[arg_id],"-eb-schedule-max-log2-exponent")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-eb-schedule-max-log2-exponent后面缺少参数值");
                param.eb_schedule_max_log2_exponent=parse_positive_float(argv[++arg_id],"-eb-schedule-max-log2-exponent");
            }
            else
                throw runtime_error("未知参数："+string(argv[arg_id]));
        }
        if(!param.got_data_dir || !param.got_field || !param.got_time_step || (!param.got_error_mode && param.point_set_config_path.empty()))
            throw runtime_error("用法：s1_compress -data-dir <目录> -field <field名称> -time-step <区间或列表> (-M <REL|ABS> <误差> | -point-set-config <JSON路径>) [-coordinate-dim <维数>] [-time-anchor-stride-experimental <步长>] [-adjust-coord-matrix-file <路径>] [-value-flow <constant_extrapolation|linear_interpolation|constant_extrapolation_spatial_residual|linear_interpolation_spatial_residual>] [-label-flow <constant_extrapolation|endpoint_agreement>] [-predictor <idw|rbf>] [-rbf-kernel <gaussian|exponential|matern32|matern52|inverse_multiquadric>] [-rbf-shape <正数>] [-rbf-nugget <正数>] [-rbf-fallback <idw|nearest>] [-prediction-min-ref-num <数量>] [-prediction-max-ref-num <数量>] [-prediction-round-num <数量>] [-round-size-growth-ratio-experimental <比例>] [-idw-power <指数>] [-zstd-level <1到22>] [-point-set-predictor <名称>] [-dag-cache <路径> | -no-dag-cache] [-kd-leaf-size <数量>] [-partition-size <数量>] [-partition-num <数量>] [-thread-num <数量>] [-partition-adaptive-predictor-experimental] [-pointwise-oracle-statistics-experimental] [-partition-adaptive-sample-num-experimental <数量>] [-partition-adaptive-min-improvement-experimental <比例>] [-partition-fitting-candidates-experimental <original|weighted|ridge|mixed|residual|extended|all>] [-round-stats] [-prediction-geometry-stats] [-label-stats] [-prediction-residual-stats] [-compression-stage-time] [-parallelism-stats] [-qb-storage-order <prediction|original>] [-normalized-weight-experimental] [-label-value-execution <staged|fused>] [-linear-previous-label-experimental] [-predictability-order-experimental] [-forward-msd-order] [-forward-direction-only-order-experimental] [-forward-mean-distance-s2-order-experimental] [-forward-mean-distance-direction-order-experimental] [-reverse-predictability-order-experimental] [-reverse-distribution-order-experimental] [-reverse-msd-order] [-reverse-direction-only-order-experimental] [-reverse-direction-order-experimental] [-reverse-direction-distribution-order-experimental] [-reference-data-order <time step列表>] [-reference-data-rbf-score] [-ordering-scope <local|global>] [-ordering-score-min-ref-num <数量>] [-ordering-candidate-num <数量>] [-ordering-score-max-ref-num <数量>] [-ordering-msd-neighbor-prefetch-num-experimental <数量>] [-msd-candidate-score-geometry-reuse-experimental] [-kd-query-diagnostics-experimental] [-kd-leaf-valid-mask-experimental] [-kd-iterative-query-experimental] [-kd-scalar-one-reference-query-experimental] [-rbf-solver <general|ldlt>] [-precomputed-neighbor-num-experimental <数量>] [-multihead-execution-head-num-experimental <数量>] [-eb-schedule <模式> <倍率>] [-eb-schedule-log2-shift <正数>] [-eb-schedule-max-log2-exponent <正数>]");
        if (param.point_set_predictor!="idw" && param.point_set_predictor!="majority" && param.point_set_predictor!="continuous_actual" && param.point_set_predictor!="continuous_reflect")
            throw runtime_error("未知的点集标签预测器："+param.point_set_predictor);
        if (param.use_partition_adaptive_predictor_experimental && param.partition_size_experimental==0 && param.partition_num_experimental==0)
            throw runtime_error("partition自适应预测器需要指定空间partition");
        if (param.thread_num>1 && param.partition_size_experimental==0 && param.partition_num_experimental==0)
            throw runtime_error("多线程模式需要指定partition数量或大小");
        if (param.print_pointwise_oracle_statistics_experimental && !param.use_partition_adaptive_predictor_experimental)
            throw runtime_error("逐点oracle统计需要启用partition自适应预测器");
        if (!param.data_dir.empty() && param.data_dir.back()!='/')
            param.data_dir+="/";
        if(!param.no_prediction_dag_cache&&param.dag_cache_path.empty())
        {
            string dag_file_name=param.field_name+".dag";
            if(param.cross_time_spatial_point_num_experimental>0)
                dag_file_name=param.field_name+".joint.dag";
            else if(param.precomputed_neighbor_num_experimental>0)
                dag_file_name=param.field_name+".precomputed"+to_string(param.precomputed_neighbor_num_experimental)+".dag";
            param.dag_cache_path=param.data_dir+".unsz_cache/"+dag_file_name;
        }
        param.coord_path=param.data_dir+"coords.dat";
        for (uint32_t time_step:time_step_vec)
            param.original_data_path_vec.push_back(param.data_dir+param.field_name+".dat."+to_string(time_step));
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
        if(param.use_reference_data_rbf_score && param.reference_time_step_vec.empty())
            throw runtime_error("-reference-data-rbf-score必须和-reference-data-order一起使用");
        if(param.use_reference_data_rbf_score && none_of(param.point_set_config_vec.begin(),param.point_set_config_vec.end(),[](const unsz::point_set_config_c &config){return config.codec==unsz::point_set_codec_e::rbf;}))
            throw runtime_error("RBF参考数据评分要求至少一个点集使用rbf");
        param.cross_time_compress=false;
        param.load_adjust_coord_matrix_file();
        if(param.prediction_min_ref_num>param.prediction_max_ref_num)
            throw runtime_error("prediction_min_ref_num不能大于prediction_max_ref_num");
        if(param.ordering_candidate_num==0)
            throw runtime_error("必须显式设置正数ordering_candidate_num");
        if(param.ordering_score_max_ref_num==0)
            throw runtime_error("必须显式设置正数ordering_score_max_ref_num");
        if(param.ordering_score_min_ref_num==0)
            throw runtime_error("必须显式设置正数ordering_score_min_ref_num");
        if(param.ordering_score_min_ref_num>param.ordering_score_max_ref_num)
            throw runtime_error("ordering_score_min_ref_num不能大于ordering_score_max_ref_num");
        if(param.prediction_min_ref_num<2&&any_of(param.point_set_config_vec.begin(),param.point_set_config_vec.end(),[](const unsz::point_set_config_c&config){return config.codec==unsz::point_set_codec_e::rbf;}))
            throw runtime_error("RBF的prediction_min_ref_num不能小于2");

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
