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
            else if(strcmp(argv[arg_id],"-ordering-score-min-ref-num")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-ordering-score-min-ref-num后面缺少参数值");
                param.ordering_score_min_ref_num=parse_positive_uint32(argv[++arg_id],"-ordering-score-min-ref-num");
            }
            else if(strcmp(argv[arg_id],"-prediction-round-num")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-prediction-round-num后面缺少参数值");
                param.prediction_round_num=parse_positive_uint32(argv[++arg_id],"-prediction-round-num");
                param.got_prediction_round_num=true;
                if(param.prediction_round_num>63)
                    throw runtime_error("-prediction-round-num必须在[1,63]内");
            }
            else if(strcmp(argv[arg_id],"-round-size-growth-ratio-experimental")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-round-size-growth-ratio-experimental后面缺少参数值");
                param.round_size_growth_ratio_experimental=parse_positive_double(argv[++arg_id],"-round-size-growth-ratio-experimental");
                if(param.round_size_growth_ratio_experimental<1 || param.round_size_growth_ratio_experimental>16)
                    throw runtime_error("-round-size-growth-ratio-experimental必须在[1,16]内");
            }
            else if (strcmp(argv[arg_id],"-idw-power")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-idw-power后面缺少参数值");
                param.idw_power=parse_positive_double(argv[++arg_id],"-idw-power");
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
                param.rbf_shape=parse_positive_double(argv[++arg_id],"-rbf-shape");
            }
            else if(strcmp(argv[arg_id],"-rbf-nugget")==0)
            {
                param.rbf_parameter_specified=true;
                if(arg_id+1>=argc)
                    throw runtime_error("-rbf-nugget后面缺少参数值");
                param.rbf_nugget=parse_positive_double(argv[++arg_id],"-rbf-nugget");
            }
            else if (strcmp(argv[arg_id],"-ordering-candidate-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-ordering-candidate-num后面缺少参数值");
                param.ordering_candidate_num=parse_positive_uint32(argv[++arg_id],"-ordering-candidate-num");
            }
            else if (strcmp(argv[arg_id],"-dag-cache")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-dag-cache后面缺少参数值");
                param.dag_cache_path=argv[++arg_id];
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
                param.partition_size=parse_positive_uint32(argv[++arg_id],"-partition-size");
            }
            else if (strcmp(argv[arg_id],"-partition-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-partition-num后面缺少参数值");
                param.partition_num=parse_positive_uint32(argv[++arg_id],"-partition-num");
            }
            else if (strcmp(argv[arg_id],"-thread-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-thread-num后面缺少参数值");
                param.thread_num=parse_positive_uint32(argv[++arg_id],"-thread-num");
            }
            else if (strcmp(argv[arg_id],"-partition-adaptive-predictor-experimental")==0)
                param.use_partition_adaptive_predictor_experimental=true;
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
                param.partition_adaptive_min_improvement_experimental=parse_positive_double(argv[++arg_id],"-partition-adaptive-min-improvement-experimental");
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
                if(param.value_flow!="constant_extrapolation"&&param.value_flow!="linear_interpolation"&&param.value_flow!="constant_extrapolation_spatial_residual"&&param.value_flow!="linear_interpolation_spatial_residual"&&param.value_flow!="joint_spatiotemporal"&&param.value_flow!="anchor_interpolation_stride_8"&&param.value_flow!="anchor_interpolation_stride_8_spatial_residual")
                    throw runtime_error("未知的value flow："+param.value_flow);
            }
            else if (strcmp(argv[arg_id],"-label-flow")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-label-flow后面缺少模式");
                param.label_flow=argv[++arg_id];
                if(param.label_flow!="constant_extrapolation"&&param.label_flow!="linear_interpolation_spatial_residual"&&param.label_flow!="anchor_interpolation_stride_8")
                    throw runtime_error("未知的label flow："+param.label_flow);
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
            else if (strcmp(argv[arg_id],"-adjust-coord-matrix-file")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-adjust-coord-matrix-file后面缺少文件路径");
                param.adjust_coord_matrix_file=argv[++arg_id];
            }
            else if (strcmp(argv[arg_id],"-compression-stage-time")==0)
                param.print_compression_stage_time=true;
            else if(strcmp(argv[arg_id],"-label-value-execution")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-label-value-execution后面缺少执行方式");
                param.label_value_execution=argv[++arg_id];
                if(param.label_value_execution!="staged"&&param.label_value_execution!="fused")
                    throw runtime_error("-label-value-execution必须是staged或fused");
            }
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
                param.eb_schedule_start_ratio=parse_positive_double(argv[++arg_id],"-eb-schedule");
                if(param.eb_schedule!="none"&&param.eb_schedule!="round"&&param.eb_schedule!="point"&&param.eb_schedule!="inverse_rank_log2"&&param.eb_schedule!="round_point"&&param.eb_schedule!="round_ratio"&&param.eb_schedule!="round_point_ratio")
                    throw runtime_error("未知的EB序列模式："+param.eb_schedule);
                if (param.eb_schedule_start_ratio>1)
                    throw runtime_error("-eb-schedule的起始倍率不能大于1");
            }
            else if(strcmp(argv[arg_id],"-eb-schedule-log2-shift")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-eb-schedule-log2-shift后面缺少参数值");
                param.eb_schedule_log2_shift=parse_positive_double(argv[++arg_id],"-eb-schedule-log2-shift");
            }
            else if(strcmp(argv[arg_id],"-eb-schedule-max-log2-exponent")==0)
            {
                if(arg_id+1>=argc)
                    throw runtime_error("-eb-schedule-max-log2-exponent后面缺少参数值");
                param.eb_schedule_max_log2_exponent=parse_positive_double(argv[++arg_id],"-eb-schedule-max-log2-exponent");
            }
            else
                throw runtime_error("未知参数："+string(argv[arg_id]));
        }
        param.prepare_cli_parameters(time_step_vec,error_bound);
    }

    int parse_args_and_select_types(int argc,char**argv)
    {
        param_c<float,float> param;
        parse_args(argc,argv,param);
        param.init();
        auto preprocessing_start_time=chrono::steady_clock::now();
        printf("使用float处理坐标和距离\n");
        unsz::compress(param,preprocessing_start_time);
        return 0;
    }
}
