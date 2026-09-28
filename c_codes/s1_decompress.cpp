#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <unordered_set>
#include "huffman.hpp"
#include "parsers.hpp"
#include "third_party/zstd/lib/zstd.h"
#include "unsz_core/unsz.hpp"

using namespace std;

vector<unsigned char> read_byte_file(string path)
{
    FILE *file=fopen(path.c_str(),"rb");
    if (!file)
        throw runtime_error("找不到文件："+path);
    if (fseek(file,0,SEEK_END)!=0)
    {
        fclose(file);
        throw runtime_error("无法定位到文件末尾："+path);
    }
    long byte_num_signed=ftell(file);
    if (byte_num_signed<0)
    {
        fclose(file);
        throw runtime_error("无法读取文件大小："+path);
    }
    if (fseek(file,0,SEEK_SET)!=0)
    {
        fclose(file);
        throw runtime_error("无法定位到文件开头："+path);
    }
    vector<unsigned char> bytes((uint64_t)byte_num_signed);
    if (fread(bytes.data(),1,bytes.size(),file)!=bytes.size())
    {
        fclose(file);
        throw runtime_error("读取文件失败："+path);
    }
    if (fclose(file)!=0)
        throw runtime_error("关闭文件失败："+path);
    return bytes;
}

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

string read_option_value(int argc,char **argv,int &arg_id,string option)
{
    if (arg_id+1>=argc || argv[arg_id+1][0]=='-')
        throw runtime_error(option+"后面缺少参数值");
    return argv[++arg_id];
}

template<typename T_coord>
void run_decompression(unsz::param_c<T_coord,float> &param,chrono::steady_clock::time_point preprocessing_start_time)
{
    double adjust_coord_time=0;
    bool joint_spatiotemporal=param.value_flow=="joint_spatiotemporal";
    if(joint_spatiotemporal)
    {
        if(param.point_set_config_vec.size()!=1)
            throw runtime_error("联合时空解压目前只支持一个点集");
        param.prepare_joint_spatiotemporal_coordinates();
    }
    if (param.if_adjust_coord)
    {
        auto adjust_coord_start_time=chrono::steady_clock::now();
        param.adjust_coord();
        adjust_coord_time=chrono::duration<double>(chrono::steady_clock::now()-adjust_coord_start_time).count();
    }
    param.load_or_generate_pred_refs_structure();
    if(param.rbf_parameter_specified||any_of(param.point_set_config_vec.begin(),param.point_set_config_vec.end(),[](const unsz::point_set_config_c &config){return config.codec==unsz::point_set_codec_e::rbf;}))
        param.prepare_rbf_weights();
    bool data_dependent_eb_schedule=param.eb_schedule=="round_point_ratio_dag" || param.eb_schedule=="round_point_ratio_dag_direct" || param.eb_schedule=="round_point_ratio_dag_variance_direct";
    if(!data_dependent_eb_schedule && param.cached_pred_eb_multiplier_vec.size()!=param.point_num)
        param.cached_pred_eb_multiplier_vec=param.generate_pred_eb_multiplier_vec();
    if (param.print_prediction_dag_statistics)
        param.print_prediction_dag_statistics_result();
    double preprocessing_time=chrono::duration<double>(chrono::steady_clock::now()-preprocessing_start_time).count()-adjust_coord_time;
    printf("preprocessing time excluding adjust coord= %f s\n",preprocessing_time);
    if(joint_spatiotemporal)
    {
        using param_type=unsz::param_c<T_coord,float>;
        auto codec=param.point_set_config_vec[0].codec;
        if(codec==unsz::point_set_codec_e::idw)
            param.decompress_joint_spatiotemporal_files(&param_type::template decompress_point_sets<false,unsz::time_dimension_mode_e::spatial,unsz::time_dimension_mode_e::not_applicable,unsz::label_value_execution_e::no_label,unsz::point_set_codec_e::idw>);
        else if(codec==unsz::point_set_codec_e::rbf)
            param.decompress_joint_spatiotemporal_files(&param_type::template decompress_point_sets<false,unsz::time_dimension_mode_e::spatial,unsz::time_dimension_mode_e::not_applicable,unsz::label_value_execution_e::no_label,unsz::point_set_codec_e::rbf>);
        else
            throw runtime_error("单点集联合时空解压只支持IDW或RBF预测器");
        return;
    }
    uint64_t total_point_num=0;
    uint32_t validated_file_num=0;
    double total_squared_error=0,total_decompression_time=0,max_error_ratio=0,max_error=0,max_error_abs_eb=0;
    uint32_t max_error_point_id=0,max_error_point_set_id=0;
    string max_error_data_path;
    float total_min_data=numeric_limits<float>::infinity(),total_max_data=-numeric_limits<float>::infinity();
    using param_type=unsz::param_c<T_coord,float>;
    if(param.point_set_config_vec.size()!=1&&param.point_set_config_vec.size()!=2)
        throw runtime_error("解压只支持一个或两个点集");
    void(param_type::*decompress_field)()=param.point_set_config_vec.size()==1?&param_type::decompress_single_point_set:&param_type::decompress_two_point_sets;
    for (uint32_t file_id=0;file_id<param.original_data_path_vec.size();file_id++)
    {
        param.current_time_step=param.time_step_vec[file_id];
        string compressed_path=param.original_data_path_vec[file_id]+".unsz";
        string decompressed_path=param.original_data_path_vec[file_id]+".unsz.bin";
        param.current_zstd_bs_vec=read_byte_file(compressed_path);

        auto decompression_start_time=chrono::steady_clock::now();
        (param.*decompress_field)();
        param.save_time_reconstruction();
        double decompression_time=chrono::duration<double>(chrono::steady_clock::now()-decompression_start_time).count();
        total_decompression_time+=decompression_time;
        printf("file decompression time= %f s\n",decompression_time);
        FILE *f_decompressed=fopen(decompressed_path.c_str(),"wb");
        if (!f_decompressed)
            throw runtime_error("无法创建解压文件："+decompressed_path);
        if (fwrite(param.current_decompressed_data.data(),sizeof(float),param.point_num,f_decompressed)!=param.point_num)
        {
            fclose(f_decompressed);
            throw runtime_error("写入解压文件失败："+decompressed_path);
        }
        if (fclose(f_decompressed)!=0)
            throw runtime_error("关闭解压文件失败："+decompressed_path);

        vector<float> original_data(param.point_num);
        FILE *f_original=fopen(param.original_data_path_vec[file_id].c_str(),"rb");
        if (!f_original)
        {
            printf("Decompressed %s（原始文件不存在，跳过误差验证）\n",decompressed_path.c_str());
            param.advance_time_prediction_buffers();
            continue;
        }
        if (fread(original_data.data(),sizeof(float),param.point_num,f_original)!=param.point_num)
        {
            fclose(f_original);
            throw runtime_error("读取原始文件失败："+param.original_data_path_vec[file_id]);
        }
        if (fclose(f_original)!=0)
            throw runtime_error("关闭原始文件失败："+param.original_data_path_vec[file_id]);
        double squared_error=0;
        uint32_t primary_point_set_id=0;
        while (param.point_set_point_num_vec[primary_point_set_id]==0)
            primary_point_set_id++;
        for (uint32_t i=0;i<param.point_num;i++)
        {
            double error=(double)original_data[i]-param.current_decompressed_data[i];
            squared_error+=error*error;
            uint32_t point_set_id=param.point_set_id_vec[i];
            double validation_abs_eb=param.point_set_abs_eb_vec[point_set_id];
            double error_ratio=validation_abs_eb>0?abs(error)/validation_abs_eb:(error==0?0:numeric_limits<double>::infinity());
            if (error_ratio>max_error_ratio)
            {
                max_error_ratio=error_ratio;
                max_error=error;
                max_error_abs_eb=validation_abs_eb;
                max_error_point_id=i;
                max_error_point_set_id=point_set_id;
                max_error_data_path=param.original_data_path_vec[file_id];
            }
            if (point_set_id==primary_point_set_id)
            {
                total_min_data=min(total_min_data,original_data[i]);
                total_max_data=max(total_max_data,original_data[i]);
            }
        }
        double mse=squared_error/param.point_num;
        double psnr=mse==0?numeric_limits<double>::infinity():10*log10(pow((double)param.point_set_max_data_vec[primary_point_set_id]-param.point_set_min_data_vec[primary_point_set_id],2)/mse);
        printf("Decompressed %s, psnr= %f\n",decompressed_path.c_str(),psnr);
        total_squared_error+=squared_error;
        total_point_num+=param.point_num;
        validated_file_num++;
        param.advance_time_prediction_buffers();
    }
    printf("全部文件最大误差/用户误差上限= %.9f\n",max_error_ratio);
    if (max_error_ratio>1.01)
        throw runtime_error("全部文件解压完成后误差检查失败：文件="+max_error_data_path+"，点="+to_string(max_error_point_id)+"，点集="+to_string(max_error_point_set_id)+"，误差="+to_string(max_error)+"，绝对误差上限="+to_string(max_error_abs_eb)+"，误差/上限="+to_string(max_error_ratio));
    printf("total decompression time= %f s\n",total_decompression_time);
    if (validated_file_num>0)
    {
        double total_mse=total_squared_error/total_point_num;
        double total_psnr=total_mse==0?numeric_limits<double>::infinity():10*log10(pow((double)total_max_data-total_min_data,2)/total_mse);
        printf("total decompressed psnr= %f\n",total_psnr);
    }
}

template<typename T_coord>
void general_run_decompression(unsz::param_c<T_coord,float> &param,chrono::steady_clock::time_point preprocessing_start_time)
{
    if constexpr (is_same<T_coord,double>::value)
        run_decompression(param,preprocessing_start_time);
    else if (!param.need_double_coord())
        run_decompression(param,preprocessing_start_time);
    else
    {
        unsz::param_c<double,float> double_param(param);
        general_run_decompression(double_param,preprocessing_start_time);
    }
}

namespace unsz
{
int decompress_main(int argc,char **argv)
{
    param_c<float,float> param;
    vector<uint32_t> time_step_vec;
    float error_bound=0;
    unordered_set<string> used_option_set;
    for (int arg_id=1;arg_id<argc;arg_id++)
    {
        string option=argv[arg_id];
        if (!used_option_set.insert(option).second)
            throw runtime_error("参数重复："+option);
        if (option=="-data-dir")
        {
            param.data_dir=read_option_value(argc,argv,arg_id,option);
            param.got_data_dir=true;
        }
        else if (option=="-field")
        {
            param.field_name=read_option_value(argc,argv,arg_id,option);
            param.got_field=true;
        }
        else if (option=="-coordinate-dim")
            param.dim=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-time-step")
        {
            time_step_vec=parse_time_step_vec(read_option_value(argc,argv,arg_id,option));
            param.got_time_step=true;
        }
        else if(option=="-time-step-coordinate-file")
            param.time_step_coordinate_path=read_option_value(argc,argv,arg_id,option);
        else if (option=="-M")
        {
            param.error_mode=read_option_value(argc,argv,arg_id,option);
            error_bound=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
            if (param.error_mode!="REL" && param.error_mode!="ABS")
                throw runtime_error("-M的误差模式必须是REL或ABS："+param.error_mode);
            param.got_error_mode=true;
        }
        else if (option=="-prediction-min-ref-num")
            param.prediction_min_ref_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-prediction-max-ref-num")
            param.prediction_max_ref_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if(option=="-ordering-score-max-ref-num")
            param.ordering_score_max_ref_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if(option=="-ordering-score-min-ref-num")
            param.ordering_score_min_ref_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if(option=="-kd-iterative-query-experimental")
            param.use_kd_iterative_query_experimental=true;
        else if(option=="-kd-scalar-one-reference-query-experimental")
            param.use_kd_scalar_one_reference_query_experimental=true;
        else if(option=="-rbf-solver")
        {
            string solver=read_option_value(argc,argv,arg_id,option);
            if(solver!="general"&&solver!="ldlt")
                throw runtime_error("-rbf-solver必须是general或ldlt");
            param.use_rbf_ldlt=solver=="ldlt";
        }
        else if(option=="-multihead-execution-head-num-experimental")
            param.multihead_execution_head_num_experimental=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if(option=="-prediction-round-num")
        {
            param.prediction_round_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
            if(param.prediction_round_num>63)
                throw runtime_error("-prediction-round-num必须在[1,63]内");
        }
        else if(option=="-round-size-growth-ratio-experimental")
        {
            param.round_size_growth_ratio_experimental=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
            if(param.round_size_growth_ratio_experimental<1 || param.round_size_growth_ratio_experimental>16)
                throw runtime_error("-round-size-growth-ratio-experimental必须在[1,16]内");
        }
        else if (option=="-idw-power")
            param.idw_power=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
        else if(option=="-rbf-kernel")
        {
            param.rbf_parameter_specified=true;
            param.rbf_kernel=read_option_value(argc,argv,arg_id,option);
            if(param.rbf_kernel!="gaussian" && param.rbf_kernel!="exponential" && param.rbf_kernel!="matern32" && param.rbf_kernel!="matern52" && param.rbf_kernel!="inverse_multiquadric")
                throw runtime_error("-rbf-kernel必须是gaussian、exponential、matern32、matern52或inverse_multiquadric");
        }
        else if(option=="-rbf-shape")
        {
            param.rbf_parameter_specified=true;
            param.rbf_shape=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
        }
        else if(option=="-rbf-nugget")
        {
            param.rbf_parameter_specified=true;
            param.rbf_nugget=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
        }
        else if (option=="-ordering-candidate-num")
            param.ordering_candidate_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-dag-cache")
        {
            param.dag_cache_path=read_option_value(argc,argv,arg_id,option);
        }
        else if (option=="-kd-leaf-size")
            param.kd_tree_leaf_size=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-partition-size")
            param.partition_size_experimental=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-partition-num")
            param.partition_num_experimental=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-thread-num")
            param.thread_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-partition-adaptive-predictor-experimental")
            param.use_partition_adaptive_predictor_experimental=true;
        else if (option=="-partition-adaptive-sample-num-experimental")
            param.partition_adaptive_sample_num_experimental=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-partition-adaptive-min-improvement-experimental")
        {
            param.partition_adaptive_min_improvement_experimental=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
            if (param.partition_adaptive_min_improvement_experimental>1)
                throw runtime_error("-partition-adaptive-min-improvement-experimental不能大于1");
        }
        else if (option=="-partition-fitting-candidates-experimental")
        {
            param.partition_fitting_candidates_experimental=read_option_value(argc,argv,arg_id,option);
            if (param.partition_fitting_candidates_experimental!="original" && param.partition_fitting_candidates_experimental!="weighted" && param.partition_fitting_candidates_experimental!="ridge" && param.partition_fitting_candidates_experimental!="mixed" && param.partition_fitting_candidates_experimental!="residual" && param.partition_fitting_candidates_experimental!="extended" && param.partition_fitting_candidates_experimental!="all")
                throw runtime_error("-partition-fitting-candidates-experimental必须是original、weighted、ridge、mixed、residual、extended或all");
        }
        else if(option=="-predictor")
        {
            param.predictor=read_option_value(argc,argv,arg_id,option);
            if(param.predictor!="idw"&&param.predictor!="rbf")
                throw runtime_error("-predictor必须是idw或rbf");
            param.got_predictor=true;
        }
        else if(option=="-point-set-config")
            param.point_set_config_path=read_option_value(argc,argv,arg_id,option);
        else if (option=="-value-flow")
        {
            param.value_flow=read_option_value(argc,argv,arg_id,option);
            if(param.value_flow!="constant_extrapolation"&&param.value_flow!="linear_interpolation"&&param.value_flow!="constant_extrapolation_spatial_residual"&&param.value_flow!="linear_interpolation_spatial_residual"&&param.value_flow!="joint_spatiotemporal")
                throw runtime_error("未知的value flow："+param.value_flow);
        }
        else if (option=="-label-flow")
        {
            param.label_flow=read_option_value(argc,argv,arg_id,option);
            if(param.label_flow!="constant_extrapolation"&&param.label_flow!="endpoint_agreement")
                throw runtime_error("未知的label flow："+param.label_flow);
        }
        else if (option=="-adjust-coord-matrix-file")
            param.adjust_coord_matrix_file=read_option_value(argc,argv,arg_id,option);
        else if (option=="-parallelism-stats")
            param.print_prediction_dag_statistics=true;
        else if(option=="-decompression-stage-time")
            param.print_compression_stage_time=true;
        else if(option=="-label-value-execution")
        {
            param.label_value_execution=read_option_value(argc,argv,arg_id,option);
            if(param.label_value_execution!="staged"&&param.label_value_execution!="fused")
                throw runtime_error("-label-value-execution必须是staged或fused");
        }
        else if (option=="-linear-previous-label-experimental")
            param.use_linear_previous_label_experimental=true;
        else if(option=="-rbf-fallback")
        {
            string fallback=read_option_value(argc,argv,arg_id,option);
            if(fallback!="idw"&&fallback!="nearest")
                throw runtime_error("-rbf-fallback必须是idw或nearest");
            param.rbf_fallback_nearest=fallback=="nearest";
        }
        else if(option=="-ordering-scope")
        {
            param.ordering_scope=read_option_value(argc,argv,arg_id,option);
            if(param.ordering_scope!="local"&&param.ordering_scope!="global")
                throw runtime_error("-ordering-scope必须是local或global");
        }
        else if (option=="-eb-schedule")
        {
            param.eb_schedule=read_option_value(argc,argv,arg_id,option);
            param.eb_schedule_start_ratio=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
            if(param.eb_schedule!="none" && param.eb_schedule!="round" && param.eb_schedule!="point" && param.eb_schedule!="inverse_rank_log2" && param.eb_schedule!="round_point" && param.eb_schedule!="round_ratio" && param.eb_schedule!="round_point_ratio" && param.eb_schedule!="round_point_ratio_dag" && param.eb_schedule!="round_point_ratio_dag_direct" && param.eb_schedule!="round_point_ratio_dag_variance_direct")
                throw runtime_error("未知的EB序列模式："+param.eb_schedule);
            if (param.eb_schedule_start_ratio>1)
                throw runtime_error("-eb-schedule的起始倍率不能大于1");
        }
        else if(option=="-eb-schedule-log2-shift")
            param.eb_schedule_log2_shift=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
        else if(option=="-eb-schedule-max-log2-exponent")
            param.eb_schedule_max_log2_exponent=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
        else
            throw runtime_error("未知参数："+option);
    }
    param.prepare_cli_parameters(time_step_vec,error_bound);
    param.check_original_data_files=false;
    auto preprocessing_start_time=chrono::steady_clock::now();
    param.init();
    general_run_decompression(param,preprocessing_start_time);
    return 0;
}
}

int main(int argc,char **argv)
{
    try
    {
        return unsz::decompress_main(argc,argv);
    }
    catch (const exception &error)
    {
        fprintf(stderr,"错误：%s\n",error.what());
        return 1;
    }
}
