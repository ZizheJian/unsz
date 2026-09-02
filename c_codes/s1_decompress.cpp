#include <string>
#include <vector>
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
void run_decompression(unsz::param_c<T_coord,float> &param)
{
    if (param.if_adjust_coord)
        param.adjust_coord();
    param.load_or_generate_pred_refs_structure();
    if (param.print_prediction_dag_statistics)
        param.print_prediction_dag_statistics_result();
    uint64_t total_point_num=0;
    uint32_t validated_file_num=0;
    double total_squared_error=0,total_decompression_time=0,max_error_ratio=0,max_error=0,max_error_abs_eb=0;
    uint32_t max_error_point_id=0,max_error_point_set_id=0;
    string max_error_data_path;
    float total_min_data=numeric_limits<float>::infinity(),total_max_data=-numeric_limits<float>::infinity();
    for (uint32_t file_id=0;file_id<param.original_data_path_vec.size();file_id++)
    {
        param.current_time_step=param.time_step_vec[file_id];
        string compressed_path=param.original_data_path_vec[file_id]+".unsz";
        string decompressed_path=param.original_data_path_vec[file_id]+".unsz.bin";
        param.current_zstd_bs_vec=read_byte_file(compressed_path);

        auto decompression_start_time=chrono::steady_clock::now();
        param.decompress();
        if (param.cross_time_predictor_experimental=="previous")
        {
            param.previous_time_decompressed_data=param.current_decompressed_data;
            param.previous_time_point_set_id_vec=param.point_set_id_vec;
        }
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
    }
    printf("全部文件最大误差/用户误差上限= %.9f\n",max_error_ratio);
    if (max_error_ratio>1.1)
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
void general_run_decompression(unsz::param_c<T_coord,float> &param)
{
    if constexpr (is_same<T_coord,double>::value)
        run_decompression(param);
    else if (!param.need_double_coord())
        run_decompression(param);
    else
    {
        unsz::param_c<double,float> double_param(param);
        general_run_decompression(double_param);
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
        else if (option=="-variable")
        {
            param.variable_name=read_option_value(argc,argv,arg_id,option);
            param.got_variable=true;
        }
        else if (option=="-dim")
            param.dim=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-time-step")
        {
            time_step_vec=parse_time_step_vec(read_option_value(argc,argv,arg_id,option));
            param.got_time_step=true;
        }
        else if (option=="-M")
        {
            param.error_mode=read_option_value(argc,argv,arg_id,option);
            error_bound=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
            if (param.error_mode!="REL" && param.error_mode!="ABS")
                throw runtime_error("-M的误差模式必须是REL或ABS："+param.error_mode);
            param.got_error_mode=true;
        }
        else if (option=="-max-ref-num")
            param.max_ref_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-actual-ref-num")
            param.actual_ref_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if(option=="-predict-round-num")
        {
            param.predict_round_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
            if(param.predict_round_num>63)
                throw runtime_error("-predict-round-num必须在[1,63]内");
        }
        else if(option=="-round-size-growth-ratio-experimental")
        {
            param.round_size_growth_ratio_experimental=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
            if(param.round_size_growth_ratio_experimental<1 || param.round_size_growth_ratio_experimental>16)
                throw runtime_error("-round-size-growth-ratio-experimental必须在[1,16]内");
        }
        else if (option=="-inverse-distance-power")
            param.inverse_distance_power=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-d-only-cache-refill-threshold-experimental")
            param.d_only_cache_refill_threshold_experimental=parse_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-cross-time-spatial-point-num-experimental")
            param.cross_time_spatial_point_num_experimental=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-order-candidate-num-experimental")
            param.order_candidate_num_experimental=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-prediction-structure")
            param.prediction_structure_path=read_option_value(argc,argv,arg_id,option);
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
        else if (option=="-partition-fitting-candidates-experimental")
        {
            param.partition_fitting_candidates_experimental=read_option_value(argc,argv,arg_id,option);
            if (param.partition_fitting_candidates_experimental!="original" && param.partition_fitting_candidates_experimental!="weighted" && param.partition_fitting_candidates_experimental!="ridge" && param.partition_fitting_candidates_experimental!="mixed" && param.partition_fitting_candidates_experimental!="residual" && param.partition_fitting_candidates_experimental!="extended" && param.partition_fitting_candidates_experimental!="all")
                throw runtime_error("-partition-fitting-candidates-experimental必须是original、weighted、ridge、mixed、residual、extended或all");
        }
        else if (option=="-multi-huffman-experimental")
        {
            param.multi_huffman_mode_experimental=read_option_value(argc,argv,arg_id,option);
            if (param.multi_huffman_mode_experimental!="none" && param.multi_huffman_mode_experimental!="round" && param.multi_huffman_mode_experimental!="partition" && param.multi_huffman_mode_experimental!="partition_round")
                throw runtime_error("-multi-huffman-experimental必须是none、round、partition或partition_round");
        }
        else if (option=="-point-set-predictor")
            param.point_set_predictor=read_option_value(argc,argv,arg_id,option);
        else if (option=="-cross-time-predictor-experimental")
        {
            param.cross_time_predictor_experimental=read_option_value(argc,argv,arg_id,option);
            if (param.cross_time_predictor_experimental!="previous" && param.cross_time_predictor_experimental!="interpolation")
                throw runtime_error("-cross-time-predictor-experimental目前只支持previous或interpolation");
        }
        else if (option=="-adjust-coord")
            param.if_adjust_coord=true;
        else if (option=="-adjust-coord-matrix-file")
            param.adjust_coord_matrix_path=read_option_value(argc,argv,arg_id,option);
        else if (option=="-adjust-coord-matrix")
        {
            uint32_t parameter_num=param.dim*(param.dim+1)/2;
            if ((uint64_t)arg_id+parameter_num>=(uint64_t)argc)
                throw runtime_error("-adjust-coord-matrix后面需要"+to_string(parameter_num)+"个下三角矩阵元素；请先设置-dim");
            param.adjust_coord_matrix_parameter_vec.clear();
            for (uint32_t parameter_id=0;parameter_id<parameter_num;parameter_id++)
            {
                double value=stod(argv[++arg_id]);
                if (!isfinite(value))
                    throw runtime_error("坐标变换矩阵元素必须是有限数");
                param.adjust_coord_matrix_parameter_vec.push_back(value);
            }
            param.if_adjust_coord=true;
        }
        else if (option=="-parallelism-stats")
            param.print_prediction_dag_statistics=true;
        else if (option=="-prediction-geometry-stats")
            param.print_prediction_geometry_statistics=true;
        else if (option=="-prediction-order-layout-experimental")
            param.use_prediction_order_layout_experimental=true;
        else if (option=="-normalized-weight-experimental")
            param.use_normalized_weight_experimental=true;
        else if (option=="-prediction-order-label-experimental")
            param.use_prediction_order_label_experimental=true;
        else if (option=="-predictability-order-experimental")
            param.use_predictability_order_experimental=true;
        else if (option=="-forward-mean-distance-only-order-experimental")
            param.use_forward_mean_distance_only_order_experimental=true;
        else if (option=="-forward-s2-only-order-experimental")
            param.use_forward_s2_only_order_experimental=true;
        else if (option=="-forward-direction-only-order-experimental")
            param.use_forward_direction_only_order_experimental=true;
        else if (option=="-forward-mean-distance-s2-order-experimental")
            param.use_forward_mean_distance_s2_order_experimental=true;
        else if (option=="-forward-mean-distance-direction-order-experimental")
            param.use_forward_mean_distance_direction_order_experimental=true;
        else if (option=="-reverse-predictability-order-experimental")
            param.use_reverse_predictability_order_experimental=true;
        else if (option=="-reverse-distribution-order-experimental")
            param.use_reverse_distribution_order_experimental=true;
        else if (option=="-reverse-mean-distance-only-order-experimental")
            param.use_reverse_mean_distance_only_order_experimental=true;
        else if (option=="-reverse-reference-reuse-order-experimental")
            param.use_reverse_reference_reuse_order_experimental=true;
        else if (option=="-reverse-s2-only-order-experimental")
            param.use_reverse_s2_only_order_experimental=true;
        else if (option=="-reverse-direction-only-order-experimental")
            param.use_reverse_direction_only_order_experimental=true;
        else if (option=="-reverse-direction-order-experimental")
            param.use_reverse_direction_order_experimental=true;
        else if (option=="-reverse-direction-distribution-order-experimental")
            param.use_reverse_direction_distribution_order_experimental=true;
        else if (option=="-global-order-before-round-split-experimental")
            param.use_global_order_before_round_split_experimental=true;
        else if (option=="-eb-schedule-experimental")
        {
            param.eb_schedule_experimental=read_option_value(argc,argv,arg_id,option);
            param.eb_schedule_start_ratio=parse_positive_float(read_option_value(argc,argv,arg_id,option),option);
            if(param.eb_schedule_experimental!="none" && param.eb_schedule_experimental!="round" && param.eb_schedule_experimental!="point" && param.eb_schedule_experimental!="round_point" && param.eb_schedule_experimental!="round_ratio" && param.eb_schedule_experimental!="round_point_ratio" && param.eb_schedule_experimental!="round_point_ratio_dag" && param.eb_schedule_experimental!="round_point_ratio_dag_direct" && param.eb_schedule_experimental!="round_point_ratio_dag_variance_direct")
                throw runtime_error("未知的实验性EB序列模式："+param.eb_schedule_experimental);
            if (param.eb_schedule_start_ratio>1)
                throw runtime_error("-eb-schedule-experimental的起始倍率不能大于1");
        }
        else
            throw runtime_error("未知参数："+option);
    }
    bool has_point_set_error_bound=false;
    for (const point_set_config_c &config:param.point_set_config_vec)
        has_point_set_error_bound|=config.error_mode!=point_set_error_mode_e::use_global;
    if (!param.got_data_dir || !param.got_variable || !param.got_time_step || (!param.got_error_mode && !has_point_set_error_bound))
        throw runtime_error("用法：s1_decompress -data-dir <目录> -variable <变量名> -time-step <区间或列表> [-M <REL|ABS> <误差>] [-dim <维数>] [-adjust-coord] [-adjust-coord-matrix-file <路径>] [-cross-time-predictor-experimental previous] [-max-ref-num <数量>] [-actual-ref-num <数量>] [-predict-round-num <数量>] [-round-size-growth-ratio-experimental <比例>] [-inverse-distance-power <指数>] [-point-set-predictor <名称>] [-prediction-structure <路径>] [-kd-leaf-size <数量>] [-partition-size <数量>] [-partition-num <数量>] [-thread-num <数量>] [-partition-fitting-candidates-experimental <original|weighted|ridge|mixed|residual|extended|all>] [-multi-huffman-experimental <none|round|partition|partition_round>] [-parallelism-stats] [-prediction-geometry-stats] [-prediction-order-layout-experimental] [-normalized-weight-experimental] [-prediction-order-label-experimental] [-predictability-order-experimental] [-forward-mean-distance-only-order-experimental] [-forward-s2-only-order-experimental] [-forward-direction-only-order-experimental] [-forward-mean-distance-s2-order-experimental] [-forward-mean-distance-direction-order-experimental] [-reverse-predictability-order-experimental] [-reverse-distribution-order-experimental] [-reverse-mean-distance-only-order-experimental] [-reverse-reference-reuse-order-experimental] [-reverse-s2-only-order-experimental] [-reverse-direction-only-order-experimental] [-reverse-direction-order-experimental] [-reverse-direction-distribution-order-experimental] [-d-only-cache-refill-threshold-experimental <数量>] [-order-candidate-num-experimental <数量>] [-eb-schedule-experimental <模式> <倍率>]");
    if (param.point_set_predictor!="inverse_proportion" && param.point_set_predictor!="majority" && param.point_set_predictor!="continuous_actual" && param.point_set_predictor!="continuous_reflect")
        throw runtime_error("未知的点集标签预测器："+param.point_set_predictor);
    if(param.thread_num>1 && param.partition_size_experimental==0 && param.partition_num_experimental==0)
        throw runtime_error("多线程模式需要指定partition数量或大小");
    if (!param.data_dir.empty() && param.data_dir.back()!='/')
        param.data_dir+="/";
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
    param.cross_time_compress=false;
    param.check_original_data_files=false;
    param.load_adjust_coord_matrix_file();
    if (param.actual_ref_num>param.max_ref_num)
        throw runtime_error("actual_ref_num不能大于max_ref_num");
    param.init();
    general_run_decompression(param);
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
