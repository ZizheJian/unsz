#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <queue>
#include <numeric>
#include <random>
#include <cstdio>
#include <cstring>
#include <unordered_set>
#include "huffman.hpp"
#include "parsers.hpp"
#include "third_party/zstd/lib/zstd.h"
#include "unsz_core/unsz.hpp"

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

template<typename T_coord>
void run_compression(unsz::param_c<T_coord,float> &param,chrono::steady_clock::time_point preprocessing_start_time)
{
    double adjust_coord_time=0;
    if (param.if_adjust_coord)
    {
        auto adjust_coord_start_time=chrono::steady_clock::now();
        param.adjust_coord();
        adjust_coord_time=chrono::duration<double>(chrono::steady_clock::now()-adjust_coord_start_time).count();
    }
    param.load_or_generate_pred_refs_structure();
    if (param.cross_time_spatial_point_num_experimental>0)
    {
        uint32_t spatial_point_num=param.cross_time_spatial_point_num_experimental;
        if (param.point_num%spatial_point_num!=0)
            throw runtime_error("时空联合点数不能被空间点数整除");
        uint64_t cross_time_ref_num=0,total_ref_num=0,target_with_cross_time_ref_num=0;
        for (auto &pred_refs:param.pred_refs_vec)
        {
            uint32_t pred_original_id=param.use_prediction_order_layout_experimental?param.prediction_id_to_original_id[pred_refs.first]:pred_refs.first;
            bool has_cross_time_ref=false;
            for (int ref_id:pred_refs.second)
            {
                uint32_t ref_original_id=param.use_prediction_order_layout_experimental?param.prediction_id_to_original_id[ref_id]:ref_id;
                bool cross_time_ref=pred_original_id!=ref_original_id && pred_original_id%spatial_point_num==ref_original_id%spatial_point_num;
                cross_time_ref_num+=cross_time_ref;
                total_ref_num++;
                has_cross_time_ref|=cross_time_ref;
            }
            target_with_cross_time_ref_num+=has_cross_time_ref;
        }
        printf("cross_time_reference_targets=%llu, target_ratio=%f, reference_edges=%llu, edge_ratio=%f\n",(unsigned long long)target_with_cross_time_ref_num,(double)target_with_cross_time_ref_num/param.point_num,(unsigned long long)cross_time_ref_num,total_ref_num==0?0:(double)cross_time_ref_num/total_ref_num);
    }
    if (param.print_prediction_dag_statistics)
        param.print_prediction_dag_statistics_result();
    double preprocessing_time=chrono::duration<double>(chrono::steady_clock::now()-preprocessing_start_time).count()-adjust_coord_time;
    printf("preprocessing time excluding adjust coord= %f s\n",preprocessing_time);

    uint64_t total_original_bytes=0,total_compressed_bytes=0,total_point_num=0;
    double total_squared_error=0,total_compression_time=0,max_error_ratio=0,max_error=0,max_error_abs_eb=0;
    uint32_t max_error_point_id=0,max_error_point_set_id=0;
    string max_error_data_path;
    float total_min_data=numeric_limits<float>::infinity(),total_max_data=-numeric_limits<float>::infinity();
    for (uint32_t file_id=0;file_id<param.original_data_path_vec.size();file_id++)
    {
        auto &original_data_path=param.original_data_path_vec[file_id];
        printf("Compressing %s\n",original_data_path.c_str());
        param.compressed_data_path_vec.push_back(original_data_path+".unsz");
        param.decompressed_data_path_vec.push_back(original_data_path+".unsz.bin");
        param.current_data.resize(param.point_num);
        FILE *f_data=fopen(original_data_path.c_str(),"rb");
        if (!f_data)
            throw runtime_error("找不到原始数据文件："+original_data_path);
        if (fread(param.current_data.data(),sizeof(float),param.point_num,f_data)!=param.point_num)
        {
            fclose(f_data);
            throw runtime_error("读取原始数据文件失败："+original_data_path);
        }
        if (fclose(f_data)!=0)
            throw runtime_error("关闭原始数据文件失败："+original_data_path);
        auto compression_start_time=chrono::steady_clock::now();
        param.compress();
        double compression_time=chrono::duration<double>(chrono::steady_clock::now()-compression_start_time).count();
        total_compression_time+=compression_time;
        printf("file compression time= %f s\n",compression_time);
        if (param.use_prediction_order_layout_experimental)
        {
            auto result_reorder_start_time=chrono::steady_clock::now();
            param.restore_compression_result_to_original_order();
            printf("compression result reorder time excluded from compression= %f s\n",chrono::duration<double>(chrono::steady_clock::now()-result_reorder_start_time).count());
        }
        if (param.cross_time_predictor_experimental=="previous")
        {
            param.previous_time_decompressed_data=param.current_decompressed_data;
            param.previous_time_point_set_id_vec=param.point_set_id_vec;
        }
        if (!param.hybrid_selector_path.empty())
        {
            if (param.hybrid_selector_vec.size()!=param.point_num)
                throw runtime_error("没有生成完整的混合预测器selector");
            vector<uint8_t> selector=param.hybrid_selector_vec;
            if (param.use_prediction_order_layout_experimental)
            {
                vector<uint8_t> original_order_selector(param.point_num);
                for (uint32_t prediction_id=0;prediction_id<param.point_num;prediction_id++)
                    original_order_selector[param.prediction_id_to_original_id[prediction_id]]=selector[prediction_id];
                selector.swap(original_order_selector);
            }
            string selector_path=param.hybrid_selector_path;
            if (param.original_data_path_vec.size()>1)
                selector_path+="."+to_string(file_id);
            FILE *selector_file=fopen(selector_path.c_str(),"wb");
            if (!selector_file)
                throw runtime_error("无法创建混合预测器selector文件："+selector_path);
            if (fwrite(selector.data(),1,selector.size(),selector_file)!=selector.size())
            {
                fclose(selector_file);
                throw runtime_error("写入混合预测器selector文件失败："+selector_path);
            }
            if (fclose(selector_file)!=0)
                throw runtime_error("关闭混合预测器selector文件失败："+selector_path);
            printf("hybrid selector path=%s\n",selector_path.c_str());
        }
        total_original_bytes+=(uint64_t)param.point_num*sizeof(float);
        total_compressed_bytes+=param.current_zstd_bs_vec.size();
        total_point_num+=param.point_num;
        uint32_t primary_point_set_id=0;
        while (param.point_set_point_num_vec[primary_point_set_id]==0)
            primary_point_set_id++;
        for (uint32_t i=0;i<param.point_num;i++)
        {
            double error=(double)param.current_data[i]-param.current_decompressed_data[i];
            total_squared_error+=error*error;
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
                max_error_data_path=original_data_path;
            }
            if (param.point_set_id_vec[i]==primary_point_set_id)
            {
                total_min_data=min(total_min_data,param.current_data[i]);
                total_max_data=max(total_max_data,param.current_data[i]);
            }
        }
        FILE *f_compressed=fopen(param.compressed_data_path_vec.back().c_str(),"wb");
        if (!f_compressed)
            throw runtime_error("无法创建压缩文件："+param.compressed_data_path_vec.back());
        if (fwrite(param.current_zstd_bs_vec.data(),1,param.current_zstd_bs_vec.size(),f_compressed)!=param.current_zstd_bs_vec.size())
        {
            fclose(f_compressed);
            throw runtime_error("写入压缩文件失败："+param.compressed_data_path_vec.back());
        }
        if (fclose(f_compressed)!=0)
            throw runtime_error("关闭压缩文件失败："+param.compressed_data_path_vec.back());
    }
    printf("全部文件最大误差/用户误差上限= %.9f\n",max_error_ratio);
    if (max_error_ratio>1.1)
        throw runtime_error("全部文件压缩完成后误差检查失败：文件="+max_error_data_path+"，点="+to_string(max_error_point_id)+"，点集="+to_string(max_error_point_set_id)+"，误差="+to_string(max_error)+"，绝对误差上限="+to_string(max_error_abs_eb)+"，误差/上限="+to_string(max_error_ratio));
    double total_cr=(double)total_original_bytes/total_compressed_bytes;
    double total_mse=total_squared_error/total_point_num;
    double total_psnr=total_mse==0?numeric_limits<double>::infinity():10*log10(pow((double)total_max_data-total_min_data,2)/total_mse);
    double total_br=sizeof(float)*8/total_cr;
    printf("total cr= %f, total br= %f, total psnr= %f\n",total_cr,total_br,total_psnr);
    printf("total compression time= %f s\n",total_compression_time);

}

template<typename T_coord>
void general_run_compression(unsz::param_c<T_coord,float> &param,chrono::steady_clock::time_point preprocessing_start_time)
{
    if constexpr (is_same<T_coord,double>::value)
    {
        printf("使用double处理坐标和距离\n");
        run_compression(param,preprocessing_start_time);
    }
    else if (!param.need_double_coord())
    {
        printf("使用float处理坐标和距离\n");
        run_compression(param,preprocessing_start_time);
    }
    else
    {
        printf("坐标跨度较大，将坐标和距离计算从float升级为double\n");
        unsz::param_c<double,float> double_param(param);
        general_run_compression(double_param,preprocessing_start_time);
    }
}

namespace unsz
{
    int compress_main(int argc,char **argv)
    {
        param_c<float,float> param;
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
            else if (strcmp(argv[arg_id],"-inverse-distance-power")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-inverse-distance-power后面缺少参数值");
                param.inverse_distance_power=parse_positive_float(argv[++arg_id],"-inverse-distance-power");
            }
            else if (strcmp(argv[arg_id],"-label-ref-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-label-ref-num后面缺少参数值");
                param.point_set_ref_num=parse_positive_uint32(argv[++arg_id],"-label-ref-num");
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
            else if (strcmp(argv[arg_id],"-order-candidate-num-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-order-candidate-num-experimental后面缺少参数值");
                param.order_candidate_num_experimental=parse_positive_uint32(argv[++arg_id],"-order-candidate-num-experimental");
            }
            else if (strcmp(argv[arg_id],"-prediction-structure")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-prediction-structure后面缺少参数值");
                param.prediction_structure_path=argv[++arg_id];
            }
            else if (strcmp(argv[arg_id],"-kd-leaf-size")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-kd-leaf-size后面缺少参数值");
                param.kd_tree_leaf_size=parse_positive_uint32(argv[++arg_id],"-kd-leaf-size");
            }
            else if (strcmp(argv[arg_id],"-independent-block-size-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-independent-block-size-experimental后面缺少参数值");
                param.independent_block_size_experimental=parse_positive_uint32(argv[++arg_id],"-independent-block-size-experimental");
            }
            else if (strcmp(argv[arg_id],"-independent-block-num-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-independent-block-num-experimental后面缺少参数值");
                param.independent_block_num_experimental=parse_positive_uint32(argv[++arg_id],"-independent-block-num-experimental");
            }
            else if (strcmp(argv[arg_id],"-block-adaptive-predictor-experimental")==0)
                param.use_block_adaptive_predictor_experimental=true;
            else if (strcmp(argv[arg_id],"-pointwise-oracle-statistics-experimental")==0)
                param.print_pointwise_oracle_statistics_experimental=true;
            else if (strcmp(argv[arg_id],"-block-adaptive-sample-num-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-block-adaptive-sample-num-experimental后面缺少参数值");
                param.block_adaptive_sample_num_experimental=parse_positive_uint32(argv[++arg_id],"-block-adaptive-sample-num-experimental");
            }
            else if (strcmp(argv[arg_id],"-block-fitting-candidates-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-block-fitting-candidates-experimental后面缺少模式");
                param.block_fitting_candidates_experimental=argv[++arg_id];
                if (param.block_fitting_candidates_experimental!="original" && param.block_fitting_candidates_experimental!="weighted" && param.block_fitting_candidates_experimental!="ridge" && param.block_fitting_candidates_experimental!="mixed" && param.block_fitting_candidates_experimental!="residual" && param.block_fitting_candidates_experimental!="extended" && param.block_fitting_candidates_experimental!="all")
                    throw runtime_error("-block-fitting-candidates-experimental必须是original、weighted、ridge、mixed、residual、extended或all");
            }
            else if (strcmp(argv[arg_id],"-multi-huffman-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-multi-huffman-experimental后面缺少模式");
                param.multi_huffman_mode_experimental=argv[++arg_id];
                if (param.multi_huffman_mode_experimental!="none" && param.multi_huffman_mode_experimental!="round" && param.multi_huffman_mode_experimental!="block" && param.multi_huffman_mode_experimental!="block_round")
                    throw runtime_error("-multi-huffman-experimental必须是none、round、block或block_round");
            }
            else if (strcmp(argv[arg_id],"-point-set-predictor")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-point-set-predictor后面缺少参数值");
                param.point_set_predictor=argv[++arg_id];
            }
            else if (strcmp(argv[arg_id],"-cross-time-predictor-experimental")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-cross-time-predictor-experimental后面缺少模式");
                param.cross_time_predictor_experimental=argv[++arg_id];
                if (param.cross_time_predictor_experimental!="previous")
                    throw runtime_error("-cross-time-predictor-experimental目前只支持previous");
            }
            else if (strcmp(argv[arg_id],"-predictor")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-predictor后面缺少参数值");
                param.predictor=argv[++arg_id];
                if (param.predictor!="inverse_proportion" && param.predictor!="fitting" && param.predictor!="hybrid_loo")
                    throw runtime_error("-predictor必须是inverse_proportion、fitting或hybrid_loo");
            }
            else if (strcmp(argv[arg_id],"-hybrid-selector-path")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-hybrid-selector-path后面缺少参数值");
                param.hybrid_selector_path=argv[++arg_id];
            }
            else if (strcmp(argv[arg_id],"-round-stats")==0)
                param.print_round_statistics=true;
            else if (strcmp(argv[arg_id],"-label-stats")==0)
                param.print_label_compression_statistics=true;
            else if (strcmp(argv[arg_id],"-adjust-coord")==0)
                param.if_adjust_coord=true;
            else if (strcmp(argv[arg_id],"-adjust-coord-matrix")==0)
            {
                if (arg_id+6>=argc)
                    throw runtime_error("-adjust-coord-matrix后面需要6个下三角矩阵元素");
                param.adjust_coord_matrix_parameter_vec.clear();
                for (uint32_t parameter_id=0;parameter_id<6;parameter_id++)
                {
                    double value=stod(argv[++arg_id]);
                    if (!isfinite(value))
                        throw runtime_error("坐标变换矩阵元素必须是有限数");
                    param.adjust_coord_matrix_parameter_vec.push_back(value);
                }
                param.if_adjust_coord=true;
            }
            else if (strcmp(argv[arg_id],"-compression-stage-time")==0)
                param.print_compression_stage_time=true;
            else if (strcmp(argv[arg_id],"-parallelism-stats")==0)
                param.print_prediction_dag_statistics=true;
            else if (strcmp(argv[arg_id],"-prediction-order-layout-experimental")==0)
                param.use_prediction_order_layout_experimental=true;
            else if (strcmp(argv[arg_id],"-normalized-weight-experimental")==0)
                param.use_normalized_weight_experimental=true;
            else if (strcmp(argv[arg_id],"-prediction-order-label-experimental")==0)
                param.use_prediction_order_label_experimental=true;
            else if (strcmp(argv[arg_id],"-predictability-order-experimental")==0)
                param.use_predictability_order_experimental=true;
            else if (strcmp(argv[arg_id],"-forward-mean-distance-only-order-experimental")==0)
                param.use_forward_mean_distance_only_order_experimental=true;
            else if (strcmp(argv[arg_id],"-forward-s2-only-order-experimental")==0)
                param.use_forward_s2_only_order_experimental=true;
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
            else if (strcmp(argv[arg_id],"-reverse-reference-reuse-order-experimental")==0)
                param.use_reverse_reference_reuse_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reverse-s2-only-order-experimental")==0)
                param.use_reverse_s2_only_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reverse-direction-only-order-experimental")==0)
                param.use_reverse_direction_only_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reverse-direction-order-experimental")==0)
                param.use_reverse_direction_order_experimental=true;
            else if (strcmp(argv[arg_id],"-reverse-direction-distribution-order-experimental")==0)
                param.use_reverse_direction_distribution_order_experimental=true;
            else if (strcmp(argv[arg_id],"-eb-schedule-experimental")==0)
            {
                if (arg_id+2>=argc)
                    throw runtime_error("-eb-schedule-experimental后面需要模式和起始倍率");
                param.eb_schedule_experimental=argv[++arg_id];
                param.eb_schedule_start_ratio=parse_positive_float(argv[++arg_id],"-eb-schedule-experimental");
                if (param.eb_schedule_experimental!="none" && param.eb_schedule_experimental!="round" && param.eb_schedule_experimental!="point" && param.eb_schedule_experimental!="round_point" && param.eb_schedule_experimental!="round_ratio" && param.eb_schedule_experimental!="round_point_ratio")
                    throw runtime_error("-eb-schedule-experimental的模式必须是none、round、point、round_point、round_ratio或round_point_ratio");
                if (param.eb_schedule_start_ratio>1)
                    throw runtime_error("-eb-schedule-experimental的起始倍率不能大于1");
            }
            else
                throw runtime_error("未知参数："+string(argv[arg_id]));
        }
        if (!param.got_data_dir || !param.got_variable || !param.got_time_step || !param.got_error_mode)
            throw runtime_error("用法：s1_compress -data-dir <目录> -variable <变量名> -time-step <区间或列表> -M <REL|ABS> <误差> [-dim <维数>] [-adjust-coord] [-cross-time-predictor-experimental previous] [-predictor <inverse_proportion|fitting|hybrid_loo>] [-max-ref-num <数量>] [-actual-ref-num <数量>] [-inverse-distance-power <指数>] [-label-ref-num <数量>] [-point-set-predictor <名称>] [-prediction-structure <路径>] [-kd-leaf-size <数量>] [-independent-block-size-experimental <数量>] [-independent-block-num-experimental <数量>] [-block-adaptive-predictor-experimental] [-pointwise-oracle-statistics-experimental] [-block-adaptive-sample-num-experimental <数量>] [-block-fitting-candidates-experimental <original|weighted|ridge|mixed|residual|extended|all>] [-multi-huffman-experimental <none|round|block|block_round>] [-round-stats] [-label-stats] [-compression-stage-time] [-parallelism-stats] [-prediction-order-layout-experimental] [-normalized-weight-experimental] [-prediction-order-label-experimental] [-predictability-order-experimental] [-forward-mean-distance-only-order-experimental] [-forward-s2-only-order-experimental] [-forward-direction-only-order-experimental] [-forward-mean-distance-s2-order-experimental] [-forward-mean-distance-direction-order-experimental] [-reverse-predictability-order-experimental] [-reverse-distribution-order-experimental] [-reverse-mean-distance-only-order-experimental] [-reverse-reference-reuse-order-experimental] [-reverse-s2-only-order-experimental] [-reverse-direction-only-order-experimental] [-reverse-direction-order-experimental] [-reverse-direction-distribution-order-experimental] [-d-only-cache-refill-threshold-experimental <数量>] [-order-candidate-num-experimental <数量>] [-eb-schedule-experimental <模式> <倍率>]");
        if (param.point_set_predictor!="inverse_proportion" && param.point_set_predictor!="majority" && param.point_set_predictor!="continuous_actual" && param.point_set_predictor!="continuous_reflect")
            throw runtime_error("未知的点集标签预测器："+param.point_set_predictor);
        if (!param.hybrid_selector_path.empty() && param.predictor!="hybrid_loo")
            throw runtime_error("-hybrid-selector-path只能和-predictor hybrid_loo一起使用");
        if (param.use_block_adaptive_predictor_experimental && param.independent_block_size_experimental==0 && param.independent_block_num_experimental==0)
            throw runtime_error("block自适应预测器需要指定独立空间块");
        if (param.print_pointwise_oracle_statistics_experimental && !param.use_block_adaptive_predictor_experimental)
            throw runtime_error("逐点oracle统计需要启用block自适应预测器");
        if (!param.data_dir.empty() && param.data_dir.back()!='/')
            param.data_dir+="/";
        param.coord_path=param.data_dir+"coords.dat";
        for (uint32_t time_step:time_step_vec)
            param.original_data_path_vec.push_back(param.data_dir+param.variable_name+".dat."+to_string(time_step));
        if (param.error_mode=="REL")
        {
            param.rel_eb=error_bound;
            param.got_rel_eb=true;
        }
        else
        {
            param.abs_eb=error_bound;
            param.got_abs_eb=true;
        }
        param.cross_time_compress=false;
        if (param.actual_ref_num>param.max_ref_num)
            throw runtime_error("actual_ref_num不能大于max_ref_num");
        if (param.point_set_ref_num>param.max_ref_num)
            param.point_set_ref_num=param.max_ref_num;

        auto preprocessing_start_time=chrono::steady_clock::now();
        param.init();
        general_run_compression(param,preprocessing_start_time);

    
    
    

    

    

        // //////// 预测、量化 ////////
        // vector<float> compressed_data_vec=data_vec;
        // vector<int64_t> qb_vec;
        // qb_vec.reserve(node_num);
        // clear_valid_num(block_vec);
        // for (int i=0;i<round_num;i++)
        // {
        //     printf("Compressing round %d\n",i);
        //     if (i==0)
        //     {
        //         for (int j=0;j<round_ids[i].size();j++)
        //         {
        //             int64_t qb=round(compressed_data_vec[round_ids[i][j]]/(2*abs_eb));
        //             qb_vec.push_back(qb);
        //             compressed_data_vec[round_ids[i][j]]=qb*2*abs_eb;
        //             block_vec[id_vec2[round_ids[i][j]]].valid_num=1;
        //         }
        //         update_valid_num(block_vec,0);
        //     }
        //     else
        //     {
        //         for (int j=0;j<round_ids[i].size();j++)
        //         {
        //             int tgt_id=round_ids[i][j];
        //             priority_queue<pair<float,int>> nearest_vec;
        //             find_nearest(nearest_vec,4,block_vec,0,coord_vec,coord_vec.data()+tgt_id*3,true);
        //             vector<float> ref_data,ref_dist;
        //             while (!nearest_vec.empty())
        //             {
        //                 int ref_id=nearest_vec.top().second;
        //                 nearest_vec.pop();
        //                 ref_data.push_back(compressed_data_vec[ref_id]);
        //                 ref_dist.push_back(sqrt(get_dist_to_node(coord_vec.data()+tgt_id*3,coord_vec.data()+ref_id*3)));
        //             }
        //             float total_weight=0;
        //             for (auto dist:ref_dist)
        //                 total_weight+=pow(dist,-3.61);
        //             float pred=0;
        //             for (int k=0;k<ref_data.size();k++)
        //                 pred+=ref_data[k]*pow(ref_dist[k],-3.61)/total_weight;
        //             int64_t qb=round((compressed_data_vec[tgt_id]-pred)/(2*abs_eb));
        //             qb_vec.push_back(qb);
        //             compressed_data_vec[tgt_id]=pred+qb*2*abs_eb;
        //             block_vec[id_vec2[tgt_id]].valid_num=1;
        //             update_valid_num(block_vec,0);
        //         }
        //     }
        // }
        // printf("qb_vec.size()=%zu\n",qb_vec.size());
        // printf("node_num=%d\n",node_num);
        // for (int i=0;i<node_num;i++)
        //     if (abs(data_vec[i]-compressed_data_vec[i])>abs_eb*1.001)
        //         throw runtime_error("压缩后误差超过绝对误差阈值");
        return 0;
    }
}

int main(int argc,char **argv)
{
    try
    {
        return unsz::compress_main(argc,argv);
    }
    catch (const exception &error)
    {
        fprintf(stderr,"错误：%s\n",error.what());
        return 1;
    }
}
