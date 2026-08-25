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
    if (expression.size()>=5 && (expression.front()=='[' || expression.front()=='(') && (expression.back()==']' || expression.back()==')'))
    {
        uint64_t comma_position=expression.find(',');
        if (comma_position==string::npos || expression.find(',',comma_position+1)!=string::npos)
            throw runtime_error("时间步区间必须包含且只包含一个逗号："+expression);
        uint32_t left=parse_uint32(expression.substr(1,comma_position-1),"时间步左端点");
        uint32_t right=parse_uint32(expression.substr(comma_position+1,expression.size()-comma_position-2),"时间步右端点");
        uint64_t first=(uint64_t)left+(expression.front()=='(');
        uint64_t end=(uint64_t)right+(expression.back()==']');
        if (first>=end)
            throw runtime_error("时间步区间为空或端点顺序错误："+expression);
        result.reserve(end-first);
        for (uint64_t time_step=first;time_step<end;time_step++)
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
        throw runtime_error("时间步必须写成区间'[a,b]'、'(a,b]'、'[a,b)'、'(a,b)'或列表'{a1,a2,...}'："+expression);
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
    param.generate_pred_refs_structure();
    double preprocessing_time=chrono::duration<double>(chrono::steady_clock::now()-preprocessing_start_time).count()-adjust_coord_time;
    printf("preprocessing time excluding adjust coord= %f s\n",preprocessing_time);

    uint64_t total_original_bytes=0,total_compressed_bytes=0,total_point_num=0;
    double total_squared_error=0,total_compression_time=0;
    float total_min_data=numeric_limits<float>::infinity(),total_max_data=-numeric_limits<float>::infinity();
    for (auto &original_data_path:param.original_data_path_vec)
    {
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
            else if (strcmp(argv[arg_id],"-label-ref-num")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-label-ref-num后面缺少参数值");
                param.point_set_ref_num=parse_positive_uint32(argv[++arg_id],"-label-ref-num");
            }
            else if (strcmp(argv[arg_id],"-point-set-predictor")==0)
            {
                if (arg_id+1>=argc)
                    throw runtime_error("-point-set-predictor后面缺少参数值");
                param.point_set_predictor=argv[++arg_id];
            }
            else if (strcmp(argv[arg_id],"-round-stats")==0)
                param.print_round_statistics=true;
            else if (strcmp(argv[arg_id],"-label-stats")==0)
                param.print_label_compression_statistics=true;
            else if (strcmp(argv[arg_id],"-adjust-coord")==0)
                param.if_adjust_coord=true;
            else
                throw runtime_error("未知参数："+string(argv[arg_id]));
        }
        if (!param.got_data_dir || !param.got_variable || !param.got_time_step || !param.got_error_mode)
            throw runtime_error("用法：s1_compress -data-dir <目录> -variable <变量名> -time-step <区间或列表> -M <REL|ABS> <误差> [-adjust-coord] [-max-ref-num <数量>] [-actual-ref-num <数量>] [-label-ref-num <数量>] [-point-set-predictor <名称>] [-round-stats] [-label-stats]");
        if (param.point_set_predictor!="inverse_proportion" && param.point_set_predictor!="majority" && param.point_set_predictor!="continuous_actual" && param.point_set_predictor!="continuous_reflect")
            throw runtime_error("未知的点集标签预测器："+param.point_set_predictor);
        if (!param.data_dir.empty() && param.data_dir.back()!='/')
            param.data_dir+="/";
        param.coord_path=param.data_dir+"coords.dat";
        for (uint32_t time_step:time_step_vec)
            param.original_data_path_vec.push_back(param.data_dir+param.variable_name+".dat."+to_string(time_step));
        param.dim=3;
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
