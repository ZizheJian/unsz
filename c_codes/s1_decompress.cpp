#include <string>
#include <vector>
#include <limits>
#include <cmath>
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
    param.generate_pred_refs_structure();
    uint64_t total_point_num=0;
    uint32_t validated_file_num=0;
    double total_squared_error=0;
    float total_min_data=numeric_limits<float>::infinity(),total_max_data=-numeric_limits<float>::infinity();
    for (uint32_t file_id=0;file_id<param.original_data_path_vec.size();file_id++)
    {
        string compressed_path=param.original_data_path_vec[file_id]+".unsz";
        string decompressed_path=param.original_data_path_vec[file_id]+".unsz.bin";
        param.current_zstd_bs_vec=read_byte_file(compressed_path);

        param.decompress();
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
            if (abs(error)>param.point_set_abs_eb_vec[point_set_id]*1.001)
                throw runtime_error("解压结果超过点集"+to_string(point_set_id)+"的误差上限");
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
        else if (option=="-label-ref-num")
            param.point_set_ref_num=parse_positive_uint32(read_option_value(argc,argv,arg_id,option),option);
        else if (option=="-point-set-predictor")
            param.point_set_predictor=read_option_value(argc,argv,arg_id,option);
        else if (option=="-adjust-coord")
            param.if_adjust_coord=true;
        else
            throw runtime_error("未知参数："+option);
    }
    if (!param.got_data_dir || !param.got_variable || !param.got_time_step || !param.got_error_mode)
        throw runtime_error("用法：s1_decompress -data-dir <目录> -variable <变量名> -time-step <区间或列表> -M <REL|ABS> <误差> [-adjust-coord] [-max-ref-num <数量>] [-actual-ref-num <数量>] [-label-ref-num <数量>] [-point-set-predictor <名称>]");
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
    param.check_original_data_files=false;
    if (param.actual_ref_num>param.max_ref_num)
        throw runtime_error("actual_ref_num不能大于max_ref_num");
    if (param.point_set_ref_num>param.max_ref_num)
        param.point_set_ref_num=param.max_ref_num;
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
