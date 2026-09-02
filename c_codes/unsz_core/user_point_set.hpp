#pragma once

// 用户在这里按优先级定义点集。第一个返回true的分类函数决定点集编号，全部返回false的数据属于最后一个点集。
template<typename T_data>
bool classify0(T_data value)
{
    return value>=-1000;
}

template<typename T_data>
vector<point_set_classifier_t<T_data>> get_user_point_set_classifier_vec()
{
    return {classify0<T_data>};
}

// 配置数量必须等于分类函数数量加1。use_default和use_global分别表示沿用命令行的-predictor和-M。
inline vector<point_set_config_c> get_user_point_set_config_vec()
{
    return {
        {point_set_codec_e::use_default,point_set_error_mode_e::use_global,0},
        {point_set_codec_e::use_default,point_set_error_mode_e::use_global,0}
    };
}
