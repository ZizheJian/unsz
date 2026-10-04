#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_cli_parameters(const vector<uint32_t>&selected_time_step_vec,T_data error_bound)
    {
        if(!got_data_dir||!got_field||!got_time_step)
            throw runtime_error("必须设置-data-dir、-field和-time-step");
        if(data_dir.empty()||field_name.empty())
            throw runtime_error("-data-dir和-field的参数值不能为空");
        bool got_point_set_config=!point_set_config_path.empty();
        if(got_error_mode==got_point_set_config)
            throw runtime_error("必须且只能在-M和-point-set-config之间选择一个");
        if(got_error_mode&&!got_predictor)
            throw runtime_error("使用-M时必须通过-predictor显式指定预测器");
        if(thread_num>1&&partition_size==0&&partition_num==0)
            throw runtime_error("多线程模式需要指定partition数量或大小");
        if(data_dir.back()!='/')
            data_dir+="/";
        coord_path=data_dir+"coords.dat";
        original_data_path_vec.clear();
        original_data_path_vec.reserve(selected_time_step_vec.size());
        for(uint32_t time_step:selected_time_step_vec)
            original_data_path_vec.push_back(data_dir+field_name+".dat."+to_string(time_step));
        time_step_vec=selected_time_step_vec;
        if(time_step_vec.size()==1)
        {
            label_flow="none";
            value_flow="none";
        }
        if(got_error_mode&&error_mode=="REL")
        {
            rel_eb=error_bound;
            got_rel_eb=true;
        }
        else if(got_error_mode)
        {
            abs_eb=error_bound;
            got_abs_eb=true;
        }
        configure_point_sets();
        if(label_flow=="anchor_interpolation_stride_8"&&point_set_config_vec.size()==1)
            throw runtime_error("label锚点内插流程要求field包含两个点集");
        if(point_set_config_vec.size()==1)
            resolve_single_point_set_time_mode();
        if(prediction_min_ref_num>prediction_max_ref_num)
            throw runtime_error("prediction_min_ref_num不能大于prediction_max_ref_num");
        bool uses_rbf=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c&config){return config.codec==point_set_codec_e::rbf;});
        if(uses_rbf&&prediction_min_ref_num!=prediction_max_ref_num)
            throw runtime_error("RBF要求prediction_min_ref_num与prediction_max_ref_num相等");
        if(ordering_candidate_num==0)
            throw runtime_error("必须显式设置正数ordering_candidate_num");
        if(ordering_score_max_ref_num==0)
            throw runtime_error("必须显式设置正数ordering_score_max_ref_num");
        if(ordering_score_min_ref_num==0)
            throw runtime_error("必须显式设置正数ordering_score_min_ref_num");
        if(ordering_score_min_ref_num>ordering_score_max_ref_num)
            throw runtime_error("ordering_score_min_ref_num不能大于ordering_score_max_ref_num");
        if(ordering_scope=="global"&&got_prediction_round_num)
            throw runtime_error("global排序不使用-prediction-round-num");
        if(ordering_scope=="local"&&!got_prediction_round_num)
            throw runtime_error("local排序必须显式设置-prediction-round-num");
        if(ordering_scope=="global")
            prediction_round_num=1;
        if(uses_rbf&&prediction_min_ref_num<2)
            throw runtime_error("RBF的prediction_min_ref_num不能小于2");
    }
}
