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
        if(point_set_config_vec.size()==1)
            label_flow="none";
        if(label_flow=="anchor_interpolation_stride_8"&&point_set_config_vec.size()==1)
            throw runtime_error("label锚点内插流程要求field包含两个点集");
        if(point_set_config_vec.size()==1)
            resolve_single_point_set_time_mode();
        prediction_round_num=1;
    }
}
