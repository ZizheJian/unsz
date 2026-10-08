#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_cli_parameters(const vector<uint32_t>&selected_time_step_vec,T_data error_bound)
    {
        if(!got_data_dir||!got_field||!got_time_step)
            throw runtime_error("必须设置-data-dir、field和-time-step");
        if(data_dir.empty())
            throw runtime_error("-data-dir的参数值不能为空");
        if(field_name_vec.empty())
            field_name_vec.push_back(field_name);
        if(any_of(field_name_vec.begin(),field_name_vec.end(),[](const string&name){return name.empty();}))
            throw runtime_error("field名称不能为空");
        bool got_point_set_config=!point_set_config_path.empty();
        if(got_error_mode==got_point_set_config)
            throw runtime_error("必须且只能在-M和-point-set-config之间选择一个");
        if(thread_num>1&&partition_size==0&&partition_num==0)
            throw runtime_error("多线程模式需要指定partition数量或大小");
        if(data_dir.back()!='/')
            data_dir+="/";
        coord_path=data_dir+"coords.dat";
        configured_time_step_vec=selected_time_step_vec;
        time_step_vec=configured_time_step_vec;
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
        configured_point_set_rule_vec=point_set_rule_vec;
        configured_point_set_config_vec=point_set_config_vec;
        point_set_source_id_vec.resize(point_set_config_vec.size());
        iota(point_set_source_id_vec.begin(),point_set_source_id_vec.end(),0);
        if(point_set_config_vec.size()==1)
            label_flow="none";
        if(point_set_config_vec.size()==1&&point_set_config_vec[0].codec==point_set_codec_e::constant)
            value_flow="none";
        if(label_flow=="anchor_interpolation_stride_8"&&point_set_config_vec.size()==1)
            throw runtime_error("label锚点内插流程要求field包含两个点集");
        if(point_set_config_vec.size()==1)
            resolve_single_point_set_time_mode();
        configured_label_flow=label_flow;
        configured_value_flow=value_flow;
        activate_field(0);
        prediction_round_num=1;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::activate_field(uint32_t field_id)
    {
        if(field_id>=field_name_vec.size())
            throw runtime_error("field下标越界");
        field_name=field_name_vec[field_id];
        point_set_rule_vec=configured_point_set_rule_vec;
        point_set_config_vec=configured_point_set_config_vec;
        point_set_source_id_vec.resize(point_set_config_vec.size());
        iota(point_set_source_id_vec.begin(),point_set_source_id_vec.end(),0);
        time_step_vec=configured_time_step_vec;
        original_data_path_vec.clear();
        original_data_path_vec.reserve(time_step_vec.size());
        for(uint32_t time_step:time_step_vec)
            original_data_path_vec.push_back(data_dir+field_name+".dat."+to_string(time_step));
        label_flow=configured_label_flow;
        value_flow=configured_value_flow;
        compression_file_point_set_min_data_vec.clear();
        compression_file_point_set_max_data_vec.clear();
        compression_file_point_set_abs_eb_vec.clear();
        compression_file_point_set_codec_vec.clear();
        if(field_id<compression_field_statistics_vec.size())
        {
            const compression_field_statistics_c<T_data>&statistics=compression_field_statistics_vec[field_id];
            compression_file_point_set_min_data_vec=statistics.point_set_min_data_vec;
            compression_file_point_set_max_data_vec=statistics.point_set_max_data_vec;
            compression_file_point_set_abs_eb_vec=statistics.point_set_abs_eb_vec;
            compression_file_point_set_codec_vec=statistics.point_set_codec_vec;
            if(!statistics.point_set_configs.empty())
            {
                point_set_config_vec=statistics.point_set_configs;
                point_set_rule_vec=statistics.point_set_rules;
                point_set_source_id_vec=statistics.point_set_source_ids;
            }
        }
        if(point_set_config_vec.size()==1)
            label_flow="none";
        if(point_set_config_vec.size()==1&&point_set_config_vec[0].codec==point_set_codec_e::constant)
            value_flow="none";
        reconstructed_time_data_vec.clear();
        reconstructed_time_point_set_id_vec.clear();
        temporal_value_dependency_vec.clear();
        temporal_label_dependency_vec.clear();
        temporal_value_remaining_outdegree_vec.clear();
        temporal_label_remaining_outdegree_vec.clear();
    }
}
