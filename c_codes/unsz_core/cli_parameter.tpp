#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_cli_parameters(const vector<uint32_t>&selected_time_step_vec,T_data error_bound)
    {
        if(!got_data_dir||!got_field||(!got_time_step&&tensor_metadata_path.empty()))
            throw runtime_error("必须设置-data-dir、field，并通过-time-step或-tensor-metadata定义切片");
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
        if(field_worker_num==1&&!field_worker_id_vec.empty())
            throw runtime_error("-field-worker-map只能和大于1的-field-worker-num一起使用");
        if(field_worker_num>1)
        {
            if(field_name_vec.size()<2)
                throw runtime_error("多个field worker至少需要两个field");
            if(field_worker_id_vec.size()!=field_name_vec.size())
                throw runtime_error("-field-worker-map必须为每个field提供一个worker编号");
            if(any_of(field_worker_id_vec.begin(),field_worker_id_vec.end(),[this](uint32_t worker_id){return worker_id>=field_worker_num;}))
                throw runtime_error("-field-worker-map包含越界的worker编号");
        }
        if(data_dir.back()!='/')
            data_dir+="/";
        if(output_dir.empty())
            output_dir=data_dir;
        else
        {
            filesystem::create_directories(output_dir);
            if(output_dir.back()!='/')
                output_dir+="/";
        }
        coord_path=data_dir+"coords.dat";
        if(!tensor_metadata_path.empty()&&got_time_step)
            throw runtime_error("-time-step和-tensor-metadata不能同时使用");
        if(!tensor_metadata_path.empty()&&!time_step_coordinate_path.empty())
            throw runtime_error("tensor metadata已经定义轴坐标，不能再使用-time-step-coordinate-file");
        if(tensor_metadata_path.empty())
        {
            configured_time_step_vec=selected_time_step_vec;
            time_step_vec=configured_time_step_vec;
        }
        else
            load_tensor_metadata();
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
        current_field_uses_tensor=false;
        current_tensor_shape={{0,0}};
        current_tensor_axis_coordinates={};
        current_tensor_dependency_plan_key.clear();
        current_active_level_count_path.clear();
        current_active_level_count_offset=0;
        current_active_level_counts=nullptr;
        tensor_flow_selected=false;
        auto tensor_config_iterator=tensor_field_config_map.find(field_name);
        if(tensor_config_iterator==tensor_field_config_map.end())
            time_step_vec=configured_time_step_vec;
        else
            activate_tensor_field(tensor_config_iterator->second);
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
        if(point_set_config_vec.size()==1)
            label_flow="none";
        if(point_set_config_vec.size()==1&&point_set_config_vec[0].codec==point_set_codec_e::constant)
            value_flow="none";
        if(time_step_vec.size()==1)
        {
            label_flow="none";
            value_flow="none";
        }
        else if(point_set_config_vec.size()==1)
            resolve_single_point_set_time_mode();
        reconstructed_time_data_vec.clear();
        reconstructed_time_point_set_id_vec.clear();
        temporal_value_dependency_vec.clear();
        temporal_label_dependency_vec.clear();
        temporal_value_remaining_outdegree_vec.clear();
        temporal_label_remaining_outdegree_vec.clear();
    }
}
