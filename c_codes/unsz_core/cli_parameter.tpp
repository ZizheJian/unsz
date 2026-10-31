#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_cli_parameters(const vector<uint32_t>&selected_slice_vec,T_data error_bound)
    {
        if(!got_data_dir||!got_field||(!got_slice&&tensor_metadata_path.empty()))
            throw runtime_error("必须设置-data-dir、field，并通过-slice或-tensor-metadata定义切片");
        if(data_dir.empty())
            throw runtime_error("-data-dir的参数值不能为空");
        if(field_name_vec.empty())
            field_name_vec.push_back(field_name);
        if(any_of(field_name_vec.begin(),field_name_vec.end(),[](const string&name){return name.empty();}))
            throw runtime_error("field名称不能为空");
        bool got_label_config=!label_config_path.empty();
        if(got_error_mode==got_label_config)
            throw runtime_error("必须且只能在-M和-label-config之间选择一个");
        if(thread_num>1&&partition_num==0)
            throw runtime_error("多线程模式需要指定partition数量");
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
        if(!tensor_metadata_path.empty()&&got_slice)
            throw runtime_error("-slice和-tensor-metadata不能同时使用");
        if(!tensor_metadata_path.empty()&&!slice_coordinate_path.empty())
            throw runtime_error("tensor metadata已经定义轴坐标，不能再使用-slice-coordinate-file");
        if(tensor_metadata_path.empty())
        {
            configured_slice_vec=selected_slice_vec;
            slice_vec=configured_slice_vec;
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
        configure_labels();
        configured_label_rule_vec=label_rule_vec;
        configured_label_config_vec=label_config_vec;
        label_source_id_vec.resize(label_config_vec.size());
        iota(label_source_id_vec.begin(),label_source_id_vec.end(),0);
        if(label_config_vec.size()==1)
            label_pipeline="independent";
        if(label_config_vec.size()==1&&label_config_vec[0].codec==label_codec_e::constant)
            value_pipeline="independent";
        if(label_pipeline=="anchor_interpolation"&&label_config_vec.size()==1)
            throw runtime_error("label锚点内插流程要求field包含两个标签");
        configured_label_pipeline=label_pipeline;
        configured_value_pipeline=value_pipeline;
        activate_field(0);
        // global和forward排序只有一轮，local排序使用默认的8轮
        if(ordering_scope!="local")
            prediction_round_num=1;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::activate_field(uint32_t field_id)
    {
        if(field_id>=field_name_vec.size())
            throw runtime_error("field下标越界");
        field_name=field_name_vec[field_id];
        label_rule_vec=configured_label_rule_vec;
        label_config_vec=configured_label_config_vec;
        label_source_id_vec.resize(label_config_vec.size());
        iota(label_source_id_vec.begin(),label_source_id_vec.end(),0);
        current_field_uses_tensor=false;
        current_tensor_shape={{0,0}};
        current_tensor_axis_coordinates={};
        current_tensor_slice_dag_key.clear();
        current_active_level_count_path.clear();
        current_active_level_count_offset=0;
        current_active_level_counts=nullptr;
        tensor_pipeline_selected=false;
        auto tensor_config_iterator=tensor_field_config_map.find(field_name);
        if(tensor_config_iterator==tensor_field_config_map.end())
            slice_vec=configured_slice_vec;
        else
            activate_tensor_field(tensor_config_iterator->second);
        original_data_path_vec.clear();
        original_data_path_vec.reserve(slice_vec.size());
        for(uint32_t slice:slice_vec)
            original_data_path_vec.push_back(data_dir+field_name+".dat."+to_string(slice));
        label_pipeline=configured_label_pipeline;
        value_pipeline=configured_value_pipeline;
        compression_file_label_min_data_vec.clear();
        compression_file_label_max_data_vec.clear();
        compression_file_label_abs_eb_vec.clear();
        compression_file_label_codec_vec.clear();
        if(label_config_vec.size()==1)
            label_pipeline="independent";
        if(label_config_vec.size()==1&&label_config_vec[0].codec==label_codec_e::constant)
            value_pipeline="independent";
        if(slice_vec.size()==1)
        {
            label_pipeline="independent";
            value_pipeline="independent";
        }
        else if(label_config_vec.size()==1)
            resolve_single_label_pipeline();
        reconstructed_slice_data_vec.clear();
        reconstructed_slice_label_id_vec.clear();
        value_reference_slices_vec.clear();
        label_reference_slices_vec.clear();
        value_slice_remaining_outdegree_vec.clear();
        label_slice_remaining_outdegree_vec.clear();
    }
}
