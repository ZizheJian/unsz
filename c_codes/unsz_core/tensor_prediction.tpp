#pragma once

namespace unsz
{
    inline string slice_dag_name(const string&pipeline)
    {
        if(pipeline=="extrapolation_spatial_residual")return "extrapolation";
        if(pipeline=="interpolation_spatial_residual")return "interpolation";
        if(pipeline=="anchor_interpolation_spatial_residual")return "anchor_interpolation";
        return pipeline;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::precompute_all_tensor_slice_dags()
    {
        for(const string&field_name:field_name_vec)
        {
            const tensor_field_config_c&config=tensor_field_config_map.at(field_name);
            if(!tensor_slice_dag_cache.count(config.slice_dag_key))
                tensor_slice_dag_cache.emplace(config.slice_dag_key,generate_all_tensor_slice_dags(config.shape,config.axis_coordinates));
        }
    }

    inline reference_slices_c build_reference_slices(const string&pipeline,uint32_t slice,uint32_t slice_index,int32_t left_source,int32_t right_source,int32_t&previous_source)
    {
        reference_slices_c dependency;
        bool anchor_pipeline=pipeline=="anchor_interpolation"||pipeline=="anchor_interpolation_spatial_residual";
        bool constant_pipeline=pipeline=="extrapolation"||pipeline=="extrapolation_spatial_residual";
        bool interpolation_pipeline=pipeline=="interpolation"||pipeline=="interpolation_spatial_residual";
        if(anchor_pipeline)
        {
            bool first_anchor=slice==0;
            bool regular_anchor=slice%8==0;
            bool can_interpolate=left_source>=0&&right_source>=0;
            if(first_anchor)
                previous_source=(int32_t)slice_index;
            else if(regular_anchor||!can_interpolate)
            {
                if(previous_source>=0)
                {
                    dependency.source_indices[0]=previous_source;
                    dependency.source_num=1;
                }
                previous_source=(int32_t)slice_index;
            }
            else
            {
                dependency.source_indices={{left_source,right_source}};
                dependency.source_num=2;
            }
            return dependency;
        }
        if(constant_pipeline)
        {
            if(previous_source>=0)
            {
                dependency.source_indices[0]=previous_source;
                dependency.source_num=1;
            }
            previous_source=(int32_t)slice_index;
            return dependency;
        }
        if(interpolation_pipeline&&left_source>=0&&right_source>=0)
        {
            dependency.source_indices={{left_source,right_source}};
            dependency.source_num=2;
        }
        return dependency;
    }

    template<typename T_coord,typename T_data>
    slice_dag_c param_c<T_coord,T_data>::generate_slice_dag(const string&pipeline) const
    {
        slice_dag_c plan;
        plan.processing_order=generate_slice_coding_order(get_slice_order(pipeline));
        plan.dependencies.assign(slice_vec.size(),{});
        map<uint32_t,uint32_t> processed_slice_indices;
        int32_t previous_source=-1;
        for(uint32_t physical_slice_index:plan.processing_order)
        {
            uint32_t slice=slice_vec[physical_slice_index];
            int32_t left_source=-1,right_source=-1;
            auto right_iterator=processed_slice_indices.lower_bound(slice);
            if(right_iterator!=processed_slice_indices.end())
                right_source=(int32_t)right_iterator->second;
            if(right_iterator!=processed_slice_indices.begin())
            {
                auto left_iterator=right_iterator;
                --left_iterator;
                left_source=(int32_t)left_iterator->second;
            }
            reference_slices_c&dependency=plan.dependencies[physical_slice_index];
            dependency=build_reference_slices(pipeline,slice,physical_slice_index,left_source,right_source,previous_source);
            if(dependency.source_num==1)
                dependency.source_weights[0]=1;
            else if(dependency.source_num==2)
            {
                uint32_t left_index=(uint32_t)dependency.source_indices[0],right_index=(uint32_t)dependency.source_indices[1];
                uint32_t left_slice=slice_vec[left_index],right_slice=slice_vec[right_index];
                double left_coordinate=slice_coordinate_vec.empty()?(double)left_slice:(double)slice_coordinate_vec[left_slice];
                double right_coordinate=slice_coordinate_vec.empty()?(double)right_slice:(double)slice_coordinate_vec[right_slice];
                double current_coordinate=slice_coordinate_vec.empty()?(double)slice:(double)slice_coordinate_vec[slice];
                double right_weight=(current_coordinate-left_coordinate)/(right_coordinate-left_coordinate);
                dependency.source_weights={{1-right_weight,right_weight}};
            }
            processed_slice_indices.emplace(slice,physical_slice_index);
        }
        return plan;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::precompute_slice_dags()
    {
        slice_dag_cache.clear();
        if(current_field_uses_tensor)
        {
            if(current_tensor_slice_dag_key.empty())
                throw runtime_error("当前field缺少tensor依赖计划缓存键");
            if(!tensor_slice_dag_cache.count(current_tensor_slice_dag_key))
                tensor_slice_dag_cache.emplace(current_tensor_slice_dag_key,generate_all_tensor_slice_dags(current_tensor_shape,current_tensor_axis_coordinates));
            return;
        }
        vector<uint32_t> sorted_slices=slice_vec;
        sort(sorted_slices.begin(),sorted_slices.end());
        bool anchor_compatible=!sorted_slices.empty()&&sorted_slices.front()==0;
        for(uint32_t slice_index=1;slice_index<sorted_slices.size();slice_index++)
            anchor_compatible&=sorted_slices[slice_index]==sorted_slices[slice_index-1]+1;
        constexpr const char*pipeline_names[]={"independent","extrapolation","interpolation","anchor_interpolation"};
        for(const char*pipeline:pipeline_names)
        {
            bool anchor_pipeline=string(pipeline).find("anchor_interpolation")!=string::npos;
            if(!anchor_pipeline||anchor_compatible)
                slice_dag_cache.emplace(pipeline,generate_slice_dag(pipeline));
        }
    }

    template<typename T_coord,typename T_data>
    const slice_dag_c&param_c<T_coord,T_data>::get_tensor_slice_dag(const tensor_pipeline_c&pipeline) const
    {
        auto cache_iterator=tensor_slice_dag_cache.find(current_tensor_slice_dag_key);
        if(cache_iterator==tensor_slice_dag_cache.end())
            throw runtime_error("没有预生成当前形状的tensor依赖计划");
        auto iterator=cache_iterator->second.find(tensor_slice_dag_id(pipeline));
        if(iterator==cache_iterator->second.end())
            throw runtime_error("没有预生成的tensor依赖计划："+tensor_pipeline_name(pipeline));
        return iterator->second;
    }

    template<typename T_coord,typename T_data>
    const slice_dag_c&param_c<T_coord,T_data>::get_slice_dag(const string&pipeline) const
    {
        auto iterator=slice_dag_cache.find(slice_dag_name(pipeline));
        if(iterator==slice_dag_cache.end())
            throw runtime_error("没有预生成的slice DAG："+pipeline);
        return iterator->second;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_reference_slices()
    {
        uint32_t slice_num=(uint32_t)slice_vec.size();
        value_reference_slices_vec.assign(slice_num,{});
        label_reference_slices_vec.assign(slice_num,{});
        value_slice_remaining_outdegree_vec.assign(slice_num,0);
        label_slice_remaining_outdegree_vec.assign(slice_num,0);
        reconstructed_slice_data_vec.assign(slice_num,{});
        reconstructed_slice_label_id_vec.assign(slice_num,{});
        previous_slice_value_source=nullptr;
        previous_slice_value_label_source=nullptr;
        previous_slice_label_source=nullptr;
        map<uint32_t,uint32_t> configured_index_by_slice,current_index_by_slice;
        for(uint32_t index=0;index<configured_slice_vec.size();index++)
            configured_index_by_slice.emplace(configured_slice_vec[index],index);
        for(uint32_t index=0;index<slice_vec.size();index++)
            current_index_by_slice.emplace(slice_vec[index],index);
        auto map_plan_dependencies=[&](const slice_dag_c&plan,vector<reference_slices_c>&dependencies)
        {
            for(uint32_t current_index=0;current_index<slice_num;current_index++)
            {
                uint32_t configured_index=configured_index_by_slice.at(slice_vec[current_index]);
                dependencies[current_index]=plan.dependencies[configured_index];
                reference_slices_c&dependency=dependencies[current_index];
                for(uint32_t source_id=0;source_id<dependency.source_num;source_id++)
                {
                    uint32_t source_configured_index=(uint32_t)dependency.source_indices[source_id];
                    uint32_t source_slice=configured_slice_vec[source_configured_index];
                    uint32_t source_current_index=current_index_by_slice.at(source_slice);
                    if(source_current_index>=current_index)
                        throw runtime_error("slice DAG的参考切片尚未重建");
                    dependency.source_indices[source_id]=(int32_t)source_current_index;
                }
            }
        };
        if(current_field_uses_tensor)
        {
            if(!tensor_pipeline_selected)
                throw runtime_error("尚未选择tensor pipeline");
            const slice_dag_c&plan=get_tensor_slice_dag(selected_tensor_pipeline);
            map_plan_dependencies(plan,value_reference_slices_vec);
            map_plan_dependencies(plan,label_reference_slices_vec);
        }
        else
        {
            map_plan_dependencies(get_slice_dag(value_pipeline),value_reference_slices_vec);
            map_plan_dependencies(get_slice_dag(label_pipeline),label_reference_slices_vec);
        }
        for(uint32_t slice_index=0;slice_index<slice_num;slice_index++)
        {
            reference_slices_c&value_dependency=value_reference_slices_vec[slice_index];
            reference_slices_c&label_dependency=label_reference_slices_vec[slice_index];
            for(uint32_t source_index=0;source_index<value_dependency.source_num;source_index++)
            {
                uint32_t source_slice_index=(uint32_t)value_dependency.source_indices[source_index];
                value_slice_remaining_outdegree_vec[source_slice_index]++;
                if(label_config_vec.size()>1)
                    label_slice_remaining_outdegree_vec[source_slice_index]++;
            }
            for(uint32_t source_index=0;source_index<label_dependency.source_num;source_index++)
                label_slice_remaining_outdegree_vec[(uint32_t)label_dependency.source_indices[source_index]]++;
        }
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_tensor_prediction()
    {
        previous_slice_value_source=nullptr;
        previous_slice_value_label_source=nullptr;
        previous_slice_label_source=nullptr;
        previous_slice_value_available=false;
        previous_slice_label_available=false;
        tensor_value_interpolation_available=false;
        tensor_label_interpolation_available=false;
        tensor_weights={{0,0}};
        const prediction_framework_c&framework=current_prediction_framework;
        if(framework.value_mode==slice_prediction_mode_e::extrapolation_spatial_residual||framework.value_mode==slice_prediction_mode_e::interpolation_spatial_residual)
        {
            current_slice_spatial_residual_data.resize(point_num);
            current_slice_spatial_residual_valid.assign(point_num,0);
        }
        const reference_slices_c&value_dependency=value_reference_slices_vec[current_slice_index];
        const reference_slices_c&label_dependency=label_reference_slices_vec[current_slice_index];
        if((framework.value_mode==slice_prediction_mode_e::extrapolation||framework.value_mode==slice_prediction_mode_e::extrapolation_spatial_residual)&&value_dependency.source_num==1)
        {
            uint32_t source_index=(uint32_t)value_dependency.source_indices[0];
            previous_slice_value_source=&reconstructed_slice_data_vec[source_index];
            previous_slice_value_available=previous_slice_value_source->size()==point_num;
            if(label_config_vec.size()>1)
            {
                previous_slice_value_label_source=&reconstructed_slice_label_id_vec[source_index];
                previous_slice_value_available&=previous_slice_value_label_source->size()==point_num;
            }
        }
        if(framework.label_mode==slice_prediction_mode_e::extrapolation&&label_dependency.source_num==1)
        {
            uint32_t source_index=(uint32_t)label_dependency.source_indices[0];
            previous_slice_label_source=&reconstructed_slice_label_id_vec[source_index];
            previous_slice_label_available=previous_slice_label_source->size()==point_num;
        }
        if((framework.value_mode==slice_prediction_mode_e::interpolation||framework.value_mode==slice_prediction_mode_e::interpolation_spatial_residual)&&value_dependency.source_num==2)
        {
            uint32_t left_source=(uint32_t)value_dependency.source_indices[0],right_source=(uint32_t)value_dependency.source_indices[1];
            const vector<T_data>&left_data=reconstructed_slice_data_vec[left_source],&right_data=reconstructed_slice_data_vec[right_source];
            tensor_value_interpolation_available=left_data.size()==point_num&&right_data.size()==point_num;
            if(label_config_vec.size()>1)
            {
                const vector<uint8_t>&left_label_ids=reconstructed_slice_label_id_vec[left_source],&right_label_ids=reconstructed_slice_label_id_vec[right_source];
                tensor_value_interpolation_available&=left_label_ids.size()==point_num&&right_label_ids.size()==point_num;
            }
            tensor_weights[0]=(T_calc)value_dependency.source_weights[0];
            tensor_weights[1]=(T_calc)value_dependency.source_weights[1];
        }
        if(framework.label_mode==slice_prediction_mode_e::label_interpolation_spatial_residual&&label_dependency.source_num==2)
        {
            uint32_t left_source=(uint32_t)label_dependency.source_indices[0],right_source=(uint32_t)label_dependency.source_indices[1];
            tensor_label_interpolation_available=reconstructed_slice_label_id_vec[left_source].size()==point_num&&reconstructed_slice_label_id_vec[right_source].size()==point_num;
        }
    }

    template<typename T_coord,typename T_data>
    template<slice_prediction_mode_e label_mode>
    bool param_c<T_coord,T_data>::predict_tensor_label(uint32_t storage_id,uint32_t label_channel_id,uint32_t&prediction)
    {
        auto get_binary_label=[label_channel_id,storage_id](const vector<uint8_t>&label_ids)
        {
            return label_ids[storage_id]==label_channel_id?0U:1U;
        };
        if constexpr(label_mode==slice_prediction_mode_e::label_interpolation_spatial_residual)
        {
            if(!tensor_label_interpolation_available)
                return false;
            const reference_slices_c&dependency=label_reference_slices_vec[current_slice_index];
            const vector<uint8_t>&left_label_ids=reconstructed_slice_label_id_vec[(uint32_t)dependency.source_indices[0]],&right_label_ids=reconstructed_slice_label_id_vec[(uint32_t)dependency.source_indices[1]];
            uint32_t left_label=get_binary_label(left_label_ids),right_label=get_binary_label(right_label_ids);
            if(left_label!=right_label)
                return false;
            prediction=left_label;
            return true;
        }
        if(!previous_slice_label_available)
            return false;
        if constexpr(label_mode==slice_prediction_mode_e::extrapolation)
        {
            prediction=get_binary_label(*previous_slice_label_source);
            return true;
        }
        return false;
    }

    template<typename T_coord,typename T_data>
    template<slice_prediction_mode_e value_mode>
    bool param_c<T_coord,T_data>::predict_tensor_value(uint32_t storage_id,uint32_t label_id,T_data&prediction)
    {
        if constexpr(value_mode==slice_prediction_mode_e::extrapolation||value_mode==slice_prediction_mode_e::extrapolation_spatial_residual)
        {
            if(!previous_slice_value_available)
                return false;
            if(label_config_vec.size()>1&&(*previous_slice_value_label_source)[storage_id]!=label_id)
                return false;
            prediction=(*previous_slice_value_source)[storage_id];
            return true;
        }
        if constexpr(value_mode==slice_prediction_mode_e::interpolation||value_mode==slice_prediction_mode_e::interpolation_spatial_residual)
        {
            if(!tensor_value_interpolation_available)
                return false;
            const reference_slices_c&dependency=value_reference_slices_vec[current_slice_index];
            uint32_t left_source=(uint32_t)dependency.source_indices[0],right_source=(uint32_t)dependency.source_indices[1];
            const vector<T_data>&left_data=reconstructed_slice_data_vec[left_source],&right_data=reconstructed_slice_data_vec[right_source];
            if(label_config_vec.size()>1)
            {
                const vector<uint8_t>&left_label_ids=reconstructed_slice_label_id_vec[left_source],&right_label_ids=reconstructed_slice_label_id_vec[right_source];
                if(left_label_ids[storage_id]!=label_id||right_label_ids[storage_id]!=label_id)
                    return false;
            }
            prediction=(T_data)(tensor_weights[0]*(T_calc)left_data[storage_id]+tensor_weights[1]*(T_calc)right_data[storage_id]);
            return true;
        }
        return false;
    }

    // 单标签时跨slice预测对整个slice要么都可用、要么都不可用。返回是否可用，以及按存放位置计算跨slice预测值的函数。
    // 算式与predict_tensor_value相同，只是把参考slice的查找移到逐点循环之外。
    template<typename T_coord,typename T_data>
    template<slice_prediction_mode_e value_mode>
    auto param_c<T_coord,T_data>::get_single_label_tensor_predictor() const
    {
        constexpr bool interpolation=value_mode==slice_prediction_mode_e::interpolation||value_mode==slice_prediction_mode_e::interpolation_spatial_residual;
        constexpr bool extrapolation=value_mode==slice_prediction_mode_e::extrapolation||value_mode==slice_prediction_mode_e::extrapolation_spatial_residual;
        bool available=false;
        const vector<T_data>*left_data=nullptr,*right_data=nullptr;
        if constexpr(extrapolation)
        {
            available=previous_slice_value_available;
            left_data=previous_slice_value_source;
        }
        else if constexpr(interpolation)
        {
            available=tensor_value_interpolation_available;
            if(available)
            {
                const reference_slices_c&dependency=value_reference_slices_vec[current_slice_index];
                left_data=&reconstructed_slice_data_vec[(uint32_t)dependency.source_indices[0]];
                right_data=&reconstructed_slice_data_vec[(uint32_t)dependency.source_indices[1]];
            }
        }
        array<T_calc,2> weights=tensor_weights;
        auto predict=[left_data,right_data,weights](uint32_t storage_id)->T_data
        {
            if constexpr(interpolation)
                return (T_data)(weights[0]*(T_calc)(*left_data)[storage_id]+weights[1]*(T_calc)(*right_data)[storage_id]);
            else if constexpr(extrapolation)
                return (*left_data)[storage_id];
            else
                return 0;
        };
        return pair<bool,decltype(predict)>(available,predict);
    }

    template<typename T_coord,typename T_data>
    template<slice_prediction_mode_e value_mode>
    void param_c<T_coord,T_data>::store_tensor_residual(uint32_t storage_id,bool used_tensor_prediction,T_data tensor_prediction)
    {
        if constexpr(value_mode!=slice_prediction_mode_e::extrapolation_spatial_residual&&value_mode!=slice_prediction_mode_e::interpolation_spatial_residual)
            return;
        if(used_tensor_prediction)
        {
            const vector<T_data>&reconstructed_data=prediction_data_uses_prediction_order?prediction_order_decompressed_data:current_decompressed_data;
            current_slice_spatial_residual_data[storage_id]=reconstructed_data[storage_id]-tensor_prediction;
            current_slice_spatial_residual_valid[storage_id]=1;
        }
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::update_tensor_prediction_state()
    {
        previous_slice_value_source=nullptr;
        previous_slice_value_label_source=nullptr;
        previous_slice_label_source=nullptr;
        if(value_slice_remaining_outdegree_vec[current_slice_index]>0)
        {
            if(prediction_data_uses_prediction_order)
                reconstructed_slice_data_vec[current_slice_index]=move(prediction_order_decompressed_data);
            else
                reconstructed_slice_data_vec[current_slice_index]=move(current_decompressed_data);
        }
        if(label_slice_remaining_outdegree_vec[current_slice_index]>0)
        {
            if(prediction_data_uses_prediction_order)
                reconstructed_slice_label_id_vec[current_slice_index]=move(prediction_order_label_id_vec);
            else
                reconstructed_slice_label_id_vec[current_slice_index]=move(label_id_vec);
        }
        const reference_slices_c&value_dependency=value_reference_slices_vec[current_slice_index];
        for(uint32_t source_index=0;source_index<value_dependency.source_num;source_index++)
        {
            uint32_t source_slice_index=(uint32_t)value_dependency.source_indices[source_index];
            if(--value_slice_remaining_outdegree_vec[source_slice_index]==0)
                vector<T_data>().swap(reconstructed_slice_data_vec[source_slice_index]);
            if(label_config_vec.size()>1&&--label_slice_remaining_outdegree_vec[source_slice_index]==0)
                vector<uint8_t>().swap(reconstructed_slice_label_id_vec[source_slice_index]);
        }
        const reference_slices_c&label_dependency=label_reference_slices_vec[current_slice_index];
        for(uint32_t source_index=0;source_index<label_dependency.source_num;source_index++)
        {
            uint32_t source_slice_index=(uint32_t)label_dependency.source_indices[source_index];
            if(--label_slice_remaining_outdegree_vec[source_slice_index]==0)
                vector<uint8_t>().swap(reconstructed_slice_label_id_vec[source_slice_index]);
        }
    }
}
