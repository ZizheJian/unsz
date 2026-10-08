#pragma once

namespace unsz
{
    inline string temporal_dependency_plan_name(const string&flow)
    {
        if(flow=="constant_extrapolation_spatial_residual")return "constant_extrapolation";
        if(flow=="linear_interpolation_spatial_residual")return "linear_interpolation";
        if(flow=="anchor_interpolation_stride_8_spatial_residual")return "anchor_interpolation_stride_8";
        return flow;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::precompute_all_tensor_dependency_plans()
    {
        for(const string&field_name:field_name_vec)
        {
            const tensor_field_config_c&config=tensor_field_config_map.at(field_name);
            if(!tensor_dependency_plan_cache.count(config.dependency_plan_key))
                tensor_dependency_plan_cache.emplace(config.dependency_plan_key,generate_all_tensor_dependency_plans(config.shape,config.axis_coordinates));
        }
    }

    inline temporal_dependency_c build_temporal_dependency(const string&flow,uint32_t time_step,uint32_t time_step_index,int32_t left_source,int32_t right_source,int32_t&previous_source)
    {
        temporal_dependency_c dependency;
        bool anchor_flow=flow=="anchor_interpolation_stride_8"||flow=="anchor_interpolation_stride_8_spatial_residual";
        bool constant_flow=flow=="constant_extrapolation"||flow=="constant_extrapolation_spatial_residual";
        bool interpolation_flow=flow=="linear_interpolation"||flow=="linear_interpolation_spatial_residual";
        if(anchor_flow)
        {
            bool first_anchor=time_step==0;
            bool regular_anchor=time_step%8==0;
            bool can_interpolate=left_source>=0&&right_source>=0;
            if(first_anchor)
                previous_source=(int32_t)time_step_index;
            else if(regular_anchor||!can_interpolate)
            {
                if(previous_source>=0)
                {
                    dependency.source_indices[0]=previous_source;
                    dependency.source_num=1;
                }
                previous_source=(int32_t)time_step_index;
            }
            else
            {
                dependency.source_indices={{left_source,right_source}};
                dependency.source_num=2;
            }
            return dependency;
        }
        if(constant_flow)
        {
            if(previous_source>=0)
            {
                dependency.source_indices[0]=previous_source;
                dependency.source_num=1;
            }
            previous_source=(int32_t)time_step_index;
            return dependency;
        }
        if(interpolation_flow&&left_source>=0&&right_source>=0)
        {
            dependency.source_indices={{left_source,right_source}};
            dependency.source_num=2;
        }
        return dependency;
    }

    template<typename T_coord,typename T_data>
    temporal_dependency_plan_c param_c<T_coord,T_data>::generate_temporal_dependency_plan(const string&flow) const
    {
        temporal_dependency_plan_c plan;
        plan.processing_order=generate_time_step_processing_order(get_time_step_order(flow));
        plan.dependencies.assign(time_step_vec.size(),{});
        map<uint32_t,uint32_t> processed_time_step_indices;
        int32_t previous_source=-1;
        for(uint32_t physical_time_step_index:plan.processing_order)
        {
            uint32_t time_step=time_step_vec[physical_time_step_index];
            int32_t left_source=-1,right_source=-1;
            auto right_iterator=processed_time_step_indices.lower_bound(time_step);
            if(right_iterator!=processed_time_step_indices.end())
                right_source=(int32_t)right_iterator->second;
            if(right_iterator!=processed_time_step_indices.begin())
            {
                auto left_iterator=right_iterator;
                --left_iterator;
                left_source=(int32_t)left_iterator->second;
            }
            temporal_dependency_c&dependency=plan.dependencies[physical_time_step_index];
            dependency=build_temporal_dependency(flow,time_step,physical_time_step_index,left_source,right_source,previous_source);
            if(dependency.source_num==1)
                dependency.source_weights[0]=1;
            else if(dependency.source_num==2)
            {
                uint32_t left_index=(uint32_t)dependency.source_indices[0],right_index=(uint32_t)dependency.source_indices[1];
                uint32_t left_time_step=time_step_vec[left_index],right_time_step=time_step_vec[right_index];
                double left_coordinate=time_step_coordinate_vec.empty()?(double)left_time_step:(double)time_step_coordinate_vec[left_time_step];
                double right_coordinate=time_step_coordinate_vec.empty()?(double)right_time_step:(double)time_step_coordinate_vec[right_time_step];
                double current_coordinate=time_step_coordinate_vec.empty()?(double)time_step:(double)time_step_coordinate_vec[time_step];
                double right_weight=(current_coordinate-left_coordinate)/(right_coordinate-left_coordinate);
                dependency.source_weights={{1-right_weight,right_weight}};
            }
            processed_time_step_indices.emplace(time_step,physical_time_step_index);
        }
        return plan;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::precompute_temporal_dependency_plans()
    {
        temporal_dependency_plan_cache.clear();
        if(current_field_uses_tensor)
        {
            if(current_tensor_dependency_plan_key.empty())
                throw runtime_error("当前field缺少tensor依赖计划缓存键");
            if(!tensor_dependency_plan_cache.count(current_tensor_dependency_plan_key))
                tensor_dependency_plan_cache.emplace(current_tensor_dependency_plan_key,generate_all_tensor_dependency_plans(current_tensor_shape,current_tensor_axis_coordinates));
            return;
        }
        vector<uint32_t> sorted_time_steps=time_step_vec;
        sort(sorted_time_steps.begin(),sorted_time_steps.end());
        bool anchor_compatible=!sorted_time_steps.empty()&&sorted_time_steps.front()==0;
        for(uint32_t time_step_index=1;time_step_index<sorted_time_steps.size();time_step_index++)
            anchor_compatible&=sorted_time_steps[time_step_index]==sorted_time_steps[time_step_index-1]+1;
        constexpr const char*flow_names[]={"none","constant_extrapolation","linear_interpolation","anchor_interpolation_stride_8"};
        for(const char*flow:flow_names)
        {
            bool anchor_flow=string(flow).find("anchor_interpolation_stride_8")!=string::npos;
            if(!anchor_flow||anchor_compatible)
                temporal_dependency_plan_cache.emplace(flow,generate_temporal_dependency_plan(flow));
        }
    }

    template<typename T_coord,typename T_data>
    const temporal_dependency_plan_c&param_c<T_coord,T_data>::get_tensor_dependency_plan(const tensor_flow_c&flow) const
    {
        auto cache_iterator=tensor_dependency_plan_cache.find(current_tensor_dependency_plan_key);
        if(cache_iterator==tensor_dependency_plan_cache.end())
            throw runtime_error("没有预生成当前形状的tensor依赖计划");
        auto iterator=cache_iterator->second.find(tensor_dependency_plan_id(flow));
        if(iterator==cache_iterator->second.end())
            throw runtime_error("没有预生成的tensor依赖计划："+tensor_flow_name(flow));
        return iterator->second;
    }

    template<typename T_coord,typename T_data>
    const temporal_dependency_plan_c&param_c<T_coord,T_data>::get_temporal_dependency_plan(const string&flow) const
    {
        auto iterator=temporal_dependency_plan_cache.find(temporal_dependency_plan_name(flow));
        if(iterator==temporal_dependency_plan_cache.end())
            throw runtime_error("没有预生成的时间依赖计划："+flow);
        return iterator->second;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_temporal_dependencies()
    {
        uint32_t time_step_num=(uint32_t)time_step_vec.size();
        temporal_value_dependency_vec.assign(time_step_num,{});
        temporal_label_dependency_vec.assign(time_step_num,{});
        temporal_value_remaining_outdegree_vec.assign(time_step_num,0);
        temporal_label_remaining_outdegree_vec.assign(time_step_num,0);
        reconstructed_time_data_vec.assign(time_step_num,{});
        reconstructed_time_point_set_id_vec.assign(time_step_num,{});
        previous_time_value_source=nullptr;
        previous_time_value_point_set_source=nullptr;
        previous_time_label_source=nullptr;
        map<uint32_t,uint32_t> configured_index_by_time_step,current_index_by_time_step;
        for(uint32_t index=0;index<configured_time_step_vec.size();index++)
            configured_index_by_time_step.emplace(configured_time_step_vec[index],index);
        for(uint32_t index=0;index<time_step_vec.size();index++)
            current_index_by_time_step.emplace(time_step_vec[index],index);
        auto map_plan_dependencies=[&](const temporal_dependency_plan_c&plan,vector<temporal_dependency_c>&dependencies)
        {
            for(uint32_t current_index=0;current_index<time_step_num;current_index++)
            {
                uint32_t configured_index=configured_index_by_time_step.at(time_step_vec[current_index]);
                dependencies[current_index]=plan.dependencies[configured_index];
                temporal_dependency_c&dependency=dependencies[current_index];
                for(uint32_t source_id=0;source_id<dependency.source_num;source_id++)
                {
                    uint32_t source_configured_index=(uint32_t)dependency.source_indices[source_id];
                    uint32_t source_time_step=configured_time_step_vec[source_configured_index];
                    uint32_t source_current_index=current_index_by_time_step.at(source_time_step);
                    if(source_current_index>=current_index)
                        throw runtime_error("时间依赖计划的参考切片尚未重建");
                    dependency.source_indices[source_id]=(int32_t)source_current_index;
                }
            }
        };
        if(current_field_uses_tensor)
        {
            if(!tensor_flow_selected)
                throw runtime_error("尚未选择tensor flow");
            const temporal_dependency_plan_c&plan=get_tensor_dependency_plan(selected_tensor_flow);
            map_plan_dependencies(plan,temporal_value_dependency_vec);
            map_plan_dependencies(plan,temporal_label_dependency_vec);
        }
        else
        {
            map_plan_dependencies(get_temporal_dependency_plan(value_flow),temporal_value_dependency_vec);
            map_plan_dependencies(get_temporal_dependency_plan(label_flow),temporal_label_dependency_vec);
        }
        for(uint32_t time_step_index=0;time_step_index<time_step_num;time_step_index++)
        {
            temporal_dependency_c&value_dependency=temporal_value_dependency_vec[time_step_index];
            temporal_dependency_c&label_dependency=temporal_label_dependency_vec[time_step_index];
            for(uint32_t source_index=0;source_index<value_dependency.source_num;source_index++)
            {
                uint32_t source_time_step_index=(uint32_t)value_dependency.source_indices[source_index];
                temporal_value_remaining_outdegree_vec[source_time_step_index]++;
                if(point_set_config_vec.size()>1)
                    temporal_label_remaining_outdegree_vec[source_time_step_index]++;
            }
            for(uint32_t source_index=0;source_index<label_dependency.source_num;source_index++)
                temporal_label_remaining_outdegree_vec[(uint32_t)label_dependency.source_indices[source_index]]++;
        }
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_temporal_prediction()
    {
        previous_time_value_source=nullptr;
        previous_time_value_point_set_source=nullptr;
        previous_time_label_source=nullptr;
        previous_time_value_available=false;
        previous_time_label_available=false;
        temporal_value_interpolation_available=false;
        temporal_label_interpolation_available=false;
        temporal_weights={{0,0}};
        const prediction_framework_c&framework=current_prediction_framework;
        if(framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual||framework.value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
        {
            current_time_spatial_residual_data.resize(point_num);
            current_time_spatial_residual_valid.assign(point_num,0);
        }
        const temporal_dependency_c&value_dependency=temporal_value_dependency_vec[current_time_step_index];
        const temporal_dependency_c&label_dependency=temporal_label_dependency_vec[current_time_step_index];
        if((framework.value_mode==time_dimension_mode_e::constant_extrapolation||framework.value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)&&value_dependency.source_num==1)
        {
            uint32_t source_index=(uint32_t)value_dependency.source_indices[0];
            previous_time_value_source=&reconstructed_time_data_vec[source_index];
            previous_time_value_available=previous_time_value_source->size()==point_num;
            if(point_set_config_vec.size()>1)
            {
                previous_time_value_point_set_source=&reconstructed_time_point_set_id_vec[source_index];
                previous_time_value_available&=previous_time_value_point_set_source->size()==point_num;
            }
        }
        if(framework.label_mode==time_dimension_mode_e::constant_extrapolation&&label_dependency.source_num==1)
        {
            uint32_t source_index=(uint32_t)label_dependency.source_indices[0];
            previous_time_label_source=&reconstructed_time_point_set_id_vec[source_index];
            previous_time_label_available=previous_time_label_source->size()==point_num;
        }
        if((framework.value_mode==time_dimension_mode_e::linear_interpolation||framework.value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)&&value_dependency.source_num==2)
        {
            uint32_t left_source=(uint32_t)value_dependency.source_indices[0],right_source=(uint32_t)value_dependency.source_indices[1];
            const vector<T_data>&left_data=reconstructed_time_data_vec[left_source],&right_data=reconstructed_time_data_vec[right_source];
            temporal_value_interpolation_available=left_data.size()==point_num&&right_data.size()==point_num;
            if(point_set_config_vec.size()>1)
            {
                const vector<uint8_t>&left_point_set_ids=reconstructed_time_point_set_id_vec[left_source],&right_point_set_ids=reconstructed_time_point_set_id_vec[right_source];
                temporal_value_interpolation_available&=left_point_set_ids.size()==point_num&&right_point_set_ids.size()==point_num;
            }
            temporal_weights[0]=(T_calc)value_dependency.source_weights[0];
            temporal_weights[1]=(T_calc)value_dependency.source_weights[1];
        }
        if(framework.label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual&&label_dependency.source_num==2)
        {
            uint32_t left_source=(uint32_t)label_dependency.source_indices[0],right_source=(uint32_t)label_dependency.source_indices[1];
            temporal_label_interpolation_available=reconstructed_time_point_set_id_vec[left_source].size()==point_num&&reconstructed_time_point_set_id_vec[right_source].size()==point_num;
        }
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e label_mode>
    bool param_c<T_coord,T_data>::predict_temporal_label(uint32_t storage_id,uint32_t label_channel_id,uint32_t&prediction)
    {
        auto get_binary_label=[label_channel_id,storage_id](const vector<uint8_t>&point_set_ids)
        {
            return point_set_ids[storage_id]==label_channel_id?0U:1U;
        };
        if constexpr(label_mode==time_dimension_mode_e::label_linear_interpolation_spatial_residual)
        {
            if(!temporal_label_interpolation_available)
                return false;
            const temporal_dependency_c&dependency=temporal_label_dependency_vec[current_time_step_index];
            const vector<uint8_t>&left_point_set_ids=reconstructed_time_point_set_id_vec[(uint32_t)dependency.source_indices[0]],&right_point_set_ids=reconstructed_time_point_set_id_vec[(uint32_t)dependency.source_indices[1]];
            uint32_t left_label=get_binary_label(left_point_set_ids),right_label=get_binary_label(right_point_set_ids);
            if(left_label!=right_label)
                return false;
            prediction=left_label;
            return true;
        }
        if(!previous_time_label_available)
            return false;
        if constexpr(label_mode==time_dimension_mode_e::constant_extrapolation)
        {
            prediction=get_binary_label(*previous_time_label_source);
            return true;
        }
        return false;
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e value_mode>
    bool param_c<T_coord,T_data>::predict_temporal_value(uint32_t storage_id,uint32_t point_set_id,T_data&prediction)
    {
        if constexpr(value_mode==time_dimension_mode_e::constant_extrapolation||value_mode==time_dimension_mode_e::constant_extrapolation_spatial_residual)
        {
            if(!previous_time_value_available)
                return false;
            if(point_set_config_vec.size()>1&&(*previous_time_value_point_set_source)[storage_id]!=point_set_id)
                return false;
            prediction=(*previous_time_value_source)[storage_id];
            return true;
        }
        if constexpr(value_mode==time_dimension_mode_e::linear_interpolation||value_mode==time_dimension_mode_e::linear_interpolation_spatial_residual)
        {
            if(!temporal_value_interpolation_available)
                return false;
            const temporal_dependency_c&dependency=temporal_value_dependency_vec[current_time_step_index];
            uint32_t left_source=(uint32_t)dependency.source_indices[0],right_source=(uint32_t)dependency.source_indices[1];
            const vector<T_data>&left_data=reconstructed_time_data_vec[left_source],&right_data=reconstructed_time_data_vec[right_source];
            if(point_set_config_vec.size()>1)
            {
                const vector<uint8_t>&left_point_set_ids=reconstructed_time_point_set_id_vec[left_source],&right_point_set_ids=reconstructed_time_point_set_id_vec[right_source];
                if(left_point_set_ids[storage_id]!=point_set_id||right_point_set_ids[storage_id]!=point_set_id)
                    return false;
            }
            prediction=(T_data)(temporal_weights[0]*(T_calc)left_data[storage_id]+temporal_weights[1]*(T_calc)right_data[storage_id]);
            return true;
        }
        return false;
    }

    template<typename T_coord,typename T_data>
    template<time_dimension_mode_e value_mode>
    void param_c<T_coord,T_data>::store_temporal_residual(uint32_t storage_id,bool used_temporal_prediction,T_data temporal_prediction)
    {
        if constexpr(value_mode!=time_dimension_mode_e::constant_extrapolation_spatial_residual&&value_mode!=time_dimension_mode_e::linear_interpolation_spatial_residual)
            return;
        if(used_temporal_prediction)
        {
            const vector<T_data>&reconstructed_data=prediction_data_uses_prediction_order?prediction_order_decompressed_data:current_decompressed_data;
            current_time_spatial_residual_data[storage_id]=reconstructed_data[storage_id]-temporal_prediction;
            current_time_spatial_residual_valid[storage_id]=1;
        }
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::update_time_prediction_state()
    {
        previous_time_value_source=nullptr;
        previous_time_value_point_set_source=nullptr;
        previous_time_label_source=nullptr;
        if(temporal_value_remaining_outdegree_vec[current_time_step_index]>0)
        {
            if(prediction_data_uses_prediction_order)
                reconstructed_time_data_vec[current_time_step_index]=move(prediction_order_decompressed_data);
            else
                reconstructed_time_data_vec[current_time_step_index]=move(current_decompressed_data);
        }
        if(temporal_label_remaining_outdegree_vec[current_time_step_index]>0)
        {
            if(prediction_data_uses_prediction_order)
                reconstructed_time_point_set_id_vec[current_time_step_index]=move(prediction_order_point_set_id_vec);
            else
                reconstructed_time_point_set_id_vec[current_time_step_index]=move(point_set_id_vec);
        }
        const temporal_dependency_c&value_dependency=temporal_value_dependency_vec[current_time_step_index];
        for(uint32_t source_index=0;source_index<value_dependency.source_num;source_index++)
        {
            uint32_t source_time_step_index=(uint32_t)value_dependency.source_indices[source_index];
            if(--temporal_value_remaining_outdegree_vec[source_time_step_index]==0)
                vector<T_data>().swap(reconstructed_time_data_vec[source_time_step_index]);
            if(point_set_config_vec.size()>1&&--temporal_label_remaining_outdegree_vec[source_time_step_index]==0)
                vector<uint8_t>().swap(reconstructed_time_point_set_id_vec[source_time_step_index]);
        }
        const temporal_dependency_c&label_dependency=temporal_label_dependency_vec[current_time_step_index];
        for(uint32_t source_index=0;source_index<label_dependency.source_num;source_index++)
        {
            uint32_t source_time_step_index=(uint32_t)label_dependency.source_indices[source_index];
            if(--temporal_label_remaining_outdegree_vec[source_time_step_index]==0)
                vector<uint8_t>().swap(reconstructed_time_point_set_id_vec[source_time_step_index]);
        }
    }
}
