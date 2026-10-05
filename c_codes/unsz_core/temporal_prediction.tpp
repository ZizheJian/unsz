#pragma once

namespace unsz
{
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
            plan.dependencies[physical_time_step_index]=build_temporal_dependency(flow,time_step,physical_time_step_index,left_source,right_source,previous_source);
            processed_time_step_indices.emplace(time_step,physical_time_step_index);
        }
        return plan;
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
        map<uint32_t,uint32_t> processed_time_step_indices;
        int32_t previous_value_source=-1,previous_label_source=-1;
        for(uint32_t time_step_index=0;time_step_index<time_step_num;time_step_index++)
        {
            uint32_t time_step=time_step_vec[time_step_index];
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
            temporal_dependency_c&value_dependency=temporal_value_dependency_vec[time_step_index];
            temporal_dependency_c&label_dependency=temporal_label_dependency_vec[time_step_index];
            value_dependency=build_temporal_dependency(value_flow,time_step,time_step_index,left_source,right_source,previous_value_source);
            label_dependency=build_temporal_dependency(label_flow,time_step,time_step_index,left_source,right_source,previous_label_source);
            for(uint32_t source_index=0;source_index<value_dependency.source_num;source_index++)
            {
                uint32_t source_time_step_index=(uint32_t)value_dependency.source_indices[source_index];
                temporal_value_remaining_outdegree_vec[source_time_step_index]++;
                if(point_set_config_vec.size()>1)
                    temporal_label_remaining_outdegree_vec[source_time_step_index]++;
            }
            for(uint32_t source_index=0;source_index<label_dependency.source_num;source_index++)
                temporal_label_remaining_outdegree_vec[(uint32_t)label_dependency.source_indices[source_index]]++;
            processed_time_step_indices.emplace(time_step,time_step_index);
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
            uint32_t left_time_step=time_step_vec[left_source],right_time_step=time_step_vec[right_source];
            T_calc right_weight;
            if(time_step_coordinate_vec.empty())
                right_weight=(T_calc)(current_time_step-left_time_step)/(right_time_step-left_time_step);
            else
            {
                T_coord left_coordinate=time_step_coordinate_vec[left_time_step],right_coordinate=time_step_coordinate_vec[right_time_step],current_coordinate=time_step_coordinate_vec[current_time_step];
                right_weight=(T_calc)((current_coordinate-left_coordinate)/(right_coordinate-left_coordinate));
            }
            temporal_weights[0]=(T_calc)1-right_weight;
            temporal_weights[1]=right_weight;
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
