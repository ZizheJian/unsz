#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_current_spatial_prediction_state(bool for_decompression)
    {
        bool uses_spatial_prediction=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c&config){return config.codec!=point_set_codec_e::constant;});
        if(uses_spatial_prediction&&!use_coordinate_transform&&activate_shared_spatial_prediction_state())
        {
            printf("Reusing shared spatial prediction DAG and weights\n");
            return;
        }
        generate_prediction_structures_and_weights();
        finalize_current_spatial_prediction_state(for_decompression);
        current_spatial_prediction_state_is_shared=uses_spatial_prediction&&!use_coordinate_transform;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::finalize_current_spatial_prediction_state(bool for_decompression)
    {
        for(uint32_t structure_id=0;structure_id<prediction_structure_vec.size();structure_id++)
        {
            prediction_structure_c<T_calc>&structure=prediction_structure_vec[structure_id];
            if(structure.eb_multiplier.size()!=structure.target_ids.size())
                structure.eb_multiplier=generate_pred_eb_multiplier_vec((uint32_t)structure.target_ids.size(),structure_id==0);
            structure.inverse_eb_multiplier.resize(structure.eb_multiplier.size());
            for(uint32_t prediction_index=0;prediction_index<structure.eb_multiplier.size();prediction_index++)
                structure.inverse_eb_multiplier[prediction_index]=1/structure.eb_multiplier[prediction_index];
        }
        remap_prediction_data_to_prediction_order(for_decompression);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::activate_shared_spatial_prediction_state()
    {
        if(!shared_spatial_prediction_state.valid)
            return false;
        prediction_structure_vec=move(shared_spatial_prediction_state.structures);
        original_to_prediction_order=move(shared_spatial_prediction_state.original_to_prediction_order);
        prediction_data_uses_prediction_order=shared_spatial_prediction_state.uses_prediction_order;
        shared_spatial_prediction_state.valid=false;
        current_spatial_prediction_state_is_shared=true;
        return true;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::release_current_spatial_prediction_state()
    {
        if(current_spatial_prediction_state_is_shared)
        {
            shared_spatial_prediction_state.structures=move(prediction_structure_vec);
            shared_spatial_prediction_state.original_to_prediction_order=move(original_to_prediction_order);
            shared_spatial_prediction_state.uses_prediction_order=prediction_data_uses_prediction_order;
            shared_spatial_prediction_state.valid=true;
        }
        else
        {
            vector<prediction_structure_c<T_calc>>().swap(prediction_structure_vec);
            vector<int>().swap(original_to_prediction_order);
        }
        prediction_data_uses_prediction_order=false;
        current_spatial_prediction_state_is_shared=false;
    }
}
