#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_current_spatial_prediction_state(bool for_decompression)
    {
        bool uses_spatial_prediction=any_of(label_config_vec.begin(),label_config_vec.end(),[](const label_config_c&config){return config.codec!=label_codec_e::constant;});
        if(uses_spatial_prediction&&!use_coordinate_rescaling)
        {
            bool generated_shared_state=false;
            call_once(shared_spatial_prediction_cache->initialization_flag,[&]()
            {
                generated_shared_state=true;
                prediction_structure_vec.reset();
                original_to_prediction_order.reset();
                generate_prediction_structures_and_weights();
                finalize_current_spatial_prediction_state(for_decompression);
                auto state=make_shared<spatial_prediction_state_c<T_calc>>();
                state->structures=prediction_structure_vec;
                state->original_to_prediction_order=original_to_prediction_order;
                state->uses_prediction_order=prediction_data_uses_prediction_order;
                shared_spatial_prediction_cache->state=move(state);
            });
            if(!activate_shared_spatial_prediction_state())
                throw runtime_error("公共空间预测DAG初始化失败");
            if(!generated_shared_state)
                printf("Reusing shared spatial prediction DAG and weights\n");
            return;
        }
        prediction_structure_vec.reset();
        original_to_prediction_order.reset();
        generate_prediction_structures_and_weights();
        finalize_current_spatial_prediction_state(for_decompression);
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
        const shared_ptr<const spatial_prediction_state_c<T_calc>>&state=shared_spatial_prediction_cache->state;
        if(!state)
            return false;
        prediction_structure_vec=state->structures;
        original_to_prediction_order=state->original_to_prediction_order;
        prediction_data_uses_prediction_order=state->uses_prediction_order;
        return true;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::release_current_spatial_prediction_state()
    {
        prediction_structure_vec.reset();
        original_to_prediction_order.reset();
        prediction_data_uses_prediction_order=false;
    }
}
