#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::remap_prediction_data_to_prediction_order(bool for_decompression)
    {
        prediction_data_uses_prediction_order=false;
        original_to_prediction_order.clear();
        uint64_t total_target_num=0;
        for(const prediction_structure_c<T_calc>&structure:prediction_structure_vec)
            total_target_num+=structure.target_ids.size();
        if(total_target_num!=point_num)
            return;
        bool has_predicted_point_set=false;
        for(const point_set_config_c&config:point_set_config_vec)
            has_predicted_point_set|=config.codec!=point_set_codec_e::constant;
        if(!has_predicted_point_set)
            return;
        original_to_prediction_order.assign(point_num,-1);
        uint32_t prediction_order_offset=0;
        for(prediction_structure_c<T_calc>&structure:prediction_structure_vec)
        {
            structure.prediction_order_offset=prediction_order_offset;
            for(uint32_t prediction_index=0;prediction_index<structure.target_ids.size();prediction_index++)
            {
                int target_id=structure.target_ids[prediction_index];
                if(target_id<0||(uint32_t)target_id>=point_num||original_to_prediction_order[target_id]>=0)
                    throw runtime_error("预测结构包含无效或重复目标点");
                original_to_prediction_order[target_id]=(int)(prediction_order_offset+prediction_index);
            }
            prediction_order_offset+=(uint32_t)structure.target_ids.size();
        }
        for(prediction_structure_c<T_calc>&structure:prediction_structure_vec)
            for(int&reference_id:structure.reference_ids)
            {
                if(reference_id<0||(uint32_t)reference_id>=point_num)
                    throw runtime_error("预测结构包含无效参考点");
                int prediction_index=original_to_prediction_order[reference_id];
                if(prediction_index<0)
                    throw runtime_error("无法将空间参考点映射到预测顺序");
                reference_id=prediction_index;
            }
        prediction_data_uses_prediction_order=true;
        if(!for_decompression)
            original_to_prediction_order.reset();
    }

}
