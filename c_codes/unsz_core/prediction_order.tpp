#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::remap_prediction_data_to_prediction_order(bool for_decompression)
    {
        prediction_data_uses_prediction_order=false;
        if(prediction_structure_vec.size()!=1)
            return;
        if(for_decompression&&(point_set_config_vec.size()!=1||point_set_config_vec.front().codec==point_set_codec_e::constant||time_step_vec.size()!=1||value_flow!="none"))
            return;
        prediction_structure_c<T_calc>&structure=prediction_structure_vec.front();
        if(structure.target_ids.size()!=point_num)
            return;
        vector<int> original_to_prediction(point_num,-1);
        for(uint32_t prediction_index=0;prediction_index<point_num;prediction_index++)
            original_to_prediction[structure.target_ids[prediction_index]]=(int)prediction_index;
        for(int&reference_id:structure.reference_ids)
        {
            int prediction_index=original_to_prediction[reference_id];
            if(prediction_index<0)
                throw runtime_error("无法将空间参考点映射到预测顺序");
            reference_id=prediction_index;
        }
        prediction_data_uses_prediction_order=true;
    }

}
