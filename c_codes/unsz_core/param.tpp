#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    template<typename T_coord2>
    param_c<T_coord,T_data>::param_c(const param_c<T_coord2,T_data> &other)
    {
        data_dir=other.data_dir;
        variable_name=other.variable_name;
        error_mode=other.error_mode;
        coord_path=other.coord_path;
        original_data_path_vec=other.original_data_path_vec;
        compressed_data_path_vec=other.compressed_data_path_vec;
        decompressed_data_path_vec=other.decompressed_data_path_vec;
        dim=other.dim;
        point_num=other.point_num;
        rel_eb=other.rel_eb;
        abs_eb=other.abs_eb;
        got_rel_eb=other.got_rel_eb;
        got_abs_eb=other.got_abs_eb;
        got_data_dir=other.got_data_dir;
        got_variable=other.got_variable;
        got_time_step=other.got_time_step;
        got_error_mode=other.got_error_mode;
        point_set_rel_eb_config_vec=other.point_set_rel_eb_config_vec;
        point_set_abs_eb_config_vec=other.point_set_abs_eb_config_vec;
        point_set_codec_config_vec=other.point_set_codec_config_vec;
        check_original_data_files=other.check_original_data_files;
        cross_time_compress=other.cross_time_compress;
        if_adjust_coord=other.if_adjust_coord;
        print_round_statistics=other.print_round_statistics;
        print_label_compression_statistics=other.print_label_compression_statistics;
        predict_round_num=other.predict_round_num;
        max_ref_num=other.max_ref_num;
        actual_ref_num=other.actual_ref_num;
        inverse_distance_power=other.inverse_distance_power;
        zstd_compression_level=other.zstd_compression_level;
        point_set_ref_num=other.point_set_ref_num;
        point_set_predictor=other.point_set_predictor;
        predictor=other.predictor;

        coord_array.resize(other.coord_array.n0,other.coord_array.n1);
        coord_array.data.assign(other.coord_array.data.begin(),other.coord_array.data.end());
        coord_array_backup.resize(other.coord_array_backup.n0,other.coord_array_backup.n1);
        coord_array_backup.data.assign(other.coord_array_backup.data.begin(),other.coord_array_backup.data.end());
        min_coord.assign(other.min_coord.begin(),other.min_coord.end());
        max_coord.assign(other.max_coord.begin(),other.max_coord.end());
        round_target_ids=other.round_target_ids;
        pred_refs_vec=other.pred_refs_vec;

        current_data=other.current_data;
        current_decompressed_data=other.current_decompressed_data;
        current_min_data=other.current_min_data;
        current_max_data=other.current_max_data;
        current_abs_eb=other.current_abs_eb;
        qb_vec=other.qb_vec;
        current_zstd_bs_vec=other.current_zstd_bs_vec;
        point_set_id_vec=other.point_set_id_vec;
        point_set_prediction_value_vec=other.point_set_prediction_value_vec;
        point_set_point_num_vec=other.point_set_point_num_vec;
        point_set_min_data_vec=other.point_set_min_data_vec;
        point_set_max_data_vec=other.point_set_max_data_vec;
        point_set_abs_eb_vec=other.point_set_abs_eb_vec;
        point_set_codec_vec=other.point_set_codec_vec;
        current_valid_node_num=other.current_valid_node_num;
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::need_double_coord()
    {
        if (!is_same<T_coord,float>::value)
            return false;
        double max_squared_distance=0;
        for (uint32_t d=0;d<dim;d++)
        {
            double coord_range=(double)max_coord[d]-min_coord[d];
            max_squared_distance+=coord_range*coord_range;
        }
        if (!isfinite(max_squared_distance))
            return true;
        if (max_squared_distance==0)
            return false;
        double min_possible_weight=pow(max_squared_distance,-inverse_distance_power/2);
        return min_possible_weight<numeric_limits<float>::min();
    }
}
