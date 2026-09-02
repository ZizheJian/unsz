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
        prediction_structure_path=other.prediction_structure_path;
        hybrid_selector_path=other.hybrid_selector_path;
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
        cross_time_predictor_experimental=other.cross_time_predictor_experimental;
        if_adjust_coord=other.if_adjust_coord;
        adjust_coord_matrix_parameter_vec=other.adjust_coord_matrix_parameter_vec;
        print_round_statistics=other.print_round_statistics;
        print_prediction_geometry_statistics=other.print_prediction_geometry_statistics;
        print_label_compression_statistics=other.print_label_compression_statistics;
        print_compression_stage_time=other.print_compression_stage_time;
        print_prediction_dag_statistics=other.print_prediction_dag_statistics;
        print_pointwise_oracle_statistics_experimental=other.print_pointwise_oracle_statistics_experimental;
        use_prediction_order_layout_experimental=other.use_prediction_order_layout_experimental;
        use_normalized_weight_experimental=other.use_normalized_weight_experimental;
        use_prediction_order_label_experimental=other.use_prediction_order_label_experimental;
        use_predictability_order_experimental=other.use_predictability_order_experimental;
        use_forward_mean_distance_only_order_experimental=other.use_forward_mean_distance_only_order_experimental;
        use_forward_s2_only_order_experimental=other.use_forward_s2_only_order_experimental;
        use_forward_direction_only_order_experimental=other.use_forward_direction_only_order_experimental;
        use_forward_mean_distance_s2_order_experimental=other.use_forward_mean_distance_s2_order_experimental;
        use_forward_mean_distance_direction_order_experimental=other.use_forward_mean_distance_direction_order_experimental;
        use_reverse_predictability_order_experimental=other.use_reverse_predictability_order_experimental;
        use_reverse_distribution_order_experimental=other.use_reverse_distribution_order_experimental;
        use_reverse_mean_distance_only_order_experimental=other.use_reverse_mean_distance_only_order_experimental;
        use_reverse_reference_reuse_order_experimental=other.use_reverse_reference_reuse_order_experimental;
        use_reverse_s2_only_order_experimental=other.use_reverse_s2_only_order_experimental;
        use_reverse_direction_only_order_experimental=other.use_reverse_direction_only_order_experimental;
        use_reverse_direction_order_experimental=other.use_reverse_direction_order_experimental;
        use_reverse_direction_distribution_order_experimental=other.use_reverse_direction_distribution_order_experimental;
        use_global_order_before_round_split_experimental=other.use_global_order_before_round_split_experimental;
        d_only_cache_refill_threshold_experimental=other.d_only_cache_refill_threshold_experimental;
        order_candidate_num_experimental=other.order_candidate_num_experimental;
        cross_time_spatial_point_num_experimental=other.cross_time_spatial_point_num_experimental;
        kd_tree_leaf_size=other.kd_tree_leaf_size;
        independent_block_size_experimental=other.independent_block_size_experimental;
        independent_block_num_experimental=other.independent_block_num_experimental;
        use_block_adaptive_predictor_experimental=other.use_block_adaptive_predictor_experimental;
        block_adaptive_sample_num_experimental=other.block_adaptive_sample_num_experimental;
        block_fitting_candidates_experimental=other.block_fitting_candidates_experimental;
        multi_huffman_mode_experimental=other.multi_huffman_mode_experimental;
        eb_schedule_experimental=other.eb_schedule_experimental;
        eb_schedule_start_ratio=other.eb_schedule_start_ratio;
        predict_round_num=other.predict_round_num;
        max_ref_num=other.max_ref_num;
        actual_ref_num=other.actual_ref_num;
        inverse_distance_power=other.inverse_distance_power;
        zstd_compression_level=other.zstd_compression_level;
        point_set_predictor=other.point_set_predictor;
        predictor=other.predictor;

        coord_array.resize(other.coord_array.n0,other.coord_array.n1);
        coord_array.data.assign(other.coord_array.data.begin(),other.coord_array.data.end());
        coord_array_backup.resize(other.coord_array_backup.n0,other.coord_array_backup.n1);
        coord_array_backup.data.assign(other.coord_array_backup.data.begin(),other.coord_array_backup.data.end());
        min_coord.assign(other.min_coord.begin(),other.min_coord.end());
        max_coord.assign(other.max_coord.begin(),other.max_coord.end());
        round_target_ids=other.round_target_ids;
        pred_target_ids=other.pred_target_ids;
        pred_ref_ids=other.pred_ref_ids;
        pred_ref_begin=other.pred_ref_begin;
        pred_ref_weights=other.pred_ref_weights;
        pred_ref_label_normalized_weights=other.pred_ref_label_normalized_weights;
        pred_ref_value_normalized_weights=other.pred_ref_value_normalized_weights;
        prediction_id_to_original_id=other.prediction_id_to_original_id;
        original_id_to_prediction_id=other.original_id_to_prediction_id;
        independent_block_point_ids_vec=other.independent_block_point_ids_vec;
        point_independent_block_id_vec=other.point_independent_block_id_vec;
        independent_block_root_ids_vec=other.independent_block_root_ids_vec;
        block_prediction_config_id_vec=other.block_prediction_config_id_vec;
        block_adaptive_sample_mask=other.block_adaptive_sample_mask;

        current_data=other.current_data;
        current_decompressed_data=other.current_decompressed_data;
        previous_time_decompressed_data=other.previous_time_decompressed_data;
        previous_time_point_set_id_vec=other.previous_time_point_set_id_vec;
        time_step_vec=other.time_step_vec;
        reconstructed_time_step_vec=other.reconstructed_time_step_vec;
        reconstructed_time_data_vec=other.reconstructed_time_data_vec;
        reconstructed_time_point_set_id_vec=other.reconstructed_time_point_set_id_vec;
        current_time_step=other.current_time_step;
        interpolation_left_history_id=other.interpolation_left_history_id;
        interpolation_right_history_id=other.interpolation_right_history_id;
        interpolation_right_weight=other.interpolation_right_weight;
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
        hybrid_selector_vec=other.hybrid_selector_vec;
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
