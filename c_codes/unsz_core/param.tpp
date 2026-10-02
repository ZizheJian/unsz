#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    template<typename T_coord2>
    param_c<T_coord,T_data>::param_c(const param_c<T_coord2,T_data> &other)
    {
        data_dir=other.data_dir;
        field_name=other.field_name;
        error_mode=other.error_mode;
        coord_path=other.coord_path;
        time_step_coordinate_path=other.time_step_coordinate_path;
        dag_cache_path=other.dag_cache_path;
        adjust_coord_matrix_file=other.adjust_coord_matrix_file;
        adjust_coord_training_type=other.adjust_coord_training_type;
        point_set_config_path=other.point_set_config_path;
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
        got_field=other.got_field;
        got_time_step=other.got_time_step;
        got_error_mode=other.got_error_mode;
        got_predictor=other.got_predictor;
        got_prediction_round_num=other.got_prediction_round_num;
        point_set_rule_vec=other.point_set_rule_vec;
        point_set_config_vec=other.point_set_config_vec;
        check_original_data_files=other.check_original_data_files;
        value_flow=other.value_flow;
        label_flow=other.label_flow;
        if_adjust_coord=other.if_adjust_coord;
        adjust_coord_matrix_parameter_vec=other.adjust_coord_matrix_parameter_vec;
        print_compression_stage_time=other.print_compression_stage_time;
        label_value_execution=other.label_value_execution;
        ordering_scope=other.ordering_scope;
        ordering_score_min_ref_num=other.ordering_score_min_ref_num;
        ordering_candidate_num=other.ordering_candidate_num;
        ordering_score_max_ref_num=other.ordering_score_max_ref_num;
        kd_tree_leaf_size=other.kd_tree_leaf_size;
        partition_size=other.partition_size;
        partition_num=other.partition_num;
        thread_num=other.thread_num;
        use_partition_adaptive_predictor_experimental=other.use_partition_adaptive_predictor_experimental;
        partition_adaptive_sample_num_experimental=other.partition_adaptive_sample_num_experimental;
        partition_adaptive_min_improvement_experimental=other.partition_adaptive_min_improvement_experimental;
        partition_fitting_candidates_experimental=other.partition_fitting_candidates_experimental;
        eb_schedule=other.eb_schedule;
        eb_schedule_start_ratio=other.eb_schedule_start_ratio;
        eb_schedule_log2_shift=other.eb_schedule_log2_shift;
        eb_schedule_max_log2_exponent=other.eb_schedule_max_log2_exponent;
        prediction_round_num=other.prediction_round_num;
        round_size_growth_ratio_experimental=other.round_size_growth_ratio_experimental;
        prediction_min_ref_num=other.prediction_min_ref_num;
        prediction_max_ref_num=other.prediction_max_ref_num;
        idw_power=other.idw_power;
        rbf_kernel=other.rbf_kernel;
        rbf_shape=other.rbf_shape;
        rbf_nugget=other.rbf_nugget;
        rbf_parameter_specified=other.rbf_parameter_specified;
        zstd_compression_level=other.zstd_compression_level;
        predictor=other.predictor;

        coord_array.resize(other.coord_array.n0,other.coord_array.n1);
        coord_array.data.assign(other.coord_array.data.begin(),other.coord_array.data.end());
        min_coord.assign(other.min_coord.begin(),other.min_coord.end());
        max_coord.assign(other.max_coord.begin(),other.max_coord.end());
        partition_point_ids_vec=other.partition_point_ids_vec;
        point_partition_id_vec=other.point_partition_id_vec;
        prediction_structure_vec.resize(other.prediction_structure_vec.size());
        for(uint32_t structure_id=0;structure_id<prediction_structure_vec.size();structure_id++)
        {
            auto&structure=prediction_structure_vec[structure_id];
            const auto&other_structure=other.prediction_structure_vec[structure_id];
            structure.target_id_groups=other_structure.target_id_groups;
            structure.target_ids=other_structure.target_ids;
            structure.reference_ids=other_structure.reference_ids;
            structure.reference_offsets=other_structure.reference_offsets;
            structure.prediction_weights.assign(other_structure.prediction_weights.begin(),other_structure.prediction_weights.end());
            structure.rbf_fallback_flags=other_structure.rbf_fallback_flags;
            structure.eb_multiplier=other_structure.eb_multiplier;
        }
        partition_prediction_config_id_vec=other.partition_prediction_config_id_vec;
        partition_adaptive_sample_mask=other.partition_adaptive_sample_mask;

        current_data=other.current_data;
        current_decompressed_data=other.current_decompressed_data;
        previous_time_decompressed_data=other.previous_time_decompressed_data;
        previous_time_point_set_id_vec=other.previous_time_point_set_id_vec;
        time_step_vec=other.time_step_vec;
        time_step_coordinate_vec.assign(other.time_step_coordinate_vec.begin(),other.time_step_coordinate_vec.end());
        reconstructed_time_step_vec=other.reconstructed_time_step_vec;
        reconstructed_time_data_vec=other.reconstructed_time_data_vec;
        reconstructed_time_point_set_id_vec=other.reconstructed_time_point_set_id_vec;
        previous_time_value_history_id=other.previous_time_value_history_id;
        previous_time_label_history_id=other.previous_time_label_history_id;
        current_time_spatial_residual_data=other.current_time_spatial_residual_data;
        current_time_spatial_residual_valid=other.current_time_spatial_residual_valid;
        current_time_step=other.current_time_step;
        temporal_history_ids=other.temporal_history_ids;
        temporal_weights[0]=(T_calc)other.temporal_weights[0];
        temporal_weights[1]=(T_calc)other.temporal_weights[1];
        previous_time_value_available=other.previous_time_value_available;
        previous_time_label_available=other.previous_time_label_available;
        temporal_interpolation_available=other.temporal_interpolation_available;
        current_compressed_bytes=other.current_compressed_bytes;
        point_set_id_vec=other.point_set_id_vec;
        point_set_point_num_vec=other.point_set_point_num_vec;
        point_set_min_data_vec=other.point_set_min_data_vec;
        point_set_max_data_vec=other.point_set_max_data_vec;
        point_set_abs_eb_vec=other.point_set_abs_eb_vec;
        point_set_codec_vec=other.point_set_codec_vec;
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::need_double_coord()
    {
        return false;
    }
}
