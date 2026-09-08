// 将所有点分成多轮，确定预测顺序，并为每个点寻找已经完成预测的最近参考点。
#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_pred_refs_structure()
    {
        if (use_global_order_before_round_split_experimental && (partition_size_experimental>0 || partition_num_experimental>0))
            throw runtime_error("全局排序后分轮实验目前不能和partition同时使用");
        if(cache_subpartition_num_experimental>0 && (partition_size_experimental>0 || partition_num_experimental>0))
            throw runtime_error("cache subpartition实验不能和独立partition同时使用");
        if(cache_subpartition_num_experimental>0 && (cache_subpartition_num_experimental&(cache_subpartition_num_experimental-1))!=0)
            throw runtime_error("cache subpartition数量必须是2的幂");
        if (partition_size_experimental>0 || partition_num_experimental>0)
        {
            generate_partition_pred_refs_structure();
            return;
        }
        if (point_num<=0)
            throw runtime_error("point_num必须是正数");
        if (predict_round_num<=0 || predict_round_num>63)
            throw runtime_error("predict_round_num必须在[1,63]内");
        if (max_ref_num<=0)
            throw runtime_error("max_ref_num必须是正数");
        bool use_reference_data_order=!order_reference_time_step_vec_experimental.empty();
        uint32_t experimental_order_num=use_predictability_order_experimental+use_forward_mean_distance_only_order_experimental+use_forward_s2_only_order_experimental+use_forward_direction_only_order_experimental+use_forward_mean_distance_s2_order_experimental+use_forward_mean_distance_direction_order_experimental+use_reverse_predictability_order_experimental+use_reverse_distribution_order_experimental+use_reverse_mean_distance_only_order_experimental+use_reverse_reference_reuse_order_experimental+use_reverse_s2_only_order_experimental+use_reverse_direction_only_order_experimental+use_reverse_direction_order_experimental+use_reverse_direction_distribution_order_experimental+use_reference_data_order;
        if (experimental_order_num>1)
            throw runtime_error("实验性预测顺序不能同时启用多个");
        bool use_reverse_d_only=use_reverse_mean_distance_only_order_experimental || use_reverse_reference_reuse_order_experimental || use_reference_data_order;
        if (order_candidate_num_experimental>0 && !use_reverse_d_only)
            throw runtime_error("D-only排序候选数实验只能和反向D-only顺序一起使用");
        if (use_reverse_d_only && d_only_cache_refill_threshold_experimental>max_ref_num)
            throw runtime_error("D-only缓存重查阈值不能大于max_ref_num");
        if (use_global_order_before_round_split_experimental && !use_reverse_mean_distance_only_order_experimental)
            throw runtime_error("全局排序后分轮实验目前只能和反向D-only顺序同时使用");
        if (use_reference_data_order && prediction_structure_path.empty())
            throw runtime_error("参考数据辅助顺序必须通过-prediction-structure保存公共预测结构");

        vector<vector<T_data>> order_reference_data_vec;
        vector<vector<double>> order_reference_point_set_range_vec;
        if (use_reference_data_order)
        {
            printf("Loading %llu reference time steps for reference-data-assisted common-order search\n",(unsigned long long)order_reference_time_step_vec_experimental.size());
            order_reference_data_vec.resize(order_reference_time_step_vec_experimental.size(),vector<T_data>(point_num));
            order_reference_point_set_range_vec.resize(order_reference_time_step_vec_experimental.size(),vector<double>(2));
            for (uint32_t reference_file_id=0;reference_file_id<order_reference_time_step_vec_experimental.size();reference_file_id++)
            {
                uint32_t time_step=order_reference_time_step_vec_experimental[reference_file_id];
                string path=data_dir+variable_name+".dat."+to_string(time_step);
                FILE *file=fopen(path.c_str(),"rb");
                if (!file)
                    throw runtime_error("找不到预测顺序参考文件："+path);
                if (fread(order_reference_data_vec[reference_file_id].data(),sizeof(T_data),point_num,file)!=point_num)
                {
                    fclose(file);
                    throw runtime_error("读取预测顺序参考文件失败："+path);
                }
                if (fclose(file)!=0)
                    throw runtime_error("关闭预测顺序参考文件失败："+path);
                vector<double> point_set_min(2,numeric_limits<double>::infinity()),point_set_max(2,-numeric_limits<double>::infinity());
                for (T_data value:order_reference_data_vec[reference_file_id])
                {
                    uint32_t point_set_id=get_point_set_id(value);
                    if (point_set_id>=2)
                        throw runtime_error("参考数据辅助顺序目前只支持两个点集");
                    point_set_min[point_set_id]=min(point_set_min[point_set_id],(double)value);
                    point_set_max[point_set_id]=max(point_set_max[point_set_id],(double)value);
                }
                for (uint32_t point_set_id=0;point_set_id<2;point_set_id++)
                    if (isfinite(point_set_min[point_set_id]))
                        order_reference_point_set_range_vec[reference_file_id][point_set_id]=point_set_max[point_set_id]-point_set_min[point_set_id];
            }
        }

        printf("Building kd-tree\n");
        auto start_time=chrono::steady_clock::now();
        block_tree_c<T_coord> tree(coord_array,min_coord,max_coord,kd_tree_leaf_size);
        auto tree_end_time=chrono::steady_clock::now();
        printf("KD-tree blocks: %llu, time: %f s\n",(unsigned long long)tree.get_block_num(),chrono::duration<double>(tree_end_time-start_time).count());

        printf("Arranging rounds of prediction\n");
        round_target_ids.clear();
        round_target_ids.resize(predict_round_num);
        vector<int> shuffled_ids(point_num);
        iota(shuffled_ids.begin(),shuffled_ids.end(),0);
        mt19937 rng(0);
        shuffle(shuffled_ids.begin(),shuffled_ids.end(),rng);

        uint64_t previous_end=0;
        vector<uint32_t> round_target_num_vec=get_round_target_num_vec(point_num);
        printf("Round size growth ratio: %.9g\n",round_size_growth_ratio_experimental);
        for(uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            uint64_t current_end=previous_end+round_target_num_vec[round_id];
            round_target_ids[round_id].assign(shuffled_ids.begin()+previous_end,shuffled_ids.begin()+current_end);
            previous_end=current_end;
            printf("Round %u target num: %llu\n",round_id,(unsigned long long)round_target_ids[round_id].size());
        }
        if (previous_end!=point_num)
            throw runtime_error("预测轮次没有覆盖所有点");
        if (use_global_order_before_round_split_experimental)
        {
            for (uint32_t round_id=0;round_id+1<predict_round_num;round_id++)
                round_target_ids[round_id].clear();
            round_target_ids.back()=move(shuffled_ids);
            printf("Using one global D-only order before splitting it into prediction rounds\n");
        }

        vector<T_coord> center_coord(dim);
        for (uint32_t d=0;d<dim;d++)
            center_coord[d]=(min_coord[d]+max_coord[d])/2;

        auto sorting_start_time=chrono::steady_clock::now();
        printf("Sorting targets in each round\n");
        unique_ptr<block_tree_c<T_coord>> reference_tree;
        bool use_reverse_geometry_score=use_reverse_distribution_order_experimental || use_reverse_mean_distance_only_order_experimental || use_reverse_s2_only_order_experimental || use_reverse_direction_only_order_experimental || use_reverse_direction_order_experimental || use_reverse_direction_distribution_order_experimental;
        if (use_reverse_predictability_order_experimental || use_reverse_geometry_score || use_reverse_d_only)
        {
            reference_tree=make_unique<block_tree_c<T_coord>>(tree);
            if (use_reverse_distribution_order_experimental)
                printf("Using reverse distance-distribution elimination with %u local candidates and references\n",max_ref_num);
            else if (use_reverse_mean_distance_only_order_experimental)
                printf("Using reverse mean-distance-only elimination with %u local candidates and %u references\n",order_candidate_num_experimental>0?order_candidate_num_experimental:max_ref_num,max_ref_num);
            else if (use_reference_data_order)
                printf("Using reverse reference-data-assisted elimination with %u local candidates, %u references, %llu reference time steps and %s error scoring\n",order_candidate_num_experimental>0?order_candidate_num_experimental:max_ref_num,max_ref_num,(unsigned long long)order_reference_data_vec.size(),use_reference_data_rbf_score_experimental?"RBF":"IDW");
            else if (use_reverse_reference_reuse_order_experimental)
                printf("Using reverse reference-reuse-first elimination with %u local candidates and %u references\n",order_candidate_num_experimental>0?order_candidate_num_experimental:max_ref_num,max_ref_num);
            else if (use_reverse_s2_only_order_experimental)
                printf("Using reverse S2-only elimination with %u local candidates and references\n",max_ref_num);
            else if (use_reverse_direction_only_order_experimental)
                printf("Using reverse direction-concentration elimination with %u local candidates and references\n",max_ref_num);
            else if (use_reverse_direction_order_experimental)
                printf("Using reverse nearest-distance and direction-concentration elimination with %u local candidates and references\n",max_ref_num);
            else if (use_reverse_direction_distribution_order_experimental)
                printf("Using reverse direction-concentration and distribution elimination with %u local candidates and references\n",max_ref_num);
            else
                printf("Using reverse predictability elimination with %u local candidates\n",max_ref_num);
            uint32_t reverse_ref_query_num=min<uint64_t>((uint64_t)max_ref_num+1,point_num);
            vector<double> d_only_cached_score;
            vector<double> d_only_cached_reference_data_error;
            vector<uint32_t> d_only_cached_ref_num;
            vector<uint8_t> d_only_cache_initialized;
            vector<uint8_t> reference_valid;
            vector<int> d_only_cached_ref_ids;
            vector<double> d_only_cached_ref_weight;
            uint64_t d_only_cache_hit_num=0,d_only_cache_reduced_ref_hit_num=0,d_only_cache_miss_num=0;
            vector<int> previous_route_ref_ids;
            uint64_t selected_reference_overlap_sum=0,selected_reference_overlap_num=0;
            uint64_t reference_score_evaluation_num=0;
            uint64_t reference_score_cache_hit_num=0,reference_list_check_num=0,reference_list_unchanged_num=0;
            uint64_t reference_weight_calculation_num=0,reference_weight_reduced_list_reuse_num=0;
            if (use_reverse_d_only)
            {
                d_only_cached_score.resize(point_num);
                if (use_reference_data_order)
                    d_only_cached_reference_data_error.resize(point_num);
                d_only_cached_ref_num.resize(point_num);
                d_only_cache_initialized.assign(point_num,0);
                reference_valid.assign(point_num,1);
                d_only_cached_ref_ids.resize((uint64_t)point_num*reverse_ref_query_num);
                if (use_reference_data_order)
                    d_only_cached_ref_weight.resize((uint64_t)point_num*reverse_ref_query_num);
                printf("D-only order candidates: %u, cache refill threshold: %u\n",order_candidate_num_experimental>0?order_candidate_num_experimental:max_ref_num,d_only_cache_refill_threshold_experimental>0?d_only_cache_refill_threshold_experimental:max_ref_num);
            }
            bool reverse_center_seed_selected=false;
            for (int round_id=predict_round_num-1;round_id>=0;round_id--)
            {
                auto &target_ids=round_target_ids[round_id];
                if (target_ids.empty())
                    continue;
                tree.clear_all_valid_num();
                for (int original_id:target_ids)
                {
                    tree.set_point_valid(original_id,true);
                }
                tree.update_all_valid_num(0);
                vector<int> removed_target_ids;
                removed_target_ids.reserve(target_ids.size());
                T_coord *query_coord=center_coord.data();
                nearest_queue_c<T_coord> nearest_queue;
                nearest_queue_c<T_coord> d_only_ref_queue;
                while (removed_target_ids.size()<target_ids.size())
                {
                    uint32_t order_candidate_num=order_candidate_num_experimental>0?order_candidate_num_experimental:max_ref_num;
                    uint32_t candidate_num=min<uint32_t>(order_candidate_num,target_ids.size()-removed_target_ids.size());
                    tree.find_original_nearest(nearest_queue,coord_array,candidate_num,0,query_coord,true);
                    if (nearest_queue.empty())
                        throw runtime_error("无法找到尚未剔除的预测目标");
                    int original_id=-1;
                    T_coord best_ref_dist=numeric_limits<T_coord>::infinity(),best_route_dist=numeric_limits<T_coord>::infinity();
                    if (!reverse_center_seed_selected)
                    {
                        while (!nearest_queue.empty())
                        {
                            int candidate_id=nearest_queue.top().second;
                            T_coord route_dist=nearest_queue.top().first;
                            nearest_queue.pop();
                            if (original_id==-1 || route_dist<best_route_dist || (route_dist==best_route_dist && candidate_id<original_id))
                            {
                                original_id=candidate_id;
                                best_route_dist=route_dist;
                            }
                        }
                        reverse_center_seed_selected=true;
                    }
                    else if (use_reverse_d_only)
                    {
                        double best_mean_target_ref_squared_dist=numeric_limits<double>::infinity(),best_reference_data_error=numeric_limits<double>::infinity();
                        uint32_t best_reference_overlap=0;
                        while (!nearest_queue.empty())
                        {
                            int candidate_id=nearest_queue.top().second;
                            T_coord route_dist=nearest_queue.top().first;
                            nearest_queue.pop();
                            uint64_t cache_begin=(uint64_t)candidate_id*reverse_ref_query_num;
                            uint32_t cached_ref_num=d_only_cached_ref_num[candidate_id],valid_cached_ref_num=0;
                            bool cache_was_initialized=d_only_cache_initialized[candidate_id];
                            if (cache_was_initialized)
                                for (uint32_t ref_index=0;ref_index<cached_ref_num;ref_index++)
                                {
                                    int cached_ref_id=d_only_cached_ref_ids[cache_begin+ref_index];
                                    if (reference_valid[cached_ref_id])
                                    {
                                        d_only_cached_ref_ids[cache_begin+valid_cached_ref_num]=cached_ref_id;
                                        if (use_reference_data_order)
                                            d_only_cached_ref_weight[cache_begin+valid_cached_ref_num]=d_only_cached_ref_weight[cache_begin+ref_index];
                                        valid_cached_ref_num++;
                                    }
                                }
                            bool reference_list_unchanged=cache_was_initialized&&valid_cached_ref_num==cached_ref_num;
                            uint32_t required_cached_ref_num=d_only_cache_refill_threshold_experimental>0?min(d_only_cache_refill_threshold_experimental,cached_ref_num):cached_ref_num;
                            bool cache_valid=d_only_cache_initialized[candidate_id] && cached_ref_num>0 && valid_cached_ref_num>=required_cached_ref_num;
                            double mean_target_ref_squared_dist;
                            if (cache_valid)
                            {
                                if (valid_cached_ref_num==cached_ref_num)
                                    mean_target_ref_squared_dist=d_only_cached_score[candidate_id];
                                else
                                {
                                    double target_ref_squared_dist_sum=0;
                                    for (uint32_t ref_index=0;ref_index<valid_cached_ref_num;ref_index++)
                                        target_ref_squared_dist_sum+=(double)get_dist_to_coord(coord_array[candidate_id],coord_array[d_only_cached_ref_ids[cache_begin+ref_index]],dim);
                                    mean_target_ref_squared_dist=target_ref_squared_dist_sum/valid_cached_ref_num;
                                    d_only_cached_score[candidate_id]=mean_target_ref_squared_dist;
                                    d_only_cached_ref_num[candidate_id]=valid_cached_ref_num;
                                    d_only_cache_reduced_ref_hit_num++;
                                }
                                d_only_cache_hit_num++;
                            }
                            else
                            {
                                reference_tree->find_original_nearest(d_only_ref_queue,coord_array,reverse_ref_query_num,0,coord_array[candidate_id],true);
                                double target_ref_squared_dist_sum=0;
                                uint32_t ref_num=0;
                                while (!d_only_ref_queue.empty())
                                {
                                    int ref_id=d_only_ref_queue.top().second;
                                    T_coord squared_dist=d_only_ref_queue.top().first;
                                    d_only_ref_queue.pop();
                                    if (ref_id!=candidate_id)
                                    {
                                        target_ref_squared_dist_sum+=(double)squared_dist;
                                        d_only_cached_ref_ids[cache_begin+ref_num]=ref_id;
                                        if (use_reference_data_order)
                                        {
                                            d_only_cached_ref_weight[cache_begin+ref_num]=squared_dist==0?numeric_limits<double>::infinity():pow((double)squared_dist,-inverse_distance_power/2);
                                            reference_weight_calculation_num++;
                                        }
                                        ref_num++;
                                    }
                                }
                                mean_target_ref_squared_dist=ref_num>0?target_ref_squared_dist_sum/ref_num:numeric_limits<double>::infinity();
                                d_only_cached_score[candidate_id]=mean_target_ref_squared_dist;
                                d_only_cached_ref_num[candidate_id]=ref_num;
                                d_only_cache_initialized[candidate_id]=1;
                                d_only_cache_miss_num++;
                            }
                            uint32_t reference_overlap=0;
                            if (use_reverse_reference_reuse_order_experimental)
                                for (uint32_t ref_index=0;ref_index<d_only_cached_ref_num[candidate_id];ref_index++)
                                    if (find(previous_route_ref_ids.begin(),previous_route_ref_ids.end(),d_only_cached_ref_ids[cache_begin+ref_index])!=previous_route_ref_ids.end())
                                        reference_overlap++;
                            double reference_data_error=0;
                            if (use_reference_data_order)
                            {
                                if (cache_was_initialized)
                                {
                                    reference_list_check_num++;
                                    if (reference_list_unchanged)
                                        reference_list_unchanged_num++;
                                }
                                if (reference_list_unchanged)
                                {
                                    reference_data_error=d_only_cached_reference_data_error[candidate_id];
                                    reference_score_cache_hit_num++;
                                }
                                else
                                {
                                    uint32_t score_ref_num=d_only_cached_ref_num[candidate_id];
                                    if (cache_valid)
                                        reference_weight_reduced_list_reuse_num+=score_ref_num;
                                    vector<int> valid_ref_ids,rbf_weight_ref_ids;
                                    vector<double> rbf_weights;
                                    bool rbf_weights_valid=false;
                                    for (uint32_t reference_file_id=0;reference_file_id<order_reference_data_vec.size();reference_file_id++)
                                    {
                                        auto &reference_data=order_reference_data_vec[reference_file_id];
                                        uint32_t target_point_set_id=get_point_set_id(reference_data[candidate_id]);
                                        double weight_sum=0,prediction=0;
                                        uint32_t valid_ref_num=0;
                                        bool exact_prediction=false;
                                        valid_ref_ids.clear();
                                        for (uint32_t ref_index=0;ref_index<score_ref_num&&valid_ref_num<actual_ref_num;ref_index++)
                                        {
                                            int ref_id=d_only_cached_ref_ids[cache_begin+ref_index];
                                            if (get_point_set_id(reference_data[ref_id])!=target_point_set_id)
                                                continue;
                                            valid_ref_ids.push_back(ref_id);
                                            double weight=d_only_cached_ref_weight[cache_begin+ref_index];
                                            if (weight==numeric_limits<double>::infinity())
                                            {
                                                prediction=reference_data[ref_id];
                                                exact_prediction=true;
                                                break;
                                            }
                                            weight_sum+=weight;
                                            prediction+=weight*reference_data[ref_id];
                                            valid_ref_num++;
                                        }
                                        bool rbf_point_set=point_set_config_vec[target_point_set_id].codec==point_set_codec_e::rbf_fitting;
                                        bool try_rbf=use_reference_data_rbf_score_experimental&&rbf_point_set;
                                        bool incomplete_rbf_refs=valid_ref_num<min(actual_ref_num,score_ref_num);
                                        if(try_rbf&&(use_rbf_filtered_fallback_idw_experimental||use_rbf_filtered_fallback_nearest_experimental)&&incomplete_rbf_refs)
                                            try_rbf=false;
                                        bool used_rbf=false;
                                        if(try_rbf)
                                        {
                                            if(valid_ref_ids!=rbf_weight_ref_ids)
                                            {
                                                rbf_weights_valid=calculate_rbf_fitting_weights(candidate_id,valid_ref_ids,rbf_weights);
                                                rbf_weight_ref_ids=valid_ref_ids;
                                            }
                                            if(rbf_weights_valid&&rbf_weights.size()==valid_ref_ids.size())
                                            {
                                                prediction=0;
                                                for(uint32_t ref_index=0;ref_index<valid_ref_ids.size();ref_index++)
                                                    prediction+=rbf_weights[ref_index]*reference_data[valid_ref_ids[ref_index]];
                                                exact_prediction=true;
                                                used_rbf=true;
                                            }
                                        }
                                        if(use_reference_data_rbf_score_experimental&&rbf_point_set&&use_rbf_filtered_fallback_nearest_experimental&&!used_rbf&&!valid_ref_ids.empty())
                                        {
                                            int nearest_ref_id=valid_ref_ids[0];
                                            double nearest_squared_distance=(double)get_dist_to_coord(coord_array[candidate_id],coord_array[nearest_ref_id],dim);
                                            for(uint32_t ref_index=1;ref_index<valid_ref_ids.size();ref_index++)
                                            {
                                                int ref_id=valid_ref_ids[ref_index];
                                                double squared_distance=(double)get_dist_to_coord(coord_array[candidate_id],coord_array[ref_id],dim);
                                                if(squared_distance<nearest_squared_distance)
                                                {
                                                    nearest_ref_id=ref_id;
                                                    nearest_squared_distance=squared_distance;
                                                }
                                            }
                                            prediction=reference_data[nearest_ref_id];
                                            exact_prediction=true;
                                        }
                                        double normalized_error=1;
                                        if (exact_prediction||weight_sum>0)
                                        {
                                            if (!exact_prediction)
                                                prediction/=weight_sum;
                                            double error=abs((double)reference_data[candidate_id]-prediction);
                                            double data_range=order_reference_point_set_range_vec[reference_file_id][target_point_set_id];
                                            normalized_error=data_range>0?error/data_range:(error==0?0:1);
                                        }
                                        reference_data_error+=normalized_error;
                                    }
                                    reference_data_error/=order_reference_data_vec.size();
                                    d_only_cached_reference_data_error[candidate_id]=reference_data_error;
                                    reference_score_evaluation_num++;
                                }
                            }
                            bool better=original_id==-1;
                            bool route_better=route_dist<best_route_dist || (route_dist==best_route_dist && candidate_id<original_id);
                            bool d_better=mean_target_ref_squared_dist<best_mean_target_ref_squared_dist || (mean_target_ref_squared_dist==best_mean_target_ref_squared_dist && route_better);
                            if (use_reverse_reference_reuse_order_experimental)
                                better=better || reference_overlap>best_reference_overlap || (reference_overlap==best_reference_overlap && d_better);
                            else if (use_reference_data_order)
                                better=better || reference_data_error<best_reference_data_error || (reference_data_error==best_reference_data_error && d_better);
                            else
                                better=better || d_better;
                            if (better)
                            {
                                original_id=candidate_id;
                                best_mean_target_ref_squared_dist=mean_target_ref_squared_dist;
                                best_reference_data_error=reference_data_error;
                                best_reference_overlap=reference_overlap;
                                best_route_dist=route_dist;
                            }
                        }
                        if (use_reverse_reference_reuse_order_experimental && !previous_route_ref_ids.empty())
                        {
                            selected_reference_overlap_sum+=best_reference_overlap;
                            selected_reference_overlap_num++;
                        }
                    }
                    else if (use_reverse_geometry_score)
                    {
                        struct candidate_score_c
                        {
                            int id;
                            T_coord route_dist;
                            double first_score,second_score;
                            uint32_t first_rank=0,second_rank=0;
                        };
                        vector<candidate_score_c> candidate_scores;
                        candidate_scores.reserve(nearest_queue.size());
                        while (!nearest_queue.empty())
                        {
                            int candidate_id=nearest_queue.top().second;
                            T_coord route_dist=nearest_queue.top().first;
                            nearest_queue.pop();
                            nearest_queue_c<T_coord> ref_queue;
                            reference_tree->find_original_nearest(ref_queue,coord_array,reverse_ref_query_num,0,coord_array[candidate_id],true);
                            vector<int> ref_ids;
                            ref_ids.reserve(ref_queue.size());
                            while (!ref_queue.empty())
                            {
                                if (ref_queue.top().second!=candidate_id)
                                    ref_ids.push_back(ref_queue.top().second);
                                ref_queue.pop();
                            }
                            double nearest_ref_squared_dist=numeric_limits<double>::infinity(),mean_target_ref_squared_dist=numeric_limits<double>::infinity(),distribution_score=numeric_limits<double>::infinity(),direction_concentration=numeric_limits<double>::infinity();
                            if (!ref_ids.empty())
                            {
                                double target_ref_squared_dist_sum=0;
                                vector<double> direction_sum(dim,0);
                                uint32_t direction_num=0;
                                for (int ref_id:ref_ids)
                                {
                                    double squared_dist=(double)get_dist_to_coord(coord_array[candidate_id],coord_array[ref_id],dim);
                                    nearest_ref_squared_dist=min(nearest_ref_squared_dist,squared_dist);
                                    target_ref_squared_dist_sum+=squared_dist;
                                    if (squared_dist>0)
                                    {
                                        double inverse_dist=1/sqrt(squared_dist);
                                        for (uint32_t d=0;d<dim;d++)
                                            direction_sum[d]+=((double)coord_array[ref_id][d]-coord_array[candidate_id][d])*inverse_dist;
                                        direction_num++;
                                    }
                                }
                                mean_target_ref_squared_dist=target_ref_squared_dist_sum/ref_ids.size();
                                if (direction_num>0)
                                {
                                    double direction_squared_norm=0;
                                    for (double direction:direction_sum)
                                        direction_squared_norm+=direction*direction;
                                    direction_concentration=sqrt(direction_squared_norm)/direction_num;
                                }
                                if ((use_reverse_distribution_order_experimental || use_reverse_s2_only_order_experimental || use_reverse_direction_distribution_order_experimental) && ref_ids.size()>=2)
                                {
                                    vector<double> relative_coord_sum(dim,0);
                                    double relative_squared_norm_sum=0;
                                    for (int ref_id:ref_ids)
                                        for (uint32_t d=0;d<dim;d++)
                                        {
                                            double relative_coord=(double)coord_array[ref_id][d]-coord_array[ref_ids[0]][d];
                                            relative_coord_sum[d]+=relative_coord;
                                            relative_squared_norm_sum+=relative_coord*relative_coord;
                                        }
                                    double ref_ref_squared_dist_sum=ref_ids.size()*relative_squared_norm_sum;
                                    for (double coord_sum:relative_coord_sum)
                                        ref_ref_squared_dist_sum-=coord_sum*coord_sum;
                                    uint32_t ref_pair_num=ref_ids.size()*(ref_ids.size()-1)/2;
                                    double mean_ref_ref_squared_dist=ref_ref_squared_dist_sum/ref_pair_num;
                                    if (mean_ref_ref_squared_dist>0)
                                        distribution_score=2*mean_target_ref_squared_dist/mean_ref_ref_squared_dist;
                                }
                            }
                            double first_score=(use_reverse_distribution_order_experimental || use_reverse_mean_distance_only_order_experimental)?mean_target_ref_squared_dist:(use_reverse_s2_only_order_experimental?distribution_score:((use_reverse_direction_only_order_experimental || use_reverse_direction_distribution_order_experimental)?direction_concentration:nearest_ref_squared_dist));
                            double second_score=(use_reverse_distribution_order_experimental || use_reverse_direction_distribution_order_experimental)?distribution_score:direction_concentration;
                            candidate_scores.push_back({candidate_id,route_dist,first_score,second_score});
                        }
                        vector<uint32_t> score_order(candidate_scores.size());
                        iota(score_order.begin(),score_order.end(),0);
                        sort(score_order.begin(),score_order.end(),[&](uint32_t a,uint32_t b){return candidate_scores[a].first_score<candidate_scores[b].first_score || (candidate_scores[a].first_score==candidate_scores[b].first_score && candidate_scores[a].id<candidate_scores[b].id);});
                        uint32_t current_rank=0;
                        for (uint32_t rank=0;rank<score_order.size();rank++)
                        {
                            if (rank>0 && candidate_scores[score_order[rank]].first_score>candidate_scores[score_order[rank-1]].first_score)
                                current_rank=rank;
                            candidate_scores[score_order[rank]].first_rank=current_rank;
                        }
                        bool use_second_score=use_reverse_distribution_order_experimental || use_reverse_direction_order_experimental || use_reverse_direction_distribution_order_experimental;
                        if (use_second_score)
                        {
                            sort(score_order.begin(),score_order.end(),[&](uint32_t a,uint32_t b){return candidate_scores[a].second_score<candidate_scores[b].second_score || (candidate_scores[a].second_score==candidate_scores[b].second_score && candidate_scores[a].id<candidate_scores[b].id);});
                            current_rank=0;
                            for (uint32_t rank=0;rank<score_order.size();rank++)
                            {
                                if (rank>0 && candidate_scores[score_order[rank]].second_score>candidate_scores[score_order[rank-1]].second_score)
                                    current_rank=rank;
                                candidate_scores[score_order[rank]].second_rank=current_rank;
                            }
                        }
                        uint32_t best_rank_sum=numeric_limits<uint32_t>::max();
                        for (candidate_score_c &candidate:candidate_scores)
                        {
                            uint32_t rank_sum=candidate.first_rank+candidate.second_rank;
                            if (original_id==-1 || rank_sum<best_rank_sum || (rank_sum==best_rank_sum && (candidate.route_dist<best_route_dist || (candidate.route_dist==best_route_dist && candidate.id<original_id))))
                            {
                                original_id=candidate.id;
                                best_rank_sum=rank_sum;
                                best_route_dist=candidate.route_dist;
                            }
                        }
                    }
                    else
                    {
                        while (!nearest_queue.empty())
                        {
                            int candidate_id=nearest_queue.top().second;
                            T_coord route_dist=nearest_queue.top().first;
                            nearest_queue.pop();
                            nearest_queue_c<T_coord> ref_queue;
                            reference_tree->find_original_nearest(ref_queue,coord_array,2,0,coord_array[candidate_id],true);
                            T_coord ref_dist=numeric_limits<T_coord>::infinity();
                            while (!ref_queue.empty())
                            {
                                if (ref_queue.top().second!=candidate_id)
                                    ref_dist=min(ref_dist,ref_queue.top().first);
                                ref_queue.pop();
                            }
                            if (original_id==-1 || ref_dist<best_ref_dist || (ref_dist==best_ref_dist && (route_dist<best_route_dist || (route_dist==best_route_dist && candidate_id<original_id))))
                            {
                                original_id=candidate_id;
                                best_ref_dist=ref_dist;
                                best_route_dist=route_dist;
                            }
                        }
                    }
                    removed_target_ids.push_back(original_id);
                    tree.update_point_valid_num_to_root(original_id,-1);
                    reference_tree->update_point_valid_num_to_root(original_id,-1);
                    if (use_reverse_d_only)
                        reference_valid[original_id]=0;
                    if (use_reverse_reference_reuse_order_experimental)
                    {
                        uint64_t cache_begin=(uint64_t)original_id*reverse_ref_query_num;
                        if (!d_only_cache_initialized[original_id])
                        {
                            reference_tree->find_original_nearest(d_only_ref_queue,coord_array,reverse_ref_query_num,0,coord_array[original_id],true);
                            uint32_t ref_num=0;
                            while (!d_only_ref_queue.empty())
                            {
                                int ref_id=d_only_ref_queue.top().second;
                                d_only_ref_queue.pop();
                                if (ref_id!=original_id)
                                    d_only_cached_ref_ids[cache_begin+ref_num++]=ref_id;
                            }
                            d_only_cached_ref_num[original_id]=ref_num;
                            d_only_cache_initialized[original_id]=1;
                        }
                        previous_route_ref_ids.assign(d_only_cached_ref_ids.begin()+cache_begin,d_only_cached_ref_ids.begin()+cache_begin+d_only_cached_ref_num[original_id]);
                    }
                    query_coord=coord_array[original_id];
                }
                reverse(removed_target_ids.begin(),removed_target_ids.end());
                target_ids.swap(removed_target_ids);
                printf("Reverse-sorted round %u\n",round_id);
            }
            if (use_reverse_d_only)
            {
                printf("D-only score cache: %llu hits, %llu misses\n",(unsigned long long)d_only_cache_hit_num,(unsigned long long)d_only_cache_miss_num);
                printf("D-only reduced-reference cache hits: %llu\n",(unsigned long long)d_only_cache_reduced_ref_hit_num);
                if (use_reverse_reference_reuse_order_experimental)
                    printf("Selected adjacent reference overlap: %f / %u\n",selected_reference_overlap_num?(double)selected_reference_overlap_sum/selected_reference_overlap_num:0,max_ref_num);
                if (use_reference_data_order)
                {
                    printf("Reference-data predictability score evaluations: %llu\n",(unsigned long long)reference_score_evaluation_num);
                    printf("Reference-data neighbor lists unchanged after previous evaluation: %llu / %llu (%.6f%%)\n",(unsigned long long)reference_list_unchanged_num,(unsigned long long)reference_list_check_num,reference_list_check_num?100.0*reference_list_unchanged_num/reference_list_check_num:0);
                    uint64_t reference_score_request_num=reference_score_evaluation_num+reference_score_cache_hit_num;
                    printf("Reference-data score cache hits: %llu / %llu (%.6f%%)\n",(unsigned long long)reference_score_cache_hit_num,(unsigned long long)reference_score_request_num,reference_score_request_num?100.0*reference_score_cache_hit_num/reference_score_request_num:0);
                    printf("Reference-data raw IDW weights calculated: %llu, reused after neighbor reduction: %llu\n",(unsigned long long)reference_weight_calculation_num,(unsigned long long)reference_weight_reduced_list_reuse_num);
                }
            }
        }
        else
        {
            bool use_forward_new_geometry_score=use_forward_s2_only_order_experimental || use_forward_direction_only_order_experimental || use_forward_mean_distance_s2_order_experimental || use_forward_mean_distance_direction_order_experimental;
            bool use_forward_predictability_score=use_predictability_order_experimental || use_forward_mean_distance_only_order_experimental || use_forward_new_geometry_score;
            if (use_forward_predictability_score)
            {
                reference_tree=make_unique<block_tree_c<T_coord>>(tree);
                reference_tree->clear_all_valid_num();
                if (use_predictability_order_experimental)
                    printf("Using geometry-based predictability-aware target order with %u local candidates\n",max_ref_num);
                else if (use_forward_mean_distance_only_order_experimental)
                    printf("Using forward mean-distance-only target order with %u local candidates and references\n",max_ref_num);
                else if (use_forward_s2_only_order_experimental)
                    printf("Using forward S2-only target order with %u local candidates and references\n",max_ref_num);
                else if (use_forward_direction_only_order_experimental)
                    printf("Using forward direction-concentration-only target order with %u local candidates and references\n",max_ref_num);
                else if (use_forward_mean_distance_s2_order_experimental)
                    printf("Using forward mean-distance and S2 target order with %u local candidates and references\n",max_ref_num);
                else
                    printf("Using forward mean-distance and direction-concentration target order with %u local candidates and references\n",max_ref_num);
            }
            for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
            {
                auto &target_ids=round_target_ids[round_id];
                if (target_ids.empty())
                    continue;
                tree.clear_all_valid_num();
                for (int original_id:target_ids)
                {
                    tree.set_point_valid(original_id,true);
                }
                tree.update_all_valid_num(0);

                vector<int> sorted_target_ids;
                sorted_target_ids.reserve(target_ids.size());
                T_coord *query_coord=center_coord.data();
                while (sorted_target_ids.size()<target_ids.size())
                {
                    nearest_queue_c<T_coord> nearest_queue;
                    uint32_t candidate_num=use_forward_predictability_score?min<uint32_t>(max_ref_num,target_ids.size()-sorted_target_ids.size()):1;
                    tree.find_original_nearest(nearest_queue,coord_array,candidate_num,0,query_coord,true);
                    if (nearest_queue.empty())
                        throw runtime_error("无法找到尚未排序的预测目标");
                    int original_id=-1;
                    T_coord best_route_dist=numeric_limits<T_coord>::infinity();
                    if (use_forward_new_geometry_score)
                    {
                        struct candidate_score_c
                        {
                            int id;
                            T_coord route_dist;
                            double first_score,second_score;
                            uint32_t first_rank=0,second_rank=0;
                        };
                        vector<candidate_score_c> candidate_scores;
                        candidate_scores.reserve(nearest_queue.size());
                        while (!nearest_queue.empty())
                        {
                            int candidate_id=nearest_queue.top().second;
                            T_coord route_dist=nearest_queue.top().first;
                            nearest_queue.pop();
                            nearest_queue_c<T_coord> ref_queue;
                            reference_tree->find_original_nearest(ref_queue,coord_array,max_ref_num,0,coord_array[candidate_id],true);
                            vector<int> ref_ids;
                            ref_ids.reserve(ref_queue.size());
                            while (!ref_queue.empty())
                            {
                                ref_ids.push_back(ref_queue.top().second);
                                ref_queue.pop();
                            }
                            double mean_target_ref_squared_dist=numeric_limits<double>::infinity(),distribution_score=numeric_limits<double>::infinity(),direction_concentration=numeric_limits<double>::infinity();
                            if (!ref_ids.empty())
                            {
                                double target_ref_squared_dist_sum=0;
                                vector<double> direction_sum(dim,0);
                                uint32_t direction_num=0;
                                for (int ref_id:ref_ids)
                                {
                                    double squared_dist=(double)get_dist_to_coord(coord_array[candidate_id],coord_array[ref_id],dim);
                                    target_ref_squared_dist_sum+=squared_dist;
                                    if (squared_dist>0)
                                    {
                                        double inverse_dist=1/sqrt(squared_dist);
                                        for (uint32_t d=0;d<dim;d++)
                                            direction_sum[d]+=((double)coord_array[ref_id][d]-coord_array[candidate_id][d])*inverse_dist;
                                        direction_num++;
                                    }
                                }
                                mean_target_ref_squared_dist=target_ref_squared_dist_sum/ref_ids.size();
                                if (direction_num>0)
                                {
                                    double direction_squared_norm=0;
                                    for (double direction:direction_sum)
                                        direction_squared_norm+=direction*direction;
                                    direction_concentration=sqrt(direction_squared_norm)/direction_num;
                                }
                                if ((use_forward_s2_only_order_experimental || use_forward_mean_distance_s2_order_experimental) && ref_ids.size()>=2)
                                {
                                    vector<double> relative_coord_sum(dim,0);
                                    double relative_squared_norm_sum=0;
                                    for (int ref_id:ref_ids)
                                        for (uint32_t d=0;d<dim;d++)
                                        {
                                            double relative_coord=(double)coord_array[ref_id][d]-coord_array[ref_ids[0]][d];
                                            relative_coord_sum[d]+=relative_coord;
                                            relative_squared_norm_sum+=relative_coord*relative_coord;
                                        }
                                    double ref_ref_squared_dist_sum=ref_ids.size()*relative_squared_norm_sum;
                                    for (double coord_sum:relative_coord_sum)
                                        ref_ref_squared_dist_sum-=coord_sum*coord_sum;
                                    uint32_t ref_pair_num=ref_ids.size()*(ref_ids.size()-1)/2;
                                    double mean_ref_ref_squared_dist=ref_ref_squared_dist_sum/ref_pair_num;
                                    if (mean_ref_ref_squared_dist>0)
                                        distribution_score=2*mean_target_ref_squared_dist/mean_ref_ref_squared_dist;
                                }
                            }
                            double first_score=use_forward_s2_only_order_experimental?distribution_score:(use_forward_direction_only_order_experimental?direction_concentration:mean_target_ref_squared_dist);
                            double second_score=use_forward_mean_distance_s2_order_experimental?distribution_score:direction_concentration;
                            candidate_scores.push_back({candidate_id,route_dist,first_score,second_score});
                        }
                        vector<uint32_t> score_order(candidate_scores.size());
                        iota(score_order.begin(),score_order.end(),0);
                        sort(score_order.begin(),score_order.end(),[&](uint32_t a,uint32_t b){return candidate_scores[a].first_score<candidate_scores[b].first_score || (candidate_scores[a].first_score==candidate_scores[b].first_score && candidate_scores[a].id<candidate_scores[b].id);});
                        uint32_t current_rank=0;
                        for (uint32_t rank=0;rank<score_order.size();rank++)
                        {
                            if (rank>0 && candidate_scores[score_order[rank]].first_score>candidate_scores[score_order[rank-1]].first_score)
                                current_rank=rank;
                            candidate_scores[score_order[rank]].first_rank=current_rank;
                        }
                        bool use_second_score=use_forward_mean_distance_s2_order_experimental || use_forward_mean_distance_direction_order_experimental;
                        if (use_second_score)
                        {
                            sort(score_order.begin(),score_order.end(),[&](uint32_t a,uint32_t b){return candidate_scores[a].second_score<candidate_scores[b].second_score || (candidate_scores[a].second_score==candidate_scores[b].second_score && candidate_scores[a].id<candidate_scores[b].id);});
                            current_rank=0;
                            for (uint32_t rank=0;rank<score_order.size();rank++)
                            {
                                if (rank>0 && candidate_scores[score_order[rank]].second_score>candidate_scores[score_order[rank-1]].second_score)
                                    current_rank=rank;
                                candidate_scores[score_order[rank]].second_rank=current_rank;
                            }
                        }
                        uint32_t best_rank_sum=0;
                        for (candidate_score_c &candidate:candidate_scores)
                        {
                            uint32_t rank_sum=candidate.first_rank+candidate.second_rank;
                            if (original_id==-1 || rank_sum>best_rank_sum || (rank_sum==best_rank_sum && (candidate.route_dist<best_route_dist || (candidate.route_dist==best_route_dist && candidate.id<original_id))))
                            {
                                original_id=candidate.id;
                                best_rank_sum=rank_sum;
                                best_route_dist=candidate.route_dist;
                            }
                        }
                    }
                    else
                    {
                        double best_predictability_score=-numeric_limits<double>::infinity();
                        while (!nearest_queue.empty())
                        {
                            int candidate_id=nearest_queue.top().second;
                            T_coord route_dist=nearest_queue.top().first;
                            nearest_queue.pop();
                            double predictability_score=0;
                            if (use_forward_predictability_score)
                            {
                                nearest_queue_c<T_coord> ref_queue;
                                uint32_t ref_query_num=use_forward_mean_distance_only_order_experimental?max_ref_num:1;
                                reference_tree->find_original_nearest(ref_queue,coord_array,ref_query_num,0,coord_array[candidate_id],true);
                                if (ref_queue.empty())
                                    predictability_score=numeric_limits<double>::infinity();
                                else if (use_forward_mean_distance_only_order_experimental)
                                {
                                    double squared_dist_sum=0;
                                    uint32_t ref_num=ref_queue.size();
                                    while (!ref_queue.empty())
                                    {
                                        squared_dist_sum+=(double)ref_queue.top().first;
                                        ref_queue.pop();
                                    }
                                    predictability_score=squared_dist_sum/ref_num;
                                }
                                else
                                    predictability_score=ref_queue.top().first;
                            }
                            if (original_id==-1 || predictability_score>best_predictability_score || (predictability_score==best_predictability_score && (route_dist<best_route_dist || (route_dist==best_route_dist && candidate_id<original_id))))
                            {
                                original_id=candidate_id;
                                best_predictability_score=predictability_score;
                                best_route_dist=route_dist;
                            }
                        }
                    }
                    sorted_target_ids.push_back(original_id);
                    tree.update_point_valid_num_to_root(original_id,-1);
                    if (use_forward_predictability_score)
                        reference_tree->update_point_valid_num_to_root(original_id,1);
                    query_coord=coord_array[original_id];
                }
                target_ids.swap(sorted_target_ids);
                printf("Sorted round %u\n",round_id);
            }
        }

        if (use_global_order_before_round_split_experimental)
        {
            vector<int> global_target_ids=move(round_target_ids.back());
            uint64_t begin=0;
            for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
            {
                uint64_t end=begin+round_target_num_vec[round_id];
                round_target_ids[round_id].assign(global_target_ids.begin()+begin,global_target_ids.begin()+end);
                begin=end;
                vector<T_coord> round_min_coord(dim,numeric_limits<T_coord>::infinity()),round_max_coord(dim,-numeric_limits<T_coord>::infinity());
                for (int target_id:round_target_ids[round_id])
                    for (uint32_t d=0;d<dim;d++)
                    {
                        round_min_coord[d]=min(round_min_coord[d],coord_array[target_id][d]);
                        round_max_coord[d]=max(round_max_coord[d],coord_array[target_id][d]);
                    }
                printf("Global-order round %u coordinate span ratio:",round_id);
                for (uint32_t d=0;d<dim;d++)
                {
                    double full_span=(double)max_coord[d]-min_coord[d];
                    printf(" %f",full_span>0?((double)round_max_coord[d]-round_min_coord[d])/full_span:0);
                }
                printf("\n");
            }
            if (begin!=point_num)
                throw runtime_error("全局预测顺序重新分轮后没有覆盖全部点");
        }

        if(cache_subpartition_num_experimental>0)
        {
            vector<int> cache_subpartition_root_ids(1,0);
            while(cache_subpartition_root_ids.size()<cache_subpartition_num_experimental)
            {
                vector<int> child_ids;
                child_ids.reserve(cache_subpartition_root_ids.size()*2);
                for(int root_id:cache_subpartition_root_ids)
                {
                    auto &root_block=tree.structure->block_vec[root_id];
                    if(root_block.left_child_id==-1 || root_block.right_child_id==-1)
                        throw runtime_error("kd-tree深度不足，无法生成指定数量的cache subpartition");
                    child_ids.push_back(root_block.left_child_id);
                    child_ids.push_back(root_block.right_child_id);
                }
                cache_subpartition_root_ids.swap(child_ids);
            }
            vector<uint32_t> point_cache_subpartition_id(point_num,numeric_limits<uint32_t>::max());
            for(uint32_t cache_subpartition_id=0;cache_subpartition_id<cache_subpartition_root_ids.size();cache_subpartition_id++)
            {
                auto &root_block=tree.structure->block_vec[cache_subpartition_root_ids[cache_subpartition_id]];
                for(uint32_t tree_id=root_block.index_begin;tree_id<root_block.index_end;tree_id++)
                    point_cache_subpartition_id[tree.structure->tree_id_to_original_id[tree_id]]=cache_subpartition_id;
            }
            uint32_t first_grouped_round=predict_round_num>3?predict_round_num-3:0;
            printf("Grouping rounds %u-%u by %u cache subpartitions and enabling prediction-order layout\n",first_grouped_round,predict_round_num-1,cache_subpartition_num_experimental);
            for(uint32_t round_id=first_grouped_round;round_id<predict_round_num;round_id++)
            {
                vector<vector<int>> grouped_target_ids(cache_subpartition_num_experimental);
                for(int target_id:round_target_ids[round_id])
                {
                    uint32_t cache_subpartition_id=point_cache_subpartition_id[target_id];
                    if(cache_subpartition_id>=cache_subpartition_num_experimental)
                        throw runtime_error("cache subpartition没有覆盖全部预测目标");
                    grouped_target_ids[cache_subpartition_id].push_back(target_id);
                }
                vector<int> reordered_target_ids;
                reordered_target_ids.reserve(round_target_ids[round_id].size());
                uint32_t min_target_num=point_num,max_target_num=0;
                for(auto &target_ids:grouped_target_ids)
                {
                    min_target_num=min<uint32_t>(min_target_num,target_ids.size());
                    max_target_num=max<uint32_t>(max_target_num,target_ids.size());
                    reordered_target_ids.insert(reordered_target_ids.end(),target_ids.begin(),target_ids.end());
                }
                round_target_ids[round_id].swap(reordered_target_ids);
                printf("Cache-subpartition round %u target_num_range=[%u,%u]\n",round_id,min_target_num,max_target_num);
            }
        }

        auto sorting_end_time=chrono::steady_clock::now();
        printf("Target sorting time: %f s\n",chrono::duration<double>(sorting_end_time-sorting_start_time).count());
        auto reference_generation_start_time=chrono::steady_clock::now();
        printf("Generating prediction references\n");
        pred_target_ids.clear();
        pred_target_ids.reserve(point_num);
        pred_ref_ids.clear();
        pred_ref_ids.reserve((uint64_t)point_num*max_ref_num);
        pred_ref_begin.clear();
        pred_ref_begin.reserve((uint64_t)point_num+1);
        pred_ref_begin.push_back(0);
        pred_ref_weights.clear();
        pred_ref_weights.reserve((uint64_t)point_num*max_ref_num);
        tree.clear_all_valid_num();
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            for (int pred_id:round_target_ids[round_id])
            {
                nearest_queue_c<T_coord> nearest_queue;
                tree.find_original_nearest(nearest_queue,coord_array,max_ref_num,0,coord_array[pred_id],true);
                pred_target_ids.push_back(pred_id);
                vector<int> ref_ids;
                ref_ids.reserve(nearest_queue.size());
                while (!nearest_queue.empty())
                {
                    ref_ids.push_back(nearest_queue.top().second);
                    nearest_queue.pop();
                }
                reverse(ref_ids.begin(),ref_ids.end());
                for(int ref_id:ref_ids)
                {
                    double dist=(double)get_dist_to_coord(coord_array[pred_id],coord_array[ref_id],dim);
                    pred_ref_ids.push_back(ref_id);
                    pred_ref_weights.push_back(dist==0?numeric_limits<double>::infinity():pow(dist,-inverse_distance_power/2));
                }
                pred_ref_begin.push_back(pred_ref_ids.size());

                tree.update_point_valid_num_to_root(pred_id,1);
            }
        }

        if (pred_target_ids.size()!=point_num || pred_ref_begin.size()!=(uint64_t)point_num+1 || pred_ref_ids.size()!=pred_ref_weights.size())
            throw runtime_error("预测参考结构没有覆盖所有点");
        if (!prediction_structure_path.empty())
            save_pred_refs_structure();
        print_prediction_geometry_statistics_result();
        prepare_pred_refs_structure();
        auto end_time=chrono::steady_clock::now();
        printf("Prediction reference generation time: %f s\n",chrono::duration<double>(end_time-reference_generation_start_time).count());
        printf("Generated %llu prediction entries, total time: %f s\n",(unsigned long long)pred_target_ids.size(),chrono::duration<double>(end_time-start_time).count());
    }
}
