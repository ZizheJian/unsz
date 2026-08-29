// 将所有点分成多轮，确定预测顺序，并为每个点寻找已经完成预测的最近参考点。
#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_pred_refs_structure()
    {
        if (use_global_order_before_round_split_experimental && (independent_block_size_experimental>0 || independent_block_num_experimental>0))
            throw runtime_error("全局排序后分轮实验目前不能和独立分区同时使用");
        if (independent_block_size_experimental>0 || independent_block_num_experimental>0)
        {
            generate_independent_block_pred_refs_structure();
            return;
        }
        if (point_num<=0)
            throw runtime_error("point_num必须是正数");
        if (predict_round_num<=0 || predict_round_num>63)
            throw runtime_error("predict_round_num必须在[1,63]内");
        if (max_ref_num<=0)
            throw runtime_error("max_ref_num必须是正数");
        uint32_t experimental_order_num=use_predictability_order_experimental+use_forward_mean_distance_only_order_experimental+use_forward_s2_only_order_experimental+use_forward_direction_only_order_experimental+use_forward_mean_distance_s2_order_experimental+use_forward_mean_distance_direction_order_experimental+use_reverse_predictability_order_experimental+use_reverse_distribution_order_experimental+use_reverse_mean_distance_only_order_experimental+use_reverse_reference_reuse_order_experimental+use_reverse_s2_only_order_experimental+use_reverse_direction_only_order_experimental+use_reverse_direction_order_experimental+use_reverse_direction_distribution_order_experimental;
        if (experimental_order_num>1)
            throw runtime_error("实验性预测顺序不能同时启用多个");
        bool use_reverse_d_only=use_reverse_mean_distance_only_order_experimental || use_reverse_reference_reuse_order_experimental;
        if (order_candidate_num_experimental>0 && !use_reverse_d_only)
            throw runtime_error("D-only排序候选数实验只能和反向D-only顺序一起使用");
        if (use_reverse_d_only && d_only_cache_refill_threshold_experimental>max_ref_num)
            throw runtime_error("D-only缓存重查阈值不能大于max_ref_num");
        if (use_global_order_before_round_split_experimental && !use_reverse_mean_distance_only_order_experimental)
            throw runtime_error("全局排序后分轮实验目前只能和反向D-only顺序同时使用");

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
        vector<uint32_t> round_target_num_vec(predict_round_num);
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            uint64_t divisor=uint64_t(1)<<(predict_round_num-1-round_id);
            uint64_t current_end=((uint64_t)point_num+divisor-1)/divisor;
            current_end=min(current_end,(uint64_t)point_num);
            round_target_ids[round_id].assign(shuffled_ids.begin()+previous_end,shuffled_ids.begin()+current_end);
            round_target_num_vec[round_id]=round_target_ids[round_id].size();
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
            vector<uint32_t> d_only_cached_ref_num;
            vector<uint8_t> d_only_cache_initialized;
            vector<uint8_t> reference_valid;
            vector<int> d_only_cached_ref_ids;
            uint64_t d_only_cache_hit_num=0,d_only_cache_reduced_ref_hit_num=0,d_only_cache_miss_num=0;
            vector<int> previous_route_ref_ids;
            uint64_t selected_reference_overlap_sum=0,selected_reference_overlap_num=0;
            if (use_reverse_d_only)
            {
                d_only_cached_score.resize(point_num);
                d_only_cached_ref_num.resize(point_num);
                d_only_cache_initialized.assign(point_num,0);
                reference_valid.assign(point_num,1);
                d_only_cached_ref_ids.resize((uint64_t)point_num*reverse_ref_query_num);
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
                        double best_mean_target_ref_squared_dist=numeric_limits<double>::infinity();
                        uint32_t best_reference_overlap=0;
                        while (!nearest_queue.empty())
                        {
                            int candidate_id=nearest_queue.top().second;
                            T_coord route_dist=nearest_queue.top().first;
                            nearest_queue.pop();
                            uint64_t cache_begin=(uint64_t)candidate_id*reverse_ref_query_num;
                            uint32_t cached_ref_num=d_only_cached_ref_num[candidate_id],valid_cached_ref_num=0;
                            if (d_only_cache_initialized[candidate_id])
                                for (uint32_t ref_index=0;ref_index<cached_ref_num;ref_index++)
                                {
                                    int cached_ref_id=d_only_cached_ref_ids[cache_begin+ref_index];
                                    if (reference_valid[cached_ref_id])
                                        d_only_cached_ref_ids[cache_begin+valid_cached_ref_num++]=cached_ref_id;
                                }
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
                            bool better=original_id==-1;
                            bool route_better=route_dist<best_route_dist || (route_dist==best_route_dist && candidate_id<original_id);
                            bool d_better=mean_target_ref_squared_dist<best_mean_target_ref_squared_dist || (mean_target_ref_squared_dist==best_mean_target_ref_squared_dist && route_better);
                            if (use_reverse_reference_reuse_order_experimental)
                                better=better || reference_overlap>best_reference_overlap || (reference_overlap==best_reference_overlap && d_better);
                            else
                                better=better || d_better;
                            if (better)
                            {
                                original_id=candidate_id;
                                best_mean_target_ref_squared_dist=mean_target_ref_squared_dist;
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

        auto sorting_end_time=chrono::steady_clock::now();
        printf("Target sorting time: %f s\n",chrono::duration<double>(sorting_end_time-sorting_start_time).count());
        auto reference_generation_start_time=chrono::steady_clock::now();
        printf("Generating prediction references\n");
        pred_refs_vec.clear();
        pred_refs_vec.reserve(point_num);
        pred_ref_weights_vec.clear();
        pred_ref_weights_vec.reserve(point_num);
        tree.clear_all_valid_num();
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            for (int pred_id:round_target_ids[round_id])
            {
                nearest_queue_c<T_coord> nearest_queue;
                tree.find_original_nearest(nearest_queue,coord_array,max_ref_num,0,coord_array[pred_id],true);
                pred_refs_vec.push_back({pred_id,vector<int>()});
                pred_ref_weights_vec.push_back(vector<double>());
                auto &ref_ids=pred_refs_vec.back().second;
                ref_ids.reserve(nearest_queue.size());
                while (!nearest_queue.empty())
                {
                    ref_ids.push_back(nearest_queue.top().second);
                    nearest_queue.pop();
                }
                reverse(ref_ids.begin(),ref_ids.end());
                auto &ref_weights=pred_ref_weights_vec.back();
                ref_weights.reserve(ref_ids.size());
                for (int ref_id:ref_ids)
                {
                    double dist=(double)get_dist_to_coord(coord_array[pred_id],coord_array[ref_id],dim);
                    ref_weights.push_back(dist==0?numeric_limits<double>::infinity():pow(dist,-inverse_distance_power/2));
                }

                tree.update_point_valid_num_to_root(pred_id,1);
            }
        }

        if (pred_refs_vec.size()!=point_num || pred_ref_weights_vec.size()!=point_num)
            throw runtime_error("预测参考结构没有覆盖所有点");
        if (!prediction_structure_path.empty())
            save_pred_refs_structure();
        prepare_pred_refs_structure();
        auto end_time=chrono::steady_clock::now();
        printf("Prediction reference generation time: %f s\n",chrono::duration<double>(end_time-reference_generation_start_time).count());
        printf("Generated %llu prediction entries, total time: %f s\n",(unsigned long long)pred_refs_vec.size(),chrono::duration<double>(end_time-start_time).count());
    }
}
