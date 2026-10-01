// 生成global或local预测顺序、参考关系与预测权重。
#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_pred_refs_structure()
    {
        if(ordering_scope=="global"&&(partition_size_experimental>0||partition_num_experimental>0))
            throw runtime_error("global排序目前不能和partition同时使用");
        if(partition_size_experimental>0||partition_num_experimental>0)
        {
            generate_partition_pred_refs_structure();
            return;
        }
        if(point_num==0)
            throw runtime_error("point_num必须是正数");
        if(prediction_round_num==0||prediction_round_num>63)
            throw runtime_error("prediction_round_num必须在[1,63]内");
        if(prediction_max_ref_num==0)
            throw runtime_error("prediction_max_ref_num必须是正数");
        if(ordering_score_min_ref_num>ordering_score_max_ref_num)
            throw runtime_error("MSD排序缓存重查阈值不能大于MSD评分参考点数");
        if(ordering_scope!="global"&&ordering_scope!="local")
            throw runtime_error("ordering_scope必须是global或local");
        printf("Building kd-tree\n");
        auto generation_start_time=chrono::steady_clock::now();
        block_tree_c<T_coord> tree(coord_array,min_coord,max_coord,kd_tree_leaf_size);
        auto tree_end_time=chrono::steady_clock::now();
        printf("KD-tree blocks: %llu, time: %f s\n",(unsigned long long)tree.get_block_num(),chrono::duration<double>(tree_end_time-generation_start_time).count());
        bool uses_idw=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::idw;});
        bool uses_rbf=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::rbf;});
        if(uses_idw&&uses_rbf)
            throw runtime_error("同一DAG不能同时生成IDW和RBF预测权重");
        if(ordering_scope=="global")
            generate_global_pred_refs_structure(tree,generation_start_time,uses_idw,uses_rbf);
        else
            generate_local_pred_refs_structure(tree,generation_start_time,uses_idw,uses_rbf);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_global_pred_refs_structure(block_tree_c<T_coord> &tree,chrono::steady_clock::time_point generation_start_time,bool uses_idw,bool uses_rbf)
    {
        printf("Arranging global prediction order\n");
        prediction_target_id_groups.assign(1,vector<int>(point_num));
        iota(prediction_target_id_groups.front().begin(),prediction_target_id_groups.front().end(),0);

        // global排序会倒序写入参考关系，因此先为每个预测位置确定固定的连续数组区间。
        pred_ref_begin.resize((uint64_t)point_num+1);
        pred_ref_begin[0]=0;
        for(uint32_t prediction_index=0;prediction_index<point_num;prediction_index++)
            pred_ref_begin[prediction_index+1]=pred_ref_begin[prediction_index]+min<uint32_t>(prediction_index,prediction_max_ref_num);
        pred_ref_ids.resize(pred_ref_begin.back());
        pred_ref_prediction_weights.clear();
        if(uses_idw)
            pred_ref_prediction_weights.resize(pred_ref_ids.size());
        pred_ref_rbf_fallback_vec.clear();
        vector<T_coord> prediction_ref_squared_distances;
        if(uses_rbf)
            prediction_ref_squared_distances.resize(pred_ref_ids.size());
        vector<T_coord> prediction_ref_coords;
        if(uses_rbf)
            prediction_ref_coords.resize((uint64_t)pred_ref_ids.size()*dim);

        sort_targets_by_reverse_msd<true>(tree,tree,uses_idw,uses_rbf,prediction_ref_coords,prediction_ref_squared_distances);
        pred_target_ids=prediction_target_id_groups.front();

        auto reference_generation_start_time=chrono::steady_clock::now();
        printf("Preparing prediction weights from sorting geometry\n");
        if(uses_rbf)
        {
            uint64_t geometry_bytes=(uint64_t)prediction_ref_coords.size()*sizeof(T_coord)+(uint64_t)prediction_ref_squared_distances.size()*sizeof(T_coord);
            printf("Stored temporary RBF geometry: coordinates=%llu,distances=%llu,bytes=%llu\n",(unsigned long long)prediction_ref_coords.size(),(unsigned long long)prediction_ref_squared_distances.size(),(unsigned long long)geometry_bytes);
            calculate_batched_rbf_prediction_weights(prediction_ref_coords,prediction_ref_squared_distances);
        }
        printf("Reused sorting candidate queries for prediction references: %llu targets, squared distances: %llu, additional kd-tree queries: 0\n",(unsigned long long)(point_num>0?point_num-1:0),(unsigned long long)pred_ref_ids.size());
        if(!dag_cache_path.empty())
            save_pred_refs_structure();
        prepare_pred_refs_structure();
        auto end_time=chrono::steady_clock::now();
        printf("Prediction reference generation time: %f s\n",chrono::duration<double>(end_time-reference_generation_start_time).count());
        printf("Generated %llu prediction entries, total time: %f s\n",(unsigned long long)pred_target_ids.size(),chrono::duration<double>(end_time-generation_start_time).count());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_local_pred_refs_structure(block_tree_c<T_coord> &tree,chrono::steady_clock::time_point generation_start_time,bool uses_idw,bool uses_rbf)
    {
        printf("Arranging local prediction rounds\n");
        prediction_target_id_groups.assign(prediction_round_num,vector<int>());
        vector<int> shuffled_ids(point_num);
        iota(shuffled_ids.begin(),shuffled_ids.end(),0);
        mt19937 rng(0);
        shuffle(shuffled_ids.begin(),shuffled_ids.end(),rng);
        uint64_t previous_end=0;
        vector<uint32_t> round_target_num_vec=get_round_target_num_vec(point_num);
        printf("Round size growth ratio: %.9g\n",round_size_growth_ratio_experimental);
        for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
        {
            uint64_t current_end=previous_end+round_target_num_vec[round_id];
            prediction_target_id_groups[round_id].assign(shuffled_ids.begin()+previous_end,shuffled_ids.begin()+current_end);
            previous_end=current_end;
            printf("Round %u target num: %llu\n",round_id,(unsigned long long)prediction_target_id_groups[round_id].size());
        }
        if(previous_end!=point_num)
            throw runtime_error("预测轮次没有覆盖所有点");

        pred_target_ids.clear();
        pred_ref_ids.clear();
        pred_ref_begin.clear();
        pred_ref_begin.push_back(0);
        pred_ref_prediction_weights.clear();
        pred_ref_rbf_fallback_vec.clear();
        pred_target_ids.reserve(point_num);
        pred_ref_ids.reserve((uint64_t)point_num*prediction_max_ref_num);
        pred_ref_begin.reserve((uint64_t)point_num+1);
        if(uses_idw||uses_rbf)
            pred_ref_prediction_weights.reserve((uint64_t)point_num*prediction_max_ref_num);
        vector<T_coord> prediction_ref_coords,prediction_ref_squared_distances;
        if(uses_rbf)
        {
            prediction_ref_coords.reserve((uint64_t)point_num*prediction_max_ref_num*dim);
            prediction_ref_squared_distances.reserve((uint64_t)point_num*prediction_max_ref_num);
        }

        block_tree_c<T_coord> reference_tree(tree);
        sort_targets_by_reverse_msd<false>(tree,reference_tree,uses_idw,uses_rbf,prediction_ref_coords,prediction_ref_squared_distances);

        // local排序阶段只确定目标顺序；按照正向轮次重新遍历，才能得到真正已经可用的参考点。
        auto reference_generation_start_time=chrono::steady_clock::now();
        printf("Generating prediction references\n");
        tree.clear_all_valid_num();
        for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
            for(int pred_id:prediction_target_id_groups[round_id])
            {
                pred_target_ids.push_back(pred_id);
                vector<int> ref_ids;
                vector<T_coord> current_target_ref_squared_distances;
                nearest_queue_c<T_coord> nearest_queue;
                tree.find_original_nearest(nearest_queue,coord_array,prediction_max_ref_num,0,coord_array[pred_id],true);
                ref_ids.reserve(nearest_queue.size());
                current_target_ref_squared_distances.reserve(nearest_queue.size());
                while(!nearest_queue.empty())
                {
                    ref_ids.push_back(nearest_queue.top().second);
                    current_target_ref_squared_distances.push_back(nearest_queue.top().first);
                    nearest_queue.pop();
                }
                reverse(ref_ids.begin(),ref_ids.end());
                reverse(current_target_ref_squared_distances.begin(),current_target_ref_squared_distances.end());
                pred_ref_ids.insert(pred_ref_ids.end(),ref_ids.begin(),ref_ids.end());
                pred_ref_begin.push_back(pred_ref_ids.size());
                if(uses_idw)
                    append_normalized_idw_prediction_weights(pred_id,array_view_c<int>(ref_ids.data(),ref_ids.size()),pred_ref_prediction_weights,array_view_c<T_coord>(current_target_ref_squared_distances.data(),current_target_ref_squared_distances.size()));
                else if(uses_rbf)
                {
                    uint64_t coordinate_begin=prediction_ref_coords.size();
                    prediction_ref_coords.resize(coordinate_begin+(uint64_t)ref_ids.size()*dim);
                    for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
                    {
                        int ref_id=ref_ids[ref_index];
                        for(uint32_t dimension=0;dimension<dim;dimension++)
                            prediction_ref_coords[coordinate_begin+(uint64_t)ref_index*dim+dimension]=coord_array[ref_id][dimension];
                    }
                    prediction_ref_squared_distances.insert(prediction_ref_squared_distances.end(),current_target_ref_squared_distances.begin(),current_target_ref_squared_distances.end());
                }
                tree.update_point_valid_num_to_root(pred_id,1);
            }
        if(uses_rbf)
        {
            uint64_t geometry_bytes=(uint64_t)prediction_ref_coords.size()*sizeof(T_coord)+(uint64_t)prediction_ref_squared_distances.size()*sizeof(T_coord);
            printf("Stored temporary RBF geometry: coordinates=%llu,distances=%llu,bytes=%llu\n",(unsigned long long)prediction_ref_coords.size(),(unsigned long long)prediction_ref_squared_distances.size(),(unsigned long long)geometry_bytes);
            calculate_batched_rbf_prediction_weights(prediction_ref_coords,prediction_ref_squared_distances);
        }
        if(!dag_cache_path.empty())
            save_pred_refs_structure();
        prepare_pred_refs_structure();
        auto end_time=chrono::steady_clock::now();
        printf("Prediction reference generation time: %f s\n",chrono::duration<double>(end_time-reference_generation_start_time).count());
        printf("Generated %llu prediction entries, total time: %f s\n",(unsigned long long)pred_target_ids.size(),chrono::duration<double>(end_time-generation_start_time).count());
    }

    template<typename T_coord,typename T_data>
    template<bool global_order>
    void param_c<T_coord,T_data>::sort_targets_by_reverse_msd(block_tree_c<T_coord> &target_tree,block_tree_c<T_coord> &reference_tree,bool uses_idw,bool uses_rbf,vector<T_coord> &prediction_ref_coords,vector<T_coord> &prediction_ref_squared_distances)
    {
        uint32_t ordering_neighbor_query_num=max(ordering_candidate_num,prediction_max_ref_num);
        bool use_candidate_score_geometry_reuse=global_order&&ordering_score_max_ref_num==1&&ordering_score_min_ref_num==1&&ordering_neighbor_query_num>=2;
        bool share_validity_tree=&reference_tree==&target_tree;
        vector<T_coord> center_coord(dim);
        for(uint32_t dimension=0;dimension<dim;dimension++)
            center_coord[dimension]=(min_coord[dimension]+max_coord[dimension])/2;
        auto sorting_start_time=chrono::steady_clock::now();
        if constexpr(global_order)
            printf("Sorting global prediction targets\n");
        else
            printf("Sorting targets in each local round\n");
        if(share_validity_tree)
            printf("Global MSD: target and scoring queries share one kd-tree validity state\n");
        printf("Using reverse MSD elimination with %u nearby candidates and %u references\n",ordering_candidate_num,prediction_max_ref_num);
        uint32_t score_query_neighbor_num=min<uint64_t>((uint64_t)ordering_score_max_ref_num+1,point_num);
        {
            vector<T_coord> cached_msd_scores;
            vector<uint32_t> cached_ref_counts;
            vector<uint8_t> score_cache_initialized;
            vector<uint8_t> reference_is_valid;
            vector<int> cached_score_ref_ids;
            uint64_t score_cache_hit_num=0,score_cache_reduced_ref_hit_num=0,score_cache_miss_num=0;
            uint64_t candidate_kd_query_num=0;
            uint64_t score_candidate_geometry_certified_num=0,score_candidate_geometry_checked_num=0;
            uint64_t candidate_kd_sample_ns=0,candidate_kd_sample_num=0,score_kd_sample_ns=0,score_kd_sample_num=0;
            cached_msd_scores.resize(point_num);
            cached_ref_counts.resize(point_num);
            score_cache_initialized.assign(point_num,0);
            reference_is_valid.assign(point_num,1);
            cached_score_ref_ids.resize((uint64_t)point_num*score_query_neighbor_num);
            printf("MSD ordering candidates: %u, queried neighbors: %u, scoring references: %u, cache refill threshold: %u\n",ordering_candidate_num,ordering_neighbor_query_num,ordering_score_max_ref_num,ordering_score_min_ref_num);
            bool has_selected_initial_target=false;
            // local必须先排序后面的轮次，使评分参考集合包含所有更早的轮次。
            for(int round_id=(int)prediction_target_id_groups.size()-1;round_id>=0;round_id--)
            {
                auto &target_ids=prediction_target_id_groups[round_id];
                if(target_ids.empty())
                    continue;
                target_tree.clear_all_valid_num();
                for(int target_id:target_ids)
                    target_tree.set_point_valid(target_id,true);
                target_tree.update_all_valid_num(0);
                vector<int> reverse_order_target_ids;
                reverse_order_target_ids.reserve(target_ids.size());
                T_coord *candidate_query_coord=center_coord.data();
                int previous_selected_target_id=-1;
                nearest_queue_c<T_coord> candidate_queue;
                nearest_queue_c<T_coord> score_ref_queue;
                vector<pair<T_coord,int>> queried_neighbor_group;
                vector<T_coord> current_target_ref_squared_distances;
                if(use_candidate_score_geometry_reuse)
                    queried_neighbor_group.reserve(ordering_neighbor_query_num);
                if constexpr(global_order)
                    if(uses_idw)
                        current_target_ref_squared_distances.reserve(prediction_max_ref_num);
                auto query_score_references=[&](int candidate_id)
                {
                    auto execute_query=[&]()
                    {
                        reference_tree.find_original_nearest(score_ref_queue,coord_array,score_query_neighbor_num,0,coord_array[candidate_id],true);
                    };
                    if((score_cache_miss_num&255)==0)
                    {
                        auto query_start=chrono::steady_clock::now();
                        execute_query();
                        score_kd_sample_ns+=chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now()-query_start).count();
                        score_kd_sample_num++;
                    }
                    else
                        execute_query();
                };
                while(reverse_order_target_ids.size()<target_ids.size())
                {
                    uint32_t remaining_target_num=(uint32_t)(target_ids.size()-reverse_order_target_ids.size());
                    uint32_t queried_neighbor_num=min(ordering_neighbor_query_num,remaining_target_num);
                    if((candidate_kd_query_num&255)==0)
                    {
                        auto query_start=chrono::steady_clock::now();
                        target_tree.find_original_nearest(candidate_queue,coord_array,queried_neighbor_num,0,candidate_query_coord,true);
                        candidate_kd_sample_ns+=chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now()-query_start).count();
                        candidate_kd_sample_num++;
                    }
                    else
                        target_tree.find_original_nearest(candidate_queue,coord_array,queried_neighbor_num,0,candidate_query_coord,true);
                    candidate_kd_query_num++;
                    if(candidate_queue.empty())
                        throw runtime_error("无法找到尚未剔除的预测目标");
                    if constexpr(global_order)
                    {
                        // 当前查询围绕上一个反向选中的目标进行，因此最近点就是它在正向DAG中的参考点。
                        if(previous_selected_target_id>=0)
                        {
                            nearest_queue_c<T_coord> ref_queue=candidate_queue;
                            if(use_candidate_score_geometry_reuse)
                                queried_neighbor_group.clear();
                            uint32_t keep_num=min<uint32_t>(prediction_max_ref_num,ref_queue.size());
                            while(ref_queue.size()>keep_num)
                            {
                                if(use_candidate_score_geometry_reuse)
                                    queried_neighbor_group.push_back(ref_queue.top());
                                ref_queue.pop();
                            }
                            uint32_t prediction_index=remaining_target_num;
                            uint64_t ref_begin=pred_ref_begin[prediction_index];
                            if(pred_ref_begin[prediction_index+1]-ref_begin!=keep_num)
                                throw runtime_error("全局排序参考点数量与最终DAG位置不一致");
                            if(uses_idw)
                                current_target_ref_squared_distances.resize(keep_num);
                            for(uint32_t ref_index=keep_num;ref_index>0;ref_index--)
                            {
                                if(use_candidate_score_geometry_reuse)
                                    queried_neighbor_group.push_back(ref_queue.top());
                                uint64_t ref_offset=ref_begin+ref_index-1;
                                int ref_id=ref_queue.top().second;
                                pred_ref_ids[ref_offset]=ref_id;
                                if(uses_idw)
                                    current_target_ref_squared_distances[ref_index-1]=ref_queue.top().first;
                                if(uses_rbf)
                                    prediction_ref_squared_distances[ref_offset]=ref_queue.top().first;
                                if(uses_rbf)
                                    for(uint32_t dimension=0;dimension<dim;dimension++)
                                        prediction_ref_coords[ref_offset*dim+dimension]=coord_array[ref_id][dimension];
                                ref_queue.pop();
                            }
                            if(uses_idw)
                                write_normalized_idw_prediction_weights(previous_selected_target_id,array_view_c<int>(pred_ref_ids.data()+ref_begin,keep_num),pred_ref_prediction_weights.data()+ref_begin,array_view_c<T_coord>(current_target_ref_squared_distances));
                        }
                    }
                    while(candidate_queue.size()>ordering_candidate_num)
                        candidate_queue.pop();
                    int selected_target_id=-1;
                    T_coord best_path_squared_distance=numeric_limits<T_coord>::infinity();
                    if(!has_selected_initial_target)
                    {
                        // 第一个反向目标仅由包围盒中心确定，后续目标再使用MSD评分。
                        while(!candidate_queue.empty())
                        {
                            int candidate_id=candidate_queue.top().second;
                            T_coord path_squared_distance=candidate_queue.top().first;
                            candidate_queue.pop();
                            if(selected_target_id==-1||path_squared_distance<best_path_squared_distance)
                            {
                                selected_target_id=candidate_id;
                                best_path_squared_distance=path_squared_distance;
                            }
                        }
                        has_selected_initial_target=true;
                    }
                    else
                    {
                        T_coord best_msd=numeric_limits<T_coord>::infinity();
                        while(!candidate_queue.empty())
                        {
                            int candidate_id=candidate_queue.top().second;
                            T_coord path_squared_distance=candidate_queue.top().first;
                            candidate_queue.pop();
                            uint64_t cache_begin=(uint64_t)candidate_id*score_query_neighbor_num;
                            uint32_t cached_ref_num=cached_ref_counts[candidate_id],valid_cached_ref_num=0;
                            bool cache_was_initialized=score_cache_initialized[candidate_id];
                            bool reference_list_unchanged=false;
                            if(cache_was_initialized)
                            {
                                for(uint32_t ref_index=0;ref_index<cached_ref_num;ref_index++)
                                {
                                    int cached_ref_id=cached_score_ref_ids[cache_begin+ref_index];
                                    if(reference_is_valid[cached_ref_id])
                                    {
                                        cached_score_ref_ids[cache_begin+valid_cached_ref_num]=cached_ref_id;
                                        valid_cached_ref_num++;
                                    }
                                }
                                reference_list_unchanged=valid_cached_ref_num==cached_ref_num;
                            }
                            uint32_t required_cached_ref_num=min(ordering_score_min_ref_num,cached_ref_num);
                            bool cache_valid=score_cache_initialized[candidate_id]&&cached_ref_num>0&&valid_cached_ref_num>=required_cached_ref_num;
                            T_coord candidate_msd;
                            // 优先复用仍然有效的评分参考点，避免再次查询kd-tree。
                            if(cache_valid)
                            {
                                if(reference_list_unchanged)
                                    candidate_msd=cached_msd_scores[candidate_id];
                                else
                                {
                                    T_coord squared_distance_sum=0;
                                    for(uint32_t ref_index=0;ref_index<valid_cached_ref_num;ref_index++)
                                        squared_distance_sum+=get_dist_to_coord(coord_array[candidate_id],coord_array[cached_score_ref_ids[cache_begin+ref_index]],dim);
                                    candidate_msd=squared_distance_sum/valid_cached_ref_num;
                                    cached_msd_scores[candidate_id]=candidate_msd;
                                    cached_ref_counts[candidate_id]=valid_cached_ref_num;
                                    score_cache_reduced_ref_hit_num++;
                                }
                                score_cache_hit_num++;
                            }
                            else
                            {
                                // global一点MSD可先用本轮候选几何严格证明最近点；无法证明时才查询kd-tree。
                                bool certified_nearest=false;
                                int certified_ref_id=-1;
                                T_coord certified_squared_dist=numeric_limits<T_coord>::infinity();
                                if(use_candidate_score_geometry_reuse&&queried_neighbor_group.size()>1)
                                {
                                    score_candidate_geometry_checked_num++;
                                    T_coord query_radius=sqrt(queried_neighbor_group.front().first);
                                    T_coord candidate_radius=sqrt(path_squared_distance);
                                    T_coord guaranteed_radius=(query_radius-candidate_radius)*(T_coord)0.999;
                                    if(queried_neighbor_group.size()==remaining_target_num)
                                        guaranteed_radius=numeric_limits<T_coord>::infinity();
                                    if(guaranteed_radius>0)
                                    {
                                        for(const auto&peer:queried_neighbor_group)
                                        {
                                            if(peer.second==candidate_id)
                                                continue;
                                            T_coord squared_dist=get_dist_to_coord(coord_array[candidate_id],coord_array[peer.second],dim);
                                            if(squared_dist<certified_squared_dist)
                                            {
                                                certified_squared_dist=squared_dist;
                                                certified_ref_id=peer.second;
                                            }
                                        }
                                        certified_nearest=certified_ref_id>=0&&certified_squared_dist<guaranteed_radius*guaranteed_radius;
                                    }
                                }
                                if(certified_nearest)
                                {
                                    candidate_msd=certified_squared_dist;
                                    cached_msd_scores[candidate_id]=certified_squared_dist;
                                    cached_score_ref_ids[cache_begin]=certified_ref_id;
                                    cached_ref_counts[candidate_id]=1;
                                    score_cache_initialized[candidate_id]=1;
                                    score_candidate_geometry_certified_num++;
                                }
                                else
                                {
                                    query_score_references(candidate_id);
                                    T_coord squared_distance_sum=0;
                                    uint32_t ref_num=0;
                                    while(!score_ref_queue.empty())
                                    {
                                        int ref_id=score_ref_queue.top().second;
                                        T_coord squared_dist=score_ref_queue.top().first;
                                        score_ref_queue.pop();
                                        if(ref_id!=candidate_id)
                                        {
                                            squared_distance_sum+=squared_dist;
                                            cached_score_ref_ids[cache_begin+ref_num]=ref_id;
                                            ref_num++;
                                        }
                                    }
                                    candidate_msd=ref_num>0?squared_distance_sum/ref_num:numeric_limits<T_coord>::infinity();
                                    cached_msd_scores[candidate_id]=candidate_msd;
                                    cached_ref_counts[candidate_id]=ref_num;
                                    score_cache_initialized[candidate_id]=1;
                                    score_cache_miss_num++;
                                }
                            }
                            bool path_is_shorter=path_squared_distance<best_path_squared_distance;
                            bool candidate_is_better=selected_target_id==-1||candidate_msd<best_msd||(candidate_msd==best_msd&&path_is_shorter);
                            if(candidate_is_better)
                            {
                                selected_target_id=candidate_id;
                                best_msd=candidate_msd;
                                best_path_squared_distance=path_squared_distance;
                            }
                        }
                    }
                    reverse_order_target_ids.push_back(selected_target_id);
                    target_tree.update_point_valid_num_to_root(selected_target_id,-1);
                    if(!share_validity_tree)
                        reference_tree.update_point_valid_num_to_root(selected_target_id,-1);
                    reference_is_valid[selected_target_id]=0;
                    previous_selected_target_id=selected_target_id;
                    candidate_query_coord=coord_array[selected_target_id];
                }
                reverse(reverse_order_target_ids.begin(),reverse_order_target_ids.end());
                target_ids.swap(reverse_order_target_ids);
                printf("Reverse-sorted round %u\n",round_id);
            }
            printf("MSD ordering score cache: %llu hits, %llu misses\n",(unsigned long long)score_cache_hit_num,(unsigned long long)score_cache_miss_num);
            double candidate_kd_mean_us=candidate_kd_sample_num?0.001*(double)candidate_kd_sample_ns/candidate_kd_sample_num:0;
            double score_kd_mean_us=score_kd_sample_num?0.001*(double)score_kd_sample_ns/score_kd_sample_num:0;
            printf("MSD ordering KD query sample: candidate=%llu/%llu,mean=%.3f us; score=%llu/%llu,mean=%.3f us\n",(unsigned long long)candidate_kd_sample_num,(unsigned long long)candidate_kd_query_num,candidate_kd_mean_us,(unsigned long long)score_kd_sample_num,(unsigned long long)score_cache_miss_num,score_kd_mean_us);
            if(use_candidate_score_geometry_reuse)
                printf("MSD candidate-geometry score reuse: checked=%llu,certified=%llu,candidate_kd_queries=%llu,score_kd_queries=%llu,total_kd_queries=%llu\n",(unsigned long long)score_candidate_geometry_checked_num,(unsigned long long)score_candidate_geometry_certified_num,(unsigned long long)candidate_kd_query_num,(unsigned long long)score_cache_miss_num,(unsigned long long)(candidate_kd_query_num+score_cache_miss_num));
            printf("MSD ordering reduced-reference cache hits: %llu\n",(unsigned long long)score_cache_reduced_ref_hit_num);
        }

        auto sorting_end_time=chrono::steady_clock::now();
        printf("Target sorting time: %f s\n",chrono::duration<double>(sorting_end_time-sorting_start_time).count());
    }
}
