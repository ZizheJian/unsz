// 将所有点分成多轮，确定预测顺序，并为每个点寻找已经完成预测的最近参考点。
#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_pred_refs_structure()
    {
        if(ordering_scope=="global"&&(partition_size_experimental>0||partition_num_experimental>0))
            throw runtime_error("global排序目前不能和partition同时使用");
        if (partition_size_experimental>0 || partition_num_experimental>0)
        {
            generate_partition_pred_refs_structure();
            return;
        }
        if (point_num<=0)
            throw runtime_error("point_num必须是正数");
        if (prediction_round_num<=0 || prediction_round_num>63)
            throw runtime_error("prediction_round_num必须在[1,63]内");
        if (prediction_max_ref_num<=0)
            throw runtime_error("prediction_max_ref_num必须是正数");
        uint32_t effective_ordering_score_max_ref_num=ordering_score_max_ref_num;
        uint32_t candidate_limit=ordering_candidate_num;
        bool use_candidate_score_geometry_reuse=ordering_scope=="global"&&effective_ordering_score_max_ref_num==1&&ordering_score_min_ref_num==1&&candidate_limit>=prediction_max_ref_num&&candidate_limit>=2;
        if(use_kd_scalar_one_reference_query_experimental&&(ordering_scope!="global"||effective_ordering_score_max_ref_num!=1||ordering_score_min_ref_num!=1||!use_kd_iterative_query_experimental))
            throw runtime_error("标量1点参考查询只支持使用迭代kd-tree的global反向1点MSD");
        if(ordering_score_min_ref_num>effective_ordering_score_max_ref_num)
            throw runtime_error("MSD排序缓存重查阈值不能大于MSD评分参考点数");

        printf("Building kd-tree\n");
        auto start_time=chrono::steady_clock::now();
        block_tree_c<T_coord> tree(coord_array,min_coord,max_coord,kd_tree_leaf_size);
        tree.use_iterative_query=use_kd_iterative_query_experimental;
        auto tree_end_time=chrono::steady_clock::now();
        printf("KD-tree blocks: %llu, time: %f s\n",(unsigned long long)tree.get_block_num(),chrono::duration<double>(tree_end_time-start_time).count());

        printf("Arranging rounds of prediction\n");
        round_target_ids.clear();
        round_target_ids.resize(prediction_round_num);
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
            round_target_ids[round_id].assign(shuffled_ids.begin()+previous_end,shuffled_ids.begin()+current_end);
            previous_end=current_end;
            printf("Round %u target num: %llu\n",round_id,(unsigned long long)round_target_ids[round_id].size());
        }
        if (previous_end!=point_num)
            throw runtime_error("预测轮次没有覆盖所有点");
        if (ordering_scope=="global")
        {
            for (uint32_t round_id=0;round_id+1<prediction_round_num;round_id++)
                round_target_ids[round_id].clear();
            round_target_ids.back()=move(shuffled_ids);
            printf("Using one global prediction order\n");
        }

        vector<T_coord> center_coord(dim);
        for (uint32_t d=0;d<dim;d++)
            center_coord[d]=(min_coord[d]+max_coord[d])/2;

        uint32_t effective_ordering_candidate_num=ordering_candidate_num;
        bool reuse_global_ordering_candidate_refs=ordering_scope=="global"&&effective_ordering_candidate_num>=prediction_max_ref_num;
        bool generate_idw_weights=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::idw;});
        bool generate_rbf_weights=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::rbf;});
        if(generate_idw_weights&&generate_rbf_weights)
            throw runtime_error("同一DAG不能同时生成IDW和RBF预测权重");
        vector<uint64_t> order_candidate_ref_begin;
        vector<uint8_t> order_candidate_ref_num;
        vector<int> order_candidate_ref_ids;
        vector<T_coord> order_candidate_ref_squared_distances;
        if(reuse_global_ordering_candidate_refs)
        {
            order_candidate_ref_begin.assign(point_num,numeric_limits<uint64_t>::max());
            order_candidate_ref_num.assign(point_num,0);
            order_candidate_ref_ids.reserve((uint64_t)point_num*prediction_max_ref_num);
            order_candidate_ref_squared_distances.reserve((uint64_t)point_num*prediction_max_ref_num);
        }

        auto sorting_start_time=chrono::steady_clock::now();
        printf("Sorting targets in each round\n");
        unique_ptr<block_tree_c<T_coord>> reference_tree_storage;
        block_tree_c<T_coord>*reference_tree=nullptr;
        bool share_global_msd_validity_tree=ordering_scope=="global"&&effective_ordering_score_max_ref_num==1;
        if(share_global_msd_validity_tree)
        {
            reference_tree=&tree;
            printf("Global 1-point MSD: target and scoring queries share one kd-tree validity state\n");
        }
        else
        {
            reference_tree_storage=make_unique<block_tree_c<T_coord>>(tree);
            reference_tree=reference_tree_storage.get();
        }
        printf("Using reverse MSD elimination with %u local candidates and %u references\n",ordering_candidate_num,prediction_max_ref_num);
        uint32_t reverse_ref_query_num=min<uint64_t>((uint64_t)effective_ordering_score_max_ref_num+1,point_num);
        {
            vector<T_coord> ordering_score_cached_score;
            vector<uint32_t> ordering_score_cached_ref_num;
            vector<uint8_t> ordering_score_cache_initialized;
            vector<uint8_t> reference_valid;
            vector<int> ordering_score_cached_ref_ids;
            uint64_t ordering_score_cache_hit_num=0,ordering_score_cache_reduced_ref_hit_num=0,ordering_score_cache_miss_num=0;
            uint64_t candidate_kd_query_num=0;
            uint64_t score_candidate_geometry_certified_num=0,score_candidate_geometry_checked_num=0;
            uint64_t candidate_kd_sample_ns=0,candidate_kd_sample_num=0,score_kd_sample_ns=0,score_kd_sample_num=0;
            ordering_score_cached_score.resize(point_num);
            ordering_score_cached_ref_num.resize(point_num);
            ordering_score_cache_initialized.assign(point_num,0);
            reference_valid.assign(point_num,1);
            ordering_score_cached_ref_ids.resize((uint64_t)point_num*reverse_ref_query_num);
            printf("MSD ordering candidates: %u, scoring references: %u, cache refill threshold: %u\n",ordering_candidate_num,effective_ordering_score_max_ref_num,ordering_score_min_ref_num);
            bool reverse_center_seed_selected=false;
            for (int round_id=prediction_round_num-1;round_id>=0;round_id--)
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
                int previous_removed_id=-1;
                nearest_queue_c<T_coord> nearest_queue;
                nearest_queue_c<T_coord> ordering_score_ref_queue;
                vector<pair<T_coord,int>> candidate_group;
                if(use_candidate_score_geometry_reuse)
                    candidate_group.reserve(candidate_limit);
                while (removed_target_ids.size()<target_ids.size())
                {
                    uint32_t candidate_num=min<uint32_t>(effective_ordering_candidate_num,target_ids.size()-removed_target_ids.size());
                    if((candidate_kd_query_num&255)==0)
                    {
                        auto query_start=chrono::steady_clock::now();
                        tree.find_original_nearest(nearest_queue,coord_array,candidate_num,0,query_coord,true);
                        candidate_kd_sample_ns+=chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now()-query_start).count();
                        candidate_kd_sample_num++;
                    }
                    else
                        tree.find_original_nearest(nearest_queue,coord_array,candidate_num,0,query_coord,true);
                    candidate_kd_query_num++;
                    if (nearest_queue.empty())
                        throw runtime_error("无法找到尚未剔除的预测目标");
                    if(reuse_global_ordering_candidate_refs&&previous_removed_id>=0)
                    {
                        nearest_queue_c<T_coord> ref_queue=nearest_queue;
                        if(use_candidate_score_geometry_reuse)
                            candidate_group.clear();
                        uint32_t keep_num=min<uint32_t>(prediction_max_ref_num,ref_queue.size());
                        uint64_t ref_begin=order_candidate_ref_ids.size();
                        order_candidate_ref_ids.resize(ref_begin+keep_num);
                        order_candidate_ref_squared_distances.resize(ref_begin+keep_num);
                        while(ref_queue.size()>keep_num)
                        {
                            if(use_candidate_score_geometry_reuse)
                                candidate_group.push_back(ref_queue.top());
                            ref_queue.pop();
                        }
                        for(uint32_t ref_index=keep_num;ref_index>0;ref_index--)
                        {
                            if(use_candidate_score_geometry_reuse)
                                candidate_group.push_back(ref_queue.top());
                            order_candidate_ref_ids[ref_begin+ref_index-1]=ref_queue.top().second;
                            order_candidate_ref_squared_distances[ref_begin+ref_index-1]=ref_queue.top().first;
                            ref_queue.pop();
                        }
                        order_candidate_ref_begin[previous_removed_id]=ref_begin;
                        order_candidate_ref_num[previous_removed_id]=keep_num;
                    }
                    int original_id=-1;
                    T_coord best_route_dist=numeric_limits<T_coord>::infinity();
                    if (!reverse_center_seed_selected)
                    {
                        while (!nearest_queue.empty())
                        {
                            int candidate_id=nearest_queue.top().second;
                            T_coord route_dist=nearest_queue.top().first;
                            nearest_queue.pop();
                            if (original_id==-1 || route_dist<best_route_dist)
                            {
                                original_id=candidate_id;
                                best_route_dist=route_dist;
                            }
                        }
                        reverse_center_seed_selected=true;
                    }
                    else
                    {
                        T_coord best_mean_target_ref_squared_dist=numeric_limits<T_coord>::infinity();
                        while (!nearest_queue.empty())
                        {
                            int candidate_id=nearest_queue.top().second;
                            T_coord route_dist=nearest_queue.top().first;
                            nearest_queue.pop();
                            uint64_t cache_begin=(uint64_t)candidate_id*reverse_ref_query_num;
                            uint32_t cached_ref_num=ordering_score_cached_ref_num[candidate_id],valid_cached_ref_num=0;
                            bool cache_was_initialized=ordering_score_cache_initialized[candidate_id];
                            bool reference_list_unchanged=false;
                            if (cache_was_initialized)
                            {
                                for(uint32_t ref_index=0;ref_index<cached_ref_num;ref_index++)
                                {
                                    int cached_ref_id=ordering_score_cached_ref_ids[cache_begin+ref_index];
                                    if (reference_valid[cached_ref_id])
                                    {
                                        ordering_score_cached_ref_ids[cache_begin+valid_cached_ref_num]=cached_ref_id;
                                        valid_cached_ref_num++;
                                    }
                                }
                                reference_list_unchanged=valid_cached_ref_num==cached_ref_num;
                            }
                            uint32_t required_cached_ref_num=min(ordering_score_min_ref_num,cached_ref_num);
                            bool cache_valid=ordering_score_cache_initialized[candidate_id] && cached_ref_num>0 && valid_cached_ref_num>=required_cached_ref_num;
                            T_coord mean_target_ref_squared_dist;
                            if (cache_valid)
                            {
                                if (reference_list_unchanged)
                                    mean_target_ref_squared_dist=ordering_score_cached_score[candidate_id];
                                else
                                {
                                    T_coord target_ref_squared_dist_sum=0;
                                    for(uint32_t ref_index=0;ref_index<valid_cached_ref_num;ref_index++)
                                        target_ref_squared_dist_sum+=get_dist_to_coord(coord_array[candidate_id],coord_array[ordering_score_cached_ref_ids[cache_begin+ref_index]],dim);
                                    mean_target_ref_squared_dist=target_ref_squared_dist_sum/valid_cached_ref_num;
                                    ordering_score_cached_score[candidate_id]=mean_target_ref_squared_dist;
                                    ordering_score_cached_ref_num[candidate_id]=valid_cached_ref_num;
                                    ordering_score_cache_reduced_ref_hit_num++;
                                }
                                ordering_score_cache_hit_num++;
                            }
                            else
                            {
                                bool certified_nearest=false;
                                int certified_ref_id=-1;
                                T_coord certified_squared_dist=numeric_limits<T_coord>::infinity();
                                if(use_candidate_score_geometry_reuse&&candidate_group.size()>1)
                                {
                                    score_candidate_geometry_checked_num++;
                                    T_coord radius=sqrt(candidate_group.front().first),route_radius=sqrt(route_dist);
                                    T_coord margin=(radius-route_radius)*(T_coord)0.999;
                                    if(candidate_group.size()==target_ids.size()-removed_target_ids.size())
                                        margin=numeric_limits<T_coord>::infinity();
                                    if(margin>0)
                                    {
                                        for(const auto&peer:candidate_group)
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
                                        certified_nearest=certified_ref_id>=0&&certified_squared_dist<margin*margin;
                                    }
                                }
                                if(certified_nearest)
                                {
                                    mean_target_ref_squared_dist=certified_squared_dist;
                                    ordering_score_cached_score[candidate_id]=certified_squared_dist;
                                    ordering_score_cached_ref_ids[cache_begin]=certified_ref_id;
                                    ordering_score_cached_ref_num[candidate_id]=1;
                                    ordering_score_cache_initialized[candidate_id]=1;
                                    score_candidate_geometry_certified_num++;
                                }
                                else
                                {
                                    if((ordering_score_cache_miss_num&255)==0)
                                    {
                                        auto query_start=chrono::steady_clock::now();
                                        if(use_kd_scalar_one_reference_query_experimental)
                                        {
                                            T_coord nearest_distance;
                                            int nearest_original_id;
                                            if(reference_tree->find_original_nearest_other_iterative(coord_array[candidate_id],candidate_id,0,nearest_distance,nearest_original_id,true))
                                                ordering_score_ref_queue.push({nearest_distance,nearest_original_id});
                                        }
                                        else
                                            reference_tree->find_original_nearest(ordering_score_ref_queue,coord_array,reverse_ref_query_num,0,coord_array[candidate_id],true);
                                        score_kd_sample_ns+=chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now()-query_start).count();
                                        score_kd_sample_num++;
                                    }
                                    else
                                    {
                                        if(use_kd_scalar_one_reference_query_experimental)
                                        {
                                            T_coord nearest_distance;
                                            int nearest_original_id;
                                            if(reference_tree->find_original_nearest_other_iterative(coord_array[candidate_id],candidate_id,0,nearest_distance,nearest_original_id,true))
                                                ordering_score_ref_queue.push({nearest_distance,nearest_original_id});
                                        }
                                        else
                                            reference_tree->find_original_nearest(ordering_score_ref_queue,coord_array,reverse_ref_query_num,0,coord_array[candidate_id],true);
                                    }
                                    T_coord target_ref_squared_dist_sum=0;
                                    uint32_t ref_num=0;
                                    while (!ordering_score_ref_queue.empty())
                                    {
                                        int ref_id=ordering_score_ref_queue.top().second;
                                        T_coord squared_dist=ordering_score_ref_queue.top().first;
                                        ordering_score_ref_queue.pop();
                                        if (ref_id!=candidate_id)
                                        {
                                            target_ref_squared_dist_sum+=squared_dist;
                                            ordering_score_cached_ref_ids[cache_begin+ref_num]=ref_id;
                                            ref_num++;
                                        }
                                    }
                                    mean_target_ref_squared_dist=ref_num>0?target_ref_squared_dist_sum/ref_num:numeric_limits<T_coord>::infinity();
                                    ordering_score_cached_score[candidate_id]=mean_target_ref_squared_dist;
                                    ordering_score_cached_ref_num[candidate_id]=ref_num;
                                    ordering_score_cache_initialized[candidate_id]=1;
                                    ordering_score_cache_miss_num++;
                                }
                            }
                            bool route_better=route_dist<best_route_dist;
                            bool better=original_id==-1||mean_target_ref_squared_dist<best_mean_target_ref_squared_dist||(mean_target_ref_squared_dist==best_mean_target_ref_squared_dist&&route_better);
                            if (better)
                            {
                                original_id=candidate_id;
                                best_mean_target_ref_squared_dist=mean_target_ref_squared_dist;
                                best_route_dist=route_dist;
                            }
                        }
                    }
                    removed_target_ids.push_back(original_id);
                    tree.update_point_valid_num_to_root(original_id,-1);
                    if(!share_global_msd_validity_tree)
                        reference_tree->update_point_valid_num_to_root(original_id,-1);
                    reference_valid[original_id]=0;
                    previous_removed_id=original_id;
                    query_coord=coord_array[original_id];
                }
                reverse(removed_target_ids.begin(),removed_target_ids.end());
                target_ids.swap(removed_target_ids);
                printf("Reverse-sorted round %u\n",round_id);
            }
            printf("MSD ordering score cache: %llu hits, %llu misses\n",(unsigned long long)ordering_score_cache_hit_num,(unsigned long long)ordering_score_cache_miss_num);
            double candidate_kd_mean_us=candidate_kd_sample_num?0.001*(double)candidate_kd_sample_ns/candidate_kd_sample_num:0;
            double score_kd_mean_us=score_kd_sample_num?0.001*(double)score_kd_sample_ns/score_kd_sample_num:0;
            printf("MSD ordering KD query sample: candidate=%llu/%llu,mean=%.3f us; score=%llu/%llu,mean=%.3f us\n",(unsigned long long)candidate_kd_sample_num,(unsigned long long)candidate_kd_query_num,candidate_kd_mean_us,(unsigned long long)score_kd_sample_num,(unsigned long long)ordering_score_cache_miss_num,score_kd_mean_us);
            if(use_candidate_score_geometry_reuse)
                printf("MSD candidate-geometry score reuse: checked=%llu,certified=%llu,candidate_kd_queries=%llu,score_kd_queries=%llu,total_kd_queries=%llu\n",(unsigned long long)score_candidate_geometry_checked_num,(unsigned long long)score_candidate_geometry_certified_num,(unsigned long long)candidate_kd_query_num,(unsigned long long)ordering_score_cache_miss_num,(unsigned long long)(candidate_kd_query_num+ordering_score_cache_miss_num));
            printf("MSD ordering reduced-reference cache hits: %llu\n",(unsigned long long)ordering_score_cache_reduced_ref_hit_num);
        }
        if (ordering_scope=="global")
        {
            vector<int> global_target_ids=move(round_target_ids.back());
            uint64_t begin=0;
            for (uint32_t round_id=0;round_id<prediction_round_num;round_id++)
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
        pred_target_ids.clear();
        pred_target_ids.reserve(point_num);
        pred_ref_ids.clear();
        pred_ref_ids.reserve((uint64_t)point_num*prediction_max_ref_num);
        pred_ref_begin.clear();
        pred_ref_begin.reserve((uint64_t)point_num+1);
        pred_ref_begin.push_back(0);
        pred_ref_prediction_weights.clear();
        if(generate_idw_weights||generate_rbf_weights)
            pred_ref_prediction_weights.reserve((uint64_t)point_num*prediction_max_ref_num);
        pred_ref_rbf_fallback_vec.clear();
        if(generate_rbf_weights)
            pred_ref_rbf_fallback_vec.assign(point_num,0);
        vector<T_coord> rbf_reference_coords,rbf_target_ref_squared_distances;
        if(generate_rbf_weights)
        {
            rbf_reference_coords.reserve((uint64_t)point_num*prediction_max_ref_num*dim);
            rbf_target_ref_squared_distances.reserve((uint64_t)point_num*prediction_max_ref_num);
        }
        tree.clear_all_valid_num();
        uint64_t reused_reference_query_num=0;
        uint64_t reused_distance_num=0;
        for (uint32_t round_id=0;round_id<prediction_round_num;round_id++)
        {
            for (int pred_id:round_target_ids[round_id])
            {
                pred_target_ids.push_back(pred_id);
                vector<int> ref_ids;
                vector<T_coord> target_ref_squared_distances;
                uint64_t saved_ref_begin=reuse_global_ordering_candidate_refs?order_candidate_ref_begin[pred_id]:numeric_limits<uint64_t>::max();
                if(saved_ref_begin!=numeric_limits<uint64_t>::max())
                {
                    uint32_t saved_ref_num=order_candidate_ref_num[pred_id];
                    ref_ids.assign(order_candidate_ref_ids.begin()+saved_ref_begin,order_candidate_ref_ids.begin()+saved_ref_begin+saved_ref_num);
                    target_ref_squared_distances.assign(order_candidate_ref_squared_distances.begin()+saved_ref_begin,order_candidate_ref_squared_distances.begin()+saved_ref_begin+saved_ref_num);
                    reused_reference_query_num++;
                    reused_distance_num+=saved_ref_num;
                }
                else if(reuse_global_ordering_candidate_refs)
                {
                    if(pred_target_ids.size()!=1)
                        throw runtime_error("全局排序没有保存预测目标的参考点");
                }
                else
                {
                    nearest_queue_c<T_coord> nearest_queue;
                    tree.find_original_nearest(nearest_queue,coord_array,prediction_max_ref_num,0,coord_array[pred_id],true);
                    ref_ids.reserve(nearest_queue.size());
                    target_ref_squared_distances.reserve(nearest_queue.size());
                    while(!nearest_queue.empty())
                    {
                        ref_ids.push_back(nearest_queue.top().second);
                        target_ref_squared_distances.push_back(nearest_queue.top().first);
                        nearest_queue.pop();
                    }
                    reverse(ref_ids.begin(),ref_ids.end());
                    reverse(target_ref_squared_distances.begin(),target_ref_squared_distances.end());
                }
                for(int ref_id:ref_ids)
                    pred_ref_ids.push_back(ref_id);
                pred_ref_begin.push_back(pred_ref_ids.size());
                if(generate_idw_weights)
                    append_normalized_idw_prediction_weights(pred_id,array_view_c<int>(ref_ids.data(),ref_ids.size()),pred_ref_prediction_weights,array_view_c<T_coord>(target_ref_squared_distances.data(),target_ref_squared_distances.size()));
                else if(generate_rbf_weights)
                {
                    uint64_t coordinate_begin=rbf_reference_coords.size();
                    rbf_reference_coords.resize(coordinate_begin+(uint64_t)ref_ids.size()*dim);
                    for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
                    {
                        int ref_id=ref_ids[ref_index];
                        for(uint32_t dimension=0;dimension<dim;dimension++)
                            rbf_reference_coords[coordinate_begin+(uint64_t)ref_index*dim+dimension]=coord_array[ref_id][dimension];
                    }
                    rbf_target_ref_squared_distances.insert(rbf_target_ref_squared_distances.end(),target_ref_squared_distances.begin(),target_ref_squared_distances.end());
                }

                tree.update_point_valid_num_to_root(pred_id,1);
            }
        }

        if(generate_rbf_weights)
        {
            uint64_t geometry_bytes=(uint64_t)rbf_reference_coords.size()*sizeof(T_coord)+(uint64_t)rbf_target_ref_squared_distances.size()*sizeof(T_coord);
            printf("Stored temporary RBF geometry: coordinates=%llu,distances=%llu,bytes=%llu\n",(unsigned long long)rbf_reference_coords.size(),(unsigned long long)rbf_target_ref_squared_distances.size(),(unsigned long long)geometry_bytes);
            calculate_batched_rbf_prediction_weights(rbf_reference_coords,rbf_target_ref_squared_distances);
        }

        if(pred_target_ids.size()!=point_num||pred_ref_begin.size()!=(uint64_t)point_num+1||((generate_idw_weights||generate_rbf_weights)&&pred_ref_ids.size()!=pred_ref_prediction_weights.size()))
            throw runtime_error("预测参考结构没有覆盖所有点");
        if(reuse_global_ordering_candidate_refs)
            printf("Reused sorting candidate queries for prediction references: %llu targets, squared distances: %llu, additional kd-tree queries: 0\n",(unsigned long long)reused_reference_query_num,(unsigned long long)reused_distance_num);
        if (!dag_cache_path.empty())
            save_pred_refs_structure();
        prepare_pred_refs_structure();
        auto end_time=chrono::steady_clock::now();
        printf("Prediction reference generation time: %f s\n",chrono::duration<double>(end_time-reference_generation_start_time).count());
        printf("Generated %llu prediction entries, total time: %f s\n",(unsigned long long)pred_target_ids.size(),chrono::duration<double>(end_time-start_time).count());
    }
}
