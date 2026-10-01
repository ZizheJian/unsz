#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::build_partitions()
    {
        if (partition_size_experimental==0 && partition_num_experimental==0)
            return;
        if (partition_size_experimental>0 && partition_num_experimental>0)
            throw runtime_error("不能同时指定partition大小和partition数量");
        uint32_t tree_leaf_size=kd_tree_leaf_size;
        if (partition_size_experimental>0)
            tree_leaf_size=min(tree_leaf_size,partition_size_experimental);
        else
            tree_leaf_size=min(tree_leaf_size,(point_num+partition_num_experimental-1)/partition_num_experimental);
        block_tree_c<T_coord> tree(coord_array,min_coord,max_coord,tree_leaf_size);
        build_partitions(tree);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::build_partitions(block_tree_c<T_coord> &tree)
    {
        if (partition_size_experimental>0 && partition_num_experimental>0)
            throw runtime_error("不能同时指定partition大小和partition数量");
        uint32_t requested_partition_num=partition_num_experimental;
        if (requested_partition_num==0)
        {
            requested_partition_num=1;
            while (((uint64_t)point_num+requested_partition_num-1)/requested_partition_num>partition_size_experimental)
                requested_partition_num*=2;
        }
        if ((requested_partition_num&(requested_partition_num-1))!=0)
            throw runtime_error("partition数量目前必须是2的幂");
        partition_root_block_ids.assign(1,0);
        while (partition_root_block_ids.size()<requested_partition_num)
        {
            vector<int> child_ids;
            child_ids.reserve(partition_root_block_ids.size()*2);
            for (int block_root_id:partition_root_block_ids)
            {
                auto &root_block=tree.structure->block_vec[block_root_id];
                if(root_block.left_child_id==-1 || root_block.right_child_id==-1)
                    throw runtime_error("kd-tree深度不足，无法生成指定数量的partition");
                child_ids.push_back(root_block.left_child_id);
                child_ids.push_back(root_block.right_child_id);
            }
            partition_root_block_ids.swap(child_ids);
        }
        partition_point_ids_vec.assign(requested_partition_num,vector<int>());
        point_partition_id_vec.assign(point_num,numeric_limits<uint32_t>::max());
        uint32_t min_block_point_num=point_num,max_block_point_num=0;
        for (uint32_t partition_id=0;partition_id<requested_partition_num;partition_id++)
        {
            auto &root_block=tree.structure->block_vec[partition_root_block_ids[partition_id]];
            auto &point_ids=partition_point_ids_vec[partition_id];
            point_ids.reserve(root_block.index_end-root_block.index_begin);
            for(uint32_t tree_id=root_block.index_begin;tree_id<root_block.index_end;tree_id++)
            {
                int original_id=tree.structure->tree_id_to_original_id[tree_id];
                point_ids.push_back(original_id);
                point_partition_id_vec[original_id]=partition_id;
            }
            min_block_point_num=min<uint32_t>(min_block_point_num,point_ids.size());
            max_block_point_num=max<uint32_t>(max_block_point_num,point_ids.size());
        }
        for (uint32_t block_id:point_partition_id_vec)
            if (block_id==numeric_limits<uint32_t>::max())
                throw runtime_error("partition没有覆盖所有点");
        printf("Independent spatial partitions: %u, point_num_range=[%u,%u]\n",requested_partition_num,min_block_point_num,max_block_point_num);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_partition_pred_refs_structure()
    {
        auto start_time=chrono::steady_clock::now();
        uint32_t effective_ordering_score_max_ref_num=ordering_score_max_ref_num;
        if(ordering_score_min_ref_num>effective_ordering_score_max_ref_num)
            throw runtime_error("MSD排序缓存重查阈值不能大于ordering_score_max_ref_num");
        uint32_t tree_leaf_size=kd_tree_leaf_size;
        if (partition_size_experimental>0)
            tree_leaf_size=min(tree_leaf_size,partition_size_experimental);
        else
            tree_leaf_size=min(tree_leaf_size,(point_num+partition_num_experimental-1)/partition_num_experimental);
        block_tree_c<T_coord> query_tree(coord_array,min_coord,max_coord,tree_leaf_size);
        build_partitions(query_tree);
        printf("Generating fully independent partition prediction DAG with partition_num=%llu, thread_num=%u\n",(unsigned long long)partition_point_ids_vec.size(),thread_num);
        printf("Round size growth ratio: %.9g\n",round_size_growth_ratio_experimental);
        vector<vector<vector<int>>> partition_round_target_ids(partition_point_ids_vec.size(),vector<vector<int>>(prediction_round_num));
        mt19937 rng(0);
        for (uint32_t partition_id=0;partition_id<partition_point_ids_vec.size();partition_id++)
        {
            vector<int> shuffled_ids=partition_point_ids_vec[partition_id];
            shuffle(shuffled_ids.begin(),shuffled_ids.end(),rng);
            uint64_t previous_end=0;
            vector<uint32_t> round_target_num_vec=get_round_target_num_vec((uint32_t)shuffled_ids.size());
            for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
            {
                uint64_t current_end=previous_end+round_target_num_vec[round_id];
                partition_round_target_ids[partition_id][round_id].assign(shuffled_ids.begin()+previous_end,shuffled_ids.begin()+current_end);
                previous_end=current_end;
            }
            if (previous_end!=shuffled_ids.size())
                throw runtime_error("partition预测轮次没有覆盖全部点");
        }

        auto sorting_start_time=chrono::steady_clock::now();
        block_tree_c<T_coord> reference_tree(query_tree),target_tree(query_tree);
        target_tree.clear_all_valid_num();
        uint32_t reverse_ref_query_num=min<uint64_t>((uint64_t)effective_ordering_score_max_ref_num+1,point_num);
        vector<double> ordering_score_cached_score(point_num);
        vector<uint32_t> ordering_score_cached_ref_num(point_num);
        vector<uint8_t> ordering_score_cache_initialized(point_num,0),reference_valid(point_num,1);
        vector<int> ordering_score_cached_ref_ids((uint64_t)point_num*reverse_ref_query_num);
        uint64_t ordering_score_cache_hit_num=0,ordering_score_cache_reduced_ref_hit_num=0,ordering_score_cache_miss_num=0;
        printf("Independent partition MSD ordering candidates: %u, score references: %u, cache refill threshold: %u\n",ordering_candidate_num,effective_ordering_score_max_ref_num,ordering_score_min_ref_num);
        exception_ptr sorting_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(thread_num>1) reduction(+:ordering_score_cache_hit_num,ordering_score_cache_reduced_ref_hit_num,ordering_score_cache_miss_num)
        for (int64_t partition_index=0;partition_index<(int64_t)partition_point_ids_vec.size();partition_index++)
        {
            try
            {
            uint32_t partition_id=(uint32_t)partition_index;
            int block_root_id=partition_root_block_ids[partition_id];
            vector<T_coord> center_coord(dim);
            for (uint32_t d=0;d<dim;d++)
                center_coord[d]=(query_tree.get_block_min_coord(block_root_id)[d]+query_tree.get_block_max_coord(block_root_id)[d])/2;
            for (int round_id=prediction_round_num-1;round_id>=0;round_id--)
            {
                auto &target_ids=partition_round_target_ids[partition_id][round_id];
                if (target_ids.empty())
                    continue;
                for (int original_id:target_ids)
                    target_tree.set_point_valid(original_id,true);
                target_tree.update_all_valid_num(block_root_id);
                vector<int> removed_target_ids;
                removed_target_ids.reserve(target_ids.size());
                T_coord *query_coord=center_coord.data();
                while (removed_target_ids.size()<target_ids.size())
                {
                    nearest_queue_c<T_coord> candidate_queue;
                    uint32_t effective_ordering_candidate_num=ordering_candidate_num;
                    uint32_t candidate_num=min<uint32_t>(effective_ordering_candidate_num,target_ids.size()-removed_target_ids.size());
                    target_tree.find_original_nearest(candidate_queue,coord_array,candidate_num,block_root_id,query_coord,true);
                    int selected_id=-1;
                    double best_mean_squared_distance=numeric_limits<double>::infinity();
                    T_coord best_route_distance=numeric_limits<T_coord>::infinity();
                    while (!candidate_queue.empty())
                    {
                        int candidate_id=candidate_queue.top().second;
                        T_coord route_distance=candidate_queue.top().first;
                        candidate_queue.pop();
                        uint64_t cache_begin=(uint64_t)candidate_id*reverse_ref_query_num;
                        uint32_t cached_ref_num=ordering_score_cached_ref_num[candidate_id],valid_cached_ref_num=0;
                        if (ordering_score_cache_initialized[candidate_id])
                            for (uint32_t ref_index=0;ref_index<cached_ref_num;ref_index++)
                            {
                                int cached_ref_id=ordering_score_cached_ref_ids[cache_begin+ref_index];
                                if (reference_valid[cached_ref_id])
                                    ordering_score_cached_ref_ids[cache_begin+valid_cached_ref_num++]=cached_ref_id;
                            }
                        uint32_t required_cached_ref_num=min(ordering_score_min_ref_num,cached_ref_num);
                        bool cache_valid=ordering_score_cache_initialized[candidate_id] && cached_ref_num>0 && valid_cached_ref_num>=required_cached_ref_num;
                        double mean_squared_distance;
                        if (cache_valid)
                        {
                            if (valid_cached_ref_num==cached_ref_num)
                                mean_squared_distance=ordering_score_cached_score[candidate_id];
                            else
                            {
                                double squared_distance_sum=0;
                                for (uint32_t ref_index=0;ref_index<valid_cached_ref_num;ref_index++)
                                    squared_distance_sum+=(double)get_dist_to_coord(coord_array[candidate_id],coord_array[ordering_score_cached_ref_ids[cache_begin+ref_index]],dim);
                                mean_squared_distance=squared_distance_sum/valid_cached_ref_num;
                                ordering_score_cached_score[candidate_id]=mean_squared_distance;
                                ordering_score_cached_ref_num[candidate_id]=valid_cached_ref_num;
                                ordering_score_cache_reduced_ref_hit_num++;
                            }
                            ordering_score_cache_hit_num++;
                        }
                        else
                        {
                            nearest_queue_c<T_coord> reference_queue;
                            uint32_t partition_ref_query_num=min<uint32_t>(reverse_ref_query_num,partition_point_ids_vec[partition_id].size());
                            reference_tree.find_original_nearest(reference_queue,coord_array,partition_ref_query_num,block_root_id,coord_array[candidate_id],true);
                            double squared_distance_sum=0;
                            uint32_t reference_num=0;
                            while (!reference_queue.empty())
                            {
                                int ref_id=reference_queue.top().second;
                                T_coord squared_distance=reference_queue.top().first;
                                reference_queue.pop();
                                if (ref_id!=candidate_id)
                                {
                                    squared_distance_sum+=squared_distance;
                                    ordering_score_cached_ref_ids[cache_begin+reference_num]=ref_id;
                                    reference_num++;
                                }
                            }
                            mean_squared_distance=reference_num>0?squared_distance_sum/reference_num:numeric_limits<double>::infinity();
                            ordering_score_cached_score[candidate_id]=mean_squared_distance;
                            ordering_score_cached_ref_num[candidate_id]=reference_num;
                            ordering_score_cache_initialized[candidate_id]=1;
                            ordering_score_cache_miss_num++;
                        }
                        if (selected_id==-1 || mean_squared_distance<best_mean_squared_distance || (mean_squared_distance==best_mean_squared_distance && route_distance<best_route_distance))
                        {
                            selected_id=candidate_id;
                            best_mean_squared_distance=mean_squared_distance;
                            best_route_distance=route_distance;
                        }
                    }
                    if (selected_id<0)
                        throw runtime_error("partition内无法找到预测目标");
                    removed_target_ids.push_back(selected_id);
                    target_tree.update_point_valid_num_to_partition_root(selected_id,-1,block_root_id);
                    reference_tree.update_point_valid_num_to_partition_root(selected_id,-1,block_root_id);
                    reference_valid[selected_id]=0;
                    query_coord=coord_array[selected_id];
                }
                reverse(removed_target_ids.begin(),removed_target_ids.end());
                target_ids.swap(removed_target_ids);
            }
            }
            catch(...)
            {
#pragma omp critical
                if(!sorting_exception)
                    sorting_exception=current_exception();
            }
        }
        if(sorting_exception)
            rethrow_exception(sorting_exception);
        printf("Independent partition target sorting time: %f s\n",chrono::duration<double>(chrono::steady_clock::now()-sorting_start_time).count());
        printf("Independent partition MSD ordering score cache: %llu hits, %llu misses\n",(unsigned long long)ordering_score_cache_hit_num,(unsigned long long)ordering_score_cache_miss_num);
        printf("Independent partition MSD ordering reduced-reference cache hits: %llu\n",(unsigned long long)ordering_score_cache_reduced_ref_hit_num);

        vector<vector<vector<pair<int,vector<int>>>>> partition_round_entries(partition_point_ids_vec.size(),vector<vector<pair<int,vector<int>>>>(prediction_round_num));
        vector<vector<vector<vector<T_calc>>>> partition_round_weights(partition_point_ids_vec.size(),vector<vector<vector<T_calc>>>(prediction_round_num));
        bool generate_idw_weights=any_of(point_set_config_vec.begin(),point_set_config_vec.end(),[](const point_set_config_c &config){return config.codec==point_set_codec_e::idw;});
        reference_tree.clear_all_valid_num();
        exception_ptr reference_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(thread_num>1)
        for (int64_t partition_index=0;partition_index<(int64_t)partition_point_ids_vec.size();partition_index++)
        {
            try
            {
            uint32_t partition_id=(uint32_t)partition_index;
            int block_root_id=partition_root_block_ids[partition_id];
            for (uint32_t round_id=0;round_id<prediction_round_num;round_id++)
                for (int pred_id:partition_round_target_ids[partition_id][round_id])
                {
                    nearest_queue_c<T_coord> reference_queue;
                    reference_tree.find_original_nearest(reference_queue,coord_array,prediction_max_ref_num,block_root_id,coord_array[pred_id],true);
                    vector<int> ref_ids;
                    while (!reference_queue.empty())
                    {
                        ref_ids.push_back(reference_queue.top().second);
                        reference_queue.pop();
                    }
                    reverse(ref_ids.begin(),ref_ids.end());
                    vector<T_calc> ref_weights;
                    if(generate_idw_weights)
                        append_normalized_idw_prediction_weights(pred_id,array_view_c<int>(ref_ids.data(),ref_ids.size()),ref_weights);
                    partition_round_entries[partition_id][round_id].push_back({pred_id,move(ref_ids)});
                    partition_round_weights[partition_id][round_id].push_back(move(ref_weights));
                    reference_tree.update_point_valid_num_to_partition_root(pred_id,1,block_root_id);
                }
            }
            catch(...)
            {
#pragma omp critical
                if(!reference_exception)
                    reference_exception=current_exception();
            }
        }
        if(reference_exception)
            rethrow_exception(reference_exception);

        prediction_target_id_groups.assign(prediction_round_num,vector<int>());
        pred_target_ids.clear();
        pred_target_ids.reserve(point_num);
        pred_ref_ids.clear();
        pred_ref_ids.reserve((uint64_t)point_num*prediction_max_ref_num);
        pred_ref_begin.clear();
        pred_ref_begin.reserve((uint64_t)point_num+1);
        pred_ref_begin.push_back(0);
        pred_ref_prediction_weights.clear();
        pred_ref_rbf_fallback_vec.clear();
        if(generate_idw_weights)
            pred_ref_prediction_weights.reserve((uint64_t)point_num*prediction_max_ref_num);
        for (uint32_t round_id=0;round_id<prediction_round_num;round_id++)
            for (uint32_t partition_id=0;partition_id<partition_point_ids_vec.size();partition_id++)
                for (uint32_t entry_id=0;entry_id<partition_round_entries[partition_id][round_id].size();entry_id++)
                {
                    prediction_target_id_groups[round_id].push_back(partition_round_entries[partition_id][round_id][entry_id].first);
                    auto &entry=partition_round_entries[partition_id][round_id][entry_id];
                    auto &weights=partition_round_weights[partition_id][round_id][entry_id];
                    pred_target_ids.push_back(entry.first);
                    pred_ref_ids.insert(pred_ref_ids.end(),entry.second.begin(),entry.second.end());
                    if(generate_idw_weights)
                        pred_ref_prediction_weights.insert(pred_ref_prediction_weights.end(),weights.begin(),weights.end());
                    pred_ref_begin.push_back(pred_ref_ids.size());
                }
        if (pred_target_ids.size()!=point_num)
            throw runtime_error("独立partition预测DAG没有覆盖所有点");
        if (!dag_cache_path.empty())
            save_pred_refs_structure();
        prepare_pred_refs_structure();
        printf("Generated independent partition prediction entries: %llu, total time: %f s\n",(unsigned long long)pred_target_ids.size(),chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
    }
}
