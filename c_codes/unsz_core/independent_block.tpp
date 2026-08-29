#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::build_independent_blocks()
    {
        if (independent_block_size_experimental==0 && independent_block_num_experimental==0)
            return;
        if (independent_block_size_experimental>0 && independent_block_num_experimental>0)
            throw runtime_error("不能同时指定独立块大小和独立块数量");
        uint32_t tree_leaf_size=kd_tree_leaf_size;
        if (independent_block_size_experimental>0)
            tree_leaf_size=min(tree_leaf_size,independent_block_size_experimental);
        else
            tree_leaf_size=min(tree_leaf_size,(point_num+independent_block_num_experimental-1)/independent_block_num_experimental);
        block_tree_c<T_coord> tree(coord_array,min_coord,max_coord,tree_leaf_size);
        build_independent_blocks(tree);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::build_independent_blocks(block_tree_c<T_coord> &tree)
    {
        if (independent_block_size_experimental>0 && independent_block_num_experimental>0)
            throw runtime_error("不能同时指定独立块大小和独立块数量");
        uint32_t requested_block_num=independent_block_num_experimental;
        if (requested_block_num==0)
        {
            requested_block_num=1;
            while (((uint64_t)point_num+requested_block_num-1)/requested_block_num>independent_block_size_experimental)
                requested_block_num*=2;
        }
        if ((requested_block_num&(requested_block_num-1))!=0)
            throw runtime_error("独立空间块数量目前必须是2的幂");
        independent_block_root_ids_vec.assign(1,0);
        while (independent_block_root_ids_vec.size()<requested_block_num)
        {
            vector<int> child_ids;
            child_ids.reserve(independent_block_root_ids_vec.size()*2);
            for (int block_root_id:independent_block_root_ids_vec)
            {
                auto &block=tree.structure->block_vec[block_root_id];
                if (block.left_child_id==-1 || block.right_child_id==-1)
                    throw runtime_error("kd-tree深度不足，无法生成指定数量的独立块");
                child_ids.push_back(block.left_child_id);
                child_ids.push_back(block.right_child_id);
            }
            independent_block_root_ids_vec.swap(child_ids);
        }
        independent_block_point_ids_vec.assign(requested_block_num,vector<int>());
        point_independent_block_id_vec.assign(point_num,numeric_limits<uint32_t>::max());
        uint32_t min_block_point_num=point_num,max_block_point_num=0;
        for (uint32_t independent_block_id=0;independent_block_id<requested_block_num;independent_block_id++)
        {
            auto &tree_block=tree.structure->block_vec[independent_block_root_ids_vec[independent_block_id]];
            auto &point_ids=independent_block_point_ids_vec[independent_block_id];
            point_ids.reserve(tree_block.index_end-tree_block.index_begin);
            for (uint32_t tree_id=tree_block.index_begin;tree_id<tree_block.index_end;tree_id++)
            {
                int original_id=tree.structure->tree_id_to_original_id[tree_id];
                point_ids.push_back(original_id);
                point_independent_block_id_vec[original_id]=independent_block_id;
            }
            min_block_point_num=min<uint32_t>(min_block_point_num,point_ids.size());
            max_block_point_num=max<uint32_t>(max_block_point_num,point_ids.size());
        }
        for (uint32_t block_id:point_independent_block_id_vec)
            if (block_id==numeric_limits<uint32_t>::max())
                throw runtime_error("独立空间分块没有覆盖所有点");
        printf("Independent spatial blocks: %u, point_num_range=[%u,%u]\n",requested_block_num,min_block_point_num,max_block_point_num);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_independent_block_pred_refs_structure()
    {
        auto start_time=chrono::steady_clock::now();
        if (d_only_cache_refill_threshold_experimental>max_ref_num)
            throw runtime_error("D-only缓存重查阈值不能大于max_ref_num");
        uint32_t tree_leaf_size=kd_tree_leaf_size;
        if (independent_block_size_experimental>0)
            tree_leaf_size=min(tree_leaf_size,independent_block_size_experimental);
        else
            tree_leaf_size=min(tree_leaf_size,(point_num+independent_block_num_experimental-1)/independent_block_num_experimental);
        block_tree_c<T_coord> query_tree(coord_array,min_coord,max_coord,tree_leaf_size);
        build_independent_blocks(query_tree);
        printf("Generating fully independent block prediction structure with block_num=%llu\n",(unsigned long long)independent_block_point_ids_vec.size());
        vector<vector<vector<int>>> block_round_target_ids(independent_block_point_ids_vec.size(),vector<vector<int>>(predict_round_num));
        mt19937 rng(0);
        for (uint32_t independent_block_id=0;independent_block_id<independent_block_point_ids_vec.size();independent_block_id++)
        {
            vector<int> shuffled_ids=independent_block_point_ids_vec[independent_block_id];
            shuffle(shuffled_ids.begin(),shuffled_ids.end(),rng);
            uint64_t previous_end=0;
            for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
            {
                uint64_t divisor=uint64_t(1)<<(predict_round_num-1-round_id);
                uint64_t current_end=min<uint64_t>((shuffled_ids.size()+divisor-1)/divisor,shuffled_ids.size());
                block_round_target_ids[independent_block_id][round_id].assign(shuffled_ids.begin()+previous_end,shuffled_ids.begin()+current_end);
                previous_end=current_end;
            }
            if (previous_end!=shuffled_ids.size())
                throw runtime_error("独立块预测轮次没有覆盖全部块内点");
        }

        auto sorting_start_time=chrono::steady_clock::now();
        block_tree_c<T_coord> reference_tree(query_tree),target_tree(query_tree);
        target_tree.clear_all_valid_num();
        uint32_t reverse_ref_query_num=min<uint64_t>((uint64_t)max_ref_num+1,point_num);
        vector<double> d_only_cached_score(point_num);
        vector<uint32_t> d_only_cached_ref_num(point_num);
        vector<uint8_t> d_only_cache_initialized(point_num,0),reference_valid(point_num,1);
        vector<int> d_only_cached_ref_ids((uint64_t)point_num*reverse_ref_query_num);
        uint64_t d_only_cache_hit_num=0,d_only_cache_reduced_ref_hit_num=0,d_only_cache_miss_num=0;
        printf("Independent D-only order candidates: %u, cache refill threshold: %u\n",order_candidate_num_experimental>0?order_candidate_num_experimental:max_ref_num,d_only_cache_refill_threshold_experimental>0?d_only_cache_refill_threshold_experimental:max_ref_num);
        for (uint32_t independent_block_id=0;independent_block_id<independent_block_point_ids_vec.size();independent_block_id++)
        {
            int block_root_id=independent_block_root_ids_vec[independent_block_id];
            vector<T_coord> center_coord(dim);
            for (uint32_t d=0;d<dim;d++)
                center_coord[d]=(query_tree.get_block_min_coord(block_root_id)[d]+query_tree.get_block_max_coord(block_root_id)[d])/2;
            for (int round_id=predict_round_num-1;round_id>=0;round_id--)
            {
                auto &target_ids=block_round_target_ids[independent_block_id][round_id];
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
                    uint32_t order_candidate_num=order_candidate_num_experimental>0?order_candidate_num_experimental:max_ref_num;
                    uint32_t candidate_num=min<uint32_t>(order_candidate_num,target_ids.size()-removed_target_ids.size());
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
                        double mean_squared_distance;
                        if (cache_valid)
                        {
                            if (valid_cached_ref_num==cached_ref_num)
                                mean_squared_distance=d_only_cached_score[candidate_id];
                            else
                            {
                                double squared_distance_sum=0;
                                for (uint32_t ref_index=0;ref_index<valid_cached_ref_num;ref_index++)
                                    squared_distance_sum+=(double)get_dist_to_coord(coord_array[candidate_id],coord_array[d_only_cached_ref_ids[cache_begin+ref_index]],dim);
                                mean_squared_distance=squared_distance_sum/valid_cached_ref_num;
                                d_only_cached_score[candidate_id]=mean_squared_distance;
                                d_only_cached_ref_num[candidate_id]=valid_cached_ref_num;
                                d_only_cache_reduced_ref_hit_num++;
                            }
                            d_only_cache_hit_num++;
                        }
                        else
                        {
                            nearest_queue_c<T_coord> reference_queue;
                            uint32_t block_ref_query_num=min<uint32_t>(reverse_ref_query_num,independent_block_point_ids_vec[independent_block_id].size());
                            reference_tree.find_original_nearest(reference_queue,coord_array,block_ref_query_num,block_root_id,coord_array[candidate_id],true);
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
                                    d_only_cached_ref_ids[cache_begin+reference_num]=ref_id;
                                    reference_num++;
                                }
                            }
                            mean_squared_distance=reference_num>0?squared_distance_sum/reference_num:numeric_limits<double>::infinity();
                            d_only_cached_score[candidate_id]=mean_squared_distance;
                            d_only_cached_ref_num[candidate_id]=reference_num;
                            d_only_cache_initialized[candidate_id]=1;
                            d_only_cache_miss_num++;
                        }
                        if (selected_id==-1 || mean_squared_distance<best_mean_squared_distance || (mean_squared_distance==best_mean_squared_distance && (route_distance<best_route_distance || (route_distance==best_route_distance && candidate_id<selected_id))))
                        {
                            selected_id=candidate_id;
                            best_mean_squared_distance=mean_squared_distance;
                            best_route_distance=route_distance;
                        }
                    }
                    if (selected_id<0)
                        throw runtime_error("独立块内无法找到预测目标");
                    removed_target_ids.push_back(selected_id);
                    target_tree.update_point_valid_num_to_root(selected_id,-1);
                    reference_tree.update_point_valid_num_to_root(selected_id,-1);
                    reference_valid[selected_id]=0;
                    query_coord=coord_array[selected_id];
                }
                reverse(removed_target_ids.begin(),removed_target_ids.end());
                target_ids.swap(removed_target_ids);
            }
        }
        printf("Independent block target sorting time: %f s\n",chrono::duration<double>(chrono::steady_clock::now()-sorting_start_time).count());
        printf("Independent D-only score cache: %llu hits, %llu misses\n",(unsigned long long)d_only_cache_hit_num,(unsigned long long)d_only_cache_miss_num);
        printf("Independent D-only reduced-reference cache hits: %llu\n",(unsigned long long)d_only_cache_reduced_ref_hit_num);

        vector<vector<vector<pair<int,vector<int>>>>> block_round_entries(independent_block_point_ids_vec.size(),vector<vector<pair<int,vector<int>>>>(predict_round_num));
        vector<vector<vector<vector<double>>>> block_round_weights(independent_block_point_ids_vec.size(),vector<vector<vector<double>>>(predict_round_num));
        reference_tree.clear_all_valid_num();
        for (uint32_t independent_block_id=0;independent_block_id<independent_block_point_ids_vec.size();independent_block_id++)
        {
            int block_root_id=independent_block_root_ids_vec[independent_block_id];
            for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
                for (int pred_id:block_round_target_ids[independent_block_id][round_id])
                {
                    nearest_queue_c<T_coord> reference_queue;
                    reference_tree.find_original_nearest(reference_queue,coord_array,max_ref_num,block_root_id,coord_array[pred_id],true);
                    vector<int> ref_ids;
                    while (!reference_queue.empty())
                    {
                        ref_ids.push_back(reference_queue.top().second);
                        reference_queue.pop();
                    }
                    reverse(ref_ids.begin(),ref_ids.end());
                    vector<double> ref_weights;
                    ref_weights.reserve(ref_ids.size());
                    for (int ref_id:ref_ids)
                    {
                        double squared_distance=(double)get_dist_to_coord(coord_array[pred_id],coord_array[ref_id],dim);
                        ref_weights.push_back(squared_distance==0?numeric_limits<double>::infinity():pow(squared_distance,-inverse_distance_power/2));
                    }
                    block_round_entries[independent_block_id][round_id].push_back({pred_id,move(ref_ids)});
                    block_round_weights[independent_block_id][round_id].push_back(move(ref_weights));
                    reference_tree.update_point_valid_num_to_root(pred_id,1);
                }
        }

        round_target_ids.assign(predict_round_num,vector<int>());
        pred_refs_vec.clear();
        pred_ref_weights_vec.clear();
        pred_refs_vec.reserve(point_num);
        pred_ref_weights_vec.reserve(point_num);
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
            for (uint32_t independent_block_id=0;independent_block_id<independent_block_point_ids_vec.size();independent_block_id++)
                for (uint32_t entry_id=0;entry_id<block_round_entries[independent_block_id][round_id].size();entry_id++)
                {
                    round_target_ids[round_id].push_back(block_round_entries[independent_block_id][round_id][entry_id].first);
                    pred_refs_vec.push_back(move(block_round_entries[independent_block_id][round_id][entry_id]));
                    pred_ref_weights_vec.push_back(move(block_round_weights[independent_block_id][round_id][entry_id]));
                }
        if (pred_refs_vec.size()!=point_num)
            throw runtime_error("独立分块预测结构没有覆盖所有点");
        if (!prediction_structure_path.empty())
            save_pred_refs_structure();
        prepare_pred_refs_structure();
        printf("Generated independent block prediction entries: %llu, total time: %f s\n",(unsigned long long)pred_refs_vec.size(),chrono::duration<double>(chrono::steady_clock::now()-start_time).count());
    }
}
