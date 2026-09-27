// 将所有点分成多轮，确定预测顺序，并为每个点寻找已经完成预测的最近参考点。
#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_pred_refs_structure()
    {
        if(precomputed_neighbor_num_experimental>0)
        {
            generate_precomputed_neighbor_pred_refs_structure();
            return;
        }
        if(ordering_scope=="global"&&(partition_size_experimental>0||partition_num_experimental>0))
            throw runtime_error("global排序目前不能和partition同时使用");
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
        if (prediction_round_num<=0 || prediction_round_num>63)
            throw runtime_error("prediction_round_num必须在[1,63]内");
        if (prediction_max_ref_num<=0)
            throw runtime_error("prediction_max_ref_num必须是正数");
        bool use_reference_data_order=!reference_time_step_vec.empty();
        uint32_t enabled_ordering_method_num=use_predictability_order_experimental+use_forward_msd_order+use_forward_s2_only_order_experimental+use_forward_direction_only_order_experimental+use_forward_mean_distance_s2_order_experimental+use_forward_mean_distance_direction_order_experimental+use_reverse_predictability_order_experimental+use_reverse_distribution_order_experimental+use_reverse_msd_order+use_reverse_reference_reuse_order_experimental+use_reverse_s2_only_order_experimental+use_reverse_direction_only_order_experimental+use_reverse_direction_order_experimental+use_reverse_direction_distribution_order_experimental+use_reference_data_order;
        if(enabled_ordering_method_num>1)
            throw runtime_error("预测顺序生成方法不能同时启用多个");
        bool use_reverse_cached_score_order=use_reverse_msd_order || use_reverse_reference_reuse_order_experimental || use_reference_data_order;
        uint32_t effective_ordering_score_max_ref_num=ordering_score_max_ref_num;
        bool use_msd_neighbor_prefetch=ordering_msd_neighbor_prefetch_num_experimental>0;
        bool use_candidate_score_geometry_reuse=use_msd_candidate_score_geometry_reuse_experimental;
        uint32_t candidate_limit=ordering_candidate_num;
        if(use_candidate_score_geometry_reuse&&(ordering_scope!="global"||!use_reverse_msd_order||effective_ordering_score_max_ref_num!=1||ordering_score_min_ref_num!=1||candidate_limit<prediction_max_ref_num||candidate_limit<2||precomputed_neighbor_num_experimental>0))
            throw runtime_error("MSD候选几何评分复用只支持global反向1点MSD、缓存阈值1和至少2个排序候选");
        if(use_msd_neighbor_prefetch&&(ordering_scope!="global"||!use_reverse_msd_order||effective_ordering_score_max_ref_num!=1||ordering_score_min_ref_num!=1||precomputed_neighbor_num_experimental>0||ordering_msd_neighbor_prefetch_num_experimental<candidate_limit||ordering_msd_neighbor_prefetch_num_experimental>64))
            throw runtime_error("MSD近邻预取实验只支持global反向1点MSD与缓存阈值1，预取数量须覆盖排序候选且不超过64");
        if(use_kd_scalar_one_reference_query_experimental&&(ordering_scope!="global"||!use_reverse_msd_order||effective_ordering_score_max_ref_num!=1||ordering_score_min_ref_num!=1||use_msd_neighbor_prefetch||!use_kd_iterative_query_experimental))
            throw runtime_error("标量1点参考查询只支持使用迭代kd-tree的global反向1点MSD，且不能同时预取评分近邻");
        if(ordering_candidate_num>0&&!use_reverse_cached_score_order)
            throw runtime_error("ordering_candidate_num只能和反向MSD或参考数据辅助排序一起使用");
        if(use_reverse_cached_score_order&&ordering_score_min_ref_num>effective_ordering_score_max_ref_num)
            throw runtime_error("MSD/reference-data ordering缓存重查阈值不能大于MSD评分参考点数");
        if(ordering_scope=="global"&&!use_reverse_msd_order)
            throw runtime_error("global排序目前只能和反向MSD顺序同时使用");
        if (use_reference_data_order && dag_cache_path.empty())
            throw runtime_error("参考数据辅助顺序必须通过-dag-cache保存公共预测DAG");

        vector<vector<T_data>> order_reference_data_vec;
        vector<vector<double>> order_reference_point_set_range_vec;
        if (use_reference_data_order)
        {
            printf("Loading %llu reference time steps for reference-data-assisted common-order search\n",(unsigned long long)reference_time_step_vec.size());
            order_reference_data_vec.resize(reference_time_step_vec.size(),vector<T_data>(point_num));
            order_reference_point_set_range_vec.resize(reference_time_step_vec.size(),vector<double>(2));
            for (uint32_t reference_file_id=0;reference_file_id<reference_time_step_vec.size();reference_file_id++)
            {
                uint32_t time_step=reference_time_step_vec[reference_file_id];
                string path=data_dir+field_name+".dat."+to_string(time_step);
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
        if(use_kd_leaf_valid_mask_experimental)
            tree.enable_leaf_valid_mask();
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
        vector<uint64_t> order_candidate_ref_begin;
        vector<uint8_t> order_candidate_ref_num;
        vector<int> order_candidate_ref_ids;
        if(reuse_global_ordering_candidate_refs)
        {
            order_candidate_ref_begin.assign(point_num,numeric_limits<uint64_t>::max());
            order_candidate_ref_num.assign(point_num,0);
            order_candidate_ref_ids.reserve((uint64_t)point_num*prediction_max_ref_num);
        }

        auto sorting_start_time=chrono::steady_clock::now();
        printf("Sorting targets in each round\n");
        unique_ptr<block_tree_c<T_coord>> reference_tree_storage;
        block_tree_c<T_coord>*reference_tree=nullptr;
        bool share_global_msd_validity_tree=ordering_scope=="global"&&use_reverse_msd_order&&effective_ordering_score_max_ref_num==1;
        bool use_reverse_geometry_score=use_reverse_distribution_order_experimental || use_reverse_msd_order || use_reverse_s2_only_order_experimental || use_reverse_direction_only_order_experimental || use_reverse_direction_order_experimental || use_reverse_direction_distribution_order_experimental;
        if (use_reverse_predictability_order_experimental || use_reverse_geometry_score || use_reverse_cached_score_order)
        {
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
            if (use_reverse_distribution_order_experimental)
                printf("Using reverse distance-distribution elimination with %u local candidates and references\n",prediction_max_ref_num);
            else if (use_reverse_msd_order)
                printf("Using reverse MSD elimination with %u local candidates and %u references\n",ordering_candidate_num,prediction_max_ref_num);
            else if (use_reference_data_order)
                printf("Using reverse reference-data-assisted elimination with %u local candidates, %u references, %llu reference time steps and %s error scoring\n",ordering_candidate_num,prediction_max_ref_num,(unsigned long long)order_reference_data_vec.size(),use_reference_data_rbf_score?"RBF":"IDW");
            else if (use_reverse_reference_reuse_order_experimental)
                printf("Using reverse reference-reuse-first elimination with %u local candidates and %u references\n",ordering_candidate_num,prediction_max_ref_num);
            else if (use_reverse_s2_only_order_experimental)
                printf("Using reverse S2-only elimination with %u local candidates and references\n",prediction_max_ref_num);
            else if (use_reverse_direction_only_order_experimental)
                printf("Using reverse direction-concentration elimination with %u local candidates and references\n",prediction_max_ref_num);
            else if (use_reverse_direction_order_experimental)
                printf("Using reverse nearest-distance and direction-concentration elimination with %u local candidates and references\n",prediction_max_ref_num);
            else if (use_reverse_direction_distribution_order_experimental)
                printf("Using reverse direction-concentration and distribution elimination with %u local candidates and references\n",prediction_max_ref_num);
            else
                printf("Using reverse predictability elimination with %u local candidates\n",prediction_max_ref_num);
            uint32_t reverse_ref_query_num=min<uint64_t>((uint64_t)(use_msd_neighbor_prefetch?ordering_msd_neighbor_prefetch_num_experimental:(use_reverse_cached_score_order?effective_ordering_score_max_ref_num:prediction_max_ref_num))+1,point_num);
            vector<double> ordering_score_cached_score;
            vector<double> ordering_score_cached_reference_data_error;
            vector<uint32_t> ordering_score_cached_ref_num;
            vector<uint8_t> ordering_score_cache_first_valid_index;
            vector<uint8_t> ordering_score_cache_initialized;
            vector<uint8_t> reference_valid;
            vector<int> ordering_score_cached_ref_ids;
            vector<double> ordering_score_cached_ref_weight;
            uint64_t ordering_score_cache_hit_num=0,ordering_score_cache_reduced_ref_hit_num=0,ordering_score_cache_miss_num=0;
            uint64_t candidate_kd_query_num=0,candidate_cache_reuse_num=0;
            uint64_t score_candidate_geometry_certified_num=0,score_candidate_geometry_checked_num=0;
            uint64_t candidate_kd_sample_ns=0,candidate_kd_sample_num=0,score_kd_sample_ns=0,score_kd_sample_num=0;
            nearest_query_stats_c candidate_kd_stats,score_kd_stats;
            vector<int> previous_route_ref_ids;
            uint64_t selected_reference_overlap_sum=0,selected_reference_overlap_num=0;
            uint64_t reference_score_evaluation_num=0;
            uint64_t reference_score_cache_hit_num=0,reference_list_check_num=0,reference_list_unchanged_num=0;
            uint64_t reference_weight_calculation_num=0,reference_weight_reduced_list_reuse_num=0;
            if (use_reverse_cached_score_order)
            {
                ordering_score_cached_score.resize(point_num);
                if (use_reference_data_order)
                    ordering_score_cached_reference_data_error.resize(point_num);
                ordering_score_cached_ref_num.resize(point_num);
                if(use_msd_neighbor_prefetch)
                    ordering_score_cache_first_valid_index.assign(point_num,0);
                ordering_score_cache_initialized.assign(point_num,0);
                reference_valid.assign(point_num,1);
                ordering_score_cached_ref_ids.resize((uint64_t)point_num*reverse_ref_query_num);
                if (use_reference_data_order)
                    ordering_score_cached_ref_weight.resize((uint64_t)point_num*reverse_ref_query_num);
                printf("MSD/reference-data ordering order candidates: %u, MSD references: %u, cache refill threshold: %u\n",ordering_candidate_num,effective_ordering_score_max_ref_num,ordering_score_min_ref_num);
            }
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
                vector<pair<T_coord,int>> fetched_ref_buffer;
                if(use_msd_neighbor_prefetch)
                    fetched_ref_buffer.reserve(reverse_ref_query_num);
                vector<pair<T_coord,int>> candidate_group;
                if(use_candidate_score_geometry_reuse)
                    candidate_group.reserve(candidate_limit);
                while (removed_target_ids.size()<target_ids.size())
                {
                    uint32_t candidate_num=min<uint32_t>(effective_ordering_candidate_num,target_ids.size()-removed_target_ids.size());
                    bool reused_candidate_cache=false;
                    if(use_msd_neighbor_prefetch&&previous_removed_id>=0&&ordering_score_cache_initialized[previous_removed_id])
                    {
                        uint64_t cache_begin=(uint64_t)previous_removed_id*reverse_ref_query_num;
                        uint32_t cached_num=ordering_score_cached_ref_num[previous_removed_id],valid_num=0;
                        uint32_t first_index=ordering_score_cache_first_valid_index[previous_removed_id];
                        for(uint32_t ref_index=first_index;ref_index<cached_num&&valid_num<candidate_num;ref_index++)
                        {
                            int ref_id=ordering_score_cached_ref_ids[cache_begin+ref_index];
                            if(!reference_valid[ref_id])
                                continue;
                            T_coord squared_dist=get_dist_to_coord(query_coord,coord_array[ref_id],dim);
                            nearest_queue.push({squared_dist,ref_id});
                            valid_num++;
                        }
                        reused_candidate_cache=valid_num==candidate_num;
                        if(!reused_candidate_cache)
                            while(!nearest_queue.empty())
                                nearest_queue.pop();
                    }
                    if(reused_candidate_cache)
                        candidate_cache_reuse_num++;
                    else
                    {
                        nearest_query_stats_c *candidate_stats=use_kd_query_diagnostics_experimental?&candidate_kd_stats:nullptr;
                        if(candidate_stats)
                            candidate_stats->query_num++;
                        if((candidate_kd_query_num&255)==0)
                        {
                            auto query_start=chrono::steady_clock::now();
                            tree.find_original_nearest(nearest_queue,coord_array,candidate_num,0,query_coord,true,candidate_stats);
                            candidate_kd_sample_ns+=chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now()-query_start).count();
                            candidate_kd_sample_num++;
                        }
                        else
                            tree.find_original_nearest(nearest_queue,coord_array,candidate_num,0,query_coord,true,candidate_stats);
                        candidate_kd_query_num++;
                    }
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
                            ref_queue.pop();
                        }
                        order_candidate_ref_begin[previous_removed_id]=ref_begin;
                        order_candidate_ref_num[previous_removed_id]=keep_num;
                    }
                    int original_id=-1;
                    T_coord best_ref_dist=numeric_limits<T_coord>::infinity(),best_route_dist=numeric_limits<T_coord>::infinity();
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
                    else if (use_reverse_cached_score_order)
                    {
                        double best_mean_target_ref_squared_dist=numeric_limits<double>::infinity(),best_reference_data_error=numeric_limits<double>::infinity();
                        uint32_t best_reference_overlap=0;
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
                                if(use_msd_neighbor_prefetch)
                                {
                                    uint32_t previous_first_index=ordering_score_cache_first_valid_index[candidate_id],first_index=previous_first_index;
                                    while(first_index<cached_ref_num&&!reference_valid[ordering_score_cached_ref_ids[cache_begin+first_index]])
                                        first_index++;
                                    ordering_score_cache_first_valid_index[candidate_id]=(uint8_t)first_index;
                                    valid_cached_ref_num=first_index<cached_ref_num?1:0;
                                    reference_list_unchanged=first_index==previous_first_index;
                                }
                                else
                                {
                                    for(uint32_t ref_index=0;ref_index<cached_ref_num;ref_index++)
                                    {
                                        int cached_ref_id=ordering_score_cached_ref_ids[cache_begin+ref_index];
                                        if (reference_valid[cached_ref_id])
                                        {
                                            ordering_score_cached_ref_ids[cache_begin+valid_cached_ref_num]=cached_ref_id;
                                            if (use_reference_data_order)
                                                ordering_score_cached_ref_weight[cache_begin+valid_cached_ref_num]=ordering_score_cached_ref_weight[cache_begin+ref_index];
                                            valid_cached_ref_num++;
                                        }
                                    }
                                    reference_list_unchanged=valid_cached_ref_num==cached_ref_num;
                                }
                            }
                            uint32_t required_cached_ref_num=min(ordering_score_min_ref_num,cached_ref_num);
                            bool cache_valid=ordering_score_cache_initialized[candidate_id] && cached_ref_num>0 && valid_cached_ref_num>=required_cached_ref_num;
                            double mean_target_ref_squared_dist;
                            if (cache_valid)
                            {
                                if (reference_list_unchanged)
                                    mean_target_ref_squared_dist=ordering_score_cached_score[candidate_id];
                                else
                                {
                                    double target_ref_squared_dist_sum=0;
                                    if(use_msd_neighbor_prefetch)
                                        mean_target_ref_squared_dist=(double)get_dist_to_coord(coord_array[candidate_id],coord_array[ordering_score_cached_ref_ids[cache_begin+ordering_score_cache_first_valid_index[candidate_id]]],dim);
                                    else
                                    {
                                        for(uint32_t ref_index=0;ref_index<valid_cached_ref_num;ref_index++)
                                            target_ref_squared_dist_sum+=(double)get_dist_to_coord(coord_array[candidate_id],coord_array[ordering_score_cached_ref_ids[cache_begin+ref_index]],dim);
                                        mean_target_ref_squared_dist=target_ref_squared_dist_sum/valid_cached_ref_num;
                                    }
                                    ordering_score_cached_score[candidate_id]=mean_target_ref_squared_dist;
                                    if(!use_msd_neighbor_prefetch)
                                        ordering_score_cached_ref_num[candidate_id]=valid_cached_ref_num;
                                    ordering_score_cache_reduced_ref_hit_num++;
                                }
                                ordering_score_cache_hit_num++;
                            }
                            else
                            {
                                bool certified_nearest=false;
                                int certified_ref_id=-1;
                                double certified_squared_dist=numeric_limits<double>::infinity();
                                if(use_candidate_score_geometry_reuse&&candidate_group.size()>1)
                                {
                                    score_candidate_geometry_checked_num++;
                                    double radius=sqrt((double)candidate_group.front().first),route_radius=sqrt((double)route_dist);
                                    double margin=(radius-route_radius)*0.999;
                                    if(candidate_group.size()==target_ids.size()-removed_target_ids.size())
                                        margin=numeric_limits<double>::infinity();
                                    if(margin>0)
                                    {
                                        for(const auto&peer:candidate_group)
                                        {
                                            if(peer.second==candidate_id)
                                                continue;
                                            double squared_dist=(double)get_dist_to_coord(coord_array[candidate_id],coord_array[peer.second],dim);
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
                                    if(use_msd_neighbor_prefetch)
                                        ordering_score_cache_first_valid_index[candidate_id]=0;
                                    ordering_score_cache_initialized[candidate_id]=1;
                                    score_candidate_geometry_certified_num++;
                                }
                                else
                                {
                                    nearest_query_stats_c *score_stats=use_kd_query_diagnostics_experimental?&score_kd_stats:nullptr;
                                    if(score_stats)
                                        score_stats->query_num++;
                                    if((ordering_score_cache_miss_num&255)==0)
                                    {
                                        auto query_start=chrono::steady_clock::now();
                                        if(use_kd_scalar_one_reference_query_experimental)
                                        {
                                            T_coord nearest_distance;
                                            int nearest_original_id;
                                            if(reference_tree->find_original_nearest_other_iterative(coord_array[candidate_id],candidate_id,0,nearest_distance,nearest_original_id,true,score_stats))
                                                ordering_score_ref_queue.push({nearest_distance,nearest_original_id});
                                        }
                                        else
                                            reference_tree->find_original_nearest(ordering_score_ref_queue,coord_array,reverse_ref_query_num,0,coord_array[candidate_id],true,score_stats);
                                        score_kd_sample_ns+=chrono::duration_cast<chrono::nanoseconds>(chrono::steady_clock::now()-query_start).count();
                                        score_kd_sample_num++;
                                    }
                                    else
                                    {
                                        if(use_kd_scalar_one_reference_query_experimental)
                                        {
                                            T_coord nearest_distance;
                                            int nearest_original_id;
                                            if(reference_tree->find_original_nearest_other_iterative(coord_array[candidate_id],candidate_id,0,nearest_distance,nearest_original_id,true,score_stats))
                                                ordering_score_ref_queue.push({nearest_distance,nearest_original_id});
                                        }
                                        else
                                            reference_tree->find_original_nearest(ordering_score_ref_queue,coord_array,reverse_ref_query_num,0,coord_array[candidate_id],true,score_stats);
                                    }
                                    double target_ref_squared_dist_sum=0;
                                    uint32_t ref_num=0;
                                    if(use_msd_neighbor_prefetch)
                                        fetched_ref_buffer.clear();
                                    while (!ordering_score_ref_queue.empty())
                                    {
                                        int ref_id=ordering_score_ref_queue.top().second;
                                        T_coord squared_dist=ordering_score_ref_queue.top().first;
                                        ordering_score_ref_queue.pop();
                                        if (ref_id!=candidate_id)
                                        {
                                            if(use_msd_neighbor_prefetch)
                                                fetched_ref_buffer.push_back({squared_dist,ref_id});
                                            else
                                            {
                                                target_ref_squared_dist_sum+=(double)squared_dist;
                                                ordering_score_cached_ref_ids[cache_begin+ref_num]=ref_id;
                                            }
                                            if (use_reference_data_order)
                                            {
                                                ordering_score_cached_ref_weight[cache_begin+ref_num]=squared_dist==0?numeric_limits<double>::infinity():pow((double)squared_dist,-idw_power/2);
                                                reference_weight_calculation_num++;
                                            }
                                            ref_num++;
                                        }
                                    }
                                    if(use_msd_neighbor_prefetch)
                                    {
                                        reverse(fetched_ref_buffer.begin(),fetched_ref_buffer.end());
                                        for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                                            ordering_score_cached_ref_ids[cache_begin+ref_index]=fetched_ref_buffer[ref_index].second;
                                        mean_target_ref_squared_dist=ref_num>0?(double)fetched_ref_buffer[0].first:numeric_limits<double>::infinity();
                                    }
                                    else
                                        mean_target_ref_squared_dist=ref_num>0?target_ref_squared_dist_sum/ref_num:numeric_limits<double>::infinity();
                                    ordering_score_cached_score[candidate_id]=mean_target_ref_squared_dist;
                                    ordering_score_cached_ref_num[candidate_id]=ref_num;
                                    if(use_msd_neighbor_prefetch)
                                        ordering_score_cache_first_valid_index[candidate_id]=0;
                                    ordering_score_cache_initialized[candidate_id]=1;
                                    ordering_score_cache_miss_num++;
                                }
                            }
                            uint32_t reference_overlap=0;
                            if (use_reverse_reference_reuse_order_experimental)
                                for (uint32_t ref_index=0;ref_index<ordering_score_cached_ref_num[candidate_id];ref_index++)
                                    if (find(previous_route_ref_ids.begin(),previous_route_ref_ids.end(),ordering_score_cached_ref_ids[cache_begin+ref_index])!=previous_route_ref_ids.end())
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
                                    reference_data_error=ordering_score_cached_reference_data_error[candidate_id];
                                    reference_score_cache_hit_num++;
                                }
                                else
                                {
                                    uint32_t valid_ordering_score_ref_num=ordering_score_cached_ref_num[candidate_id];
                                    if (cache_valid)
                                        reference_weight_reduced_list_reuse_num+=valid_ordering_score_ref_num;
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
                                        for (uint32_t ref_index=0;ref_index<valid_ordering_score_ref_num&&valid_ref_num<prediction_max_ref_num;ref_index++)
                                        {
                                            int ref_id=ordering_score_cached_ref_ids[cache_begin+ref_index];
                                            if (get_point_set_id(reference_data[ref_id])!=target_point_set_id)
                                                continue;
                                            valid_ref_ids.push_back(ref_id);
                                            double weight=ordering_score_cached_ref_weight[cache_begin+ref_index];
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
                                        bool rbf_point_set=point_set_config_vec[target_point_set_id].codec==point_set_codec_e::rbf;
                                        bool try_rbf=use_reference_data_rbf_score&&rbf_point_set&&valid_ref_ids.size()>=prediction_min_ref_num;
                                        bool used_rbf=false;
                                        if(try_rbf)
                                        {
                                            if(valid_ref_ids!=rbf_weight_ref_ids)
                                            {
                                                rbf_weights_valid=calculate_rbf_weights(candidate_id,valid_ref_ids,rbf_weights);
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
                                        if(use_reference_data_rbf_score&&rbf_point_set&&rbf_fallback_nearest&&!used_rbf&&!valid_ref_ids.empty())
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
                                    ordering_score_cached_reference_data_error[candidate_id]=reference_data_error;
                                    reference_score_evaluation_num++;
                                }
                            }
                            bool better=original_id==-1;
                            bool route_better=route_dist<best_route_dist;
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
                            double first_score=(use_reverse_distribution_order_experimental || use_reverse_msd_order)?mean_target_ref_squared_dist:(use_reverse_s2_only_order_experimental?distribution_score:((use_reverse_direction_only_order_experimental || use_reverse_direction_distribution_order_experimental)?direction_concentration:nearest_ref_squared_dist));
                            double second_score=(use_reverse_distribution_order_experimental || use_reverse_direction_distribution_order_experimental)?distribution_score:direction_concentration;
                            candidate_scores.push_back({candidate_id,route_dist,first_score,second_score});
                        }
                        vector<uint32_t> score_order(candidate_scores.size());
                        iota(score_order.begin(),score_order.end(),0);
                        sort(score_order.begin(),score_order.end(),[&](uint32_t a,uint32_t b){return candidate_scores[a].first_score<candidate_scores[b].first_score;});
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
                            sort(score_order.begin(),score_order.end(),[&](uint32_t a,uint32_t b){return candidate_scores[a].second_score<candidate_scores[b].second_score;});
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
                            if (original_id==-1 || rank_sum<best_rank_sum || (rank_sum==best_rank_sum && candidate.route_dist<best_route_dist))
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
                            if (original_id==-1 || ref_dist<best_ref_dist || (ref_dist==best_ref_dist && route_dist<best_route_dist))
                            {
                                original_id=candidate_id;
                                best_ref_dist=ref_dist;
                                best_route_dist=route_dist;
                            }
                        }
                    }
                    removed_target_ids.push_back(original_id);
                    tree.update_point_valid_num_to_root(original_id,-1);
                    if(!share_global_msd_validity_tree)
                        reference_tree->update_point_valid_num_to_root(original_id,-1);
                    if (use_reverse_cached_score_order)
                        reference_valid[original_id]=0;
                    if (use_reverse_reference_reuse_order_experimental)
                    {
                        uint64_t cache_begin=(uint64_t)original_id*reverse_ref_query_num;
                        if (!ordering_score_cache_initialized[original_id])
                        {
                            reference_tree->find_original_nearest(ordering_score_ref_queue,coord_array,reverse_ref_query_num,0,coord_array[original_id],true);
                            uint32_t ref_num=0;
                            while (!ordering_score_ref_queue.empty())
                            {
                                int ref_id=ordering_score_ref_queue.top().second;
                                ordering_score_ref_queue.pop();
                                if (ref_id!=original_id)
                                    ordering_score_cached_ref_ids[cache_begin+ref_num++]=ref_id;
                            }
                            ordering_score_cached_ref_num[original_id]=ref_num;
                            ordering_score_cache_initialized[original_id]=1;
                        }
                        previous_route_ref_ids.assign(ordering_score_cached_ref_ids.begin()+cache_begin,ordering_score_cached_ref_ids.begin()+cache_begin+ordering_score_cached_ref_num[original_id]);
                    }
                    previous_removed_id=original_id;
                    query_coord=coord_array[original_id];
                }
                reverse(removed_target_ids.begin(),removed_target_ids.end());
                target_ids.swap(removed_target_ids);
                printf("Reverse-sorted round %u\n",round_id);
            }
            if (use_reverse_cached_score_order)
            {
                printf("MSD/reference-data ordering score cache: %llu hits, %llu misses\n",(unsigned long long)ordering_score_cache_hit_num,(unsigned long long)ordering_score_cache_miss_num);
                printf("MSD/reference-data ordering KD query sample: candidate=%llu/%llu,mean=%.3f us; score=%llu/%llu,mean=%.3f us\n",(unsigned long long)candidate_kd_sample_num,(unsigned long long)candidate_kd_query_num,candidate_kd_sample_num?0.001*(double)candidate_kd_sample_ns/candidate_kd_sample_num:0,(unsigned long long)score_kd_sample_num,(unsigned long long)ordering_score_cache_miss_num,score_kd_sample_num?0.001*(double)score_kd_sample_ns/score_kd_sample_num:0);
                if(use_msd_neighbor_prefetch)
                    printf("MSD neighbor prefetch: saved_per_query=%u,candidate_kd_queries=%llu,candidate_cache_reuses=%llu,score_kd_queries=%llu,total_kd_queries=%llu\n",ordering_msd_neighbor_prefetch_num_experimental,(unsigned long long)candidate_kd_query_num,(unsigned long long)candidate_cache_reuse_num,(unsigned long long)ordering_score_cache_miss_num,(unsigned long long)(candidate_kd_query_num+ordering_score_cache_miss_num));
                if(use_candidate_score_geometry_reuse)
                    printf("MSD candidate-geometry score reuse: checked=%llu,certified=%llu,candidate_kd_queries=%llu,score_kd_queries=%llu,total_kd_queries=%llu\n",(unsigned long long)score_candidate_geometry_checked_num,(unsigned long long)score_candidate_geometry_certified_num,(unsigned long long)candidate_kd_query_num,(unsigned long long)ordering_score_cache_miss_num,(unsigned long long)(candidate_kd_query_num+ordering_score_cache_miss_num));
                if(use_kd_query_diagnostics_experimental)
                {
                    auto print_stats=[](const char *name,const nearest_query_stats_c &stats)
                    {
                        double query_num=stats.query_num?stats.query_num:1;
                        printf("KD diagnostics %s: queries=%llu,blocks/query=%.3f,empty_prunes/query=%.3f,leaves/query=%.3f,block_distances/query=%.3f,point_checks/query=%.3f,invalid_skips/query=%.3f,point_distances/query=%.3f,queue_pushes/query=%.3f,queue_replacements/query=%.3f,far_prunes/query=%.3f\n",name,(unsigned long long)stats.query_num,stats.block_visit_num/query_num,stats.empty_block_prune_num/query_num,stats.leaf_visit_num/query_num,stats.block_distance_num/query_num,stats.point_check_num/query_num,stats.invalid_point_skip_num/query_num,stats.point_distance_num/query_num,stats.queue_push_num/query_num,stats.queue_replace_num/query_num,stats.far_child_prune_num/query_num);
                    };
                    print_stats("candidate",candidate_kd_stats);
                    print_stats("score",score_kd_stats);
                }
                printf("MSD/reference-data ordering reduced-reference cache hits: %llu\n",(unsigned long long)ordering_score_cache_reduced_ref_hit_num);
                if (use_reverse_reference_reuse_order_experimental)
                    printf("Selected adjacent reference overlap: %f / %u\n",selected_reference_overlap_num?(double)selected_reference_overlap_sum/selected_reference_overlap_num:0,prediction_max_ref_num);
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
            bool use_forward_predictability_score=use_predictability_order_experimental || use_forward_msd_order || use_forward_new_geometry_score;
            if (use_forward_predictability_score)
            {
                reference_tree_storage=make_unique<block_tree_c<T_coord>>(tree);
                reference_tree=reference_tree_storage.get();
                reference_tree->clear_all_valid_num();
                if (use_predictability_order_experimental)
                    printf("Using geometry-based predictability-aware target order with %u local candidates\n",prediction_max_ref_num);
                else if (use_forward_msd_order)
                    printf("Using forward MSD target order with %u local candidates and references\n",prediction_max_ref_num);
                else if (use_forward_s2_only_order_experimental)
                    printf("Using forward S2-only target order with %u local candidates and references\n",prediction_max_ref_num);
                else if (use_forward_direction_only_order_experimental)
                    printf("Using forward direction-concentration-only target order with %u local candidates and references\n",prediction_max_ref_num);
                else if (use_forward_mean_distance_s2_order_experimental)
                    printf("Using forward mean-distance and S2 target order with %u local candidates and references\n",prediction_max_ref_num);
                else
                    printf("Using forward mean-distance and direction-concentration target order with %u local candidates and references\n",prediction_max_ref_num);
            }
            for (uint32_t round_id=0;round_id<prediction_round_num;round_id++)
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
                    uint32_t candidate_num=use_forward_predictability_score?min<uint32_t>(prediction_max_ref_num,target_ids.size()-sorted_target_ids.size()):1;
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
                            reference_tree->find_original_nearest(ref_queue,coord_array,prediction_max_ref_num,0,coord_array[candidate_id],true);
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
                        sort(score_order.begin(),score_order.end(),[&](uint32_t a,uint32_t b){return candidate_scores[a].first_score<candidate_scores[b].first_score;});
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
                            sort(score_order.begin(),score_order.end(),[&](uint32_t a,uint32_t b){return candidate_scores[a].second_score<candidate_scores[b].second_score;});
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
                            if (original_id==-1 || rank_sum>best_rank_sum || (rank_sum==best_rank_sum && candidate.route_dist<best_route_dist))
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
                                uint32_t ref_query_num=use_forward_msd_order?prediction_max_ref_num:1;
                                reference_tree->find_original_nearest(ref_queue,coord_array,ref_query_num,0,coord_array[candidate_id],true);
                                if (ref_queue.empty())
                                    predictability_score=numeric_limits<double>::infinity();
                                else if (use_forward_msd_order)
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
                            if (original_id==-1 || predictability_score>best_predictability_score || (predictability_score==best_predictability_score && route_dist<best_route_dist))
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
            uint32_t first_grouped_round=prediction_round_num>3?prediction_round_num-3:0;
            printf("Grouping rounds %u-%u by %u cache subpartitions and reordering data by prediction order\n",first_grouped_round,prediction_round_num-1,cache_subpartition_num_experimental);
            for(uint32_t round_id=first_grouped_round;round_id<prediction_round_num;round_id++)
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
        pred_ref_ids.reserve((uint64_t)point_num*prediction_max_ref_num);
        pred_ref_begin.clear();
        pred_ref_begin.reserve((uint64_t)point_num+1);
        pred_ref_begin.push_back(0);
        pred_ref_weights.clear();
        pred_ref_weights.reserve((uint64_t)point_num*prediction_max_ref_num);
        tree.clear_all_valid_num();
        uint64_t reused_reference_query_num=0;
        for (uint32_t round_id=0;round_id<prediction_round_num;round_id++)
        {
            for (int pred_id:round_target_ids[round_id])
            {
                pred_target_ids.push_back(pred_id);
                vector<int> ref_ids;
                uint64_t saved_ref_begin=reuse_global_ordering_candidate_refs?order_candidate_ref_begin[pred_id]:numeric_limits<uint64_t>::max();
                if(saved_ref_begin!=numeric_limits<uint64_t>::max())
                {
                    uint32_t saved_ref_num=order_candidate_ref_num[pred_id];
                    ref_ids.assign(order_candidate_ref_ids.begin()+saved_ref_begin,order_candidate_ref_ids.begin()+saved_ref_begin+saved_ref_num);
                    reused_reference_query_num++;
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
                    while(!nearest_queue.empty())
                    {
                        ref_ids.push_back(nearest_queue.top().second);
                        nearest_queue.pop();
                    }
                    reverse(ref_ids.begin(),ref_ids.end());
                }
                for(int ref_id:ref_ids)
                {
                    double dist=(double)get_dist_to_coord(coord_array[pred_id],coord_array[ref_id],dim);
                    pred_ref_ids.push_back(ref_id);
                    pred_ref_weights.push_back(dist==0?numeric_limits<double>::infinity():pow(dist,-idw_power/2));
                }
                pred_ref_begin.push_back(pred_ref_ids.size());

                tree.update_point_valid_num_to_root(pred_id,1);
            }
        }

        if (pred_target_ids.size()!=point_num || pred_ref_begin.size()!=(uint64_t)point_num+1 || pred_ref_ids.size()!=pred_ref_weights.size())
            throw runtime_error("预测参考结构没有覆盖所有点");
        if(reuse_global_ordering_candidate_refs)
            printf("Reused sorting candidate queries for prediction references: %llu targets, additional kd-tree queries: 0\n",(unsigned long long)reused_reference_query_num);
        if (!dag_cache_path.empty())
            save_pred_refs_structure();
        print_prediction_geometry_statistics_result();
        prepare_pred_refs_structure();
        auto end_time=chrono::steady_clock::now();
        printf("Prediction reference generation time: %f s\n",chrono::duration<double>(end_time-reference_generation_start_time).count());
        printf("Generated %llu prediction entries, total time: %f s\n",(unsigned long long)pred_target_ids.size(),chrono::duration<double>(end_time-start_time).count());
    }
}
