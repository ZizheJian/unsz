#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_precomputed_neighbor_pred_refs_structure()
    {
        if(ordering_scope!="global"||!use_reverse_msd_order||!reference_time_step_vec.empty()||use_reverse_reference_reuse_order_experimental||partition_size_experimental>0||partition_num_experimental>0||cache_subpartition_num_experimental>0)
            throw runtime_error("预存近邻实验目前只支持单线程global反向MSD排序，不支持参考数据辅助、partition或cache subpartition");
        if(point_num==0||prediction_round_num==0||prediction_round_num>63||prediction_max_ref_num==0)
            throw runtime_error("预存近邻实验的点数、轮数或参考点数无效");
        uint32_t neighbor_limit=precomputed_neighbor_num_experimental;
        uint32_t candidate_limit=ordering_candidate_num;
        uint32_t score_limit=ordering_score_max_ref_num;
        if(neighbor_limit>256||neighbor_limit<max({candidate_limit,score_limit,prediction_max_ref_num}))
            throw runtime_error("预存近邻数量必须覆盖排序候选、MSD评分和最终参考点数，且最多为256");
        uint32_t stored_limit=min<uint32_t>(neighbor_limit,point_num-1);
        auto total_start_time=chrono::steady_clock::now();
        vector<int> neighbor_ids((uint64_t)point_num*stored_limit,-1);
        vector<T_coord> neighbor_squared_dist((uint64_t)point_num*stored_limit,0);
        vector<uint16_t> neighbor_original_rank((uint64_t)point_num*stored_limit,0);
        vector<uint32_t> neighbor_count(point_num,0);
        {
            block_tree_c<T_coord> initial_tree(coord_array,min_coord,max_coord,kd_tree_leaf_size);
            nearest_queue_c<T_coord> queue;
            for(uint32_t point_id=0;point_id<point_num;point_id++)
            {
                initial_tree.find_original_nearest(queue,coord_array,min<uint32_t>(point_num,stored_limit+1),0,coord_array[point_id],false);
                vector<pair<T_coord,int>> neighbors;
                neighbors.reserve(queue.size());
                while(!queue.empty())
                {
                    auto neighbor=queue.top();
                    queue.pop();
                    if(neighbor.second!=(int)point_id)
                        neighbors.push_back(neighbor);
                }
                sort(neighbors.begin(),neighbors.end(),[](const auto&a,const auto&b){return a.first<b.first;});
                uint64_t begin=(uint64_t)point_id*stored_limit;
                uint32_t count=min<uint32_t>(stored_limit,neighbors.size());
                neighbor_count[point_id]=count;
                for(uint32_t index=0;index<count;index++)
                {
                    neighbor_ids[begin+index]=neighbors[index].second;
                    neighbor_squared_dist[begin+index]=neighbors[index].first;
                    neighbor_original_rank[begin+index]=(uint16_t)(index+1);
                }
            }
        }
        double initial_neighbor_time=chrono::duration<double>(chrono::steady_clock::now()-total_start_time).count();
        printf("Precomputed neighbor table: points=%u,neighbors_per_point=%u,bytes=%llu,time=%f s\n",point_num,stored_limit,(unsigned long long)((uint64_t)neighbor_ids.size()*(sizeof(int)+sizeof(T_coord)+sizeof(uint16_t))+(uint64_t)point_num*sizeof(uint32_t)),initial_neighbor_time);

        vector<uint8_t> available(point_num,1);
        vector<uint16_t> pending_ref_num(point_num,0);
        vector<uint32_t> remaining_ids(point_num),remaining_position(point_num);
        iota(remaining_ids.begin(),remaining_ids.end(),0);
        iota(remaining_position.begin(),remaining_position.end(),0);
        vector<uint64_t> pending_ref_begin(point_num,numeric_limits<uint64_t>::max());
        vector<int> pending_ref_ids;
        vector<T_coord> pending_ref_squared_dist;
        pending_ref_ids.reserve((uint64_t)point_num*prediction_max_ref_num);
        pending_ref_squared_dist.reserve((uint64_t)point_num*prediction_max_ref_num);
        vector<uint16_t> scanned_max_original_rank(point_num,0),candidate_max_original_rank(point_num,0),score_max_original_rank(point_num,0),final_ref_original_rank(point_num,0);
        uint64_t invalid_entry_removed_num=0,candidate_query_num=0,score_query_num=0,candidate_restart_num=0,short_candidate_query_num=0,short_score_query_num=0,short_final_ref_num=0;
        auto get_valid_prefix=[&](int point_id,uint32_t needed)->pair<uint32_t,uint16_t>
        {
            uint64_t begin=(uint64_t)point_id*stored_limit;
            uint32_t old_count=neighbor_count[point_id],read_index=0,write_index=0;
            uint16_t last_rank=0;
            while(read_index<old_count&&write_index<needed)
            {
                int neighbor_id=neighbor_ids[begin+read_index];
                last_rank=neighbor_original_rank[begin+read_index];
                if(available[neighbor_id])
                {
                    if(write_index!=read_index)
                    {
                        neighbor_ids[begin+write_index]=neighbor_id;
                        neighbor_squared_dist[begin+write_index]=neighbor_squared_dist[begin+read_index];
                        neighbor_original_rank[begin+write_index]=last_rank;
                    }
                    write_index++;
                }
                else
                    invalid_entry_removed_num++;
                read_index++;
            }
            if(read_index>write_index)
            {
                uint32_t remaining=old_count-read_index;
                memmove(neighbor_ids.data()+begin+write_index,neighbor_ids.data()+begin+read_index,(uint64_t)remaining*sizeof(int));
                memmove(neighbor_squared_dist.data()+begin+write_index,neighbor_squared_dist.data()+begin+read_index,(uint64_t)remaining*sizeof(T_coord));
                memmove(neighbor_original_rank.data()+begin+write_index,neighbor_original_rank.data()+begin+read_index,(uint64_t)remaining*sizeof(uint16_t));
                neighbor_count[point_id]=write_index+remaining;
            }
            scanned_max_original_rank[point_id]=max(scanned_max_original_rank[point_id],last_rank);
            return {write_index,last_rank};
        };

        vector<T_coord> center_coord(dim);
        for(uint32_t d=0;d<dim;d++)
            center_coord[d]=(min_coord[d]+max_coord[d])/2;
        int seed_id=-1;
        T_coord seed_dist=numeric_limits<T_coord>::infinity();
        for(uint32_t point_id=0;point_id<point_num;point_id++)
        {
            T_coord dist=get_dist_to_coord(coord_array[point_id],center_coord.data(),dim);
            if(dist<seed_dist)
            {
                seed_dist=dist;
                seed_id=(int)point_id;
            }
        }
        vector<int> reverse_order;
        reverse_order.reserve(point_num);
        int previous_removed_id=-1;
        auto sorting_start_time=chrono::steady_clock::now();
        while(!remaining_ids.empty())
        {
            int selected_id=-1;
            if(previous_removed_id<0)
                selected_id=seed_id;
            else
            {
                auto [candidate_num,candidate_depth]=get_valid_prefix(previous_removed_id,candidate_limit);
                candidate_query_num++;
                uint16_t candidate_need_depth=candidate_num<candidate_limit&&remaining_ids.size()>=candidate_limit?(uint16_t)(stored_limit+1):candidate_depth;
                if(candidate_num==0&&!remaining_ids.empty())
                    candidate_need_depth=(uint16_t)(stored_limit+1);
                candidate_max_original_rank[previous_removed_id]=max(candidate_max_original_rank[previous_removed_id],candidate_need_depth);
                if(candidate_num<candidate_limit&&remaining_ids.size()>=candidate_limit)
                    short_candidate_query_num++;
                uint64_t candidate_begin=(uint64_t)previous_removed_id*stored_limit;
                uint32_t ref_num=min<uint32_t>(prediction_max_ref_num,candidate_num);
                if(ref_num<prediction_max_ref_num&&remaining_ids.size()>=prediction_max_ref_num)
                    short_final_ref_num++;
                pending_ref_begin[previous_removed_id]=pending_ref_ids.size();
                pending_ref_num[previous_removed_id]=(uint16_t)ref_num;
                if(ref_num>0)
                    final_ref_original_rank[previous_removed_id]=neighbor_original_rank[candidate_begin+ref_num-1];
                if(ref_num<prediction_max_ref_num&&remaining_ids.size()>=prediction_max_ref_num)
                    final_ref_original_rank[previous_removed_id]=(uint16_t)(stored_limit+1);
                for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                {
                    pending_ref_ids.push_back(neighbor_ids[candidate_begin+ref_index]);
                    pending_ref_squared_dist.push_back(neighbor_squared_dist[candidate_begin+ref_index]);
                }
                double best_score=numeric_limits<double>::infinity();
                T_coord best_route_dist=numeric_limits<T_coord>::infinity();
                for(uint32_t candidate_index=0;candidate_index<candidate_num;candidate_index++)
                {
                    int candidate_id=neighbor_ids[candidate_begin+candidate_index];
                    T_coord route_dist=neighbor_squared_dist[candidate_begin+candidate_index];
                    auto [valid_ordering_score_ref_num,score_depth]=get_valid_prefix(candidate_id,score_limit);
                    score_query_num++;
                    uint16_t score_need_depth=valid_ordering_score_ref_num<score_limit&&remaining_ids.size()>score_limit?(uint16_t)(stored_limit+1):score_depth;
                    score_max_original_rank[candidate_id]=max(score_max_original_rank[candidate_id],score_need_depth);
                    if(valid_ordering_score_ref_num<score_limit&&remaining_ids.size()>score_limit)
                        short_score_query_num++;
                    uint64_t score_begin=(uint64_t)candidate_id*stored_limit;
                    double score=numeric_limits<double>::infinity();
                    if(valid_ordering_score_ref_num>0)
                    {
                        double sum=0;
                        for(uint32_t score_index=0;score_index<valid_ordering_score_ref_num;score_index++)
                            sum+=(double)neighbor_squared_dist[score_begin+score_index];
                        score=sum/valid_ordering_score_ref_num;
                    }
                    if(selected_id<0||score<best_score||(score==best_score&&route_dist<best_route_dist))
                    {
                        selected_id=candidate_id;
                        best_score=score;
                        best_route_dist=route_dist;
                    }
                }
                if(selected_id<0)
                {
                    selected_id=(int)remaining_ids.back();
                    candidate_restart_num++;
                }
            }
            if(selected_id<0||!available[selected_id])
                throw runtime_error("预存近邻排序未能选择有效目标");
            reverse_order.push_back(selected_id);
            uint32_t remove_position=remaining_position[selected_id];
            uint32_t last_id=remaining_ids.back();
            remaining_ids[remove_position]=last_id;
            remaining_position[last_id]=remove_position;
            remaining_ids.pop_back();
            available[selected_id]=0;
            previous_removed_id=selected_id;
        }
        double sorting_time=chrono::duration<double>(chrono::steady_clock::now()-sorting_start_time).count();
        reverse(reverse_order.begin(),reverse_order.end());
        vector<uint32_t> round_size=get_round_target_num_vec(point_num);
        round_target_ids.assign(prediction_round_num,vector<int>());
        pred_target_ids.clear();
        pred_target_ids.reserve(point_num);
        pred_ref_ids.clear();
        pred_ref_ids.reserve(pending_ref_ids.size());
        pred_ref_weights.clear();
        pred_ref_weights.reserve(pending_ref_ids.size());
        pred_ref_begin.clear();
        pred_ref_begin.reserve((uint64_t)point_num+1);
        pred_ref_begin.push_back(0);
        uint32_t order_index=0;
        for(uint32_t round_id=0;round_id<prediction_round_num;round_id++)
        {
            round_target_ids[round_id].reserve(round_size[round_id]);
            for(uint32_t local_index=0;local_index<round_size[round_id];local_index++)
            {
                int target_id=reverse_order[order_index++];
                round_target_ids[round_id].push_back(target_id);
                pred_target_ids.push_back(target_id);
                uint64_t begin=pending_ref_begin[target_id];
                if(begin!=numeric_limits<uint64_t>::max())
                    for(uint32_t ref_index=0;ref_index<pending_ref_num[target_id];ref_index++)
                    {
                        int ref_id=pending_ref_ids[begin+ref_index];
                        T_coord dist=pending_ref_squared_dist[begin+ref_index];
                        pred_ref_ids.push_back(ref_id);
                        pred_ref_weights.push_back(dist==0?numeric_limits<double>::infinity():pow((double)dist,-idw_power/2));
                    }
                pred_ref_begin.push_back(pred_ref_ids.size());
            }
        }
        if(order_index!=point_num)
            throw runtime_error("预存近邻排序未覆盖全部点");
        auto print_rank_coverage=[&](const char*name,const vector<uint16_t>&depth)
        {
            uint64_t visited=0,within8=0,within16=0,within32=0,within64=0,within128=0,unmet=0;
            for(uint16_t rank:depth)
                if(rank>0)
                {
                    visited++;
                    within8+=rank<=8&&rank<=stored_limit;
                    within16+=rank<=16&&rank<=stored_limit;
                    within32+=rank<=32&&rank<=stored_limit;
                    within64+=rank<=64&&rank<=stored_limit;
                    within128+=rank<=128&&rank<=stored_limit;
                    unmet+=rank>stored_limit;
                }
            printf("Original-neighbor rank coverage: source=%s,visited_points=%llu,within_8=%llu,within_16=%llu,within_32=%llu,within_64=%llu",name,(unsigned long long)visited,(unsigned long long)within8,(unsigned long long)within16,(unsigned long long)within32,(unsigned long long)within64);
            if(stored_limit>=128)
                printf(",within_128=%llu",(unsigned long long)within128);
            printf(",unmet_after_%u=%llu\n",stored_limit,(unsigned long long)unmet);
        };
        print_rank_coverage("actually_scanned",scanned_max_original_rank);
        print_rank_coverage("candidate",candidate_max_original_rank);
        print_rank_coverage("MSD_score",score_max_original_rank);
        print_rank_coverage("final_reference",final_ref_original_rank);
        printf("Precomputed-neighbor lazy filtering: removed_entries=%llu,candidate_queries=%llu,score_queries=%llu,restarts=%llu,short_candidate=%llu,short_score=%llu,short_final_reference=%llu,sorting_time=%f s\n",(unsigned long long)invalid_entry_removed_num,(unsigned long long)candidate_query_num,(unsigned long long)score_query_num,(unsigned long long)candidate_restart_num,(unsigned long long)short_candidate_query_num,(unsigned long long)short_score_query_num,(unsigned long long)short_final_ref_num,sorting_time);
        if(!dag_cache_path.empty())
            save_pred_refs_structure();
        print_prediction_geometry_statistics_result();
        prepare_pred_refs_structure();
        printf("Generated precomputed-neighbor DAG: points=%u,references=%llu,total_time=%f s\n",point_num,(unsigned long long)pred_ref_ids.size(),chrono::duration<double>(chrono::steady_clock::now()-total_start_time).count());
    }
}
