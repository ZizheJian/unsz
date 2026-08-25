// 将所有点分成多轮，确定预测顺序，并为每个点寻找已经完成预测的最近参考点。
#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_pred_refs_structure()
    {
        if (point_num<=0)
            throw runtime_error("point_num必须是正数");
        if (predict_round_num<=0 || predict_round_num>63)
            throw runtime_error("predict_round_num必须在[1,63]内");
        if (max_ref_num<=0)
            throw runtime_error("max_ref_num必须是正数");

        printf("Building kd-tree\n");
        auto start_time=chrono::steady_clock::now();
        block_tree_c<T_coord> tree(coord_array,min_coord,max_coord);
        auto tree_end_time=chrono::steady_clock::now();
        printf("KD-tree blocks: %llu, time: %f s\n",(unsigned long long)tree.block_vec.size(),chrono::duration<double>(tree_end_time-start_time).count());

        printf("Arranging rounds of prediction\n");
        round_target_ids.clear();
        round_target_ids.resize(predict_round_num);
        vector<int> shuffled_ids(point_num);
        iota(shuffled_ids.begin(),shuffled_ids.end(),0);
        mt19937 rng(0);
        shuffle(shuffled_ids.begin(),shuffled_ids.end(),rng);

        uint64_t previous_end=0;
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            uint64_t divisor=uint64_t(1)<<(predict_round_num-1-round_id);
            uint64_t current_end=((uint64_t)point_num+divisor-1)/divisor;
            current_end=min(current_end,(uint64_t)point_num);
            round_target_ids[round_id].assign(shuffled_ids.begin()+previous_end,shuffled_ids.begin()+current_end);
            previous_end=current_end;
            printf("Round %u target num: %llu\n",round_id,(unsigned long long)round_target_ids[round_id].size());
        }
        if (previous_end!=point_num)
            throw runtime_error("预测轮次没有覆盖所有点");

        vector<T_coord> center_coord(dim);
        for (uint32_t d=0;d<dim;d++)
            center_coord[d]=(min_coord[d]+max_coord[d])/2;

        printf("Sorting targets in each round\n");
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            auto &target_ids=round_target_ids[round_id];
            if (target_ids.empty())
                continue;

            tree.clear_all_valid_num();
            for (int original_id:target_ids)
            {
                int leaf_block_id=tree.original_id_to_block_vec_id[original_id];
                tree.block_vec[leaf_block_id].valid_num=1;
            }
            tree.update_all_valid_num(0);

            vector<int> sorted_target_ids;
            sorted_target_ids.reserve(target_ids.size());
            T_coord *query_coord=center_coord.data();
            while (sorted_target_ids.size()<target_ids.size())
            {
                priority_queue<pair<T_coord,int>> nearest_queue;
                tree.find_original_nearest(nearest_queue,coord_array,1,0,query_coord,true);
                if (nearest_queue.empty())
                    throw runtime_error("无法找到尚未排序的预测目标");
                int original_id=nearest_queue.top().second;
                sorted_target_ids.push_back(original_id);
                int leaf_block_id=tree.original_id_to_block_vec_id[original_id];
                tree.update_valid_num_to_root(leaf_block_id,-1);
                query_coord=coord_array[original_id];
            }
            target_ids.swap(sorted_target_ids);
            printf("Sorted round %u\n",round_id);
        }

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
                priority_queue<pair<T_coord,int>> nearest_queue;
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

                int leaf_block_id=tree.original_id_to_block_vec_id[pred_id];
                tree.update_valid_num_to_root(leaf_block_id,1);
            }
        }

        if (pred_refs_vec.size()!=point_num || pred_ref_weights_vec.size()!=point_num)
            throw runtime_error("预测参考结构没有覆盖所有点");
        auto end_time=chrono::steady_clock::now();
        printf("Generated %llu prediction entries, total time: %f s\n",(unsigned long long)pred_refs_vec.size(),chrono::duration<double>(end_time-start_time).count());
    }
}
