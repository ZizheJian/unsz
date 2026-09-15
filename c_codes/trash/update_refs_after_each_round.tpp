#pragma once

// XXX 废弃的“每轮结束后统一更新可用参考点”实验代码。
// 基础方案会在一个点完成预测后立即将它加入可用参考点集合，因此同一轮内后面的点可以引用前面的点。
// 本实验方案在整轮目标全部选完参考点后才统一更新集合，使一轮内的点不能互相引用。
//
// 在Katrina attr0.dat.0～2、rel_eb=0.01、启用Katrina坐标变换矩阵时：
//   基础方案：BR=0.067171，PSNR=51.551982，预测DAG耗时=2.281842 s；
//   实验方案：BR=0.080790，PSNR=51.227573，预测DAG耗时=2.522493 s。
// 实验方案使BR增加约20.3%，PSNR下降约0.324 dB，预测DAG也没有加速。
//
// 逐轮独立编码表明，BR退化并不只发生在没有参考点的第0轮。第5～9轮的平均BR分别增加约
// 12.5%、15.3%、12.4%、18.1%和29.4%。每轮点数约等于此前所有轮点数之和，因此禁止使用
// 同轮参考点会在每一轮显著降低候选点密度。实验方案第1～9轮的最近参考点也始终更远。
// 收益不足且压缩性能明显下降，因此从主代码和unsz_experimental测试入口中移除。
// 本文件仅保留核心实现以便追溯，不参与构建。

namespace unsz
{
    template<typename T_coord>
    void generate_pred_refs_after_each_round(block_tree_c<T_coord> &tree,Array2D<T_coord> &coord_array,
        int max_ref_num,vector<vector<int>> &round_target_ids,vector<pair<int,vector<int>>> &pred_refs_vec)
    {
        pred_refs_vec.clear();
        tree.clear_all_valid_num();
        for (auto &target_ids:round_target_ids)
        {
            for (int pred_id:target_ids)
            {
                priority_queue<pair<T_coord,int>> nearest_queue;
                tree.find_original_nearest(nearest_queue,coord_array,max_ref_num,0,coord_array[pred_id],true);
                pred_refs_vec.push_back({pred_id,vector<int>()});
                auto &ref_ids=pred_refs_vec.back().second;
                ref_ids.reserve(nearest_queue.size());
                while (!nearest_queue.empty())
                {
                    ref_ids.push_back(nearest_queue.top().second);
                    nearest_queue.pop();
                }
                reverse(ref_ids.begin(),ref_ids.end());
            }
            for (int pred_id:target_ids)
            {
                int leaf_block_id=tree.original_id_to_block_vec_id[pred_id];
                tree.update_valid_num_to_root(leaf_block_id,1);
            }
        }
    }
}
