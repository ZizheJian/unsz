#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::print_prediction_dag_statistics_result()
    {
        if (pred_refs_vec.size()!=point_num)
            throw runtime_error("预测DAG统计要求参考结构覆盖所有点");
        vector<uint32_t> original_id_to_prediction_order(point_num,numeric_limits<uint32_t>::max());
        for (uint32_t prediction_order=0;prediction_order<point_num;prediction_order++)
        {
            int original_id=pred_refs_vec[prediction_order].first;
            if (original_id<0 || (uint32_t)original_id>=point_num || original_id_to_prediction_order[original_id]!=numeric_limits<uint32_t>::max())
                throw runtime_error("预测DAG包含无效或重复目标点");
            original_id_to_prediction_order[original_id]=prediction_order;
        }

        vector<uint32_t> point_level(point_num,0);
        vector<uint64_t> level_width_vec;
        uint64_t edge_num=0,previous_target_dependency_num=0;
        uint64_t edge_lag_count_vec[7]={0,0,0,0,0,0,0};
        for (uint32_t prediction_order=0;prediction_order<point_num;prediction_order++)
        {
            int target_id=pred_refs_vec[prediction_order].first;
            uint32_t level=0;
            bool depends_on_previous_target=false;
            for (int ref_id:pred_refs_vec[prediction_order].second)
            {
                if (ref_id<0 || (uint32_t)ref_id>=point_num)
                    throw runtime_error("预测DAG包含无效参考点");
                uint32_t ref_prediction_order=original_id_to_prediction_order[ref_id];
                if (ref_prediction_order>=prediction_order)
                    throw runtime_error("预测DAG参考点没有位于目标点之前");
                level=max(level,point_level[ref_id]+1);
                uint32_t lag=prediction_order-ref_prediction_order;
                depends_on_previous_target|=lag==1;
                uint32_t lag_bucket=lag==1?0:(lag<=4?1:(lag<=16?2:(lag<=64?3:(lag<=256?4:(lag<=1024?5:6)))));
                edge_lag_count_vec[lag_bucket]++;
                edge_num++;
            }
            point_level[target_id]=level;
            if (level_width_vec.size()<=level)
                level_width_vec.resize((uint64_t)level+1,0);
            level_width_vec[level]++;
            previous_target_dependency_num+=depends_on_previous_target;
        }

        vector<uint64_t> sorted_level_width_vec=level_width_vec;
        sort(sorted_level_width_vec.begin(),sorted_level_width_vec.end());
        uint64_t level_num=level_width_vec.size();
        uint64_t min_level_width=sorted_level_width_vec.front();
        uint64_t median_level_width=sorted_level_width_vec[(level_num-1)/2];
        uint64_t p90_level_width=sorted_level_width_vec[(level_num-1)*9/10];
        uint64_t max_level_width=sorted_level_width_vec.back();
        double work_span_parallelism=(double)point_num/level_num;
        double previous_dependency_ratio=point_num>1?(double)previous_target_dependency_num/(point_num-1):0;
        printf("Prediction DAG statistics:\n");
        printf("  nodes=%u, edges=%llu, levels=%llu, work/span parallelism=%.6f\n",point_num,(unsigned long long)edge_num,(unsigned long long)level_num,work_span_parallelism);
        printf("  level width min/median/p90/max=%llu/%llu/%llu/%llu\n",(unsigned long long)min_level_width,(unsigned long long)median_level_width,(unsigned long long)p90_level_width,(unsigned long long)max_level_width);
        printf("  targets depending directly on previous target=%llu/%u (%.6f%%)\n",(unsigned long long)previous_target_dependency_num,point_num>0?point_num-1:0,previous_dependency_ratio*100);
        const char *lag_bucket_name_vec[7]={"1","2-4","5-16","17-64","65-256","257-1024",">1024"};
        for (uint32_t bucket=0;bucket<7;bucket++)
            printf("  edge lag %s: %llu (%.6f%%)\n",lag_bucket_name_vec[bucket],(unsigned long long)edge_lag_count_vec[bucket],edge_num>0?(double)edge_lag_count_vec[bucket]/edge_num*100:0);
        uint32_t thread_num_vec[5]={2,4,8,16,32};
        for (uint32_t thread_num:thread_num_vec)
        {
            uint64_t level_barrier_step_num=0;
            for (uint64_t level_width:level_width_vec)
                level_barrier_step_num+=(level_width+thread_num-1)/thread_num;
            double level_barrier_speedup=(double)point_num/level_barrier_step_num;
            double utilization=level_barrier_speedup/thread_num*100;
            printf("  %u-thread level-barrier upper estimate: steps=%llu, speedup=%.6f, utilization=%.6f%%\n",thread_num,(unsigned long long)level_barrier_step_num,level_barrier_speedup,utilization);
        }
    }
}
