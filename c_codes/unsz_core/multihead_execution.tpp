#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_multihead_execution_order()
    {
        multihead_execution_pred_ref_ids.clear();
        multihead_qb_index_by_pred_ref_id.clear();
        if(multihead_execution_head_num_experimental==0)
            return;
        if(multihead_execution_head_num_experimental<2||multihead_execution_head_num_experimental>64)
            throw runtime_error("多头执行数量必须在[2,64]内");
        if(point_set_config_vec.size()!=1||point_set_config_vec[0].codec!=point_set_codec_e::rbf_fitting||get_prediction_framework(1).value_mode!=time_dimension_mode_e::spatial)
            throw runtime_error("多头执行实验目前只支持单点集纯空间RBF");
        if(reorder_data_by_prediction_order_experimental)
            throw runtime_error("多头执行实验暂不支持按预测顺序重排输入");
        if(partition_size_experimental>0||partition_num_experimental>0)
            throw runtime_error("多头执行实验暂不支持partition");

        uint32_t head_num=min(multihead_execution_head_num_experimental,point_num);
        block_tree_c<T_coord> spatial_tree(coord_array,min_coord,max_coord,kd_tree_leaf_size);
        vector<uint32_t> original_id_to_pred_ref_id(point_num,numeric_limits<uint32_t>::max());
        vector<uint32_t> pred_ref_id_to_head(point_num),indegree(point_num,0),child_count(point_num,0);
        for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
        {
            int target_id=pred_target_ids[pred_ref_id];
            if(target_id<0||(uint32_t)target_id>=point_num||original_id_to_pred_ref_id[target_id]!=numeric_limits<uint32_t>::max())
                throw runtime_error("多头执行发现无效或重复目标点");
            original_id_to_pred_ref_id[target_id]=pred_ref_id;
            uint32_t tree_id=spatial_tree.structure->original_id_to_tree_id[target_id];
            pred_ref_id_to_head[pred_ref_id]=min<uint32_t>(head_num-1,(uint64_t)tree_id*head_num/point_num);
        }

        for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
        {
            auto ref_ids=get_pred_ref_ids(pred_ref_id);
            uint32_t ref_num=min<uint32_t>(actual_ref_num,ref_ids.size());
            indegree[pred_ref_id]=ref_num;
            for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
            {
                uint32_t ref_pred_ref_id=original_id_to_pred_ref_id[ref_ids[ref_index]];
                if(ref_pred_ref_id>=pred_ref_id)
                    throw runtime_error("多头执行发现不满足拓扑顺序的参考点");
                child_count[ref_pred_ref_id]++;
            }
        }

        vector<uint64_t> child_begin((uint64_t)point_num+1,0),child_cursor;
        for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
            child_begin[(uint64_t)pred_ref_id+1]=child_begin[pred_ref_id]+child_count[pred_ref_id];
        child_cursor=child_begin;
        vector<uint32_t> child_pred_ref_ids(child_begin.back());
        for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
        {
            auto ref_ids=get_pred_ref_ids(pred_ref_id);
            uint32_t ref_num=min<uint32_t>(actual_ref_num,ref_ids.size());
            for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
            {
                uint32_t ref_pred_ref_id=original_id_to_pred_ref_id[ref_ids[ref_index]];
                child_pred_ref_ids[child_cursor[ref_pred_ref_id]++]=pred_ref_id;
            }
        }

        vector<priority_queue<uint32_t,vector<uint32_t>,greater<uint32_t>>> ready_queue_vec(head_num);
        for(uint32_t pred_ref_id=0;pred_ref_id<point_num;pred_ref_id++)
            if(indegree[pred_ref_id]==0)
                ready_queue_vec[pred_ref_id_to_head[pred_ref_id]].push(pred_ref_id);
        multihead_execution_pred_ref_ids.reserve(point_num);
        vector<vector<uint32_t>> head_pred_ref_ids(head_num);
        uint32_t next_head=0;
        while(multihead_execution_pred_ref_ids.size()<point_num)
        {
            uint32_t selected_head=head_num;
            for(uint32_t head_offset=0;head_offset<head_num;head_offset++)
            {
                uint32_t head=(next_head+head_offset)%head_num;
                if(!ready_queue_vec[head].empty())
                {
                    selected_head=head;
                    break;
                }
            }
            if(selected_head==head_num)
                throw runtime_error("多头执行的拓扑调度无法继续");
            uint32_t pred_ref_id=ready_queue_vec[selected_head].top();
            ready_queue_vec[selected_head].pop();
            multihead_execution_pred_ref_ids.push_back(pred_ref_id);
            head_pred_ref_ids[selected_head].push_back(pred_ref_id);
            next_head=(selected_head+1)%head_num;
            for(uint64_t child_index=child_begin[pred_ref_id];child_index<child_begin[(uint64_t)pred_ref_id+1];child_index++)
            {
                uint32_t child_pred_ref_id=child_pred_ref_ids[child_index];
                if(indegree[child_pred_ref_id]==0)
                    throw runtime_error("多头执行的子节点入度重复归零");
                indegree[child_pred_ref_id]--;
                if(indegree[child_pred_ref_id]==0)
                    ready_queue_vec[pred_ref_id_to_head[child_pred_ref_id]].push(child_pred_ref_id);
            }
        }

        multihead_qb_index_by_pred_ref_id.resize(point_num);
        uint32_t qb_index=0,min_head_size=point_num,max_head_size=0;
        for(const auto&head:head_pred_ref_ids)
        {
            min_head_size=min<uint32_t>(min_head_size,head.size());
            max_head_size=max<uint32_t>(max_head_size,head.size());
            for(uint32_t pred_ref_id:head)
                multihead_qb_index_by_pred_ref_id[pred_ref_id]=qb_index++;
        }

        vector<uint32_t> execution_position(point_num);
        for(uint32_t position=0;position<point_num;position++)
            execution_position[multihead_execution_pred_ref_ids[position]]=position;
        uint64_t immediate_dependency_num=0;
        for(uint32_t position=1;position<point_num;position++)
        {
            uint32_t pred_ref_id=multihead_execution_pred_ref_ids[position];
            auto ref_ids=get_pred_ref_ids(pred_ref_id);
            uint32_t ref_num=min<uint32_t>(actual_ref_num,ref_ids.size());
            for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                if(execution_position[original_id_to_pred_ref_id[ref_ids[ref_index]]]+1==position)
                {
                    immediate_dependency_num++;
                    break;
                }
        }
        printf("Prepared %u-head dependency-safe execution order: head_size=[%u,%u], immediate_dependency=%llu/%u (%.6f%%)\n",head_num,min_head_size,max_head_size,(unsigned long long)immediate_dependency_num,point_num-1,point_num>1?100.0*immediate_dependency_num/(point_num-1):0);
    }
}
