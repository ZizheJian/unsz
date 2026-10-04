#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    vector<vector<int>> param_c<T_coord,T_data>::build_partitions()
    {
        if(partition_size==0&&partition_num==0)
            return {};
        if(partition_size>0&&partition_num>0)
            throw runtime_error("不能同时指定partition大小和partition数量");
        uint32_t requested_partition_num=partition_num;
        if(requested_partition_num==0)
        {
            requested_partition_num=1;
            while(((uint64_t)point_num+requested_partition_num-1)/requested_partition_num>partition_size)
                requested_partition_num*=2;
        }
        if(requested_partition_num==0||requested_partition_num>point_num||(requested_partition_num&(requested_partition_num-1))!=0)
            throw runtime_error("partition数量必须是不超过点数的2的幂");

        vector<vector<int>> partition_point_ids_vec(1,vector<int>(point_num));
        iota(partition_point_ids_vec.front().begin(),partition_point_ids_vec.front().end(),0);
        while(partition_point_ids_vec.size()<requested_partition_num)
        {
            vector<vector<int>> child_point_ids_vec;
            child_point_ids_vec.reserve(partition_point_ids_vec.size()*2);
            for(vector<int>&point_ids:partition_point_ids_vec)
            {
                if(point_ids.size()<2)
                    throw runtime_error("点数不足，无法继续划分partition");
                vector<T_coord> partition_min_coord(dim,numeric_limits<T_coord>::infinity()),partition_max_coord(dim,-numeric_limits<T_coord>::infinity());
                for(int point_id:point_ids)
                    for(uint32_t dimension=0;dimension<dim;dimension++)
                    {
                        partition_min_coord[dimension]=min(partition_min_coord[dimension],coord_array[point_id][dimension]);
                        partition_max_coord[dimension]=max(partition_max_coord[dimension],coord_array[point_id][dimension]);
                    }
                uint32_t split_dimension=0;
                for(uint32_t dimension=1;dimension<dim;dimension++)
                    if(partition_max_coord[dimension]-partition_min_coord[dimension]>partition_max_coord[split_dimension]-partition_min_coord[split_dimension])
                        split_dimension=dimension;
                auto middle=point_ids.begin()+point_ids.size()/2;
                nth_element(point_ids.begin(),middle,point_ids.end(),[&](int left_id,int right_id){return coord_array[left_id][split_dimension]<coord_array[right_id][split_dimension];});
                child_point_ids_vec.emplace_back(point_ids.begin(),middle);
                child_point_ids_vec.emplace_back(middle,point_ids.end());
            }
            partition_point_ids_vec.swap(child_point_ids_vec);
        }

        vector<uint8_t> point_was_assigned(point_num,0);
        uint32_t min_partition_point_num=point_num,max_partition_point_num=0;
        for(uint32_t partition_id=0;partition_id<partition_point_ids_vec.size();partition_id++)
        {
            const vector<int>&point_ids=partition_point_ids_vec[partition_id];
            min_partition_point_num=min<uint32_t>(min_partition_point_num,point_ids.size());
            max_partition_point_num=max<uint32_t>(max_partition_point_num,point_ids.size());
            for(int point_id:point_ids)
            {
                if(point_was_assigned[point_id])
                    throw runtime_error("partition包含重复点");
                point_was_assigned[point_id]=1;
            }
        }
        if(any_of(point_was_assigned.begin(),point_was_assigned.end(),[](uint8_t assigned){return !assigned;}))
            throw runtime_error("partition没有覆盖所有点");
        printf("Independent spatial partitions: %u, point_num_range=[%u,%u]\n",(uint32_t)partition_point_ids_vec.size(),min_partition_point_num,max_partition_point_num);
        return partition_point_ids_vec;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::generate_partition_prediction_structures(bool uses_idw,bool uses_rbf)
    {
        auto generation_start_time=chrono::steady_clock::now();
        vector<vector<int>> partition_point_ids_vec=build_partitions();
        prediction_structure_vec.clear();
        prediction_structure_vec.resize(partition_point_ids_vec.size());
        printf("Generating independent partition prediction structures: partition_num=%u,thread_num=%u\n",(uint32_t)partition_point_ids_vec.size(),thread_num);
        exception_ptr generation_exception;
#pragma omp parallel for schedule(dynamic) num_threads(thread_num) if(thread_num>1)
        for(int64_t partition_index=0;partition_index<(int64_t)partition_point_ids_vec.size();partition_index++)
        {
            try
            {
                uint32_t partition_id=(uint32_t)partition_index;
                const vector<int>&point_ids=partition_point_ids_vec[partition_id];
                vector<T_coord> partition_min_coord(dim,numeric_limits<T_coord>::infinity()),partition_max_coord(dim,-numeric_limits<T_coord>::infinity());
                for(int point_id:point_ids)
                    for(uint32_t dimension=0;dimension<dim;dimension++)
                    {
                        partition_min_coord[dimension]=min(partition_min_coord[dimension],coord_array[point_id][dimension]);
                        partition_max_coord[dimension]=max(partition_max_coord[dimension],coord_array[point_id][dimension]);
                    }
                auto partition_start_time=chrono::steady_clock::now();
                block_tree_c<T_coord> tree(coord_array,partition_min_coord,partition_max_coord,point_ids,kd_tree_leaf_size);
                generate_prediction_structure_and_weights(tree,point_ids,partition_min_coord,partition_max_coord,prediction_structure_vec[partition_id],partition_start_time,uses_idw,uses_rbf);
            }
            catch(...)
            {
#pragma omp critical
                if(!generation_exception)
                    generation_exception=current_exception();
            }
        }
        if(generation_exception)
            rethrow_exception(generation_exception);
        partition_point_ids_vec.clear();
        uint64_t generated_target_num=0,generated_reference_num=0;
        for(const prediction_structure_c<T_calc>&structure:prediction_structure_vec)
        {
            generated_target_num+=structure.target_ids.size();
            generated_reference_num+=structure.reference_ids.size();
        }
        if(generated_target_num!=point_num)
            throw runtime_error("partition预测结构没有覆盖所有点");
        printf("Generated %llu partition prediction entries and %llu references, total time: %f s\n",(unsigned long long)generated_target_num,(unsigned long long)generated_reference_num,chrono::duration<double>(chrono::steady_clock::now()-generation_start_time).count());
    }
}
