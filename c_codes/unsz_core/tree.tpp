#pragma once

namespace unsz
{
    template<typename T_coord>
    void block_tree_c<T_coord>::split_block(Array2D<T_coord> &coord_array,int block_id)
    {
        uint32_t index_begin=structure->block_vec[block_id].index_begin;
        uint32_t index_end=structure->block_vec[block_id].index_end;
        T_coord *min_coord=get_block_min_coord(block_id);
        T_coord *max_coord=get_block_max_coord(block_id);
        if (index_end-index_begin<=structure->leaf_point_num)
        {
            fill(min_coord,min_coord+dim,numeric_limits<T_coord>::infinity());
            fill(max_coord,max_coord+dim,-numeric_limits<T_coord>::infinity());
            for (uint32_t tree_id=index_begin;tree_id<index_end;tree_id++)
            {
                int original_id=structure->tree_id_to_original_id[tree_id];
                structure->tree_id_to_block_vec_id[tree_id]=block_id;
                for (uint32_t d=0;d<dim;d++)
                {
                    min_coord[d]=min(min_coord[d],coord_array[original_id][d]);
                    max_coord[d]=max(max_coord[d],coord_array[original_id][d]);
                }
            }
            return;
        }
        uint32_t slice_dim=0;
        for (uint32_t d=1;d<dim;d++)
            if (max_coord[d]-min_coord[d]>max_coord[slice_dim]-min_coord[slice_dim])
                slice_dim=d;
        uint32_t partition_pos=(index_begin+index_end)/2;
        nth_element(structure->tree_id_to_original_id.begin()+index_begin,
                    structure->tree_id_to_original_id.begin()+partition_pos,
                    structure->tree_id_to_original_id.begin()+index_end,
                    [&](int id1,int id2){return coord_array[id1][slice_dim]<coord_array[id2][slice_dim] || (coord_array[id1][slice_dim]==coord_array[id2][slice_dim] && id1<id2);});
        T_coord slice_coord=coord_array[structure->tree_id_to_original_id[partition_pos]][slice_dim];
        if (partition_pos!=index_begin)//有一些点在左边
        {
            int child_id=structure->block_vec.size();
            structure->block_vec.emplace_back();
            structure->block_coord_bounds.resize(((uint64_t)child_id+1)*dim*2);
            memcpy(get_block_min_coord(child_id),min_coord,dim*sizeof(T_coord));
            memcpy(get_block_max_coord(child_id),max_coord,dim*sizeof(T_coord));
            structure->block_vec[block_id].left_child_id=child_id;
            structure->block_vec[child_id].parent_id=block_id;
            get_block_max_coord(child_id)[slice_dim]=slice_coord;
            structure->block_vec[child_id].index_begin=index_begin;
            structure->block_vec[child_id].index_end=partition_pos;
            valid_num.push_back(partition_pos-index_begin);
            split_block(coord_array,child_id);
        }
        if (partition_pos!=index_end)//有一些点在右边
        {
            int child_id=structure->block_vec.size();
            structure->block_vec.emplace_back();
            structure->block_coord_bounds.resize(((uint64_t)child_id+1)*dim*2);
            memcpy(get_block_min_coord(child_id),min_coord,dim*sizeof(T_coord));
            memcpy(get_block_max_coord(child_id),max_coord,dim*sizeof(T_coord));
            structure->block_vec[block_id].right_child_id=child_id;
            structure->block_vec[child_id].parent_id=block_id;
            get_block_min_coord(child_id)[slice_dim]=slice_coord;
            structure->block_vec[child_id].index_begin=partition_pos;
            structure->block_vec[child_id].index_end=index_end;
            valid_num.push_back(index_end-partition_pos);
            split_block(coord_array,child_id);
        }
        min_coord=get_block_min_coord(block_id);
        max_coord=get_block_max_coord(block_id);
        for(uint32_t d=0;d<dim;d++)
        {
            min_coord[d]=numeric_limits<T_coord>::infinity();
            max_coord[d]=-numeric_limits<T_coord>::infinity();
            if (structure->block_vec[block_id].left_child_id!=-1)
            {
                min_coord[d]=min(min_coord[d],get_block_min_coord(structure->block_vec[block_id].left_child_id)[d]);
                max_coord[d]=max(max_coord[d],get_block_max_coord(structure->block_vec[block_id].left_child_id)[d]);
            }
            if (structure->block_vec[block_id].right_child_id!=-1)
            {
                min_coord[d]=min(min_coord[d],get_block_min_coord(structure->block_vec[block_id].right_child_id)[d]);
                max_coord[d]=max(max_coord[d],get_block_max_coord(structure->block_vec[block_id].right_child_id)[d]);
            }
        }
    }

    template<typename T_coord>
    block_tree_c<T_coord>::block_tree_c(Array2D<T_coord> &coord_array,vector<T_coord> &min_coord,vector<T_coord> &max_coord,uint32_t leaf_point_num)
    {
        if (coord_array.n0==0 || coord_array.n1==0)
            throw runtime_error("kd-tree的点数和维数必须是正数");
        if (leaf_point_num==0)
            throw runtime_error("kd-tree叶节点点数必须是正数");
        dim=coord_array.n1;
        structure=make_shared<block_tree_structure_c<T_coord>>();
        structure->dim=dim;
        structure->leaf_point_num=leaf_point_num;
        uint64_t max_block_num=(uint64_t)coord_array.n0*2-1;
        structure->block_vec.reserve(max_block_num);
        structure->block_coord_bounds.reserve(max_block_num*dim*2);
        structure->tree_id_to_original_id.resize(coord_array.n0);
        structure->tree_id_to_block_vec_id.resize(coord_array.n0);
        for (uint32_t i=0;i<coord_array.n0;i++)
        {
            structure->tree_id_to_original_id[i]=i;
            structure->tree_id_to_block_vec_id[i]=0;
        }
        structure->block_vec.emplace_back();
        valid_num.push_back(0);
        structure->block_coord_bounds.resize((uint64_t)dim*2);
        structure->block_vec[0].parent_id=-1;
        structure->block_vec[0].left_child_id=-1;
        structure->block_vec[0].right_child_id=-1;
        memcpy(get_block_min_coord(0),min_coord.data(),dim*sizeof(T_coord));
        memcpy(get_block_max_coord(0),max_coord.data(),dim*sizeof(T_coord));
        structure->block_vec[0].index_begin=0;
        structure->block_vec[0].index_end=coord_array.n0;
        split_block(coord_array,0);
        valid_num[0]=coord_array.n0;
        point_valid.assign(coord_array.n0,1);
        structure->original_id_to_tree_id.resize(coord_array.n0);
        structure->original_id_to_block_vec_id.resize(coord_array.n0);
        structure->tree_order_coord_array.resize(coord_array.n0,dim);
        for (uint32_t i=0;i<coord_array.n0;i++)
        {
            int original_id=structure->tree_id_to_original_id[i];
            structure->original_id_to_tree_id[original_id]=i;
            structure->original_id_to_block_vec_id[original_id]=structure->tree_id_to_block_vec_id[i];
            memcpy(structure->tree_order_coord_array[i],coord_array[original_id],dim*sizeof(T_coord));
        }
    }

    template<typename T_coord>
    void block_tree_c<T_coord>::clear_all_valid_num()
    {
        fill(valid_num.begin(),valid_num.end(),0);
        fill(point_valid.begin(),point_valid.end(),0);
    }

    template<typename T_coord>
    void block_tree_c<T_coord>::set_point_valid(int original_id,bool value)
    {
        int tree_id=structure->original_id_to_tree_id[original_id];
        point_valid[tree_id]=value;
    }

    template<typename T_coord>
    void block_tree_c<T_coord>::update_all_valid_num(int block_id)
    {
        if (structure->block_vec[block_id].left_child_id==-1 && structure->block_vec[block_id].right_child_id==-1)//是叶block
        {
            valid_num[block_id]=0;
            for (uint32_t tree_id=structure->block_vec[block_id].index_begin;tree_id<structure->block_vec[block_id].index_end;tree_id++)
                valid_num[block_id]+=point_valid[tree_id];
            return;
        }
        valid_num[block_id]=0;
        if (structure->block_vec[block_id].left_child_id!=-1)
        {
            update_all_valid_num(structure->block_vec[block_id].left_child_id);
            valid_num[block_id]+=valid_num[structure->block_vec[block_id].left_child_id];
        }
        if (structure->block_vec[block_id].right_child_id!=-1)
        {
            update_all_valid_num(structure->block_vec[block_id].right_child_id);
            valid_num[block_id]+=valid_num[structure->block_vec[block_id].right_child_id];
        }
    }

    template<typename T_coord>
    void block_tree_c<T_coord>::update_point_valid_num_to_root(int original_id,int change_value)
    {
        int tree_id=structure->original_id_to_tree_id[original_id];
        int new_value=(int)point_valid[tree_id]+change_value;
        if (new_value<0 || new_value>1)
            throw runtime_error("kd-tree逐点有效状态更新越界");
        point_valid[tree_id]=new_value;
        int block_id=structure->tree_id_to_block_vec_id[tree_id];
        while (block_id!=-1)
        {
            valid_num[block_id]+=change_value;
            block_id=structure->block_vec[block_id].parent_id;
        }
    }

    // template<typename T_coord>
    // void block_tree_c<T_coord>::find_nearest(priority_queue<pair<T_coord,int>> &nearest_queue,int nearest_num,int block_id,
    //     T_coord *query_coord,bool valid_only)
    // {
    //     if ((valid_only) && (block_vec[block_id].valid_num==0))
    //         return;
    //     if (block_vec[block_id].index_end==block_vec[block_id].index_begin+1)//block内有一个点，并且这个点有效，计算距离并和已有的nearest_vec比较
    //     {
    //         T_coord dist=get_dist_to_node(query_coord,coord_array[block_vec[block_id].index_begin]);
    //         if (nearest_queue.size()<nearest_num)
    //             nearest_queue.push({dist,block_vec[block_id].index_begin});
    //         else if (dist<nearest_queue.top().first)
    //         {
    //             nearest_queue.pop();
    //             nearest_queue.push({dist,block_vec[block_id].index_begin});
    //         }
    //     }
    //     else//block内有多个点，比较到两个子块的距离，先访问近的子块，再访问远的子块
    //     {
    //         int first_child_id=block_vec[block_id].left_child_id;
    //         int second_child_id=block_vec[block_id].right_child_id;
    //         T_coord dist_first=numeric_limits<T_coord>::infinity(),dist_second=numeric_limits<T_coord>::infinity();
    //         if (first_child_id!=-1)
    //             dist_first=get_dist_to_block(query_coord,block_vec[first_child_id].min_coord.data(),block_vec[first_child_id].max_coord.data());
    //         if (second_child_id!=-1)
    //             dist_second=get_dist_to_block(query_coord,block_vec[second_child_id].min_coord.data(),block_vec[second_child_id].max_coord.data());
    //         if (dist_first>dist_second)
    //         {
    //             swap(first_child_id,second_child_id);
    //             swap(dist_first,dist_second);
    //         }
    //         if (first_child_id!=-1)
    //             find_nearest(nearest_queue,nearest_num,first_child_id,query_coord,valid_only);
    //         if ((second_child_id!=-1) && ((nearest_queue.size()<nearest_num) || (dist_second<nearest_queue.top().first)))
    //             find_nearest(nearest_queue,nearest_num,second_child_id,query_coord,valid_only);
    //     }
    // }

    template<typename T_coord>
    void block_tree_c<T_coord>::find_original_nearest(nearest_queue_c<T_coord> &nearest_queue,Array2D<T_coord> &coord_array,
        uint32_t nearest_num,int block_id,T_coord *query_coord,bool valid_only)
    {
        if ((valid_only) && (valid_num[block_id]==0))
            return;
        if (structure->block_vec[block_id].left_child_id==-1 && structure->block_vec[block_id].right_child_id==-1)
        {
            for (uint32_t tree_id=structure->block_vec[block_id].index_begin;tree_id<structure->block_vec[block_id].index_end;tree_id++)
            {
                if (valid_only && !point_valid[tree_id])
                    continue;
                int original_id=structure->tree_id_to_original_id[tree_id];
                T_coord dist=get_dist_to_coord(query_coord,structure->tree_order_coord_array[tree_id],dim);
                pair<T_coord,int> candidate={dist,original_id};
                if (nearest_queue.size()<nearest_num)
                    nearest_queue.push(candidate);
                else if (candidate<nearest_queue.top())
                {
                    nearest_queue.pop();
                    nearest_queue.push(candidate);
                }
            }
        }
        else//block内有多个点，比较到两个子块的距离，先访问近的子块，再访问远的子块
        {
            int first_child_id=structure->block_vec[block_id].left_child_id;
            int second_child_id=structure->block_vec[block_id].right_child_id;
            T_coord dist_first=numeric_limits<T_coord>::infinity(),dist_second=numeric_limits<T_coord>::infinity();
            if (first_child_id!=-1)
                dist_first=get_dist_to_block(query_coord,get_block_min_coord(first_child_id),get_block_max_coord(first_child_id),dim);
            if (second_child_id!=-1)
                dist_second=get_dist_to_block(query_coord,get_block_min_coord(second_child_id),get_block_max_coord(second_child_id),dim);
            if (dist_first>dist_second)
            {
                swap(first_child_id,second_child_id);
                swap(dist_first,dist_second);
            }
            if (first_child_id!=-1)
                find_original_nearest(nearest_queue,coord_array,nearest_num,first_child_id,query_coord,valid_only);
            if ((second_child_id!=-1) && ((nearest_queue.size()<nearest_num) || (dist_second<=nearest_queue.top().first)))
                find_original_nearest(nearest_queue,coord_array,nearest_num,second_child_id,query_coord,valid_only);
        }
    }
}
