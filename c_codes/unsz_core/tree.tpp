#pragma once

namespace unsz
{
    template<typename T_coord>
    void block_tree_c<T_coord>::split_block(Array2D<T_coord> &coord_array,int block_id)
    {
        int index_begin=block_vec[block_id].index_begin;
        int index_end=block_vec[block_id].index_end;
        auto min_coord=block_vec[block_id].min_coord;
        auto max_coord=block_vec[block_id].max_coord;
        if (index_end==index_begin+1)//block内只有一个点，无法再划分
        {
            min_coord.resize(dim);
            max_coord.resize(dim);
            memcpy(min_coord.data(),coord_array[tree_id_to_original_id[index_begin]],dim*sizeof(T_coord));
            memcpy(max_coord.data(),coord_array[tree_id_to_original_id[index_begin]],dim*sizeof(T_coord));
            tree_id_to_block_vec_id[index_begin]=block_id;
            return;
        }
        int slice_dim=0;
        for (int d=1;d<dim;d++)
            if (max_coord[d]-min_coord[d]>max_coord[slice_dim]-min_coord[slice_dim])
                slice_dim=d;
        int partition_pos=(index_begin+index_end)/2;
        nth_element(tree_id_to_original_id.begin()+index_begin,
                    tree_id_to_original_id.begin()+partition_pos,
                    tree_id_to_original_id.begin()+index_end,
                    [&](int id1,int id2){return coord_array[id1][slice_dim]<coord_array[id2][slice_dim];});
        T_coord slice_coord=coord_array[tree_id_to_original_id[partition_pos]][slice_dim];
        if (partition_pos!=index_begin)//有一些点在左边
        {
            block_vec.push_back(block_c<T_coord>());
            block_vec[block_id].left_child_id=block_vec.size()-1;
            block_vec.back().parent_id=block_id;
            block_vec.back().min_coord=min_coord;
            block_vec.back().max_coord=max_coord;
            block_vec.back().max_coord[slice_dim]=slice_coord;
            block_vec.back().index_begin=index_begin;
            block_vec.back().index_end=partition_pos;
            block_vec.back().valid_num=partition_pos-index_begin;
            split_block(coord_array,block_vec[block_id].left_child_id);
        }
        if (partition_pos!=index_end)//有一些点在右边
        {
            block_vec.push_back(block_c<T_coord>());
            block_vec[block_id].right_child_id=block_vec.size()-1;
            block_vec.back().parent_id=block_id;
            block_vec.back().min_coord=min_coord;
            block_vec.back().max_coord=max_coord;
            block_vec.back().min_coord[slice_dim]=slice_coord;
            block_vec.back().index_begin=partition_pos;
            block_vec.back().index_end=index_end;
            block_vec.back().valid_num=index_end-partition_pos;
            split_block(coord_array,block_vec[block_id].right_child_id);
        }
        for(int d=0;d<dim;d++)
        {
            min_coord[d]=numeric_limits<T_coord>::infinity();
            max_coord[d]=-numeric_limits<T_coord>::infinity();
            if (block_vec[block_id].left_child_id!=-1)
            {
                min_coord[d]=min(block_vec[block_id].min_coord[d],block_vec[block_vec[block_id].left_child_id].min_coord[d]);
                max_coord[d]=max(block_vec[block_id].max_coord[d],block_vec[block_vec[block_id].left_child_id].max_coord[d]);
            }
            if (block_vec[block_id].right_child_id!=-1)
            {
                min_coord[d]=min(block_vec[block_id].min_coord[d],block_vec[block_vec[block_id].right_child_id].min_coord[d]);
                max_coord[d]=max(block_vec[block_id].max_coord[d],block_vec[block_vec[block_id].right_child_id].max_coord[d]);
            }
        }
    }

    template<typename T_coord>
    block_tree_c<T_coord>::block_tree_c(Array2D<T_coord> &coord_array,vector<T_coord> &min_coord,vector<T_coord> &max_coord)
    {
        dim=coord_array.n1;
        tree_id_to_original_id.resize(coord_array.n0);
        tree_id_to_block_vec_id.resize(coord_array.n0);
        for (int i=0;i<coord_array.n0;i++)
        {
            tree_id_to_original_id[i]=i;
            tree_id_to_block_vec_id[i]=0;
        }
        block_vec.push_back(block_c<T_coord>());
        block_vec[0].parent_id=-1;
        block_vec[0].left_child_id=-1;
        block_vec[0].right_child_id=-1;
        block_vec[0].min_coord=min_coord;
        block_vec[0].max_coord=max_coord;
        block_vec[0].index_begin=0;
        block_vec[0].index_end=coord_array.n0;
        split_block(coord_array,0);
        original_id_to_tree_id.resize(coord_array.n0);
        original_id_to_block_vec_id.resize(coord_array.n0);
        for (int i=0;i<coord_array.n0;i++)
        {
            original_id_to_tree_id[tree_id_to_original_id[i]]=i;
            original_id_to_block_vec_id[tree_id_to_original_id[i]]=tree_id_to_block_vec_id[i];
        }
    }

    template<typename T_coord>
    void block_tree_c<T_coord>::update_all_valid_num(int block_id)
    {
        if (block_vec[block_id].left_child_id==-1 && block_vec[block_id].right_child_id==-1)//是叶block
            return;
        block_vec[block_id].valid_num=0;
        if (block_vec[block_id].left_child_id!=-1)
        {
            update_all_valid_num(block_vec[block_id].left_child_id);
            block_vec[block_id].valid_num+=block_vec[block_vec[block_id].left_child_id].valid_num;
        }
        if (block_vec[block_id].right_child_id!=-1)
        {
            update_all_valid_num(block_vec[block_id].right_child_id);
            block_vec[block_id].valid_num+=block_vec[block_vec[block_id].right_child_id].valid_num;
        }
    }

    template<typename T_coord>
    T_coord block_tree_c<T_coord>::get_dist_to_coord(T_coord *coord1,T_coord *coord2)
    {
        T_coord dist=0;
        for (int d=0;d<dim;d++)
            dist+=(coord1[d]-coord2[d])*(coord1[d]-coord2[d]);
        return dist;
    }

    template<typename T_coord>
    T_coord block_tree_c<T_coord>::get_dist_to_block(T_coord *coord,block_c<T_coord> &block)
    {
        T_coord dist=0;
        for (int d=0;d<dim;d++)
            if (coord[d]<block.min_coord[d])
                dist+=(coord[d]-block.min_coord[d])*(coord[d]-block.min_coord[d]);
            else if (coord[d]>block.max_coord[d])
                dist+=(coord[d]-block.max_coord[d])*(coord[d]-block.max_coord[d]);
        return dist;
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
    //         if (nearest_queue.size()<(size_t)nearest_num)
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
    //         if ((second_child_id!=-1) && ((nearest_queue.size()<(size_t)nearest_num) || (dist_second<nearest_queue.top().first)))
    //             find_nearest(nearest_queue,nearest_num,second_child_id,query_coord,valid_only);
    //     }
    // }

    template<typename T_coord>
    void block_tree_c<T_coord>::find_original_nearest(priority_queue<pair<T_coord,int>> &nearest_queue,Array2D<T_coord> &coord_array,
        int nearest_num,int block_id,T_coord *query_coord,bool valid_only)
    {
        if ((valid_only) && (block_vec[block_id].valid_num==0))
            return;
        if (block_vec[block_id].index_end==block_vec[block_id].index_begin+1)//block内有一个点，并且这个点有效，计算距离并和已有的nearest_vec比较
        {
            int tree_id=block_vec[block_id].index_begin;
            int original_id=tree_id_to_original_id[tree_id];
            T_coord dist=get_dist_to_coord(query_coord,coord_array[original_id]);
            if (nearest_queue.size()<(size_t)nearest_num)
                nearest_queue.push({dist,original_id});
            else if (dist<nearest_queue.top().first)
            {
                nearest_queue.pop();
                nearest_queue.push({dist,original_id});
            }
        }
        else//block内有多个点，比较到两个子块的距离，先访问近的子块，再访问远的子块
        {
            int first_child_id=block_vec[block_id].left_child_id;
            int second_child_id=block_vec[block_id].right_child_id;
            T_coord dist_first=numeric_limits<T_coord>::infinity(),dist_second=numeric_limits<T_coord>::infinity();
            if (first_child_id!=-1)
                dist_first=get_dist_to_block(query_coord,block_vec[first_child_id]);
            if (second_child_id!=-1)
                dist_second=get_dist_to_block(query_coord,block_vec[second_child_id]);
            if (dist_first>dist_second)
            {
                swap(first_child_id,second_child_id);
                swap(dist_first,dist_second);
            }
            if (first_child_id!=-1)
                find_original_nearest(nearest_queue,coord_array,nearest_num,first_child_id,query_coord,valid_only);
            if ((second_child_id!=-1) && ((nearest_queue.size()<(size_t)nearest_num) || (dist_second<nearest_queue.top().first)))
                find_original_nearest(nearest_queue,coord_array,nearest_num,second_child_id,query_coord,valid_only);
        }
    }
}