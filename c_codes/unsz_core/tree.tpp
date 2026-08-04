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
        int dim=coord_array.n1;
        if (index_end==index_begin+1)//block内只有一个点，无法再划分
        {
            min_coord.resize(dim);
            max_coord.resize(dim);
            memcpy(min_coord.data(),coord_array[point_id_on_tree_to_original_id[index_begin]],dim*sizeof(T_coord));
            memcpy(max_coord.data(),coord_array[point_id_on_tree_to_original_id[index_begin]],dim*sizeof(T_coord));
            point_id_on_tree_to_block_vec_id[index_begin]=block_id;
            return;
        }
        int slice_dim=0;
        for (int d=1;d<dim;d++)
            if (max_coord[d]-min_coord[d]>max_coord[slice_dim]-min_coord[slice_dim])
                slice_dim=d;
        int partition_pos=(index_begin+index_end)/2;
        nth_element(point_id_on_tree_to_original_id.begin()+index_begin,
                    point_id_on_tree_to_original_id.begin()+partition_pos,
                    point_id_on_tree_to_original_id.begin()+index_end,
                    [&](int id1,int id2){return coord_array[id1][slice_dim]<coord_array[id2][slice_dim];});
        T_coord slice_coord=coord_array[point_id_on_tree_to_original_id[partition_pos]][slice_dim];
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
        point_id_on_tree_to_original_id.resize(coord_array.n0);
        point_id_on_tree_to_block_vec_id.resize(coord_array.n0);
        for (int i=0;i<coord_array.n0;i++)
        {
            point_id_on_tree_to_original_id[i]=i;
            point_id_on_tree_to_block_vec_id[i]=0;
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
    }
}