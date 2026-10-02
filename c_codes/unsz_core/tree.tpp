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
                    [&](int id1,int id2){return coord_array[id1][slice_dim]<coord_array[id2][slice_dim];});
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
        vector<int> original_ids(coord_array.n0);
        iota(original_ids.begin(),original_ids.end(),0);
        initialize(coord_array,min_coord,max_coord,original_ids,leaf_point_num);
    }

    template<typename T_coord>
    block_tree_c<T_coord>::block_tree_c(Array2D<T_coord> &coord_array,vector<T_coord> &min_coord,vector<T_coord> &max_coord,const vector<int>&original_ids,uint32_t leaf_point_num)
    {
        initialize(coord_array,min_coord,max_coord,original_ids,leaf_point_num);
    }

    template<typename T_coord>
    void block_tree_c<T_coord>::initialize(Array2D<T_coord> &coord_array,vector<T_coord> &min_coord,vector<T_coord> &max_coord,const vector<int>&original_ids,uint32_t leaf_point_num)
    {
        if (coord_array.n0==0 || coord_array.n1==0)
            throw runtime_error("kd-tree的点数和维数必须是正数");
        if(original_ids.empty())
            throw runtime_error("kd-tree的点集不能为空");
        if (leaf_point_num==0)
            throw runtime_error("kd-tree叶节点点数必须是正数");
        dim=coord_array.n1;
        structure=make_shared<block_tree_structure_c<T_coord>>();
        structure->dim=dim;
        structure->leaf_point_num=leaf_point_num;
        uint64_t max_block_num=(uint64_t)original_ids.size()*2-1;
        structure->block_vec.reserve(max_block_num);
        structure->block_coord_bounds.reserve(max_block_num*dim*2);
        structure->tree_id_to_original_id=original_ids;
        structure->tree_id_to_block_vec_id.resize(original_ids.size());
        for(int original_id:original_ids)
        {
            if(original_id<0||(uint32_t)original_id>=coord_array.n0)
                throw runtime_error("kd-tree点编号越界");
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
        structure->block_vec[0].index_end=original_ids.size();
        split_block(coord_array,0);
        valid_num[0]=original_ids.size();
        point_valid.assign(original_ids.size(),1);
        structure->original_id_to_tree_id.assign(coord_array.n0,-1);
        structure->original_id_to_block_vec_id.assign(coord_array.n0,-1);
        structure->tree_order_coord_array.resize(original_ids.size(),dim);
        for (uint32_t i=0;i<original_ids.size();i++)
        {
            int original_id=structure->tree_id_to_original_id[i];
            structure->original_id_to_tree_id[original_id]=i;
            structure->original_id_to_block_vec_id[original_id]=structure->tree_id_to_block_vec_id[i];
            memcpy(structure->tree_order_coord_array[i],coord_array[original_id],dim*sizeof(T_coord));
        }
        if(leaf_point_num<=64)
            enable_leaf_valid_mask();
    }

    template<typename T_coord>
    void block_tree_c<T_coord>::enable_leaf_valid_mask()
    {
        if(structure->leaf_point_num>64)
            throw runtime_error("叶节点有效位图只支持不超过64个点的叶节点");
        leaf_valid_mask.assign(structure->block_vec.size(),0);
        for(uint32_t block_id=0;block_id<structure->block_vec.size();block_id++)
        {
            const auto &block=structure->block_vec[block_id];
            if(block.left_child_id!=-1||block.right_child_id!=-1)
                continue;
            uint32_t point_num=block.index_end-block.index_begin;
            uint64_t mask=0;
            for(uint32_t point_offset=0;point_offset<point_num;point_offset++)
                if(point_valid[block.index_begin+point_offset])
                    mask|=(uint64_t)1<<point_offset;
            leaf_valid_mask[block_id]=mask;
        }
        use_leaf_valid_mask=true;
    }

    template<typename T_coord>
    void block_tree_c<T_coord>::clear_all_valid_num()
    {
        fill(valid_num.begin(),valid_num.end(),0);
        fill(point_valid.begin(),point_valid.end(),0);
        if(use_leaf_valid_mask)
            fill(leaf_valid_mask.begin(),leaf_valid_mask.end(),0);
    }

    template<typename T_coord>
    void block_tree_c<T_coord>::set_point_valid(int original_id,bool value)
    {
        int tree_id=structure->original_id_to_tree_id[original_id];
        point_valid[tree_id]=value;
        if(use_leaf_valid_mask)
        {
            int block_id=structure->tree_id_to_block_vec_id[tree_id];
            uint32_t point_offset=tree_id-structure->block_vec[block_id].index_begin;
            uint64_t bit=(uint64_t)1<<point_offset;
            if(value)
                leaf_valid_mask[block_id]|=bit;
            else
                leaf_valid_mask[block_id]&=~bit;
        }
    }

    template<typename T_coord>
    void block_tree_c<T_coord>::update_all_valid_num(int block_id)
    {
        if (structure->block_vec[block_id].left_child_id==-1 && structure->block_vec[block_id].right_child_id==-1)//是叶block
        {
            valid_num[block_id]=0;
            uint64_t mask=0;
            for (uint32_t tree_id=structure->block_vec[block_id].index_begin;tree_id<structure->block_vec[block_id].index_end;tree_id++)
            {
                valid_num[block_id]+=point_valid[tree_id];
                if(point_valid[tree_id])
                    mask|=(uint64_t)1<<(tree_id-structure->block_vec[block_id].index_begin);
            }
            if(use_leaf_valid_mask)
                leaf_valid_mask[block_id]=mask;
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
        if(use_leaf_valid_mask)
        {
            uint32_t point_offset=tree_id-structure->block_vec[block_id].index_begin;
            uint64_t bit=(uint64_t)1<<point_offset;
            if(new_value)
                leaf_valid_mask[block_id]|=bit;
            else
                leaf_valid_mask[block_id]&=~bit;
        }
        while (true)
        {
            valid_num[block_id]+=change_value;
            if(block_id==0)
                break;
            block_id=structure->block_vec[block_id].parent_id;
            if (block_id==-1)
                throw runtime_error("kd-tree目标点无法追溯到根节点");
        }
    }

    template<typename T_coord>
    void block_tree_c<T_coord>::find_original_nearest(nearest_queue_c<T_coord> &nearest_queue,Array2D<T_coord> &coord_array,
        uint32_t nearest_num,int block_id,T_coord *query_coord,bool valid_only)
    {
        if ((valid_only) && (valid_num[block_id]==0))
            return;
        if (structure->block_vec[block_id].left_child_id==-1 && structure->block_vec[block_id].right_child_id==-1)
        {
            if(valid_only&&use_leaf_valid_mask)
            {
                uint32_t index_begin=structure->block_vec[block_id].index_begin;
                uint64_t mask=leaf_valid_mask[block_id];
                while(mask)
                {
                    uint32_t point_offset=(uint32_t)__builtin_ctzll(mask);
                    uint32_t tree_id=index_begin+point_offset;
                    mask&=mask-1;
                    int original_id=structure->tree_id_to_original_id[tree_id];
                    T_coord dist=get_dist_to_coord(query_coord,structure->tree_order_coord_array[tree_id],dim);
                    if(nearest_queue.size()<nearest_num)
                        nearest_queue.push({dist,original_id});
                    else if(dist<nearest_queue.top().first)
                    {
                        nearest_queue.pop();
                        nearest_queue.push({dist,original_id});
                    }
                }
            }
            else
            {
                for (uint32_t tree_id=structure->block_vec[block_id].index_begin;tree_id<structure->block_vec[block_id].index_end;tree_id++)
                {
                    if (valid_only && !point_valid[tree_id])
                        continue;
                    int original_id=structure->tree_id_to_original_id[tree_id];
                    T_coord dist=get_dist_to_coord(query_coord,structure->tree_order_coord_array[tree_id],dim);
                    if (nearest_queue.size()<nearest_num)
                        nearest_queue.push({dist,original_id});
                    else if (dist<nearest_queue.top().first)
                    {
                        nearest_queue.pop();
                        nearest_queue.push({dist,original_id});
                    }
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
            if ((second_child_id!=-1) && ((nearest_queue.size()<nearest_num) || (dist_second<nearest_queue.top().first)))
                find_original_nearest(nearest_queue,coord_array,nearest_num,second_child_id,query_coord,valid_only);
        }
    }

}
