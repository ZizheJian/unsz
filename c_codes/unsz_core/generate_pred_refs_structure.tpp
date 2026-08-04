// 这里我们将所有预测目标拆分成多轮
// 虽然每一轮内会不断更新已经完成预测的点集，导致同轮内存在依赖关系，看起来拆分成轮是没必要的。但实际上这会增加预测目标在参考点内部的可能性。
#pragma once

namespace unsz
{
    // template<typename T_coord,typename T_data>
    // void param_c<T_coord,T_data>::clear_all_valid_num()
    // {
    //     for (auto &block:block_vec)
    //         block.valid_num=0;
    // }

    // template<typename T_coord,typename T_data>
    // void param_c<T_coord,T_data>::update_all_valid_num(int block_id)
    // {
    //     if (block_vec[block_id].left_child_id==-1 && block_vec[block_id].right_child_id==-1)//是叶block
    //         return;
    //     else
    //     {
    //         block_vec[block_id].valid_num=0;
    //         if (block_vec[block_id].left_child_id!=-1)
    //         {
    //             update_all_valid_num(block_vec[block_id].left_child_id);
    //             block_vec[block_id].valid_num+=block_vec[block_vec[block_id].left_child_id].valid_num;
    //         }
    //         if (block_vec[block_id].right_child_id!=-1)
    //         {
    //             update_all_valid_num(block_vec[block_id].right_child_id);
    //             block_vec[block_id].valid_num+=block_vec[block_vec[block_id].right_child_id].valid_num;
    //         }
    //     }
    // }

    // template<typename T_coord,typename T_data>
    // void param_c<T_coord,T_data>::update_valid_num_to_root(int block_id,int change_value)
    // {
    //     block_vec[block_id].valid_num+=change_value;
    //     if (block_vec[block_id].parent_id>=0)//不是根节点
    //         update_valid_num_to_root(block_vec[block_id].parent_id,change_value);
    // }

    // template<typename T_coord,typename T_data>
    // T_coord param_c<T_coord,T_data>::get_dist_to_node(T_coord *coord1,T_coord *coord2)
    // {
    //     T_coord dist=0;
    //     for (int d=0;d<dim;d++)
    //         dist+=(coord1[d]-coord2[d])*(coord1[d]-coord2[d]);
    //     return dist;
    // }

    // template<typename T_coord,typename T_data>
    // T_coord param_c<T_coord,T_data>::get_dist_to_block(T_coord *coord,T_coord *min_coord,T_coord *max_coord)
    // {
    //     T_coord dist=0;
    //     for (int d=0;d<dim;d++)
    //     {
    //         if (coord[d]<min_coord[d])
    //             dist+=(coord[d]-min_coord[d])*(coord[d]-min_coord[d]);
    //         else if (coord[d]>max_coord[d])
    //             dist+=(coord[d]-max_coord[d])*(coord[d]-max_coord[d]);
    //     }
    //     return dist;
    // }

    // template<typename T_coord,typename T_data>
    // void param_c<T_coord,T_data>::find_nearest(priority_queue<pair<T_coord,int>> &nearest_queue,int nearest_num,int block_id,T_coord *query_coord,bool valid_only)
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

    // template<typename T_coord,typename T_data>
    // void param_c<T_coord,T_data>::generate_pred_refs_structure()
    // {
    //     //////// 对坐标做变换 ////////
    //     Array2D<T_coord> coord_transform_mat(3,3);
    //     coord_transform_mat[0][0]=1.2716812e+00;
    //     coord_transform_mat[1][0]=2.0797528e-01;
    //     coord_transform_mat[1][1]=1.1570321e+00;
    //     coord_transform_mat[2][0]=6.3697807e-05;
    //     coord_transform_mat[2][1]=1.0230185e-04;
    //     coord_transform_mat[2][2]=1.3503247e-05;
    //     for (int i=0;i<node_num;i++)
    //     {
    //         vector<T_coord> new_coord_vec(dim,0);
    //         for (int j=0;j<dim;j++)
    //             for (int k=0;k<dim;k++)
    //                 new_coord_vec[j]+=coord_array[i][k]*coord_transform_mat[k][j];
    //         for (int j=0;j<dim;j++)
    //             coord_array[i][j]=new_coord_vec[j];
    //     }
    //     min_coord.assign(dim,numeric_limits<T_coord>::infinity());
    //     max_coord.assign(dim,-numeric_limits<T_coord>::infinity());
    //     for (int i=0;i<node_num;i++)
    //         for (int d=0;d<dim;d++)
    //         {
    //             min_coord[d]=min(min_coord[d],coord_array[i][d]);
    //             max_coord[d]=max(max_coord[d],coord_array[i][d]);
    //         }
    //     printf("Coord range: [[%f, %f], [%f, %f], [%f, %f]]\n",min_coord[0],max_coord[0],min_coord[1],max_coord[1],min_coord[2],max_coord[2]);

    //     //////// 构建kd-tree ////////
    //     printf("Building kd-tree\n");
    //     point_id_on_tree_to_original_id.resize(node_num);
    //     for (int i=0;i<node_num;i++)
    //         point_id_on_tree_to_original_id[i]=i;
    //     block_vec.push_back(block_tree_c<T_coord>());
    //     block_vec[0].parent_id=-1;
    //     block_vec[0].left_child_id=-1;
    //     block_vec[0].right_child_id=-1;
    //     block_vec[0].min_coord=min_coord;
    //     block_vec[0].max_coord=max_coord;
    //     block_vec[0].index_begin=0;
    //     block_vec[0].index_end=node_num;
    //     split_block(0);
    //     id_map2.resize(node_num);
    //     generate_id_map2();
        
    //     //////// 重新排列坐标 ////////
    //     coord_array_backup.resize(node_num,dim);
    //     coord_array_backup.data=coord_array.data;
    //     for (int i=0;i<node_num;i++)
    //     {
    //         coord_array[i][0]=coord_array_backup[point_id_on_tree_to_original_id[i]][0];
    //         coord_array[i][1]=coord_array_backup[point_id_on_tree_to_original_id[i]][1];
    //         coord_array[i][2]=coord_array_backup[point_id_on_tree_to_original_id[i]][2];
    //     }

    //     //////// 随机分配点到每一轮的预测目标 ////////
    //     printf("Arranging rounds of prediction\n");
    //     vector<int> shuffled_ids(node_num);
    //     iota(shuffled_ids.begin(),shuffled_ids.end(),0);
    //     mt19937 rng(0); // 固定seed，方便复现实验
    //     shuffle(shuffled_ids.begin(),shuffled_ids.end(),rng);
    //     for (int i=0;i<predict_round_num;i++)
    //     {
    //         printf("Generating round %d\n",i);
    //         round_target_ids.push_back(vector<int>());
    //         int t1=1<<(predict_round_num-i);
    //         int t2=1<<(predict_round_num-i-1);
    //         if (i==0)
    //             round_target_ids[i].assign(shuffled_ids.begin(),shuffled_ids.begin()+(node_num+t2-1)/t2);
    //         else    
    //             round_target_ids[i].assign(shuffled_ids.begin()+(node_num+t1-1)/t1,shuffled_ids.begin()+(node_num+t2-1)/t2);
    //     }

    //     //////// 对每一轮内的预测目标排序 ////////
    //     for (int i=0;i<predict_round_num;i++)
    //     {
    //         printf("Sorting round %d\n",i);
    //         clear_all_valid_num();
    //         for (auto id:round_target_ids[i])
    //             block_vec[id_map2[id]].valid_num=1;
    //         update_all_valid_num(0);
    //         vector<float> query_coord(3);
    //         for (int d=0;d<3;d++)
    //             query_coord[d]=(min_coord[d]+max_coord[d])/2;
    //         priority_queue<pair<float,int>> nearest_queue;
    //         find_nearest(nearest_queue,1,0,query_coord.data(),true);
    //         round_target_ids[i][0]=nearest_queue.top().second;
    //         update_valid_num_to_root(id_map2[round_target_ids[i][0]],-1);
    //         for (int j=1;(size_t)j<round_target_ids[i].size();j++)
    //         {
    //             nearest_queue=priority_queue<pair<float,int>>();
    //             find_nearest(nearest_queue,1,0,coord_array[round_target_ids[i][j-1]],true);
    //             round_target_ids[i][j]=nearest_queue.top().second;
    //             update_valid_num_to_root(id_map2[round_target_ids[i][j]],-1);
    //         }
    //     }

    //     //////// 生成每个预测目标可以利用的参考点 ////////
    //     printf("Generating pred_refs_vec\n");
    //     clear_all_valid_num();
    //     for (int i=0;i<predict_round_num;i++)
    //     {
    //         priority_queue<pair<float,int>> nearest_queue;
    //         for (auto pred_id:round_target_ids[i])
    //         {
    //             nearest_queue=priority_queue<pair<float,int>>();
    //             find_nearest(nearest_queue,max_ref_num,0,coord_array[pred_id],true);
    //             pred_refs_vec.push_back({point_id_on_tree_to_original_id[pred_id],vector<int>()});
    //             while (!nearest_queue.empty())
    //             {
    //                 int ref_id=nearest_queue.top().second;
    //                 nearest_queue.pop();
    //                 pred_refs_vec.back().second.push_back(point_id_on_tree_to_original_id[ref_id]);
    //             }
    //             reverse(pred_refs_vec.back().second.begin(),pred_refs_vec.back().second.end());
    //             update_valid_num_to_root(id_map2[pred_id],1);
    //         }
    //     }

    //     //////// 保存pred_refs_vec到文件 ////////
    //     FILE *f_pred_refs=fopen((coord_path+".pred_refs").c_str(),"w");
    //     for (auto &p:pred_refs_vec)
    //     {
    //         fprintf(f_pred_refs,"%d",p.first);
    //         for (auto &id:p.second)
    //             fprintf(f_pred_refs," %d",id);
    //         fprintf(f_pred_refs,"\n");
    //     }
    //     fclose(f_pred_refs);

    //     //////// 还原coord_array ////////
    //     coord_array.data=coord_array_backup.data;
    // }
}