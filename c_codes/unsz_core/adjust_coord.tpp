// 目前的方案是先根据坐标建一个树，然后读取一部分文件，对每个文件创建一个树的副本，记录其中的数据点是否有效。
// 然后在每个文件中随机采样一些有效的数据点，并选择它们最近的actual_ref_num个点。
// 统计所有点对的数值差dv和坐标差dc，计算出一个变换矩阵B，使得dv≈B*dc。
// 还要保证B是一个下三角矩阵，特征值全部都是正数，并且平均特征值为1。

// 使用公共树的原因是不应该在同一轮内反复建树
// 或许可以考虑先采样数据再建树，因为建树代价太大了
// 可能不应该根据树上的点的顺序重拍坐标、数据和异常点标记，因为每一轮后续的计算量并没有显著超过重拍的计算量

#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::adjust_coord()
    {
        // 为所有time step寻找一个公共的变换矩阵
        if (!cross_time_compress)
        {
            int sample_file_num=1;
            int sample_point_num=1e5;
            vector<string> sample_file_vec;
            Array2D<T_data> sampled_data(sample_file_num,point_num);//sampled files中的所有数据
            Array2D<char> is_abnormal_vec(sample_file_num,point_num);//sampled files中的所有数据是否异常
            for (int i=0;i<sample_file_num;i++)
            {
                int index=((i+0.5)*(original_data_path_vec.size()-1))/sample_file_num;
                FILE *f_data=fopen(original_data_path_vec[index].c_str(),"rb");
                fread(sampled_data[i],sizeof(T_data),point_num,f_data);
                fclose(f_data);
                for (int j=0;j<point_num;j++)
                    is_abnormal_vec[i][j]=is_abnormal(sampled_data[i][j]);
            }
            auto start_time=chrono::steady_clock::now();
            for (int it=0;it<100;it++)
            {
                printf("Iteration %d\n",it);
                vector<block_tree_c<T_coord>> tree_vec;
                tree_vec.push_back(block_tree_c<T_coord>(coord_array,min_coord,max_coord));
                for (int i=1;i<sample_file_num;i++)
                    tree_vec.push_back(tree_vec[0]);
                int sampled_point_num=0;
                vector<vector<int>> sampled_point_id_vec(sample_file_num);//每个sampled file中被采样的点的id
                mt19937 rng(0);
                for(int file_id=0;file_id<sample_file_num;file_id++)
                {
                    auto &tree=tree_vec[file_id];
                    //////// 抽取点对中的第一个点 ////////
                    sampled_point_id_vec[file_id].reserve(point_num);
                    for (int original_id=0;original_id<point_num;original_id++)
                        if (!is_abnormal_vec[file_id][original_id])
                            sampled_point_id_vec[file_id].push_back(original_id);
                    if (sampled_point_id_vec[file_id].size()>(sample_point_num-sampled_point_num)/(sample_file_num-file_id))
                    {
                        int target_sample_num=(sample_point_num-sampled_point_num)/(sample_file_num-file_id);
                        shuffle(sampled_point_id_vec[file_id].begin(),sampled_point_id_vec[file_id].end(),rng);
                        sampled_point_id_vec[file_id].resize(target_sample_num);
                    }
                    sampled_point_num+=sampled_point_id_vec[file_id].size();
                    //////// 在树上标记有效点 ////////
                    for (int original_id=0;original_id<point_num;original_id++)
                        tree.block_vec[tree.original_id_to_tree_id[original_id]].valid_num=1-is_abnormal_vec[file_id][original_id];
                    tree.update_all_valid_num(0);
                    //////// 选择点对的第二个点 ////////
                    for (auto original_id:sampled_point_id_vec[file_id])
                    {
                        printf("point %d\n",original_id);
                        priority_queue<pair<T_coord,int>> nearest_queue;
                        tree.find_original_nearest(nearest_queue,coord_array,actual_ref_num+1,0,coord_array[original_id],true);
                    }
                }
            }
            auto end_time=chrono::steady_clock::now();
            auto duration=chrono::duration<double>(end_time-start_time).count();
            printf("Time for building kd-tree: %f s\n",duration);
            exit(0);
        }
    }
}