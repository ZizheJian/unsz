// 使用公共树的原因是没法在同一轮内反复建树
// 或许可以考虑先采样数据再建树，因为建树代价太大了

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
            for (int i=0;i<sample_file_num;i++)
            {
                int index=((i+0.5)*(original_data_path_vec.size()-1))/sample_file_num;
                sample_file_vec.push_back(original_data_path_vec[index]);
            }
            Array2D<T_coord> sampled_data(sample_file_num,node_num);
            Array2D<char> is_abnormal_vec(sample_file_num,node_num);
            for (int i=0;i<sample_file_num;i++)
            {
                FILE *f_data=fopen(sample_file_vec[i].c_str(),"rb");
                fread(sampled_data[i],sizeof(T_coord),node_num,f_data);
                fclose(f_data);
                for (int j=0;j<node_num;j++)
                    is_abnormal_vec[i][j]=is_abnormal(sampled_data[i][j]);
            }
            auto start_time=chrono::steady_clock::now();
            for (int it=0;it<100;it++)
            {
                vector<block_tree_c<T_coord>> tree_vec;
                tree_vec.push_back(block_tree_c<T_coord>(coord_array,min_coord,max_coord));
                for (int i=1;i<sample_file_num;i++)
                    tree_vec.push_back(tree_vec[0]);
                int sampled_point_num=0;
                vector<vector<int>> sampled_node_id_vec(sample_file_num);
                mt19937 rng(0);
                for(int file_id=0;file_id<sample_file_num;file_id++)
                {

                    sampled_node_id_vec[file_id].reserve(node_num);
                    for (int node_id=0;node_id<node_num;node_id++)
                        if (!is_abnormal_vec[file_id][node_id])
                            sampled_node_id_vec[file_id].push_back(node_id);
                    if (sampled_node_id_vec[file_id].size()>(sample_point_num-sampled_point_num)/(sample_file_num-file_id))
                    {
                        int target_sample_num=(sample_point_num-sampled_point_num)/(sample_file_num-file_id);
                        shuffle(sampled_node_id_vec[file_id].begin(),sampled_node_id_vec[file_id].end(),rng);
                        sampled_node_id_vec[file_id].resize(target_sample_num);
                    }
                    sampled_point_num+=sampled_node_id_vec[file_id].size();
                }
            }
            auto end_time=chrono::steady_clock::now();
            auto duration=chrono::duration<double>(end_time-start_time).count();
            printf("Time for building kd-tree: %f s\n",duration);
            exit(0);
        }
    }
}