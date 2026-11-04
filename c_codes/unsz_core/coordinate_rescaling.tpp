#pragma once

// 坐标缩放：为每个坐标轴学习一个缩放系数s_d，把坐标x换成x⊙s后再建kd-tree、生成DAG和RBF权重，数据值本身不变。
// 目标是让缩放后的距离与数值差成正比，即|Δv|≈‖Δx⊙s‖，这样沿不同坐标轴“近”的含义在数值上一致。
// 是否启用由点云形状决定，形状用flatness=sqrt(λmax/λmin)衡量，λ是坐标协方差矩阵的特征值。
// flatness与论文中的定义相同：等于1时各方向同样分散，越大点云越扁。
namespace unsz
{
    // 自动模式的经验阈值，由多个数据集上显式off/on的率失真和耗时对比确定（见README 2.3节）：
    // 原始flatness>=20：点云极扁，一定启用。原始flatness<=4/3：点云接近各向同性，不训练直接关闭。
    // 其余情况先训练，缩放后flatness至多变为原来的1/1.21才启用，否则收益抵不上单独生成DAG的成本。
    inline constexpr double coordinate_rescaling_original_flatness_enable_threshold=20;
    inline constexpr double coordinate_rescaling_original_flatness_reject_threshold=4.0/3;
    inline constexpr double coordinate_rescaling_relative_flatness_enable_threshold=1.21;
    inline constexpr double coordinate_rescaling_difficult_group_loss_power=3;//各组损失取3次方再求和，使拟合较差的组不会被其他组平均掉
    inline constexpr uint32_t coordinate_rescaling_sample_file_num=3;//最多用3个均匀分布的slice训练
    inline constexpr uint32_t coordinate_rescaling_second_sample_num=3000;//所有训练slice合计最多3000个目标点

    // 用Jacobi旋转求对称矩阵的最小和最大特征值：每次选绝对值最大的非对角元，旋转把它消为0，直到非对角元可以忽略。
    inline pair<double,double> coordinate_covariance_eigenvalue_range(vector<double> covariance,uint32_t dim)
    {
        if(dim==0||covariance.size()!=(uint64_t)dim*dim)
            throw runtime_error("坐标协方差矩阵维数不正确");
        for(uint32_t iteration=0;iteration<64*dim*dim;iteration++)
        {
            uint32_t row=0,col=0;
            double largest_off_diagonal=0,largest_diagonal=0;
            for(uint32_t i=0;i<dim;i++)
            {
                largest_diagonal=max(largest_diagonal,abs(covariance[(uint64_t)i*dim+i]));
                for(uint32_t j=i+1;j<dim;j++)
                {
                    double value=abs(covariance[(uint64_t)i*dim+j]);
                    if(value>largest_off_diagonal)
                    {
                        largest_off_diagonal=value;
                        row=i;
                        col=j;
                    }
                }
            }
            if(largest_off_diagonal<=numeric_limits<double>::epsilon()*max(1.0,largest_diagonal)*16)
                break;
            double row_value=covariance[(uint64_t)row*dim+row],col_value=covariance[(uint64_t)col*dim+col],cross_value=covariance[(uint64_t)row*dim+col];
            double angle=0.5*atan2(2*cross_value,col_value-row_value);
            double cosine=cos(angle),sine=sin(angle);
            for(uint32_t i=0;i<dim;i++)
            {
                if(i==row||i==col)
                    continue;
                double row_cross=covariance[(uint64_t)i*dim+row],col_cross=covariance[(uint64_t)i*dim+col];
                covariance[(uint64_t)i*dim+row]=covariance[(uint64_t)row*dim+i]=cosine*row_cross-sine*col_cross;
                covariance[(uint64_t)i*dim+col]=covariance[(uint64_t)col*dim+i]=sine*row_cross+cosine*col_cross;
            }
            covariance[(uint64_t)row*dim+row]=cosine*cosine*row_value-2*sine*cosine*cross_value+sine*sine*col_value;
            covariance[(uint64_t)col*dim+col]=sine*sine*row_value+2*sine*cosine*cross_value+cosine*cosine*col_value;
            covariance[(uint64_t)row*dim+col]=covariance[(uint64_t)col*dim+row]=0;
        }
        double minimum=covariance[0],maximum=covariance[0];
        for(uint32_t d=1;d<dim;d++)
        {
            minimum=min(minimum,covariance[(uint64_t)d*dim+d]);
            maximum=max(maximum,covariance[(uint64_t)d*dim+d]);
        }
        return {minimum,maximum};
    }

    // 用Welford在线算法一次遍历求坐标协方差矩阵（未除以点数，不影响特征值之比）。
    template<typename T_coord>
    vector<double> calculate_coordinate_covariance(const Array2D<T_coord>&coordinates,uint32_t dim)
    {
        vector<double> mean(dim,0),delta(dim),covariance((uint64_t)dim*dim,0);
        for(uint32_t point_id=0;point_id<coordinates.n0;point_id++)
        {
            double count=point_id+1;
            for(uint32_t d=0;d<dim;d++)
            {
                delta[d]=(double)coordinates.data[(uint64_t)point_id*dim+d]-mean[d];
                mean[d]+=delta[d]/count;
            }
            for(uint32_t row=0;row<dim;row++)
                for(uint32_t col=0;col<dim;col++)
                    covariance[(uint64_t)row*dim+col]+=delta[row]*((double)coordinates.data[(uint64_t)point_id*dim+col]-mean[col]);
        }
        for(uint32_t row=0;row<dim;row++)
            for(uint32_t col=row+1;col<dim;col++)
            {
                double symmetric_value=(covariance[(uint64_t)row*dim+col]+covariance[(uint64_t)col*dim+row])/2;
                covariance[(uint64_t)row*dim+col]=covariance[(uint64_t)col*dim+row]=symmetric_value;
            }
        return covariance;
    }

    // 返回sqrt(λmax/λmin)，λmin不为正时返回无穷大。给出diagonal时计算缩放后坐标的flatness，缩放后的协方差为D·C·D，D=diag(s)。
    inline double calculate_coordinate_flatness(const vector<double>&covariance,uint32_t dim,const vector<double>*diagonal=nullptr)
    {
        vector<double> rescaled_covariance=covariance;
        if(diagonal)
        {
            if(diagonal->size()!=dim)
                throw runtime_error("坐标缩放对角元素数量与坐标维数不一致");
            for(uint32_t row=0;row<dim;row++)
                for(uint32_t col=0;col<dim;col++)
                    rescaled_covariance[(uint64_t)row*dim+col]*=(*diagonal)[row]*(*diagonal)[col];
        }
        auto [minimum,maximum]=coordinate_covariance_eigenvalue_range(move(rescaled_covariance),dim);
        if(!(maximum>0)||!isfinite(minimum)||!isfinite(maximum))
            return 1;
        if(!(minimum>0))
            return numeric_limits<double>::infinity();
        return sqrt(maximum/minimum);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::coordinate_rescaling_value_is_trainable(T_data value) const
    {
        if(!isfinite(value))
            return false;
        if(label_rule_vec.empty())
            return label_config_vec.front().codec!=label_codec_e::constant;
        const label_rule_c&rule=label_rule_vec.front();
        bool matches=false;
        switch(rule.operator_type)
        {
            case label_rule_operator_e::less:matches=value<rule.value;break;
            case label_rule_operator_e::less_equal:matches=value<=rule.value;break;
            case label_rule_operator_e::greater:matches=value>rule.value;break;
            case label_rule_operator_e::greater_equal:matches=value>=rule.value;break;
            case label_rule_operator_e::equal:matches=value==rule.value;break;
            case label_rule_operator_e::not_equal:matches=value!=rule.value;break;
        }
        uint32_t label_id=matches?0:1;
        return label_config_vec[label_id].codec!=label_codec_e::constant;
    }

    // 训练点对的几何部分只依赖坐标，因此同一组坐标的多个field共用。
    // 先用固定种子随机保留30%的点建kd-tree，再在其中给每个训练slice抽目标点，每个目标取10个最近邻组成训练点对。
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_coordinate_rescaling_sample_geometry(uint32_t sample_file_num)
    {
        if(coordinate_rescaling_sample_target_offsets.size()==sample_file_num+1)
            return;
        uint32_t first_sample_num=max<uint32_t>(1,(uint32_t)ceil(point_num*coordinate_rescaling_first_sample_ratio));
        vector<int>first_sample_ids(point_num);
        iota(first_sample_ids.begin(),first_sample_ids.end(),0);
        mt19937 random_generator(0);
        shuffle(first_sample_ids.begin(),first_sample_ids.end(),random_generator);
        first_sample_ids.resize(first_sample_num);
        sort(first_sample_ids.begin(),first_sample_ids.end());
        vector<T_coord>active_min_coord(dim,numeric_limits<T_coord>::infinity()),active_max_coord(dim,-numeric_limits<T_coord>::infinity());
        for(int point_id:first_sample_ids)
            for(uint32_t d=0;d<dim;d++)
            {
                active_min_coord[d]=min(active_min_coord[d],coord_array[point_id][d]);
                active_max_coord[d]=max(active_max_coord[d],coord_array[point_id][d]);
            }
        block_tree_c<T_coord>tree(coord_array,active_min_coord,active_max_coord,first_sample_ids,kd_tree_leaf_size);
        coordinate_rescaling_sample_target_offsets.assign(sample_file_num+1,0);
        coordinate_rescaling_sample_target_ids.clear();
        coordinate_rescaling_sample_reference_ids.clear();
        uint32_t remaining_target_num=coordinate_rescaling_second_sample_num;
        for(uint32_t sample_file_id=0;sample_file_id<sample_file_num;sample_file_id++)
        {
            uint32_t remaining_file_num=sample_file_num-sample_file_id;
            uint32_t target_num=min<uint32_t>(first_sample_ids.size(),(remaining_target_num+remaining_file_num-1)/remaining_file_num);
            vector<int>target_ids=first_sample_ids;
            shuffle(target_ids.begin(),target_ids.end(),random_generator);
            target_ids.resize(target_num);
            remaining_target_num-=target_num;
            coordinate_rescaling_sample_target_ids.insert(coordinate_rescaling_sample_target_ids.end(),target_ids.begin(),target_ids.end());
            coordinate_rescaling_sample_target_offsets[sample_file_id+1]=(uint32_t)coordinate_rescaling_sample_target_ids.size();
            for(int target_id:target_ids)
            {
                nearest_queue_c<T_coord>nearest_queue;
                tree.find_original_nearest(nearest_queue,coord_array,rbf_reference_num+1,0,coord_array[target_id]);
                uint32_t geometric_ref_num=0;
                while(!nearest_queue.empty())
                {
                    int reference_id=nearest_queue.top().second;
                    nearest_queue.pop();
                    if(reference_id==target_id)
                        continue;
                    coordinate_rescaling_sample_reference_ids.push_back(reference_id);
                    geometric_ref_num++;
                }
                if(geometric_ref_num!=rbf_reference_num)
                    throw runtime_error("坐标缩放目标点没有找到足够的非自身几何参考点");
            }
        }
        printf("coordinate rescaling shared sample geometry: first_sample_points=%u/%u,targets=%llu,references=%llu\n",first_sample_num,point_num,(unsigned long long)coordinate_rescaling_sample_target_ids.size(),(unsigned long long)coordinate_rescaling_sample_reference_ids.size());
    }

    // 读取训练slice的数值，把每个点对展开成坐标差Δx和数值差Δv，跳过属于常量标签（如异常值）的点。
    // REL任务中每个训练slice是一组，Δv除以该slice可预测标签的值域，避免数值幅度大的slice主导训练。ABS任务不分组、不归一化。
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_coordinate_rescaling_training_data()
    {
        if(coordinate_rescaling_mode=="off")
            return;
        coordinate_rescaling_training_input_time=0;
        struct training_file_c{string path;uint32_t slice_index;};
        vector<training_file_c>training_files;
        uint32_t sample_file_num=min<uint32_t>(coordinate_rescaling_sample_file_num,configured_slice_vec.size());
        for(uint32_t sample_file_id=0;sample_file_id<sample_file_num;sample_file_id++)
        {
            double position=(sample_file_id+0.5)*(configured_slice_vec.size()-1)/sample_file_num;
            uint32_t slice_id=(uint32_t)llround(position);
            uint32_t slice=configured_slice_vec[slice_id];
            training_files.push_back({data_dir+field_name+".dat."+to_string(slice),slice_id});
        }
        if(training_files.empty())
            throw runtime_error("坐标缩放缺少训练文件");
        bool has_relative_prediction=false,has_absolute_prediction=false;
        for(const label_config_c&config:label_config_vec)
            if(config.codec!=label_codec_e::constant)
            {
                has_relative_prediction|=config.error_mode!=label_error_mode_e::absolute;
                has_absolute_prediction|=config.error_mode==label_error_mode_e::absolute;
            }
        if(has_relative_prediction&&has_absolute_prediction)
            throw runtime_error("坐标缩放训练暂不支持预测标签混用REL和ABS误差模式");
        auto sample_geometry_start_time=chrono::steady_clock::now();
        prepare_coordinate_rescaling_sample_geometry((uint32_t)training_files.size());
        double sample_geometry_time=seconds_since(sample_geometry_start_time);
        printf("coordinate rescaling first sample ratio=%.6f,planned second targets=%llu\n",coordinate_rescaling_first_sample_ratio,(unsigned long long)coordinate_rescaling_sample_target_ids.size());
        vector<double>&coordinate_differences=coordinate_rescaling_training_coordinate_differences;
        vector<double>&value_differences=coordinate_rescaling_training_value_differences;
        coordinate_differences.clear();
        value_differences.clear();
        coordinate_rescaling_training_pair_group_ids.clear();
        coordinate_rescaling_training_group_num=0;
        coordinate_differences.reserve((uint64_t)coordinate_rescaling_second_sample_num*rbf_reference_num*dim);
        value_differences.reserve((uint64_t)coordinate_rescaling_second_sample_num*rbf_reference_num);
        bool normalize_by_sample_file=has_relative_prediction;
        uint32_t normalization_group_num=normalize_by_sample_file?(uint32_t)training_files.size():0;
        vector<double> value_scale_by_group(max<uint32_t>(normalization_group_num,1),1);
        vector<T_data> minimum_by_group,maximum_by_group;
        vector<uint32_t> pair_group_ids;
        if(normalization_group_num>0)
        {
            minimum_by_group.assign(normalization_group_num,numeric_limits<T_data>::infinity());
            maximum_by_group.assign(normalization_group_num,-numeric_limits<T_data>::infinity());
            pair_group_ids.reserve((uint64_t)coordinate_rescaling_second_sample_num*rbf_reference_num);
            if(compression_file_label_min_data_vec.size()!=slice_vec.size()||compression_file_label_max_data_vec.size()!=slice_vec.size())
                throw runtime_error("坐标缩放缺少当前field的预计算值域");
            for(uint32_t sample_file_id=0;sample_file_id<training_files.size();sample_file_id++)
            {
                const training_file_c&training_file=training_files[sample_file_id];
                for(uint32_t label_id=0;label_id<label_config_vec.size();label_id++)
                    if(label_config_vec[label_id].codec!=label_codec_e::constant)
                    {
                        minimum_by_group[sample_file_id]=min(minimum_by_group[sample_file_id],compression_file_label_min_data_vec[training_file.slice_index][label_id]);
                        maximum_by_group[sample_file_id]=max(maximum_by_group[sample_file_id],compression_file_label_max_data_vec[training_file.slice_index][label_id]);
                    }
            }
        }
        vector<T_data> data(point_num);
        uint32_t sampled_target_num=0,retained_target_num=0,complete_target_num=0;
        auto pair_assembly_start_time=chrono::steady_clock::now();
        for(uint32_t sample_file_id=0;sample_file_id<training_files.size();sample_file_id++)
        {
            auto input_start_time=chrono::steady_clock::now();
            read_binary_file_exact(training_files[sample_file_id].path,data.data(),point_num);
            coordinate_rescaling_training_input_time+=seconds_since(input_start_time);
            uint32_t normalization_group_id=sample_file_id;
            uint32_t target_begin=coordinate_rescaling_sample_target_offsets[sample_file_id],target_end=coordinate_rescaling_sample_target_offsets[sample_file_id+1];
            for(uint32_t target_index=target_begin;target_index<target_end;target_index++)
            {
                int target_id=coordinate_rescaling_sample_target_ids[target_index];
                sampled_target_num++;
                if(!coordinate_rescaling_value_is_trainable(data[target_id]))
                    continue;
                uint32_t retained_ref_num=0;
                for(uint32_t ref_offset=0;ref_offset<rbf_reference_num;ref_offset++)
                {
                    int reference_id=coordinate_rescaling_sample_reference_ids[(uint64_t)target_index*rbf_reference_num+ref_offset];
                    if(!coordinate_rescaling_value_is_trainable(data[reference_id]))
                        continue;
                    for(uint32_t d=0;d<dim;d++)
                        coordinate_differences.push_back((double)coord_array[target_id][d]-(double)coord_array[reference_id][d]);
                    value_differences.push_back((double)data[target_id]-(double)data[reference_id]);
                    if(normalization_group_num>0)
                        pair_group_ids.push_back(normalization_group_id);
                    retained_ref_num++;
                }
                retained_target_num+=retained_ref_num>0;
                complete_target_num+=retained_ref_num==rbf_reference_num;
            }
        }
        if(value_differences.empty())
            throw runtime_error("坐标缩放过滤后没有保留训练点对");
        if(normalization_group_num>0)
        {
            for(uint32_t group_id=0;group_id<normalization_group_num;group_id++)
                if(maximum_by_group[group_id]>minimum_by_group[group_id])
                    value_scale_by_group[group_id]=1/((double)maximum_by_group[group_id]-(double)minimum_by_group[group_id]);
            for(uint64_t pair_id=0;pair_id<value_differences.size();pair_id++)
                value_differences[pair_id]*=value_scale_by_group[pair_group_ids[pair_id]];
        }
        if(normalize_by_sample_file)
        {
            coordinate_rescaling_training_pair_group_ids=pair_group_ids;
            coordinate_rescaling_training_group_num=normalization_group_num;
        }
        double minimum_value_scale=*min_element(value_scale_by_group.begin(),value_scale_by_group.end()),maximum_value_scale=*max_element(value_scale_by_group.begin(),value_scale_by_group.end());
        const char*normalization_name=normalize_by_sample_file?"per-field-slice predicted-value range":"none";
        printf("coordinate rescaling training normalization=%s,sample_files=%llu,value_scale_range=[%.12g,%.12g]\n",normalization_name,(unsigned long long)training_files.size(),minimum_value_scale,maximum_value_scale);
        printf("coordinate rescaling sampled targets=%u,retained targets=%u,complete targets=%u,pairs=%llu\n",sampled_target_num,retained_target_num,complete_target_num,(unsigned long long)value_differences.size());
        double pair_assembly_time=seconds_since(pair_assembly_start_time)-coordinate_rescaling_training_input_time;
        printf("coordinate rescaling training-data breakdown: shared_geometry=%.6f s,pair_assembly=%.6f s,input_excluded=%.6f s\n",sample_geometry_time,pair_assembly_time,coordinate_rescaling_training_input_time);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::restore_original_coordinates()
    {
        if(coordinates_are_original)
            return;
        coord_array=original_coord_array;
        min_coord=original_min_coord;
        max_coord=original_max_coord;
        coordinates_are_original=true;
    }

    // 用列主元高斯消元解dim×dim的正规方程，主元过小或结果非有限时返回false。
    inline bool solve_coordinate_rescaling_normal_equation(vector<double> matrix,vector<double> right_hand_side,vector<double>&solution)
    {
        uint32_t n=(uint32_t)right_hand_side.size();
        solution=move(right_hand_side);
        for(uint32_t pivot=0;pivot<n;pivot++)
        {
            uint32_t best_row=pivot;
            for(uint32_t row=pivot+1;row<n;row++)
                if(abs(matrix[(uint64_t)row*n+pivot])>abs(matrix[(uint64_t)best_row*n+pivot]))
                    best_row=row;
            double pivot_value=matrix[(uint64_t)best_row*n+pivot];
            if(!isfinite(pivot_value)||abs(pivot_value)<=numeric_limits<double>::epsilon())
                return false;
            if(best_row!=pivot)
            {
                for(uint32_t col=pivot;col<n;col++)
                    swap(matrix[(uint64_t)pivot*n+col],matrix[(uint64_t)best_row*n+col]);
                swap(solution[pivot],solution[best_row]);
            }
            for(uint32_t row=pivot+1;row<n;row++)
            {
                double ratio=matrix[(uint64_t)row*n+pivot]/matrix[(uint64_t)pivot*n+pivot];
                for(uint32_t col=pivot+1;col<n;col++)
                    matrix[(uint64_t)row*n+col]-=ratio*matrix[(uint64_t)pivot*n+col];
                solution[row]-=ratio*solution[pivot];
            }
        }
        for(int row=(int)n-1;row>=0;row--)
        {
            for(uint32_t col=row+1;col<n;col++)
                solution[row]-=matrix[(uint64_t)row*n+col]*solution[col];
            solution[row]/=matrix[(uint64_t)row*n+row];
            if(!isfinite(solution[row]))
                return false;
        }
        return true;
    }

    // 残差r_i=|Δv_i|-‖Δx_i⊙s‖。不分组时损失为Σr_i²，分组时为Σ_g(m_g)^p，m_g是第g组的平均r_i²。
    inline double coordinate_rescaling_squared_loss(const vector<double>&coordinate_differences,const vector<double>&value_differences,const vector<double>&diagonal,const vector<uint32_t>&pair_group_ids,uint32_t group_num,double group_loss_power)
    {
        uint32_t dim=(uint32_t)diagonal.size();
        vector<double>group_squared_loss(group_num,0);
        vector<uint32_t>group_pair_num(group_num,0);
        double squared_loss=0;
        for(uint64_t pair_id=0;pair_id<value_differences.size();pair_id++)
        {
            double squared_distance=0;
            for(uint32_t d=0;d<dim;d++)
            {
                double rescaled_difference=coordinate_differences[pair_id*dim+d]*diagonal[d];
                squared_distance+=rescaled_difference*rescaled_difference;
            }
            double residual=abs(value_differences[pair_id])-sqrt(squared_distance);
            if(group_num==0)
                squared_loss+=residual*residual;
            else
            {
                uint32_t group_id=pair_group_ids[pair_id];
                group_squared_loss[group_id]+=residual*residual;
                group_pair_num[group_id]++;
            }
        }
        if(group_num>0)
            for(uint32_t group_id=0;group_id<group_num;group_id++)
                if(group_pair_num[group_id]>0)
                    squared_loss+=pow(group_squared_loss[group_id]/group_pair_num[group_id],group_loss_power);
        return squared_loss;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::train_coordinate_rescaling()
    {
        vector<double>&coordinate_differences=coordinate_rescaling_training_coordinate_differences;
        vector<double>&value_differences=coordinate_rescaling_training_value_differences;
        vector<uint32_t>&pair_group_ids=coordinate_rescaling_training_pair_group_ids;
        uint32_t group_num=coordinate_rescaling_training_group_num;
        if(value_differences.empty())
            throw runtime_error("坐标缩放缺少训练数据");
        if(group_num>0&&pair_group_ids.size()!=value_differences.size())
            throw runtime_error("坐标缩放训练点对分组数量错误");
        // 用Gauss-Newton法最小化coordinate_rescaling_squared_loss，从s=1出发。
        // 分组损失Σ_g(m_g)^p对s求导后，第g组每个点对的权重为m_g^(p-1)/N_g（常数因子不影响步长），每轮按当前s更新权重。
        // 正规方程为(Σw·J·Jᵀ)Δs=-Σw·J·r，其中J_d=∂r/∂s_d=-Δx_d²·s_d/‖Δx⊙s‖。
        // 每一步做回溯线搜索：步长从1开始减半，直到损失下降，最小到1/256。总共最多计算100次损失。
        // 收敛后取绝对值，并归一化使Σs_d²=dim，因为只有各轴之间的相对比例影响最近邻关系。
        auto fit_diagonal=[&](double group_loss_power)
        {
            vector<double>diagonal(dim,1);
            uint32_t function_evaluation_num=0,accepted_step_num=0;
            for(uint32_t iteration=0;iteration<100&&function_evaluation_num<100;iteration++)
            {
                vector<double>normal_matrix((uint64_t)dim*dim,0),normal_gradient(dim,0);
                double squared_loss=0;
                vector<double>residuals(group_num>0?value_differences.size():0),group_squared_loss(group_num,0),group_pair_weight(group_num,1);
                vector<uint32_t>group_pair_num(group_num,0);
                if(group_num>0)
                {
                    for(uint64_t pair_id=0;pair_id<value_differences.size();pair_id++)
                    {
                        double squared_distance=0;
                        for(uint32_t d=0;d<dim;d++)
                        {
                            double rescaled_difference=coordinate_differences[pair_id*dim+d]*diagonal[d];
                            squared_distance+=rescaled_difference*rescaled_difference;
                        }
                        double residual=abs(value_differences[pair_id])-sqrt(squared_distance);
                        residuals[pair_id]=residual;
                        uint32_t group_id=pair_group_ids[pair_id];
                        group_squared_loss[group_id]+=residual*residual;
                        group_pair_num[group_id]++;
                    }
                    for(uint32_t group_id=0;group_id<group_num;group_id++)
                        if(group_pair_num[group_id]>0)
                        {
                            double group_mean_squared_loss=group_squared_loss[group_id]/group_pair_num[group_id];
                            squared_loss+=pow(group_mean_squared_loss,group_loss_power);
                            group_pair_weight[group_id]=pow(max(group_mean_squared_loss,numeric_limits<double>::min()),group_loss_power-1)/group_pair_num[group_id];
                        }
                }
                for(uint64_t pair_id=0;pair_id<value_differences.size();pair_id++)
                {
                    double squared_distance=0;
                    for(uint32_t d=0;d<dim;d++)
                    {
                        double rescaled_difference=coordinate_differences[pair_id*dim+d]*diagonal[d];
                        squared_distance+=rescaled_difference*rescaled_difference;
                    }
                    double distance=sqrt(squared_distance);
                    double residual=group_num>0?residuals[pair_id]:abs(value_differences[pair_id])-distance;
                    double pair_weight=group_num>0?group_pair_weight[pair_group_ids[pair_id]]:1;
                    if(group_num==0)
                        squared_loss+=residual*residual;
                    if(distance==0)
                        continue;
                    vector<double>jacobian(dim);
                    for(uint32_t d=0;d<dim;d++)
                    {
                        double difference=coordinate_differences[pair_id*dim+d];
                        jacobian[d]=-difference*difference*diagonal[d]/distance;
                        normal_gradient[d]+=pair_weight*jacobian[d]*residual;
                    }
                    for(uint32_t row=0;row<dim;row++)
                        for(uint32_t col=0;col<dim;col++)
                            normal_matrix[(uint64_t)row*dim+col]+=pair_weight*jacobian[row]*jacobian[col];
                }
                function_evaluation_num++;
                for(double&value:normal_gradient)
                    value=-value;
                vector<double>step;
                if(!solve_coordinate_rescaling_normal_equation(move(normal_matrix),move(normal_gradient),step))
                    break;
                double step_norm=0,parameter_norm=0;
                for(uint32_t d=0;d<dim;d++)
                {
                    step_norm+=step[d]*step[d];
                    parameter_norm+=diagonal[d]*diagonal[d];
                }
                if(sqrt(step_norm)<=1e-10*(sqrt(parameter_norm)+1e-10))
                    break;
                bool accepted=false;
                double step_ratio=1;
                while(function_evaluation_num<100&&step_ratio>=1.0/256)
                {
                    vector<double>candidate=diagonal;
                    for(uint32_t d=0;d<dim;d++)
                        candidate[d]+=step_ratio*step[d];
                    double candidate_loss=coordinate_rescaling_squared_loss(coordinate_differences,value_differences,candidate,pair_group_ids,group_num,group_loss_power);
                    function_evaluation_num++;
                    if(candidate_loss<squared_loss)
                    {
                        double relative_reduction=(squared_loss-candidate_loss)/max(squared_loss,numeric_limits<double>::min());
                        diagonal=move(candidate);
                        accepted_step_num++;
                        accepted=true;
                        if(relative_reduction<=1e-10)
                            iteration=100;
                        break;
                    }
                    step_ratio/=2;
                }
                if(!accepted)
                    break;
            }
            printf("coordinate rescaling optimization group loss power=%.6g,accepted steps=%u,function evaluations=%u\n",group_loss_power,accepted_step_num,function_evaluation_num);
            for(double&value:diagonal)
                value=abs(value);
            double diagonal_squared_norm=inner_product(diagonal.begin(),diagonal.end(),diagonal.begin(),0.0);
            if(!(diagonal_squared_norm>0)||!isfinite(diagonal_squared_norm))
                throw runtime_error("坐标缩放训练得到无效的对角矩阵");
            double normalization=sqrt(dim/diagonal_squared_norm);
            for(double&value:diagonal)
                value*=normalization;
            return diagonal;
        };
        vector<double>diagonal=fit_diagonal(coordinate_rescaling_difficult_group_loss_power);
        coordinate_rescaling_diagonal=diagonal;
        coordinate_rescaling_training_coordinate_differences.clear();
        coordinate_rescaling_training_coordinate_differences.shrink_to_fit();
        coordinate_rescaling_training_value_differences.clear();
        coordinate_rescaling_training_value_differences.shrink_to_fit();
        coordinate_rescaling_training_pair_group_ids.clear();
        coordinate_rescaling_training_pair_group_ids.shrink_to_fit();
        coordinate_rescaling_training_group_num=0;

        printf("coordinate rescaling diagonal=[");
        for(uint32_t d=0;d<dim;d++)
            printf("%s%.12g",d==0?"":",",coordinate_rescaling_diagonal[d]);
        printf("]\n");
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::apply_coordinate_rescaling()
    {
        if(!use_coordinate_rescaling)
            return;
        if(coordinate_rescaling_diagonal.size()!=dim)
            throw runtime_error("坐标缩放对角元素数量与坐标维数不一致");
        min_coord.assign(dim,numeric_limits<T_coord>::infinity());
        max_coord.assign(dim,-numeric_limits<T_coord>::infinity());
        for(uint32_t point_id=0;point_id<point_num;point_id++)
            for(uint32_t d=0;d<dim;d++)
            {
                coord_array[point_id][d]*=(T_coord)coordinate_rescaling_diagonal[d];
                min_coord[d]=min(min_coord[d],coord_array[point_id][d]);
                max_coord[d]=max(max_coord[d],coord_array[point_id][d]);
            }
        coordinates_are_original=false;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::write_coordinate_rescaling_file() const
    {
        if(coordinate_rescaling_mode=="off")
            return;
        string path=field_generated_path(".unsz.coordinate_rescaling.txt");
        ofstream file(path);
        if(!file)
            throw runtime_error("无法写入坐标缩放文件："+path);
        file<<"UNSZ_COORDINATE_RESCALING 3\n";
        file<<"dimension "<<dim<<"\n";
        file<<"enabled "<<(uint32_t)use_coordinate_rescaling<<"\n";
        file<<"diagonal";
        for(double value:coordinate_rescaling_diagonal)
            file<<" "<<setprecision(17)<<value;
        file<<"\n";
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::read_coordinate_rescaling_file()
    {
        if(coordinate_rescaling_mode=="off")
        {
            use_coordinate_rescaling=false;
            coordinate_rescaling_diagonal.assign(dim,1);
            return;
        }
        string path=field_generated_path(".unsz.coordinate_rescaling.txt");
        ifstream file(path);
        if(!file)
            throw runtime_error("找不到坐标缩放文件："+path);
        string magic,key;
        uint32_t version,file_dim,enabled;
        if(!(file>>magic>>version)||magic!="UNSZ_COORDINATE_RESCALING"||version!=3)
            throw runtime_error("坐标缩放文件格式不正确："+path);
        if(!(file>>key>>file_dim)||key!="dimension"||file_dim!=dim)
            throw runtime_error("坐标缩放文件维数不正确："+path);
        if(!(file>>key>>enabled)||key!="enabled"||enabled>1)
            throw runtime_error("坐标缩放文件enabled字段不正确："+path);
        if(!(file>>key)||key!="diagonal")
            throw runtime_error("坐标缩放文件缺少diagonal字段："+path);
        vector<double> diagonal(dim);
        for(double&value:diagonal)
            if(!(file>>value)||!isfinite(value))
                throw runtime_error("坐标缩放文件的对角元素不正确："+path);
        if(file>>key)
            throw runtime_error("坐标缩放文件末尾存在多余内容："+path);
        use_coordinate_rescaling=enabled;
        coordinate_rescaling_diagonal=move(diagonal);
    }
}
