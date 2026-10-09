#pragma once

namespace unsz
{
    inline constexpr double coordinate_transform_original_flatness_enable_threshold=0.05;
    inline constexpr double coordinate_transform_original_flatness_reject_threshold=0.75;
    inline constexpr double coordinate_transform_relative_flatness_enable_threshold=1.21;
    inline constexpr double coordinate_transform_difficult_group_loss_power=3;
    inline constexpr uint32_t coordinate_transform_sample_file_num=3;
    inline constexpr uint32_t coordinate_transform_second_sample_num=3000;

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

    inline double calculate_coordinate_flatness(const vector<double>&covariance,uint32_t dim,const vector<double>*diagonal=nullptr)
    {
        vector<double> transformed_covariance=covariance;
        if(diagonal)
        {
            if(diagonal->size()!=dim)
                throw runtime_error("坐标变换对角元素数量与坐标维数不一致");
            for(uint32_t row=0;row<dim;row++)
                for(uint32_t col=0;col<dim;col++)
                    transformed_covariance[(uint64_t)row*dim+col]*=(*diagonal)[row]*(*diagonal)[col];
        }
        auto [minimum,maximum]=coordinate_covariance_eigenvalue_range(move(transformed_covariance),dim);
        if(!(maximum>0)||!isfinite(minimum)||!isfinite(maximum))
            return 1;
        return sqrt(max(0.0,minimum)/maximum);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::coordinate_transform_value_is_trainable(T_data value) const
    {
        if(!isfinite(value))
            return false;
        if(point_set_rule_vec.empty())
            return point_set_config_vec.front().codec!=point_set_codec_e::constant;
        const point_set_rule_c&rule=point_set_rule_vec.front();
        bool matches=false;
        switch(rule.operator_type)
        {
            case point_set_rule_operator_e::less:matches=value<rule.value;break;
            case point_set_rule_operator_e::less_equal:matches=value<=rule.value;break;
            case point_set_rule_operator_e::greater:matches=value>rule.value;break;
            case point_set_rule_operator_e::greater_equal:matches=value>=rule.value;break;
            case point_set_rule_operator_e::equal:matches=value==rule.value;break;
            case point_set_rule_operator_e::not_equal:matches=value!=rule.value;break;
        }
        uint32_t point_set_id=matches?0:1;
        return point_set_config_vec[point_set_id].codec!=point_set_codec_e::constant;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_coordinate_transform_sample_geometry(uint32_t sample_file_num)
    {
        if(coordinate_transform_sample_target_offsets.size()==sample_file_num+1)
            return;
        uint32_t first_sample_num=max<uint32_t>(1,(uint32_t)ceil(point_num*coordinate_transform_first_sample_ratio));
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
        coordinate_transform_sample_target_offsets.assign(sample_file_num+1,0);
        coordinate_transform_sample_target_ids.clear();
        coordinate_transform_sample_reference_ids.clear();
        uint32_t remaining_target_num=coordinate_transform_second_sample_num;
        for(uint32_t sample_file_id=0;sample_file_id<sample_file_num;sample_file_id++)
        {
            uint32_t remaining_file_num=sample_file_num-sample_file_id;
            uint32_t target_num=min<uint32_t>(first_sample_ids.size(),(remaining_target_num+remaining_file_num-1)/remaining_file_num);
            vector<int>target_ids=first_sample_ids;
            shuffle(target_ids.begin(),target_ids.end(),random_generator);
            target_ids.resize(target_num);
            remaining_target_num-=target_num;
            coordinate_transform_sample_target_ids.insert(coordinate_transform_sample_target_ids.end(),target_ids.begin(),target_ids.end());
            coordinate_transform_sample_target_offsets[sample_file_id+1]=(uint32_t)coordinate_transform_sample_target_ids.size();
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
                    coordinate_transform_sample_reference_ids.push_back(reference_id);
                    geometric_ref_num++;
                }
                if(geometric_ref_num!=rbf_reference_num)
                    throw runtime_error("坐标变换目标点没有找到足够的非自身几何参考点");
            }
        }
        printf("coordinate transform shared sample geometry: first_sample_points=%u/%u,targets=%llu,references=%llu\n",first_sample_num,point_num,(unsigned long long)coordinate_transform_sample_target_ids.size(),(unsigned long long)coordinate_transform_sample_reference_ids.size());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_coordinate_transform_training_data()
    {
        if(coordinate_transform_mode=="off")
            return;
        coordinate_transform_training_input_time=0;
        struct training_file_c{string path;uint32_t time_index;};
        vector<training_file_c>training_files;
        uint32_t sample_file_num=min<uint32_t>(coordinate_transform_sample_file_num,configured_time_step_vec.size());
        for(uint32_t sample_file_id=0;sample_file_id<sample_file_num;sample_file_id++)
        {
            double position=(sample_file_id+0.5)*(configured_time_step_vec.size()-1)/sample_file_num;
            uint32_t time_id=(uint32_t)llround(position);
            uint32_t time_step=configured_time_step_vec[time_id];
            training_files.push_back({data_dir+field_name+".dat."+to_string(time_step),time_id});
        }
        if(training_files.empty())
            throw runtime_error("坐标变换缺少训练文件");
        bool has_relative_prediction=false,has_absolute_prediction=false;
        for(const point_set_config_c&config:point_set_config_vec)
            if(config.codec!=point_set_codec_e::constant)
            {
                has_relative_prediction|=config.error_mode==point_set_error_mode_e::relative;
                has_absolute_prediction|=config.error_mode==point_set_error_mode_e::absolute;
            }
        if(has_relative_prediction&&has_absolute_prediction)
            throw runtime_error("坐标变换训练暂不支持预测点集混用REL和ABS误差模式");
        auto sample_geometry_start_time=chrono::steady_clock::now();
        prepare_coordinate_transform_sample_geometry((uint32_t)training_files.size());
        double sample_geometry_time=chrono::duration<double>(chrono::steady_clock::now()-sample_geometry_start_time).count();
        printf("coordinate transform first sample ratio=%.6f,planned second targets=%llu\n",coordinate_transform_first_sample_ratio,(unsigned long long)coordinate_transform_sample_target_ids.size());
        vector<double>&coordinate_differences=coordinate_transform_training_coordinate_differences;
        vector<double>&value_differences=coordinate_transform_training_value_differences;
        coordinate_differences.clear();
        value_differences.clear();
        coordinate_transform_training_pair_group_ids.clear();
        coordinate_transform_training_group_num=0;
        coordinate_differences.reserve((uint64_t)coordinate_transform_second_sample_num*rbf_reference_num*dim);
        value_differences.reserve((uint64_t)coordinate_transform_second_sample_num*rbf_reference_num);
        bool normalize_by_sample_file=has_relative_prediction;
        uint32_t normalization_group_num=normalize_by_sample_file?(uint32_t)training_files.size():0;
        vector<double> value_scale_by_group(max<uint32_t>(normalization_group_num,1),1);
        vector<T_data> minimum_by_group,maximum_by_group;
        vector<uint32_t> pair_group_ids;
        if(normalization_group_num>0)
        {
            minimum_by_group.assign(normalization_group_num,numeric_limits<T_data>::infinity());
            maximum_by_group.assign(normalization_group_num,-numeric_limits<T_data>::infinity());
            pair_group_ids.reserve((uint64_t)coordinate_transform_second_sample_num*rbf_reference_num);
            if(compression_file_point_set_min_data_vec.size()!=time_step_vec.size()||compression_file_point_set_max_data_vec.size()!=time_step_vec.size())
                throw runtime_error("坐标变换缺少当前field的预计算值域");
            for(uint32_t sample_file_id=0;sample_file_id<training_files.size();sample_file_id++)
            {
                const training_file_c&training_file=training_files[sample_file_id];
                for(uint32_t point_set_id=0;point_set_id<point_set_config_vec.size();point_set_id++)
                    if(point_set_config_vec[point_set_id].codec!=point_set_codec_e::constant)
                    {
                        minimum_by_group[sample_file_id]=min(minimum_by_group[sample_file_id],compression_file_point_set_min_data_vec[training_file.time_index][point_set_id]);
                        maximum_by_group[sample_file_id]=max(maximum_by_group[sample_file_id],compression_file_point_set_max_data_vec[training_file.time_index][point_set_id]);
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
            coordinate_transform_training_input_time+=chrono::duration<double>(chrono::steady_clock::now()-input_start_time).count();
            uint32_t normalization_group_id=sample_file_id;
            uint32_t target_begin=coordinate_transform_sample_target_offsets[sample_file_id],target_end=coordinate_transform_sample_target_offsets[sample_file_id+1];
            for(uint32_t target_index=target_begin;target_index<target_end;target_index++)
            {
                int target_id=coordinate_transform_sample_target_ids[target_index];
                sampled_target_num++;
                if(!coordinate_transform_value_is_trainable(data[target_id]))
                    continue;
                uint32_t retained_ref_num=0;
                for(uint32_t ref_offset=0;ref_offset<rbf_reference_num;ref_offset++)
                {
                    int reference_id=coordinate_transform_sample_reference_ids[(uint64_t)target_index*rbf_reference_num+ref_offset];
                    if(!coordinate_transform_value_is_trainable(data[reference_id]))
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
            throw runtime_error("坐标变换过滤后没有保留训练点对");
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
            coordinate_transform_training_pair_group_ids=pair_group_ids;
            coordinate_transform_training_group_num=normalization_group_num;
        }
        double minimum_value_scale=*min_element(value_scale_by_group.begin(),value_scale_by_group.end()),maximum_value_scale=*max_element(value_scale_by_group.begin(),value_scale_by_group.end());
        const char*normalization_name=normalize_by_sample_file?"per-field-time-step predicted-value range":"none";
        printf("coordinate transform training normalization=%s,sample_files=%llu,value_scale_range=[%.12g,%.12g]\n",normalization_name,(unsigned long long)training_files.size(),minimum_value_scale,maximum_value_scale);
        printf("coordinate transform sampled targets=%u,retained targets=%u,complete targets=%u,pairs=%llu\n",sampled_target_num,retained_target_num,complete_target_num,(unsigned long long)value_differences.size());
        double pair_assembly_time=chrono::duration<double>(chrono::steady_clock::now()-pair_assembly_start_time).count()-coordinate_transform_training_input_time;
        printf("coordinate transform training-data breakdown: shared_geometry=%.6f s,pair_assembly=%.6f s,input_excluded=%.6f s\n",sample_geometry_time,pair_assembly_time,coordinate_transform_training_input_time);
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

    inline bool solve_coordinate_transform_normal_equation(vector<double> matrix,vector<double> right_hand_side,vector<double>&solution)
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

    inline double coordinate_transform_squared_loss(const vector<double>&coordinate_differences,const vector<double>&value_differences,const vector<double>&diagonal,const vector<uint32_t>&pair_group_ids,uint32_t group_num,double group_loss_power)
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
                double transformed_difference=coordinate_differences[pair_id*dim+d]*diagonal[d];
                squared_distance+=transformed_difference*transformed_difference;
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
    void param_c<T_coord,T_data>::train_coordinate_transform()
    {
        vector<double>&coordinate_differences=coordinate_transform_training_coordinate_differences;
        vector<double>&value_differences=coordinate_transform_training_value_differences;
        vector<uint32_t>&pair_group_ids=coordinate_transform_training_pair_group_ids;
        uint32_t group_num=coordinate_transform_training_group_num;
        if(value_differences.empty())
            throw runtime_error("坐标变换缺少训练数据");
        if(group_num>0&&pair_group_ids.size()!=value_differences.size())
            throw runtime_error("坐标变换训练点对分组数量错误");
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
                            double transformed_difference=coordinate_differences[pair_id*dim+d]*diagonal[d];
                            squared_distance+=transformed_difference*transformed_difference;
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
                        double transformed_difference=coordinate_differences[pair_id*dim+d]*diagonal[d];
                        squared_distance+=transformed_difference*transformed_difference;
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
                if(!solve_coordinate_transform_normal_equation(move(normal_matrix),move(normal_gradient),step))
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
                    double candidate_loss=coordinate_transform_squared_loss(coordinate_differences,value_differences,candidate,pair_group_ids,group_num,group_loss_power);
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
            printf("coordinate transform optimization group loss power=%.6g,accepted steps=%u,function evaluations=%u\n",group_loss_power,accepted_step_num,function_evaluation_num);
            for(double&value:diagonal)
                value=abs(value);
            double diagonal_squared_norm=inner_product(diagonal.begin(),diagonal.end(),diagonal.begin(),0.0);
            if(!(diagonal_squared_norm>0)||!isfinite(diagonal_squared_norm))
                throw runtime_error("坐标变换训练得到无效的对角矩阵");
            double normalization=sqrt(dim/diagonal_squared_norm);
            for(double&value:diagonal)
                value*=normalization;
            return diagonal;
        };
        vector<double>diagonal=fit_diagonal(coordinate_transform_difficult_group_loss_power);
        coordinate_transform_diagonal=diagonal;
        coordinate_transform_training_coordinate_differences.clear();
        coordinate_transform_training_coordinate_differences.shrink_to_fit();
        coordinate_transform_training_value_differences.clear();
        coordinate_transform_training_value_differences.shrink_to_fit();
        coordinate_transform_training_pair_group_ids.clear();
        coordinate_transform_training_pair_group_ids.shrink_to_fit();
        coordinate_transform_training_group_num=0;

        printf("coordinate transform diagonal=[");
        for(uint32_t d=0;d<dim;d++)
            printf("%s%.12g",d==0?"":",",coordinate_transform_diagonal[d]);
        printf("]\n");
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::apply_coordinate_transform()
    {
        if(!use_coordinate_transform)
            return;
        if(coordinate_transform_diagonal.size()!=dim)
            throw runtime_error("坐标变换对角元素数量与坐标维数不一致");
        min_coord.assign(dim,numeric_limits<T_coord>::infinity());
        max_coord.assign(dim,-numeric_limits<T_coord>::infinity());
        for(uint32_t point_id=0;point_id<point_num;point_id++)
            for(uint32_t d=0;d<dim;d++)
            {
                coord_array[point_id][d]*=(T_coord)coordinate_transform_diagonal[d];
                min_coord[d]=min(min_coord[d],coord_array[point_id][d]);
                max_coord[d]=max(max_coord[d],coord_array[point_id][d]);
            }
        coordinates_are_original=false;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::write_coordinate_transform_file() const
    {
        if(coordinate_transform_mode=="off")
            return;
        string path=data_dir+field_name+".unsz.coord_transform.txt";
        ofstream file(path);
        if(!file)
            throw runtime_error("无法写入坐标变换文件："+path);
        file<<"UNSZ_COORDINATE_TRANSFORM 3\n";
        file<<"dimension "<<dim<<"\n";
        file<<"enabled "<<(uint32_t)use_coordinate_transform<<"\n";
        file<<"diagonal";
        for(double value:coordinate_transform_diagonal)
            file<<" "<<setprecision(17)<<value;
        file<<"\n";
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::read_coordinate_transform_file()
    {
        if(coordinate_transform_mode=="off")
        {
            use_coordinate_transform=false;
            coordinate_transform_diagonal.assign(dim,1);
            return;
        }
        string path=data_dir+field_name+".unsz.coord_transform.txt";
        ifstream file(path);
        if(!file)
            throw runtime_error("找不到坐标变换文件："+path);
        string magic,key;
        uint32_t version,file_dim,enabled;
        if(!(file>>magic>>version)||magic!="UNSZ_COORDINATE_TRANSFORM"||version!=3)
            throw runtime_error("坐标变换文件格式不正确："+path);
        if(!(file>>key>>file_dim)||key!="dimension"||file_dim!=dim)
            throw runtime_error("坐标变换文件维数不正确："+path);
        if(!(file>>key>>enabled)||key!="enabled"||enabled>1)
            throw runtime_error("坐标变换文件enabled字段不正确："+path);
        if(!(file>>key)||key!="diagonal")
            throw runtime_error("坐标变换文件缺少diagonal字段："+path);
        vector<double> diagonal(dim);
        for(double&value:diagonal)
            if(!(file>>value)||!isfinite(value))
                throw runtime_error("坐标变换文件的对角元素不正确："+path);
        if(file>>key)
            throw runtime_error("坐标变换文件末尾存在多余内容："+path);
        use_coordinate_transform=enabled;
        coordinate_transform_diagonal=move(diagonal);
    }
}
