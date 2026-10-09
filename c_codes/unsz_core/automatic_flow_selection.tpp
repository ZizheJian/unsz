#pragma once

namespace unsz
{
    constexpr double automatic_flow_label_weight=0.56;
    constexpr double automatic_flow_max_decision_rel_eb=1e-3;

    template<typename T_data>
    struct automatic_flow_time_statistics_c
    {
        array<T_data,2> point_set_abs_eb{{0,0}};
        array<point_set_codec_e,2> point_set_codec{{point_set_codec_e::constant,point_set_codec_e::constant}};
        vector<array<uint32_t,2>> structure_point_set_point_num;
    };

    struct automatic_flow_window_c
    {
        uint32_t region_begin=0,region_end=0,start=0,width=0,quota=0;
    };

    template<typename T_calc>
    struct automatic_flow_sample_target_c
    {
        uint32_t prediction_index=0,target_sample_id=0,window_id=0;
        vector<int> reference_sample_ids;
        vector<T_calc> reference_weights;
        bool has_rbf_weights=true;
    };

    struct automatic_flow_candidate_c
    {
        string label_flow,value_flow;
        const temporal_dependency_plan_c*dependency_plan=nullptr,*label_dependency_plan=nullptr;
        tensor_flow_c tensor_flow;
        bool use_spatial_residual=false;
        double estimated_label_bytes=0,estimated_value_bytes=0,evaluation_time=0;

        double estimated_total_bytes() const{return estimated_label_bytes+estimated_value_bytes;}
        double selection_score() const{return automatic_flow_label_weight*estimated_label_bytes+estimated_value_bytes;}
    };

    template<typename T_data>
    vector<T_data> read_automatic_flow_sample_values(const string&path,const vector<uint32_t>&sample_ids,uint32_t point_num,double&input_time)
    {
        vector<T_data> result(sample_ids.size());
        if(sample_ids.empty())
            return result;
        constexpr uint64_t block_byte_num=1ULL<<20;
        uint64_t block_value_num=max<uint64_t>(1,block_byte_num/sizeof(T_data));
        vector<uint64_t> block_ids;
        block_ids.reserve(sample_ids.size());
        uint64_t previous_block=numeric_limits<uint64_t>::max();
        for(uint32_t point_id:sample_ids)
        {
            uint64_t block_id=point_id/block_value_num;
            if(block_id!=previous_block)
            {
                block_ids.push_back(block_id);
                previous_block=block_id;
            }
        }
        uint64_t touched_value_num=min<uint64_t>(point_num,block_ids.size()*block_value_num);
        if(touched_value_num*10>=point_num*3)
        {
            vector<T_data> full_data(point_num);
            auto input_start=chrono::steady_clock::now();
            read_binary_file_exact(path,full_data.data(),point_num);
            input_time+=chrono::duration<double>(chrono::steady_clock::now()-input_start).count();
            for(uint32_t sample_id=0;sample_id<sample_ids.size();sample_id++)
                result[sample_id]=full_data[sample_ids[sample_id]];
            return result;
        }
        FILE*file=fopen(path.c_str(),"rb");
        if(!file)
            throw runtime_error("找不到文件："+path);
        if(fseek(file,0,SEEK_END)!=0)
        {
            fclose(file);
            throw runtime_error("无法定位到文件末尾："+path);
        }
        long file_byte_num=ftell(file);
        if(file_byte_num<0||(uint64_t)file_byte_num!=(uint64_t)point_num*sizeof(T_data))
        {
            fclose(file);
            throw runtime_error("文件大小与预期不一致："+path);
        }
        uint32_t sample_begin=0;
        vector<T_data> block_data;
        for(uint64_t block_id:block_ids)
        {
            uint64_t value_begin=block_id*block_value_num;
            uint64_t value_num=min<uint64_t>(block_value_num,(uint64_t)point_num-value_begin);
            block_data.resize(value_num);
            auto input_start=chrono::steady_clock::now();
            bool read_failed=fseek(file,(long)(value_begin*sizeof(T_data)),SEEK_SET)!=0||fread(block_data.data(),sizeof(T_data),value_num,file)!=value_num;
            input_time+=chrono::duration<double>(chrono::steady_clock::now()-input_start).count();
            if(read_failed)
            {
                fclose(file);
                throw runtime_error("分块读取文件失败："+path);
            }
            while(sample_begin<sample_ids.size()&&sample_ids[sample_begin]<value_begin+value_num)
            {
                result[sample_begin]=block_data[sample_ids[sample_begin]-value_begin];
                sample_begin++;
            }
        }
        if(fclose(file)!=0)
            throw runtime_error("关闭文件失败："+path);
        if(sample_begin!=sample_ids.size())
            throw runtime_error("自动flow采样值没有完整读取");
        return result;
    }

    inline double automatic_flow_log2_combination(uint64_t n,uint64_t k)
    {
        if(k==0||k==n)
            return 0;
        return (lgamma((double)n+1)-lgamma((double)k+1)-lgamma((double)(n-k)+1))/log(2.0);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::validate_automatic_flow_selection() const
    {
        bool automatic_label=label_flow=="auto",automatic_value=value_flow=="auto";
        if(!automatic_label&&!automatic_value)
            return;
        if(point_set_config_vec.size()==1)
        {
            if(automatic_label||!automatic_value)
                throw runtime_error("单点集自动flow选择只支持将value设置为auto");
            if(point_set_config_vec[0].codec==point_set_codec_e::constant)
                throw runtime_error("单点集自动flow选择要求点集使用RBF");
        }
        else if(point_set_config_vec.size()==2)
        {
            if(!automatic_label||!automatic_value)
                throw runtime_error("两点集自动flow选择要求label和value同时设置为auto");
            if(point_set_config_vec[0].codec==point_set_codec_e::constant||point_set_config_vec[1].codec!=point_set_codec_e::constant)
                throw runtime_error("两点集自动flow选择要求点集0使用RBF、点集1使用constant");
        }
        else
            throw runtime_error("自动flow选择只支持一个或两个点集");
        if(time_step_vec.size()<2)
            throw runtime_error("自动flow选择至少需要两个time step");
    }

    template<typename T_coord,typename T_data>
    double param_c<T_coord,T_data>::select_automatic_flows()
    {
        bool automatic_label=label_flow=="auto",automatic_value=value_flow=="auto";
        if(!automatic_label&&!automatic_value)
            return 0;
        validate_automatic_flow_selection();
        if(compression_file_point_set_abs_eb_vec.size()!=time_step_vec.size()||compression_file_point_set_codec_vec.size()!=time_step_vec.size())
            throw runtime_error("自动flow选择缺少逐time step预统计");
        auto selection_start_time=chrono::steady_clock::now();
        double excluded_input_time=0;
        double file_statistics_time=0,sample_structure_time=0,sample_value_preparation_time=0,candidate_evaluation_time=0,shared_label_estimation_time=0;
        uint32_t time_step_num=(uint32_t)time_step_vec.size(),structure_num=(uint32_t)prediction_structure_vec.size();
        bool two_point_sets=point_set_config_vec.size()==2;
        double decision_eb_scale=1;
        if(point_set_config_vec[0].error_mode==point_set_error_mode_e::relative&&point_set_config_vec[0].eb>automatic_flow_max_decision_rel_eb)
            decision_eb_scale=automatic_flow_max_decision_rel_eb/point_set_config_vec[0].eb;
        auto sample_structure_start_time=chrono::steady_clock::now();
        vector<vector<automatic_flow_window_c>> structure_windows(structure_num);
        auto reference_original_id=[&](const prediction_structure_c<T_calc>&structure,int reference_id)
        {
            if(!prediction_data_uses_prediction_order)
                return reference_id;
            int64_t local_reference_index=(int64_t)reference_id-structure.prediction_order_offset;
            if(local_reference_index<0||local_reference_index>=(int64_t)structure.target_ids.size())
                throw runtime_error("自动flow的空间参考点跨越prediction structure");
            return structure.target_ids[(uint32_t)local_reference_index];
        };
        for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
        {
            uint32_t target_num=(uint32_t)prediction_structure_vec[structure_id].target_ids.size();
            uint32_t structure_sample_num=automatic_flow_sample_num/structure_num+(structure_id<automatic_flow_sample_num%structure_num);
            uint32_t start_num=min<uint32_t>(min(automatic_flow_start_num,structure_sample_num),target_num);
            if(start_num==0)
                continue;
            structure_windows[structure_id].resize(start_num);
            for(uint32_t window_id=0;window_id<start_num;window_id++)
            {
                automatic_flow_window_c&window=structure_windows[structure_id][window_id];
                window.region_begin=(uint64_t)target_num*window_id/start_num;
                window.region_end=(uint64_t)target_num*(window_id+1)/start_num;
                uint32_t region_size=window.region_end-window.region_begin;
                uint64_t random_value=(uint64_t)(structure_id+1)*11400714819323198485ULL+(uint64_t)(window_id+1)*7046029254386353131ULL;
                window.start=window.region_begin+(uint32_t)(random_value%region_size);
                window.quota=structure_sample_num/start_num+(window_id<structure_sample_num%start_num);
            }
        }
        sample_structure_time+=chrono::duration<double>(chrono::steady_clock::now()-sample_structure_start_time).count();
        vector<automatic_flow_time_statistics_c<T_data>> time_statistics(time_step_num);
        vector<uint32_t> full_scan_time_indices;
        vector<double> evaluation_time_step_weight(time_step_num,0);
        if(current_field_uses_tensor&&current_tensor_shape[0]>1&&current_tensor_shape[1]>1)
        {
            constexpr uint32_t sample_budget=32;
            uint64_t field_seed=1469598103934665603ULL;
            for(unsigned char character:field_name)
                field_seed=(field_seed^character)*1099511628211ULL;
            auto random_value=[&](uint32_t stratum_id,uint32_t sample_id)
            {
                uint64_t value=field_seed+(uint64_t)(stratum_id+1)*11400714819323198485ULL+(uint64_t)(sample_id+1)*7046029254386353131ULL;
                value=(value^(value>>30))*13787848793156543929ULL;
                value=(value^(value>>27))*10723151780598845931ULL;
                return value^(value>>31);
            };
            auto add_stratum=[&](uint32_t stratum_id,uint32_t stratum_size,uint32_t sample_num,const function<uint32_t(uint32_t)>&time_index_from_offset)
            {
                if(stratum_size==0||sample_num==0)return;
                sample_num=min(sample_num,stratum_size);
                double weight=(double)stratum_size/sample_num;
                for(uint32_t sample_id=0;sample_id<sample_num;sample_id++)
                {
                    uint32_t region_begin=(uint64_t)stratum_size*sample_id/sample_num,region_end=(uint64_t)stratum_size*(sample_id+1)/sample_num;
                    uint32_t offset=region_begin+(uint32_t)(random_value(stratum_id,sample_id)%(region_end-region_begin));
                    uint32_t time_index=time_index_from_offset(offset);
                    full_scan_time_indices.push_back(time_index);
                    evaluation_time_step_weight[time_index]=weight;
                }
            };
            uint32_t axis0_num=current_tensor_shape[0],axis1_num=current_tensor_shape[1];
            array<uint32_t,4>stratum_sizes{{1,axis0_num-1,axis1_num-1,(axis0_num-1)*(axis1_num-1)}};
            array<uint32_t,4>sample_nums;
            for(uint32_t stratum_id=0;stratum_id<3;stratum_id++)
                sample_nums[stratum_id]=min<uint32_t>(stratum_sizes[stratum_id],max<uint32_t>(3,((uint64_t)sample_budget*stratum_sizes[stratum_id]+time_step_num-1)/time_step_num));
            uint32_t allocated_sample_num=sample_nums[0]+sample_nums[1]+sample_nums[2];
            sample_nums[3]=min<uint32_t>(stratum_sizes[3],sample_budget-min(sample_budget,allocated_sample_num));
            add_stratum(0,stratum_sizes[0],sample_nums[0],[&](uint32_t){return 0;});
            add_stratum(1,stratum_sizes[1],sample_nums[1],[&](uint32_t offset){return tensor_slice_id(current_tensor_shape,offset+1,0);});
            add_stratum(2,stratum_sizes[2],sample_nums[2],[&](uint32_t offset){return tensor_slice_id(current_tensor_shape,0,offset+1);});
            uint32_t interior_sample_num=sample_nums[3];
            vector<uint32_t>axis1_strata(interior_sample_num);
            iota(axis1_strata.begin(),axis1_strata.end(),0);
            for(uint32_t sample_num=interior_sample_num;sample_num>1;sample_num--)
                swap(axis1_strata[sample_num-1],axis1_strata[random_value(3,sample_num)%sample_num]);
            double interior_weight=(double)stratum_sizes[3]/interior_sample_num;
            constexpr double random_scale=1.0/9007199254740992.0;
            for(uint32_t sample_id=0;sample_id<interior_sample_num;sample_id++)
            {
                double axis0_fraction=(sample_id+(double)(random_value(4,sample_id)>>11)*random_scale)/interior_sample_num;
                double axis1_fraction=(axis1_strata[sample_id]+(double)(random_value(5,sample_id)>>11)*random_scale)/interior_sample_num;
                uint32_t axis0_position=1+min<uint32_t>(axis0_num-2,(uint32_t)(axis0_fraction*(axis0_num-1)));
                uint32_t axis1_position=1+min<uint32_t>(axis1_num-2,(uint32_t)(axis1_fraction*(axis1_num-1)));
                uint32_t time_index=tensor_slice_id(current_tensor_shape,axis0_position,axis1_position);
                full_scan_time_indices.push_back(time_index);
                evaluation_time_step_weight[time_index]=interior_weight;
            }
            sort(full_scan_time_indices.begin(),full_scan_time_indices.end());
        }
        else
        {
            uint32_t full_scan_time_step_num=min<uint32_t>(time_step_num,32);
            full_scan_time_indices.reserve(full_scan_time_step_num);
            for(uint32_t scan_id=0;scan_id<full_scan_time_step_num;scan_id++)
            {
                uint32_t time_index=full_scan_time_step_num==1?0:(uint64_t)scan_id*(time_step_num-1)/(full_scan_time_step_num-1);
                if(full_scan_time_indices.empty()||full_scan_time_indices.back()!=time_index)
                    full_scan_time_indices.push_back(time_index);
            }
            double weight=(double)time_step_num/full_scan_time_indices.size();
            for(uint32_t time_index:full_scan_time_indices)
                evaluation_time_step_weight[time_index]=weight;
        }
        if(full_scan_time_indices.empty())
            throw runtime_error("自动flow没有生成评估time step");
        double evaluation_weight_sum=accumulate(evaluation_time_step_weight.begin(),evaluation_time_step_weight.end(),0.0);
        if(abs(evaluation_weight_sum-time_step_num)>1e-9*max<uint32_t>(1,time_step_num))
            throw runtime_error("自动flow评估time step权重之和不正确");
        for(uint32_t time_index:full_scan_time_indices)
            if(!(evaluation_time_step_weight[time_index]>0))
                throw runtime_error("自动flow评估time step缺少权重");
        vector<uint8_t> evaluate_time_step(time_step_num,0);
        for(uint32_t time_index:full_scan_time_indices)
        {
            if(evaluate_time_step[time_index])
                throw runtime_error("自动flow重复选择评估time step");
            evaluate_time_step[time_index]=1;
        }
        vector<automatic_flow_candidate_c> candidates;
        auto add_candidate=[&](const string&candidate_label_flow,const string&candidate_value_flow)
        {
            automatic_flow_candidate_c candidate;
            candidate.label_flow=candidate_label_flow;
            candidate.value_flow=candidate_value_flow;
            candidate.dependency_plan=&get_temporal_dependency_plan(candidate_value_flow);
            candidate.label_dependency_plan=&get_temporal_dependency_plan(candidate_label_flow);
            candidate.use_spatial_residual=candidate_value_flow=="constant_extrapolation_spatial_residual"||candidate_value_flow=="linear_interpolation_spatial_residual"||candidate_value_flow=="anchor_interpolation_stride_8_spatial_residual";
            candidates.push_back(move(candidate));
        };
        if(current_field_uses_tensor)
        {
            for(const tensor_flow_c&flow:generate_tensor_flow_candidates(current_tensor_shape))
            {
                automatic_flow_candidate_c candidate;
                candidate.label_flow=tensor_flow_name(flow);
                candidate.value_flow=candidate.label_flow;
                candidate.dependency_plan=&get_tensor_dependency_plan(flow);
                candidate.label_dependency_plan=candidate.dependency_plan;
                candidate.tensor_flow=flow;
                candidate.use_spatial_residual=flow.use_spatial_residual;
                candidates.push_back(move(candidate));
            }
        }
        else
        {
            add_candidate("none","none");
            add_candidate(two_point_sets?"constant_extrapolation":"none","constant_extrapolation");
            add_candidate(two_point_sets?"constant_extrapolation":"none","constant_extrapolation_spatial_residual");
            bool anchor_compatible=time_step_vec.front()==0;
            for(uint32_t time_index=1;time_index<time_step_num;time_index++)
                anchor_compatible&=time_step_vec[time_index]==time_step_vec[time_index-1]+1;
            if(anchor_compatible)
            {
                add_candidate(two_point_sets?"anchor_interpolation_stride_8":"none","anchor_interpolation_stride_8");
                add_candidate(two_point_sets?"anchor_interpolation_stride_8":"none","anchor_interpolation_stride_8_spatial_residual");
            }
        }
        vector<uint8_t> read_sampled_time_step=evaluate_time_step;
        for(const automatic_flow_candidate_c&candidate:candidates)
        {
            const temporal_dependency_plan_c&label_dependency_plan=*candidate.label_dependency_plan;
            for(uint32_t time_index:full_scan_time_indices)
            {
                const temporal_dependency_c&value_dependency=candidate.dependency_plan->dependencies[time_index];
                const temporal_dependency_c&label_dependency=label_dependency_plan.dependencies[time_index];
                for(uint32_t source_id=0;source_id<value_dependency.source_num;source_id++)
                    read_sampled_time_step[(uint32_t)value_dependency.source_indices[source_id]]=1;
                for(uint32_t source_id=0;source_id<label_dependency.source_num;source_id++)
                    read_sampled_time_step[(uint32_t)label_dependency.source_indices[source_id]]=1;
            }
        }
        uint32_t sampled_time_step_num=accumulate(read_sampled_time_step.begin(),read_sampled_time_step.end(),0U);
        if(two_point_sets)
            current_data.resize(point_num);
        for(uint32_t time_index:full_scan_time_indices)
        {
            current_time_step=time_step_vec[time_index];
            if(two_point_sets)
            {
                auto input_start=chrono::steady_clock::now();
                read_binary_file_exact(original_data_path_vec[time_index],current_data.data(),point_num);
                excluded_input_time+=chrono::duration<double>(chrono::steady_clock::now()-input_start).count();
                auto statistics_start_time=chrono::steady_clock::now();
                compression_file_state_c state;
                classify_current_point_sets<true,false>(state);
                file_statistics_time+=chrono::duration<double>(chrono::steady_clock::now()-statistics_start_time).count();
            }
            sample_structure_start_time=chrono::steady_clock::now();
            automatic_flow_time_statistics_c<T_data>&statistics=time_statistics[time_index];
            statistics.point_set_abs_eb=compression_file_point_set_abs_eb_vec[time_index];
            statistics.point_set_codec=compression_file_point_set_codec_vec[time_index];
            statistics.structure_point_set_point_num.resize(structure_num);
            for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
            {
                const vector<int>&target_ids=prediction_structure_vec[structure_id].target_ids;
                uint32_t first_point_set_num=two_point_sets?0:(uint32_t)target_ids.size();
                if(two_point_sets)
                    for(int target_id:target_ids)
                        first_point_set_num+=point_set_id_vec[target_id]==0;
                statistics.structure_point_set_point_num[structure_id][0]=first_point_set_num;
                statistics.structure_point_set_point_num[structure_id][1]=(uint32_t)target_ids.size()-first_point_set_num;
                for(automatic_flow_window_c&window:structure_windows[structure_id])
                {
                    if(!two_point_sets)
                    {
                        window.width=min(window.quota,window.region_end-window.region_begin);
                        continue;
                    }
                    uint32_t region_size=window.region_end-window.region_begin,first_point_set_num=0,required_width=region_size;
                    for(uint32_t offset=0;offset<region_size;offset++)
                    {
                        uint32_t prediction_index=window.region_begin+(window.start-window.region_begin+offset)%region_size;
                        if(point_set_id_vec[target_ids[prediction_index]]==0&&++first_point_set_num==window.quota)
                        {
                            required_width=offset+1;
                            break;
                        }
                    }
                    window.width=max(window.width,required_width);
                }
            }
            sample_structure_time+=chrono::duration<double>(chrono::steady_clock::now()-sample_structure_start_time).count();
        }
        sample_structure_start_time=chrono::steady_clock::now();
        vector<uint32_t> sample_original_ids;
        vector<vector<automatic_flow_sample_target_c<T_calc>>> structure_targets(structure_num);
        vector<vector<uint32_t>> structure_window_target_indices(structure_num);
        vector<vector<uint32_t>> structure_label_prediction_indices(structure_num),structure_label_target_indices(structure_num);
        for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
        {
            const prediction_structure_c<T_calc>&structure=prediction_structure_vec[structure_id];
            vector<uint32_t>&window_target_indices=structure_window_target_indices[structure_id];
            vector<uint32_t>&label_prediction_indices=structure_label_prediction_indices[structure_id];
            for(uint32_t window_id=0;window_id<structure_windows[structure_id].size();window_id++)
            {
                const automatic_flow_window_c&window=structure_windows[structure_id][window_id];
                uint32_t region_size=window.region_end-window.region_begin;
                for(uint32_t offset=0;offset<window.width;offset++)
                    window_target_indices.push_back(window.region_begin+(window.start-window.region_begin+offset)%region_size);
                for(uint32_t offset=0;offset<min(window.quota,region_size);offset++)
                    label_prediction_indices.push_back(window.region_begin+(window.start-window.region_begin+offset)%region_size);
            }
            sort(window_target_indices.begin(),window_target_indices.end());
            window_target_indices.erase(unique(window_target_indices.begin(),window_target_indices.end()),window_target_indices.end());
            sort(label_prediction_indices.begin(),label_prediction_indices.end());
            label_prediction_indices.erase(unique(label_prediction_indices.begin(),label_prediction_indices.end()),label_prediction_indices.end());
            for(uint32_t prediction_index:window_target_indices)
            {
                int target_id=structure.target_ids[prediction_index];
                sample_original_ids.push_back((uint32_t)target_id);
                typename prediction_structure_c<T_calc>::reference_range_c reference_range=structure.get_reference_range(prediction_index,rbf_reference_num,structure.uses_standard_reference_layout(rbf_reference_num));
                for(uint32_t ref_offset=0;ref_offset<reference_range.num;ref_offset++)
                {
                    int reference_id=reference_original_id(structure,structure.reference_ids[reference_range.begin+ref_offset]);
                    sample_original_ids.push_back((uint32_t)reference_id);
                }
            }
        }
        sort(sample_original_ids.begin(),sample_original_ids.end());
        sample_original_ids.erase(unique(sample_original_ids.begin(),sample_original_ids.end()),sample_original_ids.end());
        uint64_t sampled_target_num=0,sampled_window_num=0;
        for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
        {
            sampled_target_num+=structure_window_target_indices[structure_id].size();
            sampled_window_num+=structure_windows[structure_id].size();
        }
        auto find_sample_id=[&](uint32_t original_id)
        {
            auto iterator=lower_bound(sample_original_ids.begin(),sample_original_ids.end(),original_id);
            if(iterator==sample_original_ids.end()||*iterator!=original_id)
                throw runtime_error("自动flow采样编号映射不完整");
            return (uint32_t)(iterator-sample_original_ids.begin());
        };
        for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
        {
            const prediction_structure_c<T_calc>&structure=prediction_structure_vec[structure_id];
            bool standard_reference_layout=structure.uses_standard_reference_layout(rbf_reference_num);
            uint32_t window_id=0;
            for(uint32_t prediction_index:structure_window_target_indices[structure_id])
            {
                automatic_flow_sample_target_c<T_calc> target;
                target.prediction_index=prediction_index;
                target.target_sample_id=find_sample_id((uint32_t)structure.target_ids[prediction_index]);
                while(window_id+1<structure_windows[structure_id].size()&&prediction_index>=structure_windows[structure_id][window_id].region_end)
                    window_id++;
                const automatic_flow_window_c&window=structure_windows[structure_id][window_id];
                if(prediction_index<window.region_begin||prediction_index>=window.region_end)
                    throw runtime_error("自动flow采样目标不属于任何采样窗口");
                target.window_id=window_id;
                typename prediction_structure_c<T_calc>::reference_range_c reference_range=structure.get_reference_range(prediction_index,rbf_reference_num,standard_reference_layout);
                target.reference_sample_ids.reserve(reference_range.num);
                target.reference_weights.reserve(reference_range.num);
                for(uint32_t ref_offset=0;ref_offset<reference_range.num;ref_offset++)
                {
                    uint64_t reference_index=reference_range.begin+ref_offset;
                    int reference_id=reference_original_id(structure,structure.reference_ids[reference_index]);
                    target.reference_sample_ids.push_back((int)find_sample_id((uint32_t)reference_id));
                    target.reference_weights.push_back(structure.prediction_weights[reference_index]);
                }
                target.has_rbf_weights=structure.rbf_fallback_flags.empty()||!structure.rbf_fallback_flags[prediction_index];
                structure_targets[structure_id].push_back(move(target));
            }
            vector<uint32_t>&label_target_indices=structure_label_target_indices[structure_id];
            const vector<uint32_t>&window_target_indices=structure_window_target_indices[structure_id],&label_prediction_indices=structure_label_prediction_indices[structure_id];
            uint32_t label_prediction_position=0;
            for(uint32_t target_index=0;target_index<window_target_indices.size()&&label_prediction_position<label_prediction_indices.size();target_index++)
            {
                if(window_target_indices[target_index]==label_prediction_indices[label_prediction_position])
                {
                    label_target_indices.push_back(target_index);
                    label_prediction_position++;
                }
            }
            if(label_prediction_position!=label_prediction_indices.size())
                throw runtime_error("自动flow的label采样目标映射不完整");
        }
        sample_structure_time+=chrono::duration<double>(chrono::steady_clock::now()-sample_structure_start_time).count();
        vector<vector<T_data>> sampled_values(time_step_num);
        vector<vector<uint8_t>> sampled_labels(time_step_num);
        auto sample_value_start_time=chrono::steady_clock::now();
        double sample_value_input_begin=excluded_input_time;
        for(uint32_t time_index=0;time_index<time_step_num;time_index++)
        {
            if(!read_sampled_time_step[time_index])
                continue;
            sampled_values[time_index]=read_automatic_flow_sample_values<T_data>(original_data_path_vec[time_index],sample_original_ids,point_num,excluded_input_time);
            sampled_labels[time_index].resize(sample_original_ids.size());
            if(two_point_sets)
                for(uint32_t sample_id=0;sample_id<sample_original_ids.size();sample_id++)
                    sampled_labels[time_index][sample_id]=classify_point_set(sampled_values[time_index][sample_id],sample_original_ids[sample_id],time_step_vec[time_index]);
            automatic_flow_time_statistics_c<T_data>&statistics=time_statistics[time_index];
            statistics.point_set_abs_eb=compression_file_point_set_abs_eb_vec[time_index];
            statistics.point_set_codec=compression_file_point_set_codec_vec[time_index];
        }
        sample_value_preparation_time=chrono::duration<double>(chrono::steady_clock::now()-sample_value_start_time).count()-(excluded_input_time-sample_value_input_begin);
        sample_structure_start_time=chrono::steady_clock::now();
        vector<vector<vector<uint32_t>>> selected_target_indices(time_step_num,vector<vector<uint32_t>>(structure_num));
        for(uint32_t time_index=0;time_index<time_step_num;time_index++)
        {
            if(!evaluate_time_step[time_index])
                continue;
            for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
            {
                if(time_statistics[time_index].point_set_codec[0]==point_set_codec_e::constant)
                    continue;
                const vector<automatic_flow_sample_target_c<T_calc>>&targets=structure_targets[structure_id];
                vector<uint32_t> selected_per_window(structure_windows[structure_id].size(),0);
                vector<uint32_t>&selected_indices=selected_target_indices[time_index][structure_id];
                selected_indices.reserve(automatic_flow_sample_num);
                for(uint32_t target_index=0;target_index<targets.size();target_index++)
                {
                    const automatic_flow_sample_target_c<T_calc>&target=targets[target_index];
                    if(sampled_labels[time_index][target.target_sample_id]!=0||selected_per_window[target.window_id]>=structure_windows[structure_id][target.window_id].quota)
                        continue;
                    selected_per_window[target.window_id]++;
                    selected_indices.push_back(target_index);
                }
            }
        }
        sample_structure_time+=chrono::duration<double>(chrono::steady_clock::now()-sample_structure_start_time).count();
        bool saved_prediction_order=prediction_data_uses_prediction_order;
        prediction_data_uses_prediction_order=false;
        unordered_map<string,double> estimated_label_bytes_by_flow;
        auto shared_label_start_time=chrono::steady_clock::now();
        if(!two_point_sets)
            for(const automatic_flow_candidate_c&candidate:candidates)
                estimated_label_bytes_by_flow[candidate.label_flow]=0;
        else for(const automatic_flow_candidate_c&candidate:candidates)
        {
            if(estimated_label_bytes_by_flow.count(candidate.label_flow))
                continue;
            const temporal_dependency_plan_c&label_dependency_plan=*candidate.label_dependency_plan;
            double estimated_label_bytes=0;
            for(uint32_t time_index:label_dependency_plan.processing_order)
            {
                if(!evaluate_time_step[time_index])
                    continue;
                const temporal_dependency_c&dependency=label_dependency_plan.dependencies[time_index];
                double time_estimated_label_bytes=0;
                for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
                {
                    const vector<automatic_flow_sample_target_c<T_calc>>&targets=structure_targets[structure_id];
                    const vector<uint32_t>&label_target_indices=structure_label_target_indices[structure_id];
                    uint64_t label_error_num=0;
                    for(uint32_t target_index:label_target_indices)
                    {
                        const automatic_flow_sample_target_c<T_calc>&target=targets[target_index];
                        array_view_c<int> reference_ids(target.reference_sample_ids);
                        uint32_t target_sample_id=target.target_sample_id,predicted_label;
                        if(dependency.source_num==1)
                            predicted_label=sampled_labels[(uint32_t)dependency.source_indices[0]][target_sample_id];
                        else if(dependency.source_num==2)
                        {
                            uint8_t left_label=sampled_labels[(uint32_t)dependency.source_indices[0]][target_sample_id],right_label=sampled_labels[(uint32_t)dependency.source_indices[1]][target_sample_id];
                            predicted_label=left_label==right_label?left_label:predict_label(0,reference_ids.size(),target.reference_sample_ids,target.reference_weights,sampled_labels[time_index]);
                        }
                        else
                            predicted_label=predict_label(0,reference_ids.size(),target.reference_sample_ids,target.reference_weights,sampled_labels[time_index]);
                        label_error_num+=predicted_label!=sampled_labels[time_index][target_sample_id];
                    }
                    if(!label_target_indices.empty())
                    {
                        uint64_t actual_label_num=prediction_structure_vec[structure_id].target_ids.size();
                        uint64_t estimated_label_errors=(uint64_t)llround((double)label_error_num/label_target_indices.size()*actual_label_num);
                        estimated_label_errors=min<uint64_t>(estimated_label_errors,actual_label_num);
                        time_estimated_label_bytes+=automatic_flow_log2_combination(actual_label_num,estimated_label_errors)/8;
                    }
                }
                estimated_label_bytes+=time_estimated_label_bytes*evaluation_time_step_weight[time_index];
            }
            estimated_label_bytes_by_flow[candidate.label_flow]=estimated_label_bytes;
        }
        shared_label_estimation_time=chrono::duration<double>(chrono::steady_clock::now()-shared_label_start_time).count();
        candidate_evaluation_time+=shared_label_estimation_time;
        for(automatic_flow_candidate_c&candidate:candidates)
        {
            auto candidate_start_time=chrono::steady_clock::now();
            candidate.estimated_label_bytes=estimated_label_bytes_by_flow.at(candidate.label_flow);
            vector<vector<T_data>> reconstructed_sampled_values=sampled_values;
            bool use_spatial_residual=candidate.use_spatial_residual;
            vector<T_data> temporal_baseline(sample_original_ids.size()),temporal_residual(sample_original_ids.size());
            vector<uint8_t> temporal_valid(sample_original_ids.size());
            vector<vector<int32_t>> qb_vec(structure_num);
            vector<vector<T_data>> unpredictable_value_vec(structure_num);
            for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
                qb_vec[structure_id].reserve(structure_targets[structure_id].size());
            for(uint32_t time_index:candidate.dependency_plan->processing_order)
            {
                if(!evaluate_time_step[time_index])
                    continue;
                point_set_id_vec=sampled_labels[time_index];
                const temporal_dependency_c&dependency=candidate.dependency_plan->dependencies[time_index];
                array<T_calc,2> time_weights{{(T_calc)dependency.source_weights[0],(T_calc)dependency.source_weights[1]}};
                fill(temporal_valid.begin(),temporal_valid.end(),0);
                if(dependency.source_num>0)
                {
                    for(uint32_t sample_id=0;sample_id<sample_original_ids.size();sample_id++)
                    {
                        bool valid=true;
                        T_calc prediction=0;
                        for(uint32_t source_id=0;source_id<dependency.source_num;source_id++)
                        {
                            uint32_t source_time_index=(uint32_t)dependency.source_indices[source_id];
                            if(sampled_labels[source_time_index][sample_id]!=0)
                            {
                                valid=false;
                                break;
                            }
                            prediction+=time_weights[source_id]*(T_calc)reconstructed_sampled_values[source_time_index][sample_id];
                        }
                        if(valid&&sampled_labels[time_index][sample_id]==0)
                        {
                            temporal_valid[sample_id]=1;
                            temporal_baseline[sample_id]=(T_data)prediction;
                            temporal_residual[sample_id]=reconstructed_sampled_values[time_index][sample_id]-(T_data)prediction;
                        }
                    }
                }
                for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
                {
                    const vector<automatic_flow_sample_target_c<T_calc>>&targets=structure_targets[structure_id];
                    vector<int32_t>&qbs=qb_vec[structure_id];
                    vector<T_data>&unpredictable_values=unpredictable_value_vec[structure_id];
                    qbs.clear();
                    unpredictable_values.clear();
                    for(uint32_t target_index:selected_target_indices[time_index][structure_id])
                    {
                        const automatic_flow_sample_target_c<T_calc>&target=targets[target_index];
                        array_view_c<int> reference_ids(target.reference_sample_ids);
                        array_view_c<T_calc> reference_weights(target.reference_weights);
                        uint32_t target_sample_id=target.target_sample_id;
                        T_data prediction;
                        if(temporal_valid[target_sample_id])
                        {
                            prediction=temporal_baseline[target_sample_id];
                            if(use_spatial_residual)
                            {
                                prediction+=predict_value_two_point_sets_rbf(target.has_rbf_weights,reference_ids,reference_weights,temporal_residual,0,&temporal_valid);
                            }
                        }
                        else
                            prediction=predict_value_two_point_sets_rbf(target.has_rbf_weights,reference_ids,reference_weights,reconstructed_sampled_values[time_index],0);
                        T_data decision_base_abs_eb=time_statistics[time_index].point_set_abs_eb[0]*(T_data)decision_eb_scale;
                        T_data abs_eb=decision_base_abs_eb*prediction_structure_vec[structure_id].eb_multiplier[target.prediction_index];
                        T_data inverse_double_abs_eb=(T_data)0.5/decision_base_abs_eb*prediction_structure_vec[structure_id].inverse_eb_multiplier[target.prediction_index];
                        T_data reconstructed;
                        qbs.push_back(quantize_or_store_unpredictable(sampled_values[time_index][target_sample_id],prediction,abs_eb,inverse_double_abs_eb,unpredictable_values,reconstructed));
                        reconstructed_sampled_values[time_index][target_sample_id]=reconstructed;
                        if(temporal_valid[target_sample_id])
                            temporal_residual[target_sample_id]=reconstructed-temporal_baseline[target_sample_id];
                    }
                    uint32_t actual_value_num=time_statistics[time_index].structure_point_set_point_num[structure_id][0];
                    if(!qbs.empty()&&time_statistics[time_index].point_set_codec[0]!=point_set_codec_e::constant)
                    {
                        uint64_t huffman_header_byte_num=0;
                        vector<unsigned char> huffman_bytes=huffman_encode<int32_t>(qbs,nullptr,nullptr,&huffman_header_byte_num);
                        vector<unsigned char> zstd_bytes=zstd_compress(huffman_bytes,zstd_compression_level);
                        vector<unsigned char> huffman_header(huffman_bytes.begin(),huffman_bytes.begin()+huffman_header_byte_num);
                        vector<unsigned char> zstd_header=zstd_compress(huffman_header,zstd_compression_level);
                        double scalable_zstd_bytes=max<double>(0,(double)zstd_bytes.size()-zstd_header.size());
                        double scale=(double)actual_value_num/qbs.size();
                        candidate.estimated_value_bytes+=(zstd_header.size()+(scalable_zstd_bytes+unpredictable_values.size()*sizeof(T_data))*scale)*evaluation_time_step_weight[time_index];
                    }
                }
            }
            candidate.evaluation_time=chrono::duration<double>(chrono::steady_clock::now()-candidate_start_time).count();
            candidate_evaluation_time+=candidate.evaluation_time;
        }
        prediction_data_uses_prediction_order=saved_prediction_order;
        auto best=min_element(candidates.begin(),candidates.end(),[](const automatic_flow_candidate_c&a,const automatic_flow_candidate_c&b){return a.selection_score()<b.selection_score();});
        if(current_field_uses_tensor)
        {
            selected_tensor_flow=best->tensor_flow;
            tensor_flow_selected=true;
        }
        else
        {
            label_flow=best->label_flow;
            value_flow=best->value_flow;
        }
        current_data.clear();
        point_set_id_vec.clear();
        double selection_time=chrono::duration<double>(chrono::steady_clock::now()-selection_start_time).count()-excluded_input_time;
        double other_time=selection_time-file_statistics_time-sample_structure_time-sample_value_preparation_time-candidate_evaluation_time;
        printf("automatic flow decision EB: configured_mode=%s,configured_eb=%.9g,decision_eb=%.9g\n",point_set_config_vec[0].error_mode==point_set_error_mode_e::relative?"REL":"ABS",point_set_config_vec[0].eb,point_set_config_vec[0].eb*decision_eb_scale);
        printf("automatic flow sampling: full_scan_time_steps=%u,read_time_steps=%u,structures=%u,windows=%llu,target_candidates=%llu,read_points=%llu\n",(uint32_t)full_scan_time_indices.size(),sampled_time_step_num,structure_num,(unsigned long long)sampled_window_num,(unsigned long long)sampled_target_num,(unsigned long long)sample_original_ids.size());
        double point_time_num=(double)point_num*time_step_num;
        for(const automatic_flow_candidate_c&candidate:candidates)
            printf("automatic flow candidate: label=%s,value=%s,estimated_label_bytes=%.3f,estimated_value_bytes=%.3f,estimated_total_bytes=%.3f,selection_score=%.3f,estimated_label_br=%.6f,estimated_value_br=%.6f,evaluation_time=%.6f s\n",candidate.label_flow.c_str(),candidate.value_flow.c_str(),candidate.estimated_label_bytes,candidate.estimated_value_bytes,candidate.estimated_total_bytes(),candidate.selection_score(),candidate.estimated_label_bytes*8/point_time_num,candidate.estimated_value_bytes*8/point_time_num,candidate.evaluation_time);
        if(current_field_uses_tensor)
            printf("automatic tensor flow selected: %s,selection_score=%.3f\n",tensor_flow_name(selected_tensor_flow).c_str(),best->selection_score());
        else
            printf("automatic flow selected: label=%s,value=%s,selection_score=%.3f\n",label_flow.c_str(),value_flow.c_str(),best->selection_score());
        printf("automatic flow selection breakdown: file_statistics=%.6f s,sample_structure=%.6f s,sample_value_preparation=%.6f s,candidate_evaluation=%.6f s,shared_label_estimation=%.6f s,other=%.6f s\n",file_statistics_time,sample_structure_time,sample_value_preparation_time,candidate_evaluation_time,shared_label_estimation_time,other_time);
        printf("automatic flow selection time excluding data input= %f s\n",selection_time);
        printf("automatic flow data input time excluded from reported time= %f s\n",excluded_input_time);
        return selection_time;
    }
}
