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
        temporal_dependency_plan_c dependency_plan;
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
        vector<int> prediction_to_original;
        if(prediction_data_uses_prediction_order)
        {
            prediction_to_original.resize(point_num);
            for(const prediction_structure_c<T_calc>&structure:prediction_structure_vec)
                for(uint32_t prediction_index=0;prediction_index<structure.target_ids.size();prediction_index++)
                    prediction_to_original[structure.prediction_order_offset+prediction_index]=structure.target_ids[prediction_index];
        }
        bool two_point_sets=point_set_config_vec.size()==2;
        double decision_eb_scale=1;
        if(point_set_config_vec[0].error_mode==point_set_error_mode_e::relative&&point_set_config_vec[0].eb>automatic_flow_max_decision_rel_eb)
            decision_eb_scale=automatic_flow_max_decision_rel_eb/point_set_config_vec[0].eb;
        printf("automatic flow decision EB: configured_mode=%s,configured_eb=%.9g,decision_eb=%.9g\n",point_set_config_vec[0].error_mode==point_set_error_mode_e::relative?"REL":"ABS",point_set_config_vec[0].eb,point_set_config_vec[0].eb*decision_eb_scale);
        auto sample_structure_start_time=chrono::steady_clock::now();
        vector<vector<automatic_flow_window_c>> structure_windows(structure_num);
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
        uint32_t full_scan_time_step_num=min<uint32_t>(time_step_num,32);
        vector<uint32_t> full_scan_time_indices;
        full_scan_time_indices.reserve(full_scan_time_step_num);
        for(uint32_t scan_id=0;scan_id<full_scan_time_step_num;scan_id++)
        {
            uint32_t time_index=full_scan_time_step_num==1?0:(uint64_t)scan_id*(time_step_num-1)/(full_scan_time_step_num-1);
            if(full_scan_time_indices.empty()||full_scan_time_indices.back()!=time_index)
                full_scan_time_indices.push_back(time_index);
        }
        vector<uint8_t> evaluate_time_step(time_step_num,0);
        for(uint32_t time_index:full_scan_time_indices)
            evaluate_time_step[time_index]=1;
        vector<automatic_flow_candidate_c> candidates;
        auto add_candidate=[&](const string&candidate_label_flow,const string&candidate_value_flow)
        {
            automatic_flow_candidate_c candidate;
            candidate.label_flow=candidate_label_flow;
            candidate.value_flow=candidate_value_flow;
            candidate.dependency_plan=generate_temporal_dependency_plan(candidate_value_flow);
            candidates.push_back(move(candidate));
        };
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
        vector<uint8_t> read_sampled_time_step=evaluate_time_step;
        for(const automatic_flow_candidate_c&candidate:candidates)
        {
            temporal_dependency_plan_c label_dependency_plan=generate_temporal_dependency_plan(candidate.label_flow);
            for(uint32_t time_index:full_scan_time_indices)
            {
                const temporal_dependency_c&value_dependency=candidate.dependency_plan.dependencies[time_index];
                const temporal_dependency_c&label_dependency=label_dependency_plan.dependencies[time_index];
                for(uint32_t source_id=0;source_id<value_dependency.source_num;source_id++)
                    read_sampled_time_step[(uint32_t)value_dependency.source_indices[source_id]]=1;
                for(uint32_t source_id=0;source_id<label_dependency.source_num;source_id++)
                    read_sampled_time_step[(uint32_t)label_dependency.source_indices[source_id]]=1;
            }
        }
        uint32_t sampled_time_step_num=accumulate(read_sampled_time_step.begin(),read_sampled_time_step.end(),0U);
        current_data.resize(point_num);
        for(uint32_t time_index:full_scan_time_indices)
        {
            auto input_start=chrono::steady_clock::now();
            read_binary_file_exact(original_data_path_vec[time_index],current_data.data(),point_num);
            excluded_input_time+=chrono::duration<double>(chrono::steady_clock::now()-input_start).count();
            auto statistics_start_time=chrono::steady_clock::now();
            compression_file_state_c state;
            if(two_point_sets)
                classify_current_point_sets<true,false>(state);
            else
                classify_current_point_sets<false,false>(state);
            file_statistics_time+=chrono::duration<double>(chrono::steady_clock::now()-statistics_start_time).count();
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
                    int reference_id=structure.reference_ids[reference_range.begin+ref_offset];
                    if(prediction_data_uses_prediction_order)
                        reference_id=prediction_to_original[reference_id];
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
        printf("automatic flow sampling: full_scan_time_steps=%u,read_time_steps=%u,structures=%u,windows=%llu,target_candidates=%llu,read_points=%llu\n",(uint32_t)full_scan_time_indices.size(),sampled_time_step_num,structure_num,(unsigned long long)sampled_window_num,(unsigned long long)sampled_target_num,(unsigned long long)sample_original_ids.size());
        vector<int> original_to_sample(point_num,-1);
        for(uint32_t sample_id=0;sample_id<sample_original_ids.size();sample_id++)
            original_to_sample[sample_original_ids[sample_id]]=(int)sample_id;
        for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
        {
            const prediction_structure_c<T_calc>&structure=prediction_structure_vec[structure_id];
            bool standard_reference_layout=structure.uses_standard_reference_layout(rbf_reference_num);
            uint32_t window_id=0;
            for(uint32_t prediction_index:structure_window_target_indices[structure_id])
            {
                automatic_flow_sample_target_c<T_calc> target;
                target.prediction_index=prediction_index;
                target.target_sample_id=(uint32_t)original_to_sample[structure.target_ids[prediction_index]];
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
                    int reference_id=structure.reference_ids[reference_index];
                    if(prediction_data_uses_prediction_order)
                        reference_id=prediction_to_original[reference_id];
                    target.reference_sample_ids.push_back(original_to_sample[reference_id]);
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
        auto classify_value=[this](T_data value)
        {
            const point_set_rule_c&rule=point_set_rule_vec.front();
            double x=(double)value;
            switch(rule.operator_type)
            {
                case point_set_rule_operator_e::less:return (uint8_t)(x<rule.value?0:1);
                case point_set_rule_operator_e::less_equal:return (uint8_t)(x<=rule.value?0:1);
                case point_set_rule_operator_e::greater:return (uint8_t)(x>rule.value?0:1);
                case point_set_rule_operator_e::greater_equal:return (uint8_t)(x>=rule.value?0:1);
                case point_set_rule_operator_e::equal:return (uint8_t)(x==rule.value?0:1);
                case point_set_rule_operator_e::not_equal:return (uint8_t)(x!=rule.value?0:1);
            }
            return (uint8_t)1;
        };
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
                    sampled_labels[time_index][sample_id]=classify_value(sampled_values[time_index][sample_id]);
            automatic_flow_time_statistics_c<T_data>&statistics=time_statistics[time_index];
            statistics.point_set_abs_eb=compression_file_point_set_abs_eb_vec[time_index];
            statistics.point_set_codec=compression_file_point_set_codec_vec[time_index];
        }
        sample_value_preparation_time=chrono::duration<double>(chrono::steady_clock::now()-sample_value_start_time).count()-(excluded_input_time-sample_value_input_begin);
        auto temporal_weights_for=[&](const temporal_dependency_c&dependency,uint32_t time_index)
        {
            array<T_calc,2> weights{{1,0}};
            if(dependency.source_num!=2)
                return weights;
            uint32_t left_source=(uint32_t)dependency.source_indices[0],right_source=(uint32_t)dependency.source_indices[1];
            T_coord left_coordinate=time_step_coordinate_vec.empty()?(T_coord)time_step_vec[left_source]:time_step_coordinate_vec[time_step_vec[left_source]];
            T_coord right_coordinate=time_step_coordinate_vec.empty()?(T_coord)time_step_vec[right_source]:time_step_coordinate_vec[time_step_vec[right_source]];
            T_coord current_coordinate=time_step_coordinate_vec.empty()?(T_coord)time_step_vec[time_index]:time_step_coordinate_vec[time_step_vec[time_index]];
            weights[1]=(T_calc)((current_coordinate-left_coordinate)/(right_coordinate-left_coordinate));
            weights[0]=(T_calc)1-weights[1];
            return weights;
        };
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
            estimated_label_bytes_by_flow["none"]=0;
        else for(const automatic_flow_candidate_c&candidate:candidates)
        {
            if(estimated_label_bytes_by_flow.count(candidate.label_flow))
                continue;
            temporal_dependency_plan_c label_dependency_plan=generate_temporal_dependency_plan(candidate.label_flow);
            double estimated_label_bytes=0;
            for(uint32_t time_index:label_dependency_plan.processing_order)
            {
                if(!evaluate_time_step[time_index])
                    continue;
                const temporal_dependency_c&dependency=label_dependency_plan.dependencies[time_index];
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
                        if(candidate.label_flow!="none"&&dependency.source_num==1)
                            predicted_label=sampled_labels[(uint32_t)dependency.source_indices[0]][target_sample_id];
                        else if(candidate.label_flow!="none"&&dependency.source_num==2)
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
                        estimated_label_bytes+=automatic_flow_log2_combination(actual_label_num,estimated_label_errors)/8;
                    }
                }
            }
            estimated_label_bytes_by_flow[candidate.label_flow]=estimated_label_bytes;
            estimated_label_bytes_by_flow[candidate.label_flow]*=(double)time_step_num/full_scan_time_indices.size();
        }
        shared_label_estimation_time=chrono::duration<double>(chrono::steady_clock::now()-shared_label_start_time).count();
        candidate_evaluation_time+=shared_label_estimation_time;
        for(automatic_flow_candidate_c&candidate:candidates)
        {
            auto candidate_start_time=chrono::steady_clock::now();
            candidate.estimated_label_bytes=estimated_label_bytes_by_flow.at(candidate.label_flow);
            vector<vector<T_data>> reconstructed_sampled_values=sampled_values;
            bool use_spatial_residual=candidate.value_flow=="constant_extrapolation_spatial_residual"||candidate.value_flow=="linear_interpolation_spatial_residual"||candidate.value_flow=="anchor_interpolation_stride_8_spatial_residual";
            vector<T_data> temporal_baseline(sample_original_ids.size()),temporal_residual(sample_original_ids.size());
            vector<uint8_t> temporal_valid(sample_original_ids.size());
            vector<vector<int32_t>> qb_vec(structure_num);
            vector<vector<T_data>> unpredictable_value_vec(structure_num);
            for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
                qb_vec[structure_id].reserve(structure_targets[structure_id].size());
            for(uint32_t time_index:candidate.dependency_plan.processing_order)
            {
                if(!evaluate_time_step[time_index])
                    continue;
                point_set_id_vec=sampled_labels[time_index];
                const temporal_dependency_c&dependency=candidate.dependency_plan.dependencies[time_index];
                array<T_calc,2> time_weights=temporal_weights_for(dependency,time_index);
                fill(temporal_valid.begin(),temporal_valid.end(),0);
                if(candidate.value_flow!="none"&&dependency.source_num>0)
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
                        candidate.estimated_value_bytes+=zstd_header.size()+(scalable_zstd_bytes+unpredictable_values.size()*sizeof(T_data))*scale;
                    }
                }
            }
            candidate.estimated_value_bytes*=(double)time_step_num/full_scan_time_indices.size();
            candidate.evaluation_time=chrono::duration<double>(chrono::steady_clock::now()-candidate_start_time).count();
            candidate_evaluation_time+=candidate.evaluation_time;
            double point_time_num=(double)point_num*time_step_num;
            printf("automatic flow candidate: label=%s,value=%s,estimated_label_bytes=%.3f,estimated_value_bytes=%.3f,estimated_total_bytes=%.3f,selection_score=%.3f,estimated_label_br=%.6f,estimated_value_br=%.6f,evaluation_time=%.6f s\n",candidate.label_flow.c_str(),candidate.value_flow.c_str(),candidate.estimated_label_bytes,candidate.estimated_value_bytes,candidate.estimated_total_bytes(),candidate.selection_score(),candidate.estimated_label_bytes*8/point_time_num,candidate.estimated_value_bytes*8/point_time_num,candidate.evaluation_time);
        }
        prediction_data_uses_prediction_order=saved_prediction_order;
        auto best=min_element(candidates.begin(),candidates.end(),[](const automatic_flow_candidate_c&a,const automatic_flow_candidate_c&b){return a.selection_score()<b.selection_score();});
        label_flow=best->label_flow;
        value_flow=best->value_flow;
        printf("automatic flow selected: label=%s,value=%s,selection_score=%.3f\n",label_flow.c_str(),value_flow.c_str(),best->selection_score());
        current_data.clear();
        point_set_id_vec.clear();
        double selection_time=chrono::duration<double>(chrono::steady_clock::now()-selection_start_time).count()-excluded_input_time;
        double other_time=selection_time-file_statistics_time-sample_structure_time-sample_value_preparation_time-candidate_evaluation_time;
        printf("automatic flow selection breakdown: file_statistics=%.6f s,sample_structure=%.6f s,sample_value_preparation=%.6f s,candidate_evaluation=%.6f s,shared_label_estimation=%.6f s,other=%.6f s\n",file_statistics_time,sample_structure_time,sample_value_preparation_time,candidate_evaluation_time,shared_label_estimation_time,other_time);
        printf("automatic flow selection time excluding data input= %f s\n",selection_time);
        return excluded_input_time;
    }
}
