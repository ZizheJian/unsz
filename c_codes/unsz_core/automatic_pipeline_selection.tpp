#pragma once

// 自动pipeline选择：在DAG和RBF权重生成之后、正式压缩之前，为当前field选一种跨slice预测方案（pipeline）。
// 做法是在少量slice、少量点上把每个候选pipeline按正式流程试压缩一遍，估计它的标签和数值码流大小，选估计值最小的pipeline。
namespace unsz
{
    // 在14个field上以REL 1e-3做无截距最小二乘，实际标签和数值码流分别约为估计值的0.412倍和0.732倍，
    // 因此选择分数取0.56≈0.412/0.732倍的估计标签字节加估计数值字节。权重在约0.15到10.7之间时，命中实际最佳pipeline的field数都是14个中的13个。
    constexpr double automatic_pipeline_label_weight=0.56;
    // 采样规模和标签权重都是在REL 1e-3上标定的，更宽的REL EB统一按1e-3估算，正式压缩仍用用户指定的EB。
    constexpr double automatic_pipeline_max_decision_rel_eb=1e-3;

    template<typename T_data>
    struct automatic_pipeline_slice_statistics_c
    {
        array<T_data,2> label_abs_eb{{0,0}};
        array<label_codec_e,2> label_codec{{label_codec_e::constant,label_codec_e::constant}};
        vector<array<uint32_t,2>> structure_label_point_num;
    };

    // 采样窗口：每个prediction structure的预测顺序均分为若干段（region），每段取一个从start开始、长度为width的连续窗口。
    // 窗口到region_end后回绕到region_begin。窗口内的目标在预测顺序上连续，后面的目标会用到前面目标的重建值，与正式压缩一致。
    // quota是该窗口要采样的标签0目标数。两种标签时窗口需要加宽，使每个评估slice都能在窗口内找到quota个标签0目标。
    struct automatic_pipeline_window_c
    {
        uint32_t region_begin=0,region_end=0,start=0,width=0,quota=0;

        uint32_t collect_value_candidates(const vector<int>&target_ids,const vector<uint8_t>&labels,vector<uint32_t>&indices) const
        {
            // 两段各保留前quota个第一类点，足以恢复最终公共窗口中的原有选点顺序。
            uint32_t before_begin=(uint32_t)indices.size();
            for(uint32_t index=region_begin;index<start&&indices.size()-before_begin<quota;index++)
                if(labels[target_ids[index]]==0)
                    indices.push_back(index);
            uint32_t after_begin=(uint32_t)indices.size();
            for(uint32_t index=start;index<region_end&&indices.size()-after_begin<quota;index++)
                if(labels[target_ids[index]]==0)
                    indices.push_back(index);
            uint32_t after_num=(uint32_t)indices.size()-after_begin,before_num=after_begin-before_begin;
            if(after_num>=quota)
                return indices.back()-start+1;
            if(before_num>=quota-after_num)
                return region_end-start+indices[before_begin+quota-after_num-1]-region_begin+1;
            return region_end-region_begin;
        }

        bool contains(uint32_t index) const
        {
            uint32_t offset=index>=start?index-start:region_end-start+index-region_begin;
            return offset<width;
        }
    };

    template<typename T_calc>
    struct automatic_pipeline_sample_target_c
    {
        uint32_t prediction_index=0,target_sample_id=0,window_id=0;
        vector<int> reference_sample_ids;
        vector<T_calc> reference_weights;
        bool has_rbf_weights=true;
    };

    struct automatic_pipeline_candidate_c
    {
        string label_pipeline,value_pipeline;
        const slice_dag_c*slice_dag=nullptr,*label_slice_dag=nullptr;
        tensor_pipeline_c tensor_pipeline;
        bool use_spatial_residual=false;
        double estimated_label_bytes=0,estimated_value_bytes=0,evaluation_time=0;

        double estimated_total_bytes() const{return estimated_label_bytes+estimated_value_bytes;}
        double selection_score() const{return automatic_pipeline_label_weight*estimated_label_bytes+estimated_value_bytes;}
    };

    // 自动pipeline选择各步骤之间共享的采样状态和计时累加器，由select_automatic_pipelines创建，依次交给各步骤函数填写和读取。
    template<typename T_data,typename T_calc>
    struct automatic_pipeline_sample_c
    {
        uint32_t slice_num=0,structure_num=0;
        bool two_labels=false;
        double decision_eb_scale=1;//决策EB与用户EB之比
        vector<vector<automatic_pipeline_window_c>> structure_windows;//每个prediction structure的采样窗口
        vector<uint32_t> full_scan_slice_indices;//评估slice，按编号升序
        vector<double> evaluation_slice_weight;//每个slice的权重，非评估slice为0
        vector<uint8_t> evaluate_slice,read_sampled_slice;//是否评估、是否需要读取样本值
        uint32_t sampled_slice_num=0;
        vector<automatic_pipeline_slice_statistics_c<T_data>> slice_statistics;//每个评估slice的标签EB、编码方式和各structure的点数
        vector<vector<vector<uint32_t>>> selected_target_indices;//[slice][structure]的数值目标，先存预测序号，读样本后改存structure_targets中的下标
        vector<uint32_t> sample_original_ids;//样本点的原始编号，样本编号即它在这个数组中的下标
        vector<vector<automatic_pipeline_sample_target_c<T_calc>>> structure_targets;//每个structure的样本目标及其参考点
        vector<vector<uint32_t>> structure_window_target_indices,structure_label_prediction_indices,structure_label_target_indices;
        uint64_t sampled_target_num=0,sampled_window_num=0;
        vector<vector<T_data>> sampled_values;//[slice][样本编号]
        vector<vector<uint8_t>> sampled_labels;
        unordered_map<string,double> estimated_label_bytes_by_pipeline;
        double excluded_input_time=0,file_statistics_time=0,sample_structure_time=0,sample_value_preparation_time=0,candidate_evaluation_time=0,shared_label_estimation_time=0;
    };

    // 只读取采样点的数值：采样点覆盖的1MiB块不到文件的30%时按块读取，否则整个文件读入更快。
    template<typename T_data>
    vector<T_data> read_automatic_pipeline_sample_values(const string&path,const vector<uint32_t>&sample_ids,uint32_t point_num,double&input_time)
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
            input_time+=seconds_since(input_start);
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
            input_time+=seconds_since(input_start);
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
            throw runtime_error("自动pipeline采样值没有完整读取");
        return result;
    }

    // log2(C(n,k))：在n个标签中标出k个预测错误位置所需的最少比特数，用来估计标签码流大小。
    inline double automatic_pipeline_log2_combination(uint64_t n,uint64_t k)
    {
        if(k==0||k==n)
            return 0;
        return (lgamma((double)n+1)-lgamma((double)k+1)-lgamma((double)(n-k)+1))/log(2.0);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::validate_automatic_pipeline_selection() const
    {
        bool automatic_label=label_pipeline=="auto",automatic_value=value_pipeline=="auto";
        if(!automatic_label&&!automatic_value)
            return;
        if(label_config_vec.size()==1)
        {
            if(automatic_label||!automatic_value)
                throw runtime_error("单标签自动pipeline选择只支持将value设置为auto");
            if(label_config_vec[0].codec==label_codec_e::constant)
                throw runtime_error("单标签自动pipeline选择要求标签使用RBF");
        }
        else if(label_config_vec.size()==2)
        {
            if(!automatic_label||!automatic_value)
                throw runtime_error("两种标签自动pipeline选择要求label和value同时设置为auto");
            if(label_config_vec[0].codec==label_codec_e::constant||label_config_vec[1].codec!=label_codec_e::constant)
                throw runtime_error("两种标签自动pipeline选择要求标签0使用RBF、标签1使用constant");
        }
        else
            throw runtime_error("自动pipeline选择只支持一个或两个标签");
        if(slice_vec.size()<2)
            throw runtime_error("自动pipeline选择至少需要两个slice");
    }

    // 第1步：布置采样窗口。总共automatic_pipeline_sample_num个目标均分给各prediction structure，每个structure再分给automatic_pipeline_start_num个窗口。
    // 窗口起点用固定的整数哈希（乘大奇数常数，其中第一个是2^64/黄金比例）在各段内伪随机选取，保证同样的输入总是选中同样的点。
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_automatic_pipeline_windows(automatic_pipeline_sample_c<T_data,T_calc>&sample)
    {
        auto&structure_num=sample.structure_num;
        auto&structure_windows=sample.structure_windows;
        auto&sample_structure_time=sample.sample_structure_time;
        auto sample_structure_start_time=chrono::steady_clock::now();
        structure_windows.resize(structure_num);
        for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
        {
            uint32_t target_num=(uint32_t)prediction_structure_vec[structure_id].target_ids.size();
            uint32_t structure_sample_num=automatic_pipeline_sample_num/structure_num+(structure_id<automatic_pipeline_sample_num%structure_num);
            uint32_t start_num=min<uint32_t>(min(automatic_pipeline_start_num,structure_sample_num),target_num);
            if(start_num==0)
                continue;
            structure_windows[structure_id].resize(start_num);
            for(uint32_t window_id=0;window_id<start_num;window_id++)
            {
                automatic_pipeline_window_c&window=structure_windows[structure_id][window_id];
                window.region_begin=(uint64_t)target_num*window_id/start_num;
                window.region_end=(uint64_t)target_num*(window_id+1)/start_num;
                uint32_t region_size=window.region_end-window.region_begin;
                uint64_t random_value=(uint64_t)(structure_id+1)*11400714819323198485ULL+(uint64_t)(window_id+1)*7046029254386353131ULL;
                window.start=window.region_begin+(uint32_t)(random_value%region_size);
                window.quota=structure_sample_num/start_num+(window_id<structure_sample_num%start_num);
            }
        }
        sample_structure_time+=seconds_since(sample_structure_start_time);
    }

    // 第2步：选出约32个评估slice，并给每个评估slice一个权重，使权重之和等于slice总数，评估slice的估计值加权求和即为全部slice的估计值。
    // 二维tensor分层抽样：第0层是根slice，第1、2层是过根的两条轴线，第3层是内部。
    // 根slice没有跨slice参考，轴线上的slice在一些pipeline中只能沿一个轴预测，因此各层分开抽样，保证每层都被抽到，每层的权重为层大小/抽样数。
    // 内部用拉丁超立方抽样，两个轴的位置各分成同样多的区间，每个区间恰好抽一次。
    // 伪随机数由field名的FNV-1a哈希作种子，再经splitmix64混合得到，>>11后乘2^-53变成[0,1)内的小数。
    // 非tensor数据按slice编号均匀取最多32个slice，权重相同。
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::choose_automatic_pipeline_evaluation_slices(automatic_pipeline_sample_c<T_data,T_calc>&sample)
    {
        auto&slice_num=sample.slice_num;
        auto&full_scan_slice_indices=sample.full_scan_slice_indices;
        auto&evaluation_slice_weight=sample.evaluation_slice_weight;
        auto&evaluate_slice=sample.evaluate_slice;
        auto&slice_statistics=sample.slice_statistics;
        slice_statistics.resize(slice_num);
        evaluation_slice_weight.assign(slice_num,0);
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
            auto add_stratum=[&](uint32_t stratum_id,uint32_t stratum_size,uint32_t sample_num,const function<uint32_t(uint32_t)>&slice_index_from_offset)
            {
                if(stratum_size==0||sample_num==0)return;
                sample_num=min(sample_num,stratum_size);
                double weight=(double)stratum_size/sample_num;
                for(uint32_t sample_id=0;sample_id<sample_num;sample_id++)
                {
                    uint32_t region_begin=(uint64_t)stratum_size*sample_id/sample_num,region_end=(uint64_t)stratum_size*(sample_id+1)/sample_num;
                    uint32_t offset=region_begin+(uint32_t)(random_value(stratum_id,sample_id)%(region_end-region_begin));
                    uint32_t slice_index=slice_index_from_offset(offset);
                    full_scan_slice_indices.push_back(slice_index);
                    evaluation_slice_weight[slice_index]=weight;
                }
            };
            uint32_t axis0_num=current_tensor_shape[0],axis1_num=current_tensor_shape[1];
            array<uint32_t,4>stratum_sizes{{1,axis0_num-1,axis1_num-1,(axis0_num-1)*(axis1_num-1)}};
            array<uint32_t,4>sample_nums;
            for(uint32_t stratum_id=0;stratum_id<3;stratum_id++)
                sample_nums[stratum_id]=min<uint32_t>(stratum_sizes[stratum_id],max<uint32_t>(3,((uint64_t)sample_budget*stratum_sizes[stratum_id]+slice_num-1)/slice_num));
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
                uint32_t slice_index=tensor_slice_id(current_tensor_shape,axis0_position,axis1_position);
                full_scan_slice_indices.push_back(slice_index);
                evaluation_slice_weight[slice_index]=interior_weight;
            }
            sort(full_scan_slice_indices.begin(),full_scan_slice_indices.end());
        }
        else
        {
            uint32_t full_scan_slice_num=min<uint32_t>(slice_num,32);
            full_scan_slice_indices.reserve(full_scan_slice_num);
            for(uint32_t scan_id=0;scan_id<full_scan_slice_num;scan_id++)
            {
                uint32_t slice_index=full_scan_slice_num==1?0:(uint64_t)scan_id*(slice_num-1)/(full_scan_slice_num-1);
                if(full_scan_slice_indices.empty()||full_scan_slice_indices.back()!=slice_index)
                    full_scan_slice_indices.push_back(slice_index);
            }
            double weight=(double)slice_num/full_scan_slice_indices.size();
            for(uint32_t slice_index:full_scan_slice_indices)
                evaluation_slice_weight[slice_index]=weight;
        }
        if(full_scan_slice_indices.empty())
            throw runtime_error("自动pipeline没有生成评估slice");
        double evaluation_weight_sum=accumulate(evaluation_slice_weight.begin(),evaluation_slice_weight.end(),0.0);
        if(abs(evaluation_weight_sum-slice_num)>1e-9*max<uint32_t>(1,slice_num))
            throw runtime_error("自动pipeline评估slice权重之和不正确");
        for(uint32_t slice_index:full_scan_slice_indices)
            if(!(evaluation_slice_weight[slice_index]>0))
                throw runtime_error("自动pipeline评估slice缺少权重");
        evaluate_slice.assign(slice_num,0);
        for(uint32_t slice_index:full_scan_slice_indices)
        {
            if(evaluate_slice[slice_index])
                throw runtime_error("自动pipeline重复选择评估slice");
            evaluate_slice[slice_index]=1;
        }
    }

    // 第3步：列出候选pipeline。tensor数据枚举各轴的预测方式和顺序。
    // 其他数据的候选为independent（每个slice单独做空间预测）、extrapolation（用前一个slice）和anchor_interpolation，后两种各有带空间残差和不带的版本。
    // anchor_interpolation先处理0,8,16,...号锚点slice（从前一个锚点外插），再按间距4、2、1逐层在两侧已重建的slice之间线性插值，类似SZ3的分层插值。
    // 两种标签时标签pipeline跟随数值pipeline。
    template<typename T_coord,typename T_data>
    vector<automatic_pipeline_candidate_c> param_c<T_coord,T_data>::list_automatic_pipeline_candidates(automatic_pipeline_sample_c<T_data,T_calc>&sample)
    {
        auto&slice_num=sample.slice_num;
        auto&two_labels=sample.two_labels;
        vector<automatic_pipeline_candidate_c> candidates;
        auto add_candidate=[&](const string&candidate_label_pipeline,const string&candidate_value_pipeline)
        {
            automatic_pipeline_candidate_c candidate;
            candidate.label_pipeline=candidate_label_pipeline;
            candidate.value_pipeline=candidate_value_pipeline;
            candidate.slice_dag=&get_slice_dag(candidate_value_pipeline);
            candidate.label_slice_dag=&get_slice_dag(candidate_label_pipeline);
            candidate.use_spatial_residual=candidate_value_pipeline=="extrapolation_spatial_residual"||candidate_value_pipeline=="interpolation_spatial_residual"||candidate_value_pipeline=="anchor_interpolation_spatial_residual";
            candidates.push_back(move(candidate));
        };
        if(current_field_uses_tensor)
        {
            for(const tensor_pipeline_c&pipeline:generate_tensor_pipeline_candidates(current_tensor_shape))
            {
                automatic_pipeline_candidate_c candidate;
                candidate.label_pipeline=tensor_pipeline_name(pipeline);
                candidate.value_pipeline=candidate.label_pipeline;
                candidate.slice_dag=&get_tensor_slice_dag(pipeline);
                candidate.label_slice_dag=candidate.slice_dag;
                candidate.tensor_pipeline=pipeline;
                candidate.use_spatial_residual=pipeline.use_spatial_residual;
                candidates.push_back(move(candidate));
            }
        }
        else
        {
            add_candidate("independent","independent");
            add_candidate(two_labels?"extrapolation":"independent","extrapolation");
            add_candidate(two_labels?"extrapolation":"independent","extrapolation_spatial_residual");
            bool anchor_compatible=slice_vec.front()==0;
            for(uint32_t slice_index=1;slice_index<slice_num;slice_index++)
                anchor_compatible&=slice_vec[slice_index]==slice_vec[slice_index-1]+1;
            if(anchor_compatible)
            {
                add_candidate(two_labels?"anchor_interpolation":"independent","anchor_interpolation");
                add_candidate(two_labels?"anchor_interpolation":"independent","anchor_interpolation_spatial_residual");
            }
        }
        return candidates;
    }

    // 第4步：在每个评估slice上确定数值目标，统计每个structure中两个标签各有多少点（用于把样本估计放大到整个slice）。
    // 单标签时窗口就是从start开始的quota个连续目标。两种标签时要在窗口内找到quota个标签0目标，窗口宽度取所有评估slice所需宽度的最大值。
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::select_automatic_pipeline_value_targets(automatic_pipeline_sample_c<T_data,T_calc>&sample)
    {
        auto&slice_num=sample.slice_num;
        auto&structure_num=sample.structure_num;
        auto&two_labels=sample.two_labels;
        auto&structure_windows=sample.structure_windows;
        auto&full_scan_slice_indices=sample.full_scan_slice_indices;
        auto&slice_statistics=sample.slice_statistics;
        auto&selected_target_indices=sample.selected_target_indices;
        auto&excluded_input_time=sample.excluded_input_time;
        auto&file_statistics_time=sample.file_statistics_time;
        auto&sample_structure_time=sample.sample_structure_time;
        selected_target_indices.assign(slice_num,vector<vector<uint32_t>>(structure_num));
        if(two_labels)
            current_data.resize(point_num);
        for(uint32_t slice_index:full_scan_slice_indices)
        {
            current_slice=slice_vec[slice_index];
            if(two_labels)
            {
                auto input_start=chrono::steady_clock::now();
                read_binary_file_exact(original_data_path_vec[slice_index],current_data.data(),point_num);
                excluded_input_time+=seconds_since(input_start);
                auto statistics_start_time=chrono::steady_clock::now();
                compression_file_state_c state;
                classify_current_labels<true,false>(state);
                file_statistics_time+=seconds_since(statistics_start_time);
            }
            auto sample_structure_start_time=chrono::steady_clock::now();
            automatic_pipeline_slice_statistics_c<T_data>&statistics=slice_statistics[slice_index];
            statistics.label_abs_eb=compression_file_label_abs_eb_vec[slice_index];
            statistics.label_codec=compression_file_label_codec_vec[slice_index];
            statistics.structure_label_point_num.resize(structure_num);
            for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
            {
                const vector<int>&target_ids=prediction_structure_vec[structure_id].target_ids;
                uint32_t first_label_num=two_labels?0:(uint32_t)target_ids.size();
                if(two_labels)
                    for(int target_id:target_ids)
                        first_label_num+=label_id_vec[target_id]==0;
                statistics.structure_label_point_num[structure_id][0]=first_label_num;
                statistics.structure_label_point_num[structure_id][1]=(uint32_t)target_ids.size()-first_label_num;
                vector<uint32_t>&value_indices=selected_target_indices[slice_index][structure_id];
                for(automatic_pipeline_window_c&window:structure_windows[structure_id])
                {
                    if(!two_labels)
                    {
                        window.width=min(window.quota,window.region_end-window.region_begin);
                        for(uint32_t offset=0;offset<window.width;offset++)
                            value_indices.push_back(window.region_begin+(window.start-window.region_begin+offset)%(window.region_end-window.region_begin));
                        continue;
                    }
                    uint32_t required_width=window.collect_value_candidates(target_ids,label_id_vec,value_indices);
                    window.width=max(window.width,required_width);
                }
            }
            sample_structure_time+=seconds_since(sample_structure_start_time);
        }
    }

    // 第5步：按最终的公共窗口裁剪每个评估slice的数值目标。标签目标是每个窗口开头的quota个点，不区分标签。
    // 所有目标及其空间参考点合起来就是要读取数值的样本点，它们按点编号排序后重新编号为样本编号，之后的计算都在样本编号上进行。
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::collect_automatic_pipeline_sample_targets(automatic_pipeline_sample_c<T_data,T_calc>&sample)
    {
        auto&structure_num=sample.structure_num;
        auto&structure_windows=sample.structure_windows;
        auto&full_scan_slice_indices=sample.full_scan_slice_indices;
        auto&slice_statistics=sample.slice_statistics;
        auto&selected_target_indices=sample.selected_target_indices;
        auto&sample_original_ids=sample.sample_original_ids;
        auto&structure_targets=sample.structure_targets;
        auto&structure_window_target_indices=sample.structure_window_target_indices;
        auto&structure_label_prediction_indices=sample.structure_label_prediction_indices;
        auto&structure_label_target_indices=sample.structure_label_target_indices;
        auto&sampled_target_num=sample.sampled_target_num;
        auto&sampled_window_num=sample.sampled_window_num;
        auto&sample_structure_time=sample.sample_structure_time;
        auto sample_structure_start_time=chrono::steady_clock::now();
        auto reference_original_id=[&](const prediction_structure_c<T_calc>&structure,int reference_id)
        {
            if(!prediction_data_uses_prediction_order)
                return reference_id;
            int64_t local_reference_index=(int64_t)reference_id-structure.prediction_order_offset;
            if(local_reference_index<0||local_reference_index>=(int64_t)structure.target_ids.size())
                throw runtime_error("自动pipeline的空间参考点跨越prediction structure");
            return structure.target_ids[(uint32_t)local_reference_index];
        };
        structure_targets.resize(structure_num);
        structure_window_target_indices.resize(structure_num);
        structure_label_prediction_indices.resize(structure_num);
        structure_label_target_indices.resize(structure_num);
        for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
        {
            const prediction_structure_c<T_calc>&structure=prediction_structure_vec[structure_id];
            vector<uint32_t>&window_target_indices=structure_window_target_indices[structure_id];
            vector<uint32_t>&label_prediction_indices=structure_label_prediction_indices[structure_id];
            for(uint32_t slice_index:full_scan_slice_indices)
            {
                vector<uint32_t>&value_indices=selected_target_indices[slice_index][structure_id];
                if(slice_statistics[slice_index].label_codec[0]==label_codec_e::constant)
                {
                    vector<uint32_t>().swap(value_indices);
                    continue;
                }
                sort(value_indices.begin(),value_indices.end());
                uint32_t window_id=0,selected_num=0,selected_in_window=0;
                for(uint32_t prediction_index:value_indices)
                {
                    while(window_id+1<structure_windows[structure_id].size()&&prediction_index>=structure_windows[structure_id][window_id].region_end)
                    {
                        window_id++;
                        selected_in_window=0;
                    }
                    const automatic_pipeline_window_c&window=structure_windows[structure_id][window_id];
                    if(!window.contains(prediction_index)||selected_in_window>=window.quota)
                        continue;
                    selected_in_window++;
                    value_indices[selected_num++]=prediction_index;
                    window_target_indices.push_back(prediction_index);
                }
                value_indices.resize(selected_num);
            }
            for(uint32_t window_id=0;window_id<structure_windows[structure_id].size();window_id++)
            {
                const automatic_pipeline_window_c&window=structure_windows[structure_id][window_id];
                uint32_t region_size=window.region_end-window.region_begin;
                for(uint32_t offset=0;offset<min(window.quota,region_size);offset++)
                    label_prediction_indices.push_back(window.region_begin+(window.start-window.region_begin+offset)%region_size);
            }
            window_target_indices.insert(window_target_indices.end(),label_prediction_indices.begin(),label_prediction_indices.end());
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
        for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
        {
            sampled_target_num+=structure_window_target_indices[structure_id].size();
            sampled_window_num+=structure_windows[structure_id].size();
        }
        auto find_sample_id=[&](uint32_t original_id)
        {
            auto iterator=lower_bound(sample_original_ids.begin(),sample_original_ids.end(),original_id);
            if(iterator==sample_original_ids.end()||*iterator!=original_id)
                throw runtime_error("自动pipeline采样编号映射不完整");
            return (uint32_t)(iterator-sample_original_ids.begin());
        };
        for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
        {
            const prediction_structure_c<T_calc>&structure=prediction_structure_vec[structure_id];
            bool standard_reference_layout=structure.uses_standard_reference_layout(rbf_reference_num);
            uint32_t window_id=0;
            for(uint32_t prediction_index:structure_window_target_indices[structure_id])
            {
                automatic_pipeline_sample_target_c<T_calc> target;
                target.prediction_index=prediction_index;
                target.target_sample_id=find_sample_id((uint32_t)structure.target_ids[prediction_index]);
                while(window_id+1<structure_windows[structure_id].size()&&prediction_index>=structure_windows[structure_id][window_id].region_end)
                    window_id++;
                const automatic_pipeline_window_c&window=structure_windows[structure_id][window_id];
                if(prediction_index<window.region_begin||prediction_index>=window.region_end)
                    throw runtime_error("自动pipeline采样目标不属于任何采样窗口");
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
                throw runtime_error("自动pipeline的label采样目标映射不完整");
        }
        sample_structure_time+=seconds_since(sample_structure_start_time);
    }

    // 第6步：确定需要读取样本值的slice，即评估slice加上任一候选pipeline中评估slice直接参考的slice，再稀疏读取样本点的数值并分类得到标签。
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::load_automatic_pipeline_samples(automatic_pipeline_sample_c<T_data,T_calc>&sample,const vector<automatic_pipeline_candidate_c>&candidates)
    {
        auto&slice_num=sample.slice_num;
        auto&structure_num=sample.structure_num;
        auto&two_labels=sample.two_labels;
        auto&full_scan_slice_indices=sample.full_scan_slice_indices;
        auto&evaluate_slice=sample.evaluate_slice;
        auto&read_sampled_slice=sample.read_sampled_slice;
        auto&sampled_slice_num=sample.sampled_slice_num;
        auto&slice_statistics=sample.slice_statistics;
        auto&selected_target_indices=sample.selected_target_indices;
        auto&sample_original_ids=sample.sample_original_ids;
        auto&structure_window_target_indices=sample.structure_window_target_indices;
        auto&sampled_values=sample.sampled_values;
        auto&sampled_labels=sample.sampled_labels;
        auto&excluded_input_time=sample.excluded_input_time;
        auto&sample_structure_time=sample.sample_structure_time;
        auto&sample_value_preparation_time=sample.sample_value_preparation_time;
        read_sampled_slice=evaluate_slice;
        for(const automatic_pipeline_candidate_c&candidate:candidates)
        {
            const slice_dag_c&label_slice_dag=*candidate.label_slice_dag;
            for(uint32_t slice_index:full_scan_slice_indices)
            {
                const reference_slices_c&value_dependency=candidate.slice_dag->dependencies[slice_index];
                const reference_slices_c&label_dependency=label_slice_dag.dependencies[slice_index];
                for(uint32_t source_id=0;source_id<value_dependency.source_num;source_id++)
                    read_sampled_slice[(uint32_t)value_dependency.source_indices[source_id]]=1;
                for(uint32_t source_id=0;source_id<label_dependency.source_num;source_id++)
                    read_sampled_slice[(uint32_t)label_dependency.source_indices[source_id]]=1;
            }
        }
        sampled_slice_num=accumulate(read_sampled_slice.begin(),read_sampled_slice.end(),0U);
        sampled_values.resize(slice_num);
        sampled_labels.resize(slice_num);
        auto sample_value_start_time=chrono::steady_clock::now();
        double sample_value_input_begin=excluded_input_time;
        for(uint32_t slice_index=0;slice_index<slice_num;slice_index++)
        {
            if(!read_sampled_slice[slice_index])
                continue;
            sampled_values[slice_index]=read_automatic_pipeline_sample_values<T_data>(original_data_path_vec[slice_index],sample_original_ids,point_num,excluded_input_time);
            sampled_labels[slice_index].resize(sample_original_ids.size());
            if(two_labels)
                for(uint32_t sample_id=0;sample_id<sample_original_ids.size();sample_id++)
                    sampled_labels[slice_index][sample_id]=classify_label(sampled_values[slice_index][sample_id],sample_original_ids[sample_id],slice_vec[slice_index]);
            automatic_pipeline_slice_statistics_c<T_data>&statistics=slice_statistics[slice_index];
            statistics.label_abs_eb=compression_file_label_abs_eb_vec[slice_index];
            statistics.label_codec=compression_file_label_codec_vec[slice_index];
        }
        sample_value_preparation_time=seconds_since(sample_value_start_time)-(excluded_input_time-sample_value_input_begin);
        auto sample_structure_start_time=chrono::steady_clock::now();
        for(uint32_t slice_index:full_scan_slice_indices)
        {
            for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
            {
                const vector<uint32_t>&target_indices=structure_window_target_indices[structure_id];
                uint32_t target_index=0;
                for(uint32_t&prediction_index:selected_target_indices[slice_index][structure_id])
                {
                    while(target_index<target_indices.size()&&target_indices[target_index]<prediction_index)
                        target_index++;
                    if(target_index==target_indices.size()||target_indices[target_index]!=prediction_index)
                        throw runtime_error("自动pipeline的value采样目标映射不完整");
                    prediction_index=target_index;
                }
            }
        }
        sample_structure_time+=seconds_since(sample_structure_start_time);
    }

    // 第7步：估计标签码流。每种标签pipeline只估一次：有一个参考slice时沿用它的标签，有两个时一致则沿用、不一致则用空间预测，没有时用空间预测。
    // 预测错误数按样本比例放大到整个structure，码流大小取log2(C(n,k))/8字节，再按slice权重求和。
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::estimate_automatic_pipeline_label_bytes(automatic_pipeline_sample_c<T_data,T_calc>&sample,const vector<automatic_pipeline_candidate_c>&candidates)
    {
        auto&structure_num=sample.structure_num;
        auto&two_labels=sample.two_labels;
        auto&evaluation_slice_weight=sample.evaluation_slice_weight;
        auto&evaluate_slice=sample.evaluate_slice;
        auto&structure_targets=sample.structure_targets;
        auto&structure_label_target_indices=sample.structure_label_target_indices;
        auto&sampled_labels=sample.sampled_labels;
        auto&estimated_label_bytes_by_pipeline=sample.estimated_label_bytes_by_pipeline;
        auto&candidate_evaluation_time=sample.candidate_evaluation_time;
        auto&shared_label_estimation_time=sample.shared_label_estimation_time;
        auto shared_label_start_time=chrono::steady_clock::now();
        if(!two_labels)
            for(const automatic_pipeline_candidate_c&candidate:candidates)
                estimated_label_bytes_by_pipeline[candidate.label_pipeline]=0;
        else for(const automatic_pipeline_candidate_c&candidate:candidates)
        {
            if(estimated_label_bytes_by_pipeline.count(candidate.label_pipeline))
                continue;
            const slice_dag_c&label_slice_dag=*candidate.label_slice_dag;
            double estimated_label_bytes=0;
            for(uint32_t slice_index:label_slice_dag.processing_order)
            {
                if(!evaluate_slice[slice_index])
                    continue;
                const reference_slices_c&dependency=label_slice_dag.dependencies[slice_index];
                double slice_estimated_label_bytes=0;
                for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
                {
                    const vector<automatic_pipeline_sample_target_c<T_calc>>&targets=structure_targets[structure_id];
                    const vector<uint32_t>&label_target_indices=structure_label_target_indices[structure_id];
                    uint64_t label_error_num=0;
                    for(uint32_t target_index:label_target_indices)
                    {
                        const automatic_pipeline_sample_target_c<T_calc>&target=targets[target_index];
                        array_view_c<int> reference_ids(target.reference_sample_ids);
                        uint32_t target_sample_id=target.target_sample_id,predicted_label;
                        if(dependency.source_num==1)
                            predicted_label=sampled_labels[(uint32_t)dependency.source_indices[0]][target_sample_id];
                        else if(dependency.source_num==2)
                        {
                            uint8_t left_label=sampled_labels[(uint32_t)dependency.source_indices[0]][target_sample_id],right_label=sampled_labels[(uint32_t)dependency.source_indices[1]][target_sample_id];
                            predicted_label=left_label==right_label?left_label:predict_label(0,reference_ids.size(),target.reference_sample_ids,target.reference_weights,sampled_labels[slice_index]);
                        }
                        else
                            predicted_label=predict_label(0,reference_ids.size(),target.reference_sample_ids,target.reference_weights,sampled_labels[slice_index]);
                        label_error_num+=predicted_label!=sampled_labels[slice_index][target_sample_id];
                    }
                    if(!label_target_indices.empty())
                    {
                        uint64_t actual_label_num=prediction_structure_vec[structure_id].target_ids.size();
                        uint64_t estimated_label_errors=(uint64_t)llround((double)label_error_num/label_target_indices.size()*actual_label_num);
                        estimated_label_errors=min<uint64_t>(estimated_label_errors,actual_label_num);
                        slice_estimated_label_bytes+=automatic_pipeline_log2_combination(actual_label_num,estimated_label_errors)/8;
                    }
                }
                estimated_label_bytes+=slice_estimated_label_bytes*evaluation_slice_weight[slice_index];
            }
            estimated_label_bytes_by_pipeline[candidate.label_pipeline]=estimated_label_bytes;
        }
        shared_label_estimation_time=seconds_since(shared_label_start_time);
        candidate_evaluation_time+=shared_label_estimation_time;
    }

    // 第8步：估计数值码流。按候选pipeline的处理顺序逐个评估slice，用正式的跨slice预测、空间预测和量化压缩样本目标，量化后的重建值供后续预测使用。
    // 跨slice预测只用于自身和参考slice都属于标签0的样本点，其余样本点用空间预测。
    // 每个structure的量化码真的做一遍Huffman和Zstd。Huffman码表不随点数成比例增长，不放大，其余字节按实际点数/样本数放大，再按slice权重求和。
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::estimate_automatic_pipeline_value_bytes(automatic_pipeline_sample_c<T_data,T_calc>&sample,vector<automatic_pipeline_candidate_c>&candidates)
    {
        auto&structure_num=sample.structure_num;
        auto&decision_eb_scale=sample.decision_eb_scale;
        auto&evaluation_slice_weight=sample.evaluation_slice_weight;
        auto&evaluate_slice=sample.evaluate_slice;
        auto&slice_statistics=sample.slice_statistics;
        auto&selected_target_indices=sample.selected_target_indices;
        auto&sample_original_ids=sample.sample_original_ids;
        auto&structure_targets=sample.structure_targets;
        auto&sampled_values=sample.sampled_values;
        auto&sampled_labels=sample.sampled_labels;
        auto&estimated_label_bytes_by_pipeline=sample.estimated_label_bytes_by_pipeline;
        auto&candidate_evaluation_time=sample.candidate_evaluation_time;
        for(automatic_pipeline_candidate_c&candidate:candidates)
        {
            auto candidate_start_time=chrono::steady_clock::now();
            candidate.estimated_label_bytes=estimated_label_bytes_by_pipeline.at(candidate.label_pipeline);
            vector<vector<T_data>> reconstructed_sampled_values=sampled_values;
            bool use_spatial_residual=candidate.use_spatial_residual;
            vector<T_data> tensor_baseline(sample_original_ids.size()),tensor_residual(sample_original_ids.size());
            vector<uint8_t> tensor_valid(sample_original_ids.size());
            vector<vector<int32_t>> quantized_error_vec(structure_num);
            vector<vector<T_data>> unpredictable_value_vec(structure_num);
            for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
                quantized_error_vec[structure_id].reserve(structure_targets[structure_id].size());
            for(uint32_t slice_index:candidate.slice_dag->processing_order)
            {
                if(!evaluate_slice[slice_index])
                    continue;
                label_id_vec=sampled_labels[slice_index];
                const reference_slices_c&dependency=candidate.slice_dag->dependencies[slice_index];
                array<T_calc,2> slice_weights{{(T_calc)dependency.source_weights[0],(T_calc)dependency.source_weights[1]}};
                fill(tensor_valid.begin(),tensor_valid.end(),0);
                if(dependency.source_num>0)
                {
                    for(uint32_t sample_id=0;sample_id<sample_original_ids.size();sample_id++)
                    {
                        bool valid=true;
                        T_calc prediction=0;
                        for(uint32_t source_id=0;source_id<dependency.source_num;source_id++)
                        {
                            uint32_t source_slice_index=(uint32_t)dependency.source_indices[source_id];
                            if(sampled_labels[source_slice_index][sample_id]!=0)
                            {
                                valid=false;
                                break;
                            }
                            prediction+=slice_weights[source_id]*(T_calc)reconstructed_sampled_values[source_slice_index][sample_id];
                        }
                        if(valid&&sampled_labels[slice_index][sample_id]==0)
                        {
                            tensor_valid[sample_id]=1;
                            tensor_baseline[sample_id]=(T_data)prediction;
                            tensor_residual[sample_id]=reconstructed_sampled_values[slice_index][sample_id]-(T_data)prediction;
                        }
                    }
                }
                for(uint32_t structure_id=0;structure_id<structure_num;structure_id++)
                {
                    const vector<automatic_pipeline_sample_target_c<T_calc>>&targets=structure_targets[structure_id];
                    vector<int32_t>&quantized_errors=quantized_error_vec[structure_id];
                    vector<T_data>&unpredictable_values=unpredictable_value_vec[structure_id];
                    quantized_errors.clear();
                    unpredictable_values.clear();
                    for(uint32_t target_index:selected_target_indices[slice_index][structure_id])
                    {
                        const automatic_pipeline_sample_target_c<T_calc>&target=targets[target_index];
                        array_view_c<int> reference_ids(target.reference_sample_ids);
                        array_view_c<T_calc> reference_weights(target.reference_weights);
                        uint32_t target_sample_id=target.target_sample_id;
                        T_data prediction;
                        if(tensor_valid[target_sample_id])
                        {
                            prediction=tensor_baseline[target_sample_id];
                            if(use_spatial_residual)
                            {
                                prediction+=predict_value_with_labels_rbf(target.has_rbf_weights,reference_ids,reference_weights,tensor_residual,0,&tensor_valid);
                            }
                        }
                        else
                            prediction=predict_value_with_labels_rbf(target.has_rbf_weights,reference_ids,reference_weights,reconstructed_sampled_values[slice_index],0);
                        T_data decision_base_abs_eb=slice_statistics[slice_index].label_abs_eb[0]*(T_data)decision_eb_scale;
                        T_data abs_eb=decision_base_abs_eb*prediction_structure_vec[structure_id].eb_multiplier[target.prediction_index];
                        T_data inverse_double_abs_eb=(T_data)0.5/decision_base_abs_eb*prediction_structure_vec[structure_id].inverse_eb_multiplier[target.prediction_index];
                        T_data reconstructed;
                        quantized_errors.push_back(quantize_or_store_unpredictable(sampled_values[slice_index][target_sample_id],prediction,abs_eb,inverse_double_abs_eb,unpredictable_values,reconstructed));
                        reconstructed_sampled_values[slice_index][target_sample_id]=reconstructed;
                        if(tensor_valid[target_sample_id])
                            tensor_residual[target_sample_id]=reconstructed-tensor_baseline[target_sample_id];
                    }
                    uint32_t actual_value_num=slice_statistics[slice_index].structure_label_point_num[structure_id][0];
                    if(!quantized_errors.empty()&&slice_statistics[slice_index].label_codec[0]!=label_codec_e::constant)
                    {
                        uint64_t huffman_header_byte_num=0;
                        vector<unsigned char> huffman_bytes=huffman_encode<int32_t>(quantized_errors,nullptr,nullptr,&huffman_header_byte_num);
                        vector<unsigned char> zstd_bytes=zstd_compress(huffman_bytes,zstd_compression_level);
                        vector<unsigned char> huffman_header(huffman_bytes.begin(),huffman_bytes.begin()+huffman_header_byte_num);
                        vector<unsigned char> zstd_header=zstd_compress(huffman_header,zstd_compression_level);
                        double scalable_zstd_bytes=max<double>(0,(double)zstd_bytes.size()-zstd_header.size());
                        double scale=(double)actual_value_num/quantized_errors.size();
                        candidate.estimated_value_bytes+=(zstd_header.size()+(scalable_zstd_bytes+unpredictable_values.size()*sizeof(T_data))*scale)*evaluation_slice_weight[slice_index];
                    }
                }
            }
            candidate.evaluation_time=seconds_since(candidate_start_time);
            candidate_evaluation_time+=candidate.evaluation_time;
        }
    }

    template<typename T_coord,typename T_data>
    double param_c<T_coord,T_data>::select_automatic_pipelines()
    {
        bool automatic_label=label_pipeline=="auto",automatic_value=value_pipeline=="auto";
        if(!automatic_label&&!automatic_value)
            return 0;
        validate_automatic_pipeline_selection();
        if(compression_file_label_abs_eb_vec.size()!=slice_vec.size()||compression_file_label_codec_vec.size()!=slice_vec.size())
            throw runtime_error("自动pipeline选择缺少逐slice预统计");
        auto selection_start_time=chrono::steady_clock::now();
        automatic_pipeline_sample_c<T_data,T_calc> sample;
        sample.slice_num=(uint32_t)slice_vec.size();
        sample.structure_num=(uint32_t)prediction_structure_vec.size();
        sample.two_labels=label_config_vec.size()==2;
        if(label_config_vec[0].error_mode!=label_error_mode_e::absolute&&label_config_vec[0].eb>automatic_pipeline_max_decision_rel_eb)
            sample.decision_eb_scale=automatic_pipeline_max_decision_rel_eb/label_config_vec[0].eb;
        prepare_automatic_pipeline_windows(sample);
        choose_automatic_pipeline_evaluation_slices(sample);
        vector<automatic_pipeline_candidate_c> candidates=list_automatic_pipeline_candidates(sample);
        select_automatic_pipeline_value_targets(sample);
        collect_automatic_pipeline_sample_targets(sample);
        load_automatic_pipeline_samples(sample,candidates);
        // 下面复用正式压缩的空间预测函数，它们从成员变量读取参考点的标签编号。
        // 样本的参考点用样本编号表示，所以这里暂时关闭预测顺序编号，并在估计每个slice前把label_id_vec换成样本标签，结束后恢复。
        bool saved_prediction_order=prediction_data_uses_prediction_order;
        prediction_data_uses_prediction_order=false;
        estimate_automatic_pipeline_label_bytes(sample,candidates);
        estimate_automatic_pipeline_value_bytes(sample,candidates);
        prediction_data_uses_prediction_order=saved_prediction_order;
        auto&slice_num=sample.slice_num;
        auto&structure_num=sample.structure_num;
        auto&decision_eb_scale=sample.decision_eb_scale;
        auto&full_scan_slice_indices=sample.full_scan_slice_indices;
        auto&sampled_slice_num=sample.sampled_slice_num;
        auto&sample_original_ids=sample.sample_original_ids;
        auto&sampled_target_num=sample.sampled_target_num;
        auto&sampled_window_num=sample.sampled_window_num;
        auto&excluded_input_time=sample.excluded_input_time;
        auto&file_statistics_time=sample.file_statistics_time;
        auto&sample_structure_time=sample.sample_structure_time;
        auto&sample_value_preparation_time=sample.sample_value_preparation_time;
        auto&candidate_evaluation_time=sample.candidate_evaluation_time;
        auto&shared_label_estimation_time=sample.shared_label_estimation_time;
        // 第9步：选分数最小的候选，并恢复被用作临时空间的成员变量。
        auto best=min_element(candidates.begin(),candidates.end(),[](const automatic_pipeline_candidate_c&a,const automatic_pipeline_candidate_c&b){return a.selection_score()<b.selection_score();});
        if(current_field_uses_tensor)
        {
            selected_tensor_pipeline=best->tensor_pipeline;
            tensor_pipeline_selected=true;
        }
        else
        {
            label_pipeline=best->label_pipeline;
            value_pipeline=best->value_pipeline;
        }
        current_data.clear();
        label_id_vec.clear();
        double selection_time=seconds_since(selection_start_time)-excluded_input_time;
        double other_time=selection_time-file_statistics_time-sample_structure_time-sample_value_preparation_time-candidate_evaluation_time;
        printf("automatic pipeline decision EB: configured_mode=%s,configured_eb=%.9g,decision_eb=%.9g\n",label_config_vec[0].error_mode==label_error_mode_e::relative?"REL":label_config_vec[0].error_mode==label_error_mode_e::field_relative?"FIELD_REL":"ABS",label_config_vec[0].eb,label_config_vec[0].eb*decision_eb_scale);
        printf("automatic pipeline sampling: full_scan_slices=%u,read_slices=%u,structures=%u,windows=%llu,target_candidates=%llu,read_points=%llu\n",(uint32_t)full_scan_slice_indices.size(),sampled_slice_num,structure_num,(unsigned long long)sampled_window_num,(unsigned long long)sampled_target_num,(unsigned long long)sample_original_ids.size());
        double point_slice_num=(double)point_num*slice_num;
        for(const automatic_pipeline_candidate_c&candidate:candidates)
            printf("automatic pipeline candidate: label=%s,value=%s,estimated_label_bytes=%.3f,estimated_value_bytes=%.3f,estimated_total_bytes=%.3f,selection_score=%.3f,estimated_label_br=%.6f,estimated_value_br=%.6f,evaluation_time=%.6f s\n",candidate.label_pipeline.c_str(),candidate.value_pipeline.c_str(),candidate.estimated_label_bytes,candidate.estimated_value_bytes,candidate.estimated_total_bytes(),candidate.selection_score(),candidate.estimated_label_bytes*8/point_slice_num,candidate.estimated_value_bytes*8/point_slice_num,candidate.evaluation_time);
        if(current_field_uses_tensor)
            printf("automatic tensor pipeline selected: %s,selection_score=%.3f\n",tensor_pipeline_name(selected_tensor_pipeline).c_str(),best->selection_score());
        else
            printf("automatic pipeline selected: label=%s,value=%s,selection_score=%.3f\n",label_pipeline.c_str(),value_pipeline.c_str(),best->selection_score());
        printf("automatic pipeline selection breakdown: file_statistics=%.6f s,sample_structure=%.6f s,sample_value_preparation=%.6f s,candidate_evaluation=%.6f s,shared_label_estimation=%.6f s,other=%.6f s\n",file_statistics_time,sample_structure_time,sample_value_preparation_time,candidate_evaluation_time,shared_label_estimation_time,other_time);
        printf("automatic pipeline selection time excluding data input= %f s\n",selection_time);
        printf("automatic pipeline data input time excluded from reported time= %f s\n",excluded_input_time);
        return selection_time;
    }
}
