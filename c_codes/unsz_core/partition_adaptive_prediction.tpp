#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    template<typename T_system>
    bool param_c<T_coord,T_data>::solve_linear_system(vector<vector<T_system>>&A,vector<T_system>&b,vector<T_system>&x)
    {
        uint32_t n=A.size();
        if(n==0||b.size()!=n)
            return false;
        for(auto&row:A)
            if(row.size()!=n)
                return false;
        for(uint32_t col=0;col<n;col++)
        {
            uint32_t pivot=col;
            for(uint32_t row=col+1;row<n;row++)
                if(abs(A[row][col])>abs(A[pivot][col]))
                    pivot=row;
            if(abs(A[pivot][col])<(T_system)1e-12)
                return false;
            if(pivot!=col)
            {
                swap(A[col],A[pivot]);
                swap(b[col],b[pivot]);
            }
            T_system divisor=A[col][col];
            for(uint32_t i=col;i<n;i++)
                A[col][i]/=divisor;
            b[col]/=divisor;
            for(uint32_t row=0;row<n;row++)
            {
                if(row==col)
                    continue;
                T_system factor=A[row][col];
                for(uint32_t i=col;i<n;i++)
                    A[row][i]-=factor*A[col][i];
                b[row]-=factor*b[col];
            }
        }
        x.assign(b.begin(),b.end());
        return true;
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::should_use_selected_adaptive_prediction(int pred_id) const
    {
        return use_partition_adaptive_predictor_experimental&&!partition_adaptive_sample_mask[pred_id];
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_selected_adaptive_value(int pred_id,array_view_c<int> ref_ids,vector<T_data>&data)
    {
        uint32_t group_id=point_partition_id_vec[pred_id];
        return predict_partition_adaptive_value(pred_id,ref_ids,data,partition_prediction_config_id_vec[group_id]);
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_partition_idw_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double power)
    {
        if(ref_ids.empty())
            return 0;
        T_calc total_weight=0,prediction=0;
        for(int ref_id:ref_ids)
        {
            T_calc squared_distance=(T_calc)get_dist_to_coord(coord_array[pred_id],coord_array[ref_id],dim);
            if(squared_distance==0)
                return data[ref_id];
            T_calc weight=pow(squared_distance,(T_calc)(-power/2));
            if(isinf(weight))
                return data[ref_id];
            total_weight+=weight;
            prediction+=weight*data[ref_id];
        }
        return total_weight>0?(T_data)(prediction/total_weight):(T_data)0;
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_partition_idw_value_and_loo_error(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double power,T_data &prediction,double &loo_error)
    {
        if(ref_ids.size()<2)
            return false;
        prediction=predict_partition_idw_value(pred_id,ref_ids,data,power);
        if(!isfinite(prediction))
            return false;
        vector<int> loo_ref_ids;
        loo_ref_ids.reserve(ref_ids.size()-1);
        loo_error=0;
        for(uint32_t held_id=0;held_id<ref_ids.size();held_id++)
        {
            loo_ref_ids.clear();
            for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
                if(ref_index!=held_id)
                    loo_ref_ids.push_back(ref_ids[ref_index]);
            int held_ref_id=ref_ids[held_id];
            T_data loo_prediction=predict_partition_idw_value(held_ref_id,loo_ref_ids,data,power);
            loo_error+=abs((double)data[held_ref_id]-loo_prediction);
        }
        return isfinite(loo_error);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_partition_linear_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction,double weight_power,double ridge,bool normalize_coord)
    {
        if(ref_ids.size()<=dim)
            return false;
        uint32_t system_size=dim+1;
        vector<vector<double>> A(system_size,vector<double>(system_size,0));
        vector<double> b(system_size,0),x,scale(dim,normalize_coord?0:1),squared_distance_vec(ref_ids.size());
        double min_squared_distance=numeric_limits<double>::infinity();
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            for(uint32_t d=0;d<dim;d++)
            {
                double coordinate_difference=(double)coord_array[ref_id][d]-coord_array[pred_id][d];
                if(normalize_coord)
                    scale[d]=max(scale[d],abs(coordinate_difference));
                squared_distance_vec[ref_index]+=coordinate_difference*coordinate_difference;
            }
            if(squared_distance_vec[ref_index]==0)
            {
                prediction=data[ref_id];
                return true;
            }
            min_squared_distance=min(min_squared_distance,squared_distance_vec[ref_index]);
        }
        if(normalize_coord)
            for(double coordinate_scale:scale)
                if(!(coordinate_scale>0))
                    return false;
        double weight_sum=0;
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            vector<double> coordinate_difference(dim);
            for(uint32_t d=0;d<dim;d++)
                coordinate_difference[d]=((double)coord_array[ref_id][d]-coord_array[pred_id][d])/scale[d];
            double weight=weight_power>0?pow(squared_distance_vec[ref_index]/min_squared_distance,-weight_power/2):1;
            if(!(weight>0)||!isfinite(weight))
                return false;
            weight_sum+=weight;
            for(uint32_t row=0;row<dim;row++)
            {
                for(uint32_t col=0;col<dim;col++)
                    A[row][col]+=weight*coordinate_difference[row]*coordinate_difference[col];
                A[row][dim]+=weight*coordinate_difference[row];
                A[dim][row]+=weight*coordinate_difference[row];
                b[row]+=weight*coordinate_difference[row]*data[ref_id];
            }
            A[dim][dim]+=weight;
            b[dim]+=weight*data[ref_id];
        }
        for(uint32_t d=0;d<dim;d++)
            A[d][d]+=ridge*weight_sum;
        if(!solve_linear_system(A,b,x)||!isfinite(x[dim]))
            return false;
        prediction=(T_data)x[dim];
        return isfinite(prediction);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_partition_linear_fitting_value_and_loo_error(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction,double &loo_error,double weight_power,double ridge,bool normalize_coord)
    {
        uint32_t system_size=dim+1;
        if(ref_ids.size()<=system_size)
            return false;
        vector<double> scale(dim,normalize_coord?0:1),squared_distance_vec(ref_ids.size());
        double min_squared_distance=numeric_limits<double>::infinity();
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            for(uint32_t d=0;d<dim;d++)
            {
                double coordinate_difference=(double)coord_array[ref_id][d]-coord_array[pred_id][d];
                if(normalize_coord)
                    scale[d]=max(scale[d],abs(coordinate_difference));
                squared_distance_vec[ref_index]+=coordinate_difference*coordinate_difference;
            }
            if(squared_distance_vec[ref_index]==0)
                return false;
            min_squared_distance=min(min_squared_distance,squared_distance_vec[ref_index]);
        }
        if(normalize_coord)
            for(double coordinate_scale:scale)
                if(!(coordinate_scale>0))
                    return false;
        vector<vector<double>> design(ref_ids.size(),vector<double>(system_size,1));
        vector<double> weight_vec(ref_ids.size(),1);
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            for(uint32_t d=0;d<dim;d++)
                design[ref_index][d]=((double)coord_array[ref_ids[ref_index]][d]-coord_array[pred_id][d])/scale[d];
            if(weight_power>0)
                weight_vec[ref_index]=pow(squared_distance_vec[ref_index]/min_squared_distance,-weight_power/2);
            if(!(weight_vec[ref_index]>0)||!isfinite(weight_vec[ref_index]))
                return false;
        }
        vector<vector<double>> augmented(system_size,vector<double>(system_size*2+1,0));
        double weight_sum=accumulate(weight_vec.begin(),weight_vec.end(),0.0);
        for(uint32_t row=0;row<system_size;row++)
        {
            for(uint32_t col=0;col<system_size;col++)
                for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
                    augmented[row][col]+=weight_vec[ref_index]*design[ref_index][row]*design[ref_index][col];
            for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
                augmented[row][system_size]+=weight_vec[ref_index]*design[ref_index][row]*data[ref_ids[ref_index]];
            if(row<dim)
                augmented[row][row]+=ridge*weight_sum;
            augmented[row][system_size+1+row]=1;
        }
        for(uint32_t col=0;col<system_size;col++)
        {
            uint32_t pivot=col;
            for(uint32_t row=col+1;row<system_size;row++)
                if(abs(augmented[row][col])>abs(augmented[pivot][col]))
                    pivot=row;
            if(abs(augmented[pivot][col])<1e-12)
                return false;
            if(pivot!=col)
                swap(augmented[pivot],augmented[col]);
            double divisor=augmented[col][col];
            for(uint32_t i=0;i<augmented[col].size();i++)
                augmented[col][i]/=divisor;
            for(uint32_t row=0;row<system_size;row++)
            {
                if(row==col)
                    continue;
                double factor=augmented[row][col];
                for(uint32_t i=0;i<augmented[row].size();i++)
                    augmented[row][i]-=factor*augmented[col][i];
            }
        }
        vector<double> coefficient(system_size);
        for(uint32_t row=0;row<system_size;row++)
            coefficient[row]=augmented[row][system_size];
        prediction=(T_data)coefficient[dim];
        if(!isfinite(prediction))
            return false;
        loo_error=0;
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            double fitted_value=0,leverage=0;
            for(uint32_t row=0;row<system_size;row++)
            {
                fitted_value+=design[ref_index][row]*coefficient[row];
                for(uint32_t col=0;col<system_size;col++)
                    leverage+=weight_vec[ref_index]*design[ref_index][row]*augmented[row][system_size+1+col]*design[ref_index][col];
            }
            double denominator=1-leverage;
            if(!(denominator>1e-8)||!isfinite(denominator))
                return false;
            loo_error+=abs(((double)data[ref_ids[ref_index]]-fitted_value)/denominator);
        }
        return isfinite(loo_error);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_partition_continuous_mix_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double idw_power,double fitting_weight_power,T_data &prediction)
    {
        T_data idw_prediction,fitting_prediction;
        double idw_loo_error,fitting_loo_error;
        if(!predict_partition_idw_value_and_loo_error(pred_id,ref_ids,data,idw_power,idw_prediction,idw_loo_error))
            return false;
        if(!predict_partition_linear_fitting_value_and_loo_error(pred_id,ref_ids,data,fitting_prediction,fitting_loo_error,fitting_weight_power,0,true))
            return false;
        double error_sum=idw_loo_error+fitting_loo_error;
        double fitting_ratio=error_sum>0?idw_loo_error/error_sum:0.5;
        prediction=(T_data)((1-fitting_ratio)*idw_prediction+fitting_ratio*fitting_prediction);
        return isfinite(prediction);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_partition_idw_residual_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double idw_power,double residual_weight_power,T_data &prediction)
    {
        if(ref_ids.size()<=dim)
            return false;
        T_data idw_prediction=predict_partition_idw_value(pred_id,ref_ids,data,idw_power);
        if(!isfinite(idw_prediction))
            return false;
        vector<double> residual_vec(ref_ids.size());
        vector<int> loo_ref_ids;
        loo_ref_ids.reserve(ref_ids.size()-1);
        for(uint32_t held_id=0;held_id<ref_ids.size();held_id++)
        {
            loo_ref_ids.clear();
            for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
                if(ref_index!=held_id)
                    loo_ref_ids.push_back(ref_ids[ref_index]);
            int held_ref_id=ref_ids[held_id];
            T_data loo_prediction=predict_partition_idw_value(held_ref_id,loo_ref_ids,data,idw_power);
            residual_vec[held_id]=(double)data[held_ref_id]-loo_prediction;
        }
        uint32_t system_size=dim+1;
        vector<double> scale(dim,0),squared_distance_vec(ref_ids.size());
        double min_squared_distance=numeric_limits<double>::infinity();
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            for(uint32_t d=0;d<dim;d++)
            {
                double dc=(double)coord_array[ref_id][d]-coord_array[pred_id][d];
                scale[d]=max(scale[d],abs(dc));
                squared_distance_vec[ref_index]+=dc*dc;
            }
            if(squared_distance_vec[ref_index]==0)
            {
                prediction=data[ref_id];
                return true;
            }
            min_squared_distance=min(min_squared_distance,squared_distance_vec[ref_index]);
        }
        for(double value:scale)
            if(!(value>0))
                return false;
        vector<vector<double>> A(system_size,vector<double>(system_size,0));
        vector<double> b(system_size,0),x;
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            vector<double> dc(dim);
            for(uint32_t d=0;d<dim;d++)
                dc[d]=((double)coord_array[ref_id][d]-coord_array[pred_id][d])/scale[d];
            double weight=residual_weight_power>0?pow(squared_distance_vec[ref_index]/min_squared_distance,-residual_weight_power/2):1;
            if(!(weight>0)||!isfinite(weight))
                return false;
            for(uint32_t row=0;row<dim;row++)
            {
                for(uint32_t col=0;col<dim;col++)
                    A[row][col]+=weight*dc[row]*dc[col];
                A[row][dim]+=weight*dc[row];
                A[dim][row]+=weight*dc[row];
                b[row]+=weight*dc[row]*residual_vec[ref_index];
            }
            A[dim][dim]+=weight;
            b[dim]+=weight*residual_vec[ref_index];
        }
        if(!solve_linear_system(A,b,x)||!isfinite(x[dim]))
            return false;
        prediction=(T_data)((double)idw_prediction+x[dim]);
        return isfinite(prediction);
    }

    template<typename T_coord,typename T_data>
    vector<partition_prediction_config_c> param_c<T_coord,T_data>::get_partition_prediction_config_vec()
    {
        vector<partition_prediction_config_c> config_vec;
        for (double power:{2.5,3.0,3.5,5.0,7.5})
            for (uint32_t ref_num:{4u,6u,8u,10u})
                config_vec.push_back({ref_num,power,false});
        config_vec.push_back({10,3.5,true});
        bool include_weighted=partition_fitting_candidates_experimental=="weighted" || partition_fitting_candidates_experimental=="mixed" || partition_fitting_candidates_experimental=="extended" || partition_fitting_candidates_experimental=="all";
        if (include_weighted)
        {
            config_vec.push_back({10,3.5,true,0,0,true});
            for (double weight_power:{1.0,2.0,3.5})
                config_vec.push_back({10,3.5,true,weight_power,0,true});
        }
        if (partition_fitting_candidates_experimental=="ridge" || partition_fitting_candidates_experimental=="all")
            for (double ridge:{1e-6,1e-4,1e-2})
                config_vec.push_back({10,3.5,true,0,ridge,true});
        if (partition_fitting_candidates_experimental=="mixed" || partition_fitting_candidates_experimental=="extended" || partition_fitting_candidates_experimental=="all")
            for (double weight_power:{1.0,2.0,3.5})
                config_vec.push_back({10,3.5,true,weight_power,0,true,partition_fitting_method_e::continuous_mix});
        if (partition_fitting_candidates_experimental=="residual" || partition_fitting_candidates_experimental=="extended" || partition_fitting_candidates_experimental=="all")
            for (double weight_power:{0.0,1.0,2.0,3.5})
                config_vec.push_back({10,3.5,true,weight_power,0,true,partition_fitting_method_e::idw_residual});
        if (partition_fitting_candidates_experimental!="original" && partition_fitting_candidates_experimental!="weighted" && partition_fitting_candidates_experimental!="ridge" && partition_fitting_candidates_experimental!="mixed" && partition_fitting_candidates_experimental!="residual" && partition_fitting_candidates_experimental!="extended" && partition_fitting_candidates_experimental!="all")
            throw runtime_error("未知的partition拟合候选模式："+partition_fitting_candidates_experimental);
        return config_vec;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_partition_adaptive_sample_mask()
    {
        if (!use_partition_adaptive_predictor_experimental)
            return;
        if (partition_point_ids_vec.empty())
            throw runtime_error("partition自适应预测器需要启用partition");
        partition_adaptive_sample_mask.assign(point_num,0);
        for (auto &point_ids:partition_point_ids_vec)
        {
            uint32_t sample_num=min<uint32_t>(partition_adaptive_sample_num_experimental,point_ids.size());
            for (uint32_t sample_id=0;sample_id<sample_num;sample_id++)
            {
                uint64_t point_index=((uint64_t)sample_id*2+1)*point_ids.size()/(sample_num*2);
                partition_adaptive_sample_mask[point_ids[point_index]]=1;
            }
        }
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_partition_adaptive_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,uint8_t config_id)
    {
        vector<partition_prediction_config_c> config_vec=get_partition_prediction_config_vec();
        if (config_id>=config_vec.size())
            throw runtime_error("partition预测配置编号越界");
        return predict_partition_adaptive_value(pred_id,ref_ids,data,config_vec[config_id]);
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_partition_adaptive_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,const partition_prediction_config_c &config)
    {
        uint32_t ref_num=min<uint32_t>(config.ref_num,ref_ids.size());
        if (ref_num==0)
            return 0;
        vector<int> selected_ref_ids(ref_ids.begin(),ref_ids.begin()+ref_num);
        if (config.use_fitting)
        {
            T_data fitting_prediction;
            bool fitting_succeeded=false;
            if (config.fitting_method==partition_fitting_method_e::direct)
                fitting_succeeded=predict_partition_linear_fitting_value(pred_id,selected_ref_ids,data,fitting_prediction,config.fitting_weight_power,config.fitting_ridge,config.normalize_fitting_coord);
            else if (config.fitting_method==partition_fitting_method_e::continuous_mix)
                fitting_succeeded=predict_partition_continuous_mix_value(pred_id,selected_ref_ids,data,config.idw_power,config.fitting_weight_power,fitting_prediction);
            else if (config.fitting_method==partition_fitting_method_e::idw_residual)
                fitting_succeeded=predict_partition_idw_residual_fitting_value(pred_id,selected_ref_ids,data,config.idw_power,config.fitting_weight_power,fitting_prediction);
            if (fitting_succeeded)
                return fitting_prediction;
        }
        return predict_partition_idw_value(pred_id,selected_ref_ids,data,config.idw_power);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::select_partition_prediction_configs()
    {
        if (!use_partition_adaptive_predictor_experimental)
            return;
        if (point_set_point_num_vec.size()!=1)
            throw runtime_error("partition自适应预测器当前只支持单点集数据");
        vector<partition_prediction_config_c> config_vec=get_partition_prediction_config_vec();
        partition_prediction_config_id_vec.assign(partition_point_ids_vec.size(),0);
        prepare_partition_adaptive_sample_mask();
        vector<vector<double>> partition_score_vec(partition_point_ids_vec.size(),vector<double>(config_vec.size(),0));
        vector<uint32_t> partition_sample_num_vec(partition_point_ids_vec.size());
        for (uint32_t partition_id=0;partition_id<partition_point_ids_vec.size();partition_id++)
        {
            auto &point_ids=partition_point_ids_vec[partition_id];
            vector<int> sample_ids;
            sample_ids.reserve(min<uint32_t>(partition_adaptive_sample_num_experimental,point_ids.size()));
            for (int point_id:point_ids)
                if (partition_adaptive_sample_mask[point_id])
                    sample_ids.push_back(point_id);
            if (sample_ids.size()<=10)
                throw runtime_error("partition自适应预测器的采样点数必须大于10");
            partition_sample_num_vec[partition_id]=(uint32_t)sample_ids.size();
            vector<double> &score_vec=partition_score_vec[partition_id];
            for (int target_id:sample_ids)
            {
                vector<pair<double,int>> distance_id_vec;
                distance_id_vec.reserve(sample_ids.size()-1);
                for (int ref_id:sample_ids)
                    if (ref_id!=target_id)
                        distance_id_vec.push_back({(double)get_dist_to_coord(coord_array[target_id],coord_array[ref_id],dim),ref_id});
                uint32_t nearest_num=min<uint32_t>(10,distance_id_vec.size());
                partial_sort(distance_id_vec.begin(),distance_id_vec.begin()+nearest_num,distance_id_vec.end());
                vector<int> ref_ids;
                ref_ids.reserve(nearest_num);
                for (uint32_t ref_index=0;ref_index<nearest_num;ref_index++)
                    ref_ids.push_back(distance_id_vec[ref_index].second);
                for (uint32_t config_id=0;config_id<config_vec.size();config_id++)
                {
                    T_data prediction=predict_partition_adaptive_value(target_id,ref_ids,current_data,config_id);
                    double qb=round(((double)current_data[target_id]-prediction)/(2*point_set_abs_eb_vec[0]));
                    score_vec[config_id]+=isfinite(qb)?log2(1+abs(qb)):numeric_limits<double>::max()/sample_ids.size();
                }
            }
        }
        vector<double> field_score_vec(config_vec.size(),0);
        for (const vector<double> &score_vec:partition_score_vec)
            for (uint32_t config_id=0;config_id<config_vec.size();config_id++)
                field_score_vec[config_id]+=score_vec[config_id];
        uint32_t field_config_id=min_element(field_score_vec.begin(),field_score_vec.end())-field_score_vec.begin();
        printf("partition_adaptive_field_config: config=%u,score=%.6f,min_local_improvement=%.6f\n",field_config_id,field_score_vec[field_config_id],partition_adaptive_min_improvement_experimental);
        for (uint32_t partition_id=0;partition_id<partition_point_ids_vec.size();partition_id++)
        {
            vector<double> &score_vec=partition_score_vec[partition_id];
            uint32_t local_config_id=min_element(score_vec.begin(),score_vec.end())-score_vec.begin();
            double field_config_local_score=score_vec[field_config_id];
            double local_improvement=field_config_local_score>0?(field_config_local_score-score_vec[local_config_id])/field_config_local_score:0;
            uint32_t selected_config_id=local_config_id;
            if(local_config_id!=field_config_id&&local_improvement<partition_adaptive_min_improvement_experimental)
                selected_config_id=field_config_id;
            partition_prediction_config_id_vec[partition_id]=(uint8_t)selected_config_id;
            auto &selected_config=config_vec[selected_config_id];
            const char *predictor_name;
            if(!selected_config.use_fitting)
                predictor_name="idw";
            else if(selected_config.fitting_method==partition_fitting_method_e::continuous_mix)
                predictor_name="continuous_mix";
            else if(selected_config.fitting_method==partition_fitting_method_e::idw_residual)
                predictor_name="idw_residual_fitting";
            else if(selected_config.fitting_weight_power>0)
                predictor_name="weighted_fitting";
            else if(selected_config.fitting_ridge>0)
                predictor_name="ridge_fitting";
            else
                predictor_name="fitting";
            printf("partition_adaptive_config: partition=%u,sample_num=%u,config=%u,local_config=%u,field_config=%u,local_improvement=%.6f,predictor=%s,ref_num=%u,power=%.6g,fitting_weight_power=%.6g,ridge=%.6g,normalize_coord=%u,score=%.6f\n",partition_id,partition_sample_num_vec[partition_id],selected_config_id,local_config_id,field_config_id,local_improvement,predictor_name,selected_config.ref_num,selected_config.idw_power,selected_config.fitting_weight_power,selected_config.fitting_ridge,selected_config.normalize_fitting_coord,score_vec[selected_config_id]);
        }
    }
}
