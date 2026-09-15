#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    uint32_t param_c<T_coord,T_data>::get_point_set_id(T_data val)
    {
        double value=(double)val;
        for(uint32_t rule_id=0;rule_id<point_set_rule_vec.size();rule_id++)
        {
            const point_set_rule_c &rule=point_set_rule_vec[rule_id];
            bool matches=rule.operator_type==point_set_rule_operator_e::less?value<rule.value:
                rule.operator_type==point_set_rule_operator_e::less_equal?value<=rule.value:
                rule.operator_type==point_set_rule_operator_e::greater?value>rule.value:
                rule.operator_type==point_set_rule_operator_e::greater_equal?value>=rule.value:
                rule.operator_type==point_set_rule_operator_e::equal?value==rule.value:value!=rule.value;
            if(matches)
                return rule_id;
        }
        return (uint32_t)point_set_rule_vec.size();
    }

    template<typename T_coord,typename T_data>
    uint32_t param_c<T_coord,T_data>::predict_binary_label(array_view_c<int> ref_ids,array_view_c<double> ref_weights,const vector<uint8_t> &label_vec,bool use_majority)
    {
        if (ref_ids.empty())
            return 0;
        if (use_majority)
        {
            uint32_t ref_num=min<uint32_t>(actual_ref_num,ref_ids.size());
            uint32_t vote_num[2]={0,0};
            for (uint32_t i=0;i<ref_num;i++)
                vote_num[label_vec[ref_ids[i]]]++;
            if (vote_num[0]==vote_num[1])
                return label_vec[ref_ids[0]];//票数相同时采用最近参考点的标签
            return vote_num[1]>vote_num[0];
        }
        double weight_sum[2]={0,0};
        uint32_t ref_num=min<uint32_t>(actual_ref_num,ref_ids.size());
        for (uint32_t i=0;i<ref_num;i++)
        {
            int ref_id=ref_ids[i];
            double weight=ref_weights[i];
            if (isinf(weight))
                return label_vec[ref_id];
            weight_sum[label_vec[ref_id]]+=weight;
        }
        return weight_sum[1]>weight_sum[0];
    }

    template<typename T_coord,typename T_data>
    double param_c<T_coord,T_data>::predict_binary_label_value(array_view_c<int> ref_ids,array_view_c<double> ref_weights,const vector<double> &label_value_vec)
    {
        if (ref_ids.empty())
            return 0;
        double total_weight=0,pred=0;
        uint32_t ref_num=min<uint32_t>(actual_ref_num,ref_ids.size());
        for (uint32_t i=0;i<ref_num;i++)
        {
            int ref_id=ref_ids[i];
            double weight=ref_weights[i];
            if (isinf(weight))
                return label_value_vec[ref_id];
            total_weight+=weight;
            pred+=weight*label_value_vec[ref_id];
        }
        if (total_weight==0)
            return 0;
        return min(1.0,max(0.0,pred/total_weight));
    }

    template<typename T_coord,typename T_data>
    double param_c<T_coord,T_data>::predict_binary_label_value_normalized(array_view_c<int> ref_ids,array_view_c<double> ref_weights,const vector<double> &label_value_vec)
    {
        if (ref_ids.empty())
            return 0;
        double pred=0;
        uint32_t ref_num=min<uint32_t>(actual_ref_num,ref_ids.size());
        for (uint32_t i=0;i<ref_num;i++)
            pred+=ref_weights[i]*label_value_vec[ref_ids[i]];
        return min(1.0,max(0.0,pred));
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_binary_label_and_inverse_value(array_view_c<int> ref_ids,array_view_c<double> ref_weights,array_view_c<double> normalized_ref_weights,const vector<T_data> &data,uint32_t value_point_set_id,bool predict_value,uint32_t &predicted_label,T_data &predicted_value)
    {
        uint32_t label_ref_num=min<uint32_t>(actual_ref_num,ref_ids.size());
        predicted_label=0;
        predicted_value=0;
        if(!normalized_ref_weights.empty())
        {
            if(label_ref_num==4)
            {
                int ref_id0=ref_ids[0],ref_id1=ref_ids[1],ref_id2=ref_ids[2],ref_id3=ref_ids[3];
                uint32_t point_set_id0=point_set_id_vec[ref_id0],point_set_id1=point_set_id_vec[ref_id1],point_set_id2=point_set_id_vec[ref_id2],point_set_id3=point_set_id_vec[ref_id3];
                double weight0=normalized_ref_weights[0],weight1=normalized_ref_weights[1],weight2=normalized_ref_weights[2],weight3=normalized_ref_weights[3];
                double label1_weight=(point_set_id0?weight0:0)+(point_set_id1?weight1:0)+(point_set_id2?weight2:0)+(point_set_id3?weight3:0);
                double label0_weight=(point_set_id0?0:weight0)+(point_set_id1?0:weight1)+(point_set_id2?0:weight2)+(point_set_id3?0:weight3);
                predicted_label=label1_weight>label0_weight;
                bool all_value_refs_match=point_set_id0==value_point_set_id && point_set_id1==value_point_set_id && point_set_id2==value_point_set_id && point_set_id3==value_point_set_id;
                if(predict_value && all_value_refs_match)
                {
                    predicted_value=(T_data)(weight0*data[ref_id0]+weight1*data[ref_id1]+weight2*data[ref_id2]+weight3*data[ref_id3]);
                    return true;
                }
                if(!predict_value)
                    return false;
                double value_weight_sum=0,weighted_value_sum=0;
                uint32_t value_ref_num=0;
                for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
                {
                    int ref_id=ref_ids[ref_index];
                    if(point_set_id_vec[ref_id]!=value_point_set_id)
                        continue;
                    double weight=ref_weights[ref_index];
                    value_ref_num++;
                    if(isinf(weight))
                    {
                        predicted_value=data[ref_id];
                        return false;
                    }
                    value_weight_sum+=weight;
                    weighted_value_sum+=weight*data[ref_id];
                    if(value_ref_num==actual_ref_num)
                        break;
                }
                if(value_ref_num>0)
                    predicted_value=(T_data)(weighted_value_sum/value_weight_sum);
                return false;
            }
            double label_weight_sum[2]={0,0},normalized_prediction=0;
            bool all_value_refs_match=predict_value && label_ref_num>0;
            auto add_normalized_ref=[&](uint32_t ref_index)
            {
                int ref_id=ref_ids[ref_index];
                uint32_t point_set_id=point_set_id_vec[ref_id];
                if(point_set_id>1)
                    throw runtime_error("融合两点集预测遇到越界的点集编号");
                double weight=normalized_ref_weights[ref_index];
                label_weight_sum[point_set_id]+=weight;
                if(point_set_id==value_point_set_id)
                    normalized_prediction+=weight*data[ref_id];
                else
                    all_value_refs_match=false;
            };
            for(uint32_t ref_index=0;ref_index<label_ref_num;ref_index++)
                add_normalized_ref(ref_index);
            predicted_label=label_weight_sum[1]>label_weight_sum[0];
            if(all_value_refs_match)
            {
                predicted_value=(T_data)normalized_prediction;
                return true;
            }
            if(!predict_value)
                return false;
            double value_weight_sum=0,weighted_value_sum=0;
            uint32_t value_ref_num=0;
            for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
            {
                int ref_id=ref_ids[ref_index];
                if(point_set_id_vec[ref_id]!=value_point_set_id)
                    continue;
                double weight=ref_weights[ref_index];
                value_ref_num++;
                if(isinf(weight))
                {
                    predicted_value=data[ref_id];
                    return false;
                }
                value_weight_sum+=weight;
                weighted_value_sum+=weight*data[ref_id];
                if(value_ref_num==actual_ref_num)
                    break;
            }
            if(value_ref_num>0)
                predicted_value=(T_data)(weighted_value_sum/value_weight_sum);
            return false;
        }
        double label_weight_sum[2]={0,0},value_weight_sum=0,weighted_value_sum=0;
        uint32_t value_ref_num=0;
        bool label_fixed=false,value_fixed=false;
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            uint32_t point_set_id=point_set_id_vec[ref_id];
            if(point_set_id>1)
                throw runtime_error("融合两点集预测遇到越界的点集编号");
            double weight=ref_weights[ref_index];
            if(ref_index<label_ref_num && !label_fixed)
            {
                if(isinf(weight))
                {
                    predicted_label=point_set_id;
                    label_fixed=true;
                }
                else
                    label_weight_sum[point_set_id]+=weight;
            }
            if(predict_value && point_set_id==value_point_set_id && value_ref_num<actual_ref_num)
            {
                value_ref_num++;
                if(!value_fixed)
                {
                    if(isinf(weight))
                    {
                        predicted_value=data[ref_id];
                        value_fixed=true;
                    }
                    else
                    {
                        value_weight_sum+=weight;
                        weighted_value_sum+=weight*data[ref_id];
                    }
                }
            }
        }
        if(!label_fixed)
            predicted_label=label_weight_sum[1]>label_weight_sum[0];
        if(predict_value && !value_fixed && value_ref_num>0)
            predicted_value=(T_data)(weighted_value_sum/value_weight_sum);
        return false;
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_known_point_set_inverse_value(array_view_c<int> ref_ids,array_view_c<double> ref_weights,array_view_c<double> normalized_ref_weights,const vector<T_data> &data,uint32_t value_point_set_id,bool predict_value,T_data &predicted_value)
    {
        predicted_value=0;
        if(!predict_value)
            return false;
        uint32_t normalized_ref_num=min<uint32_t>(actual_ref_num,ref_ids.size());
        if(!normalized_ref_weights.empty()&&normalized_ref_num>0)
        {
            bool all_value_refs_match=true;
            for(uint32_t ref_index=0;ref_index<normalized_ref_num;ref_index++)
                all_value_refs_match&=point_set_id_vec[ref_ids[ref_index]]==value_point_set_id;
            if(all_value_refs_match)
            {
                double prediction=0;
                if(normalized_ref_num==4)
                {
                    prediction+=normalized_ref_weights[0]*data[ref_ids[0]];
                    prediction+=normalized_ref_weights[1]*data[ref_ids[1]];
                    prediction+=normalized_ref_weights[2]*data[ref_ids[2]];
                    prediction+=normalized_ref_weights[3]*data[ref_ids[3]];
                }
                else
                    for(uint32_t ref_index=0;ref_index<normalized_ref_num;ref_index++)
                        prediction+=normalized_ref_weights[ref_index]*data[ref_ids[ref_index]];
                predicted_value=(T_data)prediction;
                return true;
            }
        }
        double value_weight_sum=0,weighted_value_sum=0;
        uint32_t value_ref_num=0;
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            if(point_set_id_vec[ref_id]!=value_point_set_id)
                continue;
            double weight=ref_weights[ref_index];
            value_ref_num++;
            if(isinf(weight))
            {
                predicted_value=data[ref_id];
                return false;
            }
            value_weight_sum+=weight;
            weighted_value_sum+=weight*data[ref_id];
            if(value_ref_num==actual_ref_num)
                break;
        }
        if(value_ref_num>0)
            predicted_value=(T_data)(weighted_value_sum/value_weight_sum);
        return false;
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_inverse_value_from_coords(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double power)
    {
        if (ref_ids.empty())
            return 0;
        if (!(power>0))
            power=inverse_distance_power;
        double total_weight=0,prediction=0;
        for (int ref_id:ref_ids)
        {
            double squared_dist=(double)get_dist_to_coord(coord_array[pred_id],coord_array[ref_id],dim);
            if (squared_dist==0)
                return data[ref_id];
            double weight=pow(squared_dist,-power/2);
            if (isinf(weight))
                return data[ref_id];
            total_weight+=weight;
            prediction+=weight*data[ref_id];
        }
        return total_weight>0?(T_data)(prediction/total_weight):0;
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_inverse_value_and_loo_error(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double power,T_data &prediction,double &loo_error)
    {
        if (ref_ids.size()<2)
            return false;
        prediction=predict_inverse_value_from_coords(pred_id,ref_ids,data,power);
        if (!isfinite(prediction))
            return false;
        vector<int> loo_ref_ids;
        loo_ref_ids.reserve(ref_ids.size()-1);
        loo_error=0;
        for (uint32_t held_id=0;held_id<ref_ids.size();held_id++)
        {
            loo_ref_ids.clear();
            for (uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
                if (ref_index!=held_id)
                    loo_ref_ids.push_back(ref_ids[ref_index]);
            int held_ref_id=ref_ids[held_id];
            T_data loo_prediction=predict_inverse_value_from_coords(held_ref_id,loo_ref_ids,data,power);
            loo_error+=abs((double)data[held_ref_id]-loo_prediction);
        }
        return isfinite(loo_error);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_continuous_mix_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double idw_power,double fitting_weight_power,T_data &prediction)
    {
        T_data idw_prediction,fitting_prediction;
        double idw_loo_error,fitting_loo_error;
        if (!predict_inverse_value_and_loo_error(pred_id,ref_ids,data,idw_power,idw_prediction,idw_loo_error))
            return false;
        if (!predict_fitting_value_and_loo_error(pred_id,ref_ids,data,fitting_prediction,fitting_loo_error,fitting_weight_power,0,true))
            return false;
        double error_sum=idw_loo_error+fitting_loo_error;
        double fitting_ratio=error_sum>0?idw_loo_error/error_sum:0.5;
        prediction=(T_data)((1-fitting_ratio)*idw_prediction+fitting_ratio*fitting_prediction);
        return isfinite(prediction);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_idw_residual_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,double idw_power,double residual_weight_power,T_data &prediction)
    {
        if (ref_ids.size()<=dim)
            return false;
        T_data idw_prediction=predict_inverse_value_from_coords(pred_id,ref_ids,data,idw_power);
        if (!isfinite(idw_prediction))
            return false;
        vector<double> residual_vec(ref_ids.size());
        vector<int> loo_ref_ids;
        loo_ref_ids.reserve(ref_ids.size()-1);
        for (uint32_t held_id=0;held_id<ref_ids.size();held_id++)
        {
            loo_ref_ids.clear();
            for (uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
                if (ref_index!=held_id)
                    loo_ref_ids.push_back(ref_ids[ref_index]);
            int held_ref_id=ref_ids[held_id];
            T_data loo_prediction=predict_inverse_value_from_coords(held_ref_id,loo_ref_ids,data,idw_power);
            residual_vec[held_id]=(double)data[held_ref_id]-loo_prediction;
        }
        uint32_t system_size=dim+1;
        vector<double> scale(dim,0),squared_distance_vec(ref_ids.size());
        double min_squared_distance=numeric_limits<double>::infinity();
        for (uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            for (uint32_t d=0;d<dim;d++)
            {
                double dc=(double)coord_array[ref_id][d]-coord_array[pred_id][d];
                scale[d]=max(scale[d],abs(dc));
                squared_distance_vec[ref_index]+=dc*dc;
            }
            if (squared_distance_vec[ref_index]==0)
            {
                prediction=data[ref_id];
                return true;
            }
            min_squared_distance=min(min_squared_distance,squared_distance_vec[ref_index]);
        }
        for (double value:scale)
            if (!(value>0))
                return false;
        vector<vector<double>> A(system_size,vector<double>(system_size,0));
        vector<double> b(system_size,0),x;
        for (uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            vector<double> dc(dim);
            for (uint32_t d=0;d<dim;d++)
                dc[d]=((double)coord_array[ref_id][d]-coord_array[pred_id][d])/scale[d];
            double weight=residual_weight_power>0?pow(squared_distance_vec[ref_index]/min_squared_distance,-residual_weight_power/2):1;
            if (!(weight>0) || !isfinite(weight))
                return false;
            for (uint32_t row=0;row<dim;row++)
            {
                for (uint32_t col=0;col<dim;col++)
                    A[row][col]+=weight*dc[row]*dc[col];
                A[row][dim]+=weight*dc[row];
                A[dim][row]+=weight*dc[row];
                b[row]+=weight*dc[row]*residual_vec[ref_index];
            }
            A[dim][dim]+=weight;
            b[dim]+=weight*residual_vec[ref_index];
        }
        if (!solve_linear_system(move(A),move(b),x) || !isfinite(x[dim]))
            return false;
        prediction=(T_data)((double)idw_prediction+x[dim]);
        return isfinite(prediction);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction,double weight_power,double ridge,bool normalize_coord)
    {
        if (ref_ids.size()<=dim)
            return false;
        uint32_t system_size=dim+1;
        vector<vector<double>> A(system_size,vector<double>(system_size,0));
        vector<double> b(system_size,0),x;
        vector<double> scale(dim,1);
        double min_squared_distance=numeric_limits<double>::infinity();
        if (normalize_coord || weight_power>0 || ridge>0)
        {
            fill(scale.begin(),scale.end(),0);
            for (int ref_id:ref_ids)
            {
                double squared_distance=0;
                for (uint32_t d=0;d<dim;d++)
                {
                    double dc=(double)coord_array[ref_id][d]-coord_array[pred_id][d];
                    scale[d]=max(scale[d],abs(dc));
                    squared_distance+=dc*dc;
                }
                if (squared_distance==0)
                {
                    prediction=data[ref_id];
                    return true;
                }
                min_squared_distance=min(min_squared_distance,squared_distance);
            }
            for (double value:scale)
                if (!(value>0))
                    return false;
        }
        double weight_sum=0;
        for (int ref_id:ref_ids)
        {
            vector<double> dc(dim);
            double squared_distance=0;
            for (uint32_t d=0;d<dim;d++)
            {
                double original_dc=(double)coord_array[ref_id][d]-coord_array[pred_id][d];
                squared_distance+=original_dc*original_dc;
                dc[d]=original_dc/scale[d];
            }
            double weight=weight_power>0?pow(squared_distance/min_squared_distance,-weight_power/2):1;
            if (!(weight>0) || !isfinite(weight))
                return false;
            weight_sum+=weight;
            for (uint32_t row=0;row<dim;row++)
            {
                for (uint32_t col=0;col<dim;col++)
                    A[row][col]+=weight*dc[row]*dc[col];
                A[row][dim]+=weight*dc[row];
                A[dim][row]+=weight*dc[row];
                b[row]+=weight*dc[row]*data[ref_id];
            }
            A[dim][dim]+=weight;
            b[dim]+=weight*data[ref_id];
        }
        for (uint32_t d=0;d<dim;d++)
            A[d][d]+=ridge*weight_sum;
        if (!solve_linear_system(move(A),move(b),x) || !isfinite(x[dim]))
            return false;
        prediction=(T_data)x[dim];
        return isfinite(prediction);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_quadratic_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction,double ridge)
    {
        uint32_t system_size=(dim+1)*(dim+2)/2;
        uint32_t ref_num=(uint32_t)ref_ids.size();
        if(ref_num<system_size)
            return false;
        vector<double> scale(dim,0);
        for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            double squared_distance=0;
            for(uint32_t d=0;d<dim;d++)
            {
                double dc=(double)coord_array[ref_id][d]-coord_array[pred_id][d];
                scale[d]=max(scale[d],abs(dc));
                squared_distance+=dc*dc;
            }
            if(squared_distance==0)
            {
                prediction=data[ref_id];
                return true;
            }
        }
        for(double value:scale)
            if(!(value>0))
                return false;
        vector<vector<double>> A(system_size,vector<double>(system_size,0));
        vector<double> b(system_size,0),feature(system_size),x;
        uint32_t constant_id=system_size-1;
        for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            uint32_t feature_id=0;
            for(uint32_t d=0;d<dim;d++)
                feature[feature_id++]=((double)coord_array[ref_id][d]-coord_array[pred_id][d])/scale[d];
            for(uint32_t row=0;row<dim;row++)
                for(uint32_t col=row;col<dim;col++)
                    feature[feature_id++]=feature[row]*feature[col];
            feature[feature_id++]=1;
            if(feature_id!=system_size)
                throw runtime_error("二次拟合的特征数量不正确");
            for(uint32_t row=0;row<system_size;row++)
            {
                for(uint32_t col=0;col<system_size;col++)
                    A[row][col]+=feature[row]*feature[col];
                b[row]+=feature[row]*data[ref_id];
            }
        }
        double ridge_scale=ridge*ref_num;
        for(uint32_t row=0;row<constant_id;row++)
            A[row][row]+=ridge_scale;
        if(!solve_linear_system(move(A),move(b),x) || !isfinite(x[constant_id]))
            return false;
        prediction=(T_data)x[constant_id];
        return isfinite(prediction);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::calculate_rbf_fitting_weights(int pred_id,array_view_c<int> ref_ids,vector<double> &weights)
    {
        uint32_t ref_num=(uint32_t)ref_ids.size();
        if(ref_num<2)
            return false;
        weights.assign(ref_num,0);
        vector<double> target_squared_distance(ref_num,0);
        double mean_squared_distance=0;
        for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            for(uint32_t d=0;d<dim;d++)
            {
                double dc=(double)coord_array[ref_id][d]-coord_array[pred_id][d];
                target_squared_distance[ref_index]+=dc*dc;
            }
            if(target_squared_distance[ref_index]==0)
            {
                weights[ref_index]=1;
                return true;
            }
            mean_squared_distance+=target_squared_distance[ref_index];
        }
        mean_squared_distance/=ref_num;
        if(!(mean_squared_distance>0) || !isfinite(mean_squared_distance))
            return false;
        auto kernel=[&](double normalized_squared_distance)
        {
            double scaled_squared_distance=rbf_shape*rbf_shape*normalized_squared_distance;
            if(rbf_kernel=="gaussian")
                return exp(-scaled_squared_distance);
            if(rbf_kernel=="inverse_multiquadric")
                return 1/sqrt(1+scaled_squared_distance);
            double scaled_distance=sqrt(scaled_squared_distance);
            if(rbf_kernel=="exponential")
                return exp(-scaled_distance);
            if(rbf_kernel=="matern32")
            {
                double value=sqrt(3.0)*scaled_distance;
                return (1+value)*exp(-value);
            }
            if(rbf_kernel=="matern52")
            {
                double value=sqrt(5.0)*scaled_distance;
                return (1+value+value*value/3)*exp(-value);
            }
            throw runtime_error("未知的RBF核函数："+rbf_kernel);
        };
        uint32_t trend_size=1;
        if(rbf_polynomial_degree>=1)
            trend_size+=dim;
        if(rbf_polynomial_degree>=2)
            trend_size+=dim*(dim+1)/2;
        if(ref_num<trend_size)
            return false;
        uint32_t system_size=ref_num+trend_size;
        vector<vector<double>> A(system_size,vector<double>(system_size,0));
        vector<double> target_feature(system_size,0),x,trend_feature(trend_size);
        double distance_scale=sqrt(mean_squared_distance);
        for(uint32_t row=0;row<ref_num;row++)
        {
            int row_ref_id=ref_ids[row];
            for(uint32_t col=0;col<ref_num;col++)
            {
                int col_ref_id=ref_ids[col];
                double squared_distance=0;
                for(uint32_t d=0;d<dim;d++)
                {
                    double dc=(double)coord_array[row_ref_id][d]-coord_array[col_ref_id][d];
                    squared_distance+=dc*dc;
                }
                A[row][col]=kernel(squared_distance/mean_squared_distance);
            }
            A[row][row]+=rbf_nugget;
            uint32_t trend_id=0;
            trend_feature[trend_id++]=1;
            if(rbf_polynomial_degree>=1)
                for(uint32_t d=0;d<dim;d++)
                    trend_feature[trend_id++]=((double)coord_array[row_ref_id][d]-coord_array[pred_id][d])/distance_scale;
            if(rbf_polynomial_degree>=2)
                for(uint32_t first_dim=0;first_dim<dim;first_dim++)
                    for(uint32_t second_dim=first_dim;second_dim<dim;second_dim++)
                        trend_feature[trend_id++]=trend_feature[1+first_dim]*trend_feature[1+second_dim];
            if(trend_id!=trend_size)
                throw runtime_error("RBF趋势项数量不正确");
            for(uint32_t current_trend_id=0;current_trend_id<trend_size;current_trend_id++)
            {
                A[row][ref_num+current_trend_id]=trend_feature[current_trend_id];
                A[ref_num+current_trend_id][row]=trend_feature[current_trend_id];
            }
            target_feature[row]=kernel(target_squared_distance[row]/mean_squared_distance);
        }
        target_feature[ref_num]=1;
        if(!solve_linear_system(move(A),move(target_feature),x))
            return false;
        for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
        {
            if(!isfinite(x[ref_index]))
                return false;
            weights[ref_index]=x[ref_index];
        }
        return true;
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_rbf_fitting_value(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction)
    {
        vector<double> weights;
        if(!calculate_rbf_fitting_weights(pred_id,ref_ids,weights))
            return false;
        double predicted_value=0;
        for(uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
            predicted_value+=weights[ref_index]*data[ref_ids[ref_index]];
        prediction=(T_data)predicted_value;
        return isfinite(prediction);
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::predict_fitting_value_and_loo_error(int pred_id,array_view_c<int> ref_ids,vector<T_data> &data,T_data &prediction,double &loo_error,double weight_power,double ridge,bool normalize_coord)
    {
        uint32_t system_size=dim+1;
        if (ref_ids.size()<=system_size)
            return false;
        vector<double> scale(dim,normalize_coord?0:1),squared_distance_vec(ref_ids.size());
        double min_squared_distance=numeric_limits<double>::infinity();
        for (uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            int ref_id=ref_ids[ref_index];
            for (uint32_t d=0;d<dim;d++)
            {
                double dc=(double)coord_array[ref_id][d]-coord_array[pred_id][d];
                if (normalize_coord)
                    scale[d]=max(scale[d],abs(dc));
                squared_distance_vec[ref_index]+=dc*dc;
            }
            if (squared_distance_vec[ref_index]==0)
                return false;
            min_squared_distance=min(min_squared_distance,squared_distance_vec[ref_index]);
        }
        if (normalize_coord)
            for (double value:scale)
                if (!(value>0))
                    return false;
        vector<vector<double>> design(ref_ids.size(),vector<double>(system_size,1));
        vector<double> weight_vec(ref_ids.size(),1);
        for (uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            for (uint32_t d=0;d<dim;d++)
                design[ref_index][d]=((double)coord_array[ref_ids[ref_index]][d]-coord_array[pred_id][d])/scale[d];
            if (weight_power>0)
                weight_vec[ref_index]=pow(squared_distance_vec[ref_index]/min_squared_distance,-weight_power/2);
            if (!(weight_vec[ref_index]>0) || !isfinite(weight_vec[ref_index]))
                return false;
        }
        vector<vector<double>> augmented(system_size,vector<double>(system_size*2+1,0));
        double weight_sum=accumulate(weight_vec.begin(),weight_vec.end(),0.0);
        for (uint32_t row=0;row<system_size;row++)
        {
            for (uint32_t col=0;col<system_size;col++)
                for (uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
                    augmented[row][col]+=weight_vec[ref_index]*design[ref_index][row]*design[ref_index][col];
            for (uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
                augmented[row][system_size]+=weight_vec[ref_index]*design[ref_index][row]*data[ref_ids[ref_index]];
            if (row<dim)
                augmented[row][row]+=ridge*weight_sum;
            augmented[row][system_size+1+row]=1;
        }
        for (uint32_t col=0;col<system_size;col++)
        {
            uint32_t pivot=col;
            for (uint32_t row=col+1;row<system_size;row++)
                if (abs(augmented[row][col])>abs(augmented[pivot][col]))
                    pivot=row;
            if (abs(augmented[pivot][col])<1e-12)
                return false;
            if (pivot!=col)
                swap(augmented[pivot],augmented[col]);
            double divisor=augmented[col][col];
            for (uint32_t i=0;i<augmented[col].size();i++)
                augmented[col][i]/=divisor;
            for (uint32_t row=0;row<system_size;row++)
            {
                if (row==col)
                    continue;
                double factor=augmented[row][col];
                for (uint32_t i=0;i<augmented[row].size();i++)
                    augmented[row][i]-=factor*augmented[col][i];
            }
        }
        vector<double> coefficient(system_size);
        for (uint32_t row=0;row<system_size;row++)
            coefficient[row]=augmented[row][system_size];
        prediction=(T_data)coefficient[dim];
        if (!isfinite(prediction))
            return false;
        loo_error=0;
        for (uint32_t ref_index=0;ref_index<ref_ids.size();ref_index++)
        {
            double fitted_value=0,leverage=0;
            for (uint32_t row=0;row<system_size;row++)
            {
                fitted_value+=design[ref_index][row]*coefficient[row];
                for (uint32_t col=0;col<system_size;col++)
                    leverage+=weight_vec[ref_index]*design[ref_index][row]*augmented[row][system_size+1+col]*design[ref_index][col];
            }
            double denominator=1-leverage;
            if (!(denominator>1e-8) || !isfinite(denominator))
                return false;
            loo_error+=abs(((double)data[ref_ids[ref_index]]-fitted_value)/denominator);
        }
        return isfinite(loo_error);
    }

    template<typename T_coord,typename T_data>
    __attribute__((always_inline)) inline T_data param_c<T_coord,T_data>::predict_single_point_set_rbf_value(uint32_t pred_ref_id,vector<T_data> &data)
    {
        uint64_t ref_begin=pred_ref_begin[pred_ref_id],ref_end=pred_ref_begin[pred_ref_id+1];
        uint32_t ref_num=min<uint32_t>(actual_ref_num,(uint32_t)(ref_end-ref_begin));
        if(ref_num==0)
            return 0;
        const int *ref_ids=pred_ref_ids.data()+ref_begin;
        if(isfinite(pred_ref_rbf_weights[ref_begin]))
        {
            const double *weights=pred_ref_rbf_weights.data()+ref_begin;
            double prediction=0;
            if(ref_num==10)
                prediction=weights[0]*data[ref_ids[0]]+weights[1]*data[ref_ids[1]]+weights[2]*data[ref_ids[2]]+weights[3]*data[ref_ids[3]]+weights[4]*data[ref_ids[4]]+weights[5]*data[ref_ids[5]]+weights[6]*data[ref_ids[6]]+weights[7]*data[ref_ids[7]]+weights[8]*data[ref_ids[8]]+weights[9]*data[ref_ids[9]];
            else
                for(uint32_t ref_index=0;ref_index<ref_num;ref_index++)
                    prediction+=weights[ref_index]*data[ref_ids[ref_index]];
            if(isfinite(prediction))
                return (T_data)prediction;
        }
        array_view_c<int> ref_id_view(pred_ref_ids.data()+ref_begin,(uint32_t)(ref_end-ref_begin));
        array_view_c<double> ref_weight_view(pred_ref_weights.data()+ref_begin,(uint32_t)(ref_end-ref_begin));
        return predict_value(pred_ref_id,pred_target_ids[pred_ref_id],ref_id_view,ref_weight_view,0,data);
    }

    template<typename T_coord,typename T_data>
    __attribute__((always_inline)) inline T_data param_c<T_coord,T_data>::predict_single_point_set_rbf10_value(uint32_t pred_ref_id,uint64_t ref_begin,vector<T_data> &data)
    {
        const int *ref_ids=pred_ref_ids.data()+ref_begin;
        const double *weights=pred_ref_rbf_weights.data()+ref_begin;
        double prediction=weights[0]*data[ref_ids[0]]+weights[1]*data[ref_ids[1]]+weights[2]*data[ref_ids[2]]+weights[3]*data[ref_ids[3]]+weights[4]*data[ref_ids[4]]+weights[5]*data[ref_ids[5]]+weights[6]*data[ref_ids[6]]+weights[7]*data[ref_ids[7]]+weights[8]*data[ref_ids[8]]+weights[9]*data[ref_ids[9]];
        if(__builtin_expect(isfinite(prediction),1))
            return (T_data)prediction;
        return predict_single_point_set_rbf_value(pred_ref_id,data);
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_value(uint32_t pred_ref_id,int pred_id,array_view_c<int> ref_ids,array_view_c<double> ref_weights,uint32_t point_set_id,vector<T_data> &data)
    {
        point_set_codec_e codec=point_set_codec_vec[point_set_id];
        auto predict_current_inverse=[&]()
        {
            double total_weight=0,prediction=0;
            uint32_t valid_ref_num=0;
            for (uint32_t i=0;i<ref_ids.size();i++)
            {
                int ref_id=ref_ids[i];
                if (point_set_id_vec[ref_id]!=point_set_id)
                    continue;
                double weight=ref_weights[i];
                if (isinf(weight))
                    return data[ref_id];
                total_weight+=weight;
                prediction+=weight*data[ref_id];
                if (++valid_ref_num==actual_ref_num)
                    break;
            }
            return valid_ref_num>0?(T_data)(prediction/total_weight):(T_data)0;
        };
        if(codec==point_set_codec_e::rbf_fitting)
        {
            uint32_t expected_ref_num=min<uint32_t>(actual_ref_num,ref_ids.size());
            bool can_use_precomputed=pred_ref_rbf_weights.size()==pred_ref_ids.size() && expected_ref_num>0;
            uint64_t rbf_weight_begin=pred_ref_begin[pred_ref_id];
            if(can_use_precomputed && point_set_codec_vec.size()==1)
                can_use_precomputed=isfinite(pred_ref_rbf_weights[rbf_weight_begin]);
            else
                for(uint32_t ref_index=0;can_use_precomputed && ref_index<expected_ref_num;ref_index++)
                    can_use_precomputed=point_set_id_vec[ref_ids[ref_index]]==point_set_id && isfinite(pred_ref_rbf_weights[rbf_weight_begin+ref_index]);
            if(can_use_precomputed)
            {
                double rbf_prediction=0;
                const double *rbf_weights=pred_ref_rbf_weights.data()+rbf_weight_begin;
                const int *rbf_ref_ids=ref_ids.begin();
                if(expected_ref_num==10)
                    rbf_prediction=rbf_weights[0]*data[rbf_ref_ids[0]]+rbf_weights[1]*data[rbf_ref_ids[1]]+rbf_weights[2]*data[rbf_ref_ids[2]]+rbf_weights[3]*data[rbf_ref_ids[3]]+rbf_weights[4]*data[rbf_ref_ids[4]]+rbf_weights[5]*data[rbf_ref_ids[5]]+rbf_weights[6]*data[rbf_ref_ids[6]]+rbf_weights[7]*data[rbf_ref_ids[7]]+rbf_weights[8]*data[rbf_ref_ids[8]]+rbf_weights[9]*data[rbf_ref_ids[9]];
                else
                    for(uint32_t ref_index=0;ref_index<expected_ref_num;ref_index++)
                        rbf_prediction+=rbf_weights[ref_index]*data[rbf_ref_ids[ref_index]];
                if(isfinite(rbf_prediction))
                    return (T_data)rbf_prediction;
            }
            if(use_rbf_filtered_fallback_nearest_experimental)
            {
                for(int ref_id:ref_ids)
                    if(point_set_id_vec[ref_id]==point_set_id)
                        return data[ref_id];
                return 0;
            }
        }
        if(codec==point_set_codec_e::fitting || codec==point_set_codec_e::hybrid_loo || codec==point_set_codec_e::quadratic_fitting || codec==point_set_codec_e::rbf_fitting)
        {
            vector<int> valid_ref_ids;
            valid_ref_ids.reserve(actual_ref_num);
            for (int ref_id:ref_ids)
                if (point_set_id_vec[ref_id]==point_set_id)
                {
                    valid_ref_ids.push_back(ref_id);
                    if (valid_ref_ids.size()==actual_ref_num)
                        break;
                }
            if (valid_ref_ids.empty())
                return 0;
            if(codec==point_set_codec_e::quadratic_fitting)
            {
                T_data quadratic_prediction;
                if(predict_quadratic_fitting_value(pred_id,valid_ref_ids,data,quadratic_prediction))
                    return quadratic_prediction;
            }
            else if(codec==point_set_codec_e::rbf_fitting)
            {
                T_data rbf_prediction;
                if(predict_rbf_fitting_value(pred_id,valid_ref_ids,data,rbf_prediction))
                    return rbf_prediction;
            }
            else if(codec==point_set_codec_e::fitting)
            {
                T_data fitting_prediction;
                if (predict_fitting_value(pred_id,valid_ref_ids,data,fitting_prediction))
                    return fitting_prediction;
            }
            else
            {
#pragma omp atomic update
                hybrid_prediction_num++;
                auto record_selector=[&](uint8_t selector)
                {
                    if (hybrid_selector_vec.size()==point_num)
                        hybrid_selector_vec[pred_id]=selector;
                };
                T_data inverse_prediction=predict_current_inverse();
                T_data fitting_prediction;
                double fitting_loo_error;
                if (!predict_fitting_value_and_loo_error(pred_id,valid_ref_ids,data,fitting_prediction,fitting_loo_error))
                {
                    record_selector(0);
                    return inverse_prediction;
                }
                double inverse_loo_error=0;
                vector<int> loo_ref_ids;
                loo_ref_ids.reserve(valid_ref_ids.size()-1);
                for (uint32_t held_id=0;held_id<valid_ref_ids.size();held_id++)
                {
                    loo_ref_ids.clear();
                    for (uint32_t ref_id=0;ref_id<valid_ref_ids.size();ref_id++)
                        if (ref_id!=held_id)
                            loo_ref_ids.push_back(valid_ref_ids[ref_id]);
                    int held_original_id=valid_ref_ids[held_id];
                    T_data loo_inverse_prediction=predict_inverse_value_from_coords(held_original_id,loo_ref_ids,data);
                    inverse_loo_error+=abs((double)data[held_original_id]-loo_inverse_prediction);
                }
                if (fitting_loo_error<inverse_loo_error)
                {
#pragma omp atomic update
                    hybrid_fitting_prediction_num++;
                    record_selector(1);
                    return fitting_prediction;
                }
                record_selector(0);
                return inverse_prediction;
            }
        }
        else if (codec!=point_set_codec_e::inverse_proportion)
            throw runtime_error("点集使用了不支持的数值预测器");
        return predict_current_inverse();
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_value_normalized(array_view_c<int> ref_ids,array_view_c<double> ref_weights,vector<T_data> &data)
    {
        double pred=0;
        uint32_t ref_num=min<uint32_t>(actual_ref_num,ref_ids.size());
        if(ref_num==4)
        {
            pred+=ref_weights[0]*data[ref_ids[0]];
            pred+=ref_weights[1]*data[ref_ids[1]];
            pred+=ref_weights[2]*data[ref_ids[2]];
            pred+=ref_weights[3]*data[ref_ids[3]];
        }
        else
            for (uint32_t i=0;i<ref_num;i++)
                pred+=ref_weights[i]*data[ref_ids[i]];
        return (T_data)pred;
    }

}
