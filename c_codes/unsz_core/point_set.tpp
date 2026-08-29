#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    uint32_t param_c<T_coord,T_data>::get_point_set_id(T_data val)
    {
        // 当前数据的点集划分规则。以后可以在这里扩展为更多点集，返回值必须从0开始且连续。
        return val<-1000?1:0;
    }

    template<typename T_coord,typename T_data>
    uint32_t param_c<T_coord,T_data>::predict_point_set_id(vector<int> &ref_ids,vector<double> &ref_weights,bool use_majority)
    {
        if (ref_ids.empty())
            return 0;
        if (use_majority)
        {
            uint32_t ref_num=min<uint32_t>(point_set_ref_num,ref_ids.size());
            uint32_t vote_num[2]={0,0};
            for (uint32_t i=0;i<ref_num;i++)
                vote_num[point_set_id_vec[ref_ids[i]]]++;
            if (vote_num[0]==vote_num[1])
                return point_set_id_vec[ref_ids[0]];//票数相同时采用最近参考点所属的点集
            return vote_num[1]>vote_num[0];
        }
        double weight_sum[2]={0,0};
        uint32_t ref_num=min<uint32_t>(point_set_ref_num,ref_ids.size());
        for (uint32_t i=0;i<ref_num;i++)
        {
            int ref_id=ref_ids[i];
            double weight=ref_weights[i];
            if (isinf(weight))
                return point_set_id_vec[ref_id];
            weight_sum[point_set_id_vec[ref_id]]+=weight;
        }
        return weight_sum[1]>weight_sum[0];
    }

    template<typename T_coord,typename T_data>
    double param_c<T_coord,T_data>::predict_point_set_value(vector<int> &ref_ids,vector<double> &ref_weights)
    {
        if (ref_ids.empty())
            return 0;
        double total_weight=0,pred=0;
        uint32_t ref_num=min<uint32_t>(point_set_ref_num,ref_ids.size());
        for (uint32_t i=0;i<ref_num;i++)
        {
            int ref_id=ref_ids[i];
            double weight=ref_weights[i];
            if (isinf(weight))
                return point_set_prediction_value_vec[ref_id];
            total_weight+=weight;
            pred+=weight*point_set_prediction_value_vec[ref_id];
        }
        if (total_weight==0)
            return 0;
        return min(1.0,max(0.0,pred/total_weight));
    }

    template<typename T_coord,typename T_data>
    double param_c<T_coord,T_data>::predict_point_set_value_normalized(vector<int> &ref_ids,vector<double> &ref_weights)
    {
        if (ref_ids.empty())
            return 0;
        double pred=0;
        uint32_t ref_num=min<uint32_t>(point_set_ref_num,ref_ids.size());
        for (uint32_t i=0;i<ref_num;i++)
            pred+=ref_weights[i]*point_set_prediction_value_vec[ref_ids[i]];
        return min(1.0,max(0.0,pred));
    }

    template<typename T_coord,typename T_data>
    uint32_t param_c<T_coord,T_data>::predict_point_set_id_continuous(uint32_t pred_ref_id,bool use_majority)
    {
        uint64_t begin=continuous_pred_ref_begin[pred_ref_id],end=continuous_pred_ref_begin[pred_ref_id+1];
        if (begin==end)
            return 0;
        uint64_t use_end=min<uint64_t>(end,begin+point_set_ref_num);
        if (use_majority)
        {
            uint32_t vote_num[2]={0,0};
            for (uint64_t ref_index=begin;ref_index<use_end;ref_index++)
                vote_num[point_set_id_vec[continuous_pred_ref_ids[ref_index]]]++;
            if (vote_num[0]==vote_num[1])
                return point_set_id_vec[continuous_pred_ref_ids[begin]];
            return vote_num[1]>vote_num[0];
        }
        double weight_sum[2]={0,0};
        for (uint64_t ref_index=begin;ref_index<use_end;ref_index++)
        {
            int ref_id=continuous_pred_ref_ids[ref_index];
            double weight=continuous_pred_ref_weights[ref_index];
            if (isinf(weight))
                return point_set_id_vec[ref_id];
            weight_sum[point_set_id_vec[ref_id]]+=weight;
        }
        return weight_sum[1]>weight_sum[0];
    }

    template<typename T_coord,typename T_data>
    double param_c<T_coord,T_data>::predict_point_set_value_continuous(uint32_t pred_ref_id,bool normalized)
    {
        uint64_t begin=continuous_pred_ref_begin[pred_ref_id],end=continuous_pred_ref_begin[pred_ref_id+1];
        if (begin==end)
            return 0;
        uint64_t use_end=min<uint64_t>(end,begin+point_set_ref_num);
        double total_weight=0,prediction=0;
        auto &weight_vec=normalized?continuous_pred_ref_label_normalized_weights:continuous_pred_ref_weights;
        for (uint64_t ref_index=begin;ref_index<use_end;ref_index++)
        {
            int ref_id=continuous_pred_ref_ids[ref_index];
            double weight=weight_vec[ref_index];
            if (!normalized && isinf(weight))
                return point_set_prediction_value_vec[ref_id];
            total_weight+=weight;
            prediction+=weight*point_set_prediction_value_vec[ref_id];
        }
        if (!normalized && total_weight==0)
            return 0;
        return min(1.0,max(0.0,normalized?prediction:prediction/total_weight));
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_inverse_value_from_coords(int pred_id,const vector<int> &ref_ids,vector<T_data> &data,double power)
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
    bool param_c<T_coord,T_data>::predict_inverse_value_and_loo_error(int pred_id,const vector<int> &ref_ids,vector<T_data> &data,double power,T_data &prediction,double &loo_error)
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
    bool param_c<T_coord,T_data>::predict_continuous_mix_value(int pred_id,const vector<int> &ref_ids,vector<T_data> &data,double idw_power,double fitting_weight_power,T_data &prediction)
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
    bool param_c<T_coord,T_data>::predict_idw_residual_fitting_value(int pred_id,const vector<int> &ref_ids,vector<T_data> &data,double idw_power,double residual_weight_power,T_data &prediction)
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
    bool param_c<T_coord,T_data>::predict_fitting_value(int pred_id,const vector<int> &ref_ids,vector<T_data> &data,T_data &prediction,double weight_power,double ridge,bool normalize_coord)
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
    bool param_c<T_coord,T_data>::predict_fitting_value_and_loo_error(int pred_id,const vector<int> &ref_ids,vector<T_data> &data,T_data &prediction,double &loo_error,double weight_power,double ridge,bool normalize_coord)
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
    T_data param_c<T_coord,T_data>::predict_value(int pred_id,vector<int> &ref_ids,vector<double> &ref_weights,uint32_t point_set_id,vector<T_data> &data)
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
        if (codec==point_set_codec_e::fitting || codec==point_set_codec_e::hybrid_loo)
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
            if (codec==point_set_codec_e::fitting)
            {
                T_data fitting_prediction;
                if (predict_fitting_value(pred_id,valid_ref_ids,data,fitting_prediction))
                    return fitting_prediction;
            }
            else
            {
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
    T_data param_c<T_coord,T_data>::predict_value_normalized(vector<int> &ref_ids,vector<double> &ref_weights,vector<T_data> &data)
    {
        double pred=0;
        uint32_t ref_num=min<uint32_t>(actual_ref_num,ref_ids.size());
        for (uint32_t i=0;i<ref_num;i++)
            pred+=ref_weights[i]*data[ref_ids[i]];
        return (T_data)pred;
    }

    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::predict_value_continuous(uint32_t pred_ref_id,uint32_t point_set_id,vector<T_data> &data,bool normalized)
    {
        if (point_set_codec_vec[point_set_id]!=point_set_codec_e::inverse_proportion)
            throw runtime_error("连续参考数组目前只支持反比例数值预测器");
        uint64_t begin=continuous_pred_ref_begin[pred_ref_id],end=continuous_pred_ref_begin[pred_ref_id+1];
        uint32_t valid_ref_num=0;
        double total_weight=0,prediction=0;
        auto &weight_vec=normalized?continuous_pred_ref_value_normalized_weights:continuous_pred_ref_weights;
        for (uint64_t ref_index=begin;ref_index<end;ref_index++)
        {
            int ref_id=continuous_pred_ref_ids[ref_index];
            if (!normalized && point_set_id_vec[ref_id]!=point_set_id)
                continue;
            double weight=weight_vec[ref_index];
            if (!normalized && isinf(weight))
                return data[ref_id];
            total_weight+=weight;
            prediction+=weight*data[ref_id];
            if (++valid_ref_num==actual_ref_num)
                break;
        }
        if (valid_ref_num==0)
            return 0;
        return (T_data)(normalized?prediction:prediction/total_weight);
    }

}
