#pragma once

namespace unsz
{
    inline void merge_reconstruction_statistics(reconstruction_statistics_c&total,const reconstruction_statistics_c&part)
    {
        total.point_num+=part.point_num;
        total.squared_error+=part.squared_error;
        total.minimum=min(total.minimum,part.minimum);
        total.maximum=max(total.maximum,part.maximum);
        if(part.max_error_ratio>total.max_error_ratio)
        {
            total.max_error_ratio=part.max_error_ratio;
            total.max_error=part.max_error;
            total.max_error_abs_eb=part.max_error_abs_eb;
            total.max_error_point_id=part.max_error_point_id;
            total.max_error_point_set_id=part.max_error_point_set_id;
        }
    }

    inline double reconstruction_psnr(const reconstruction_statistics_c&statistics)
    {
        if(statistics.point_num==0)
            return numeric_limits<double>::quiet_NaN();
        double mse=statistics.squared_error/statistics.point_num;
        if(mse==0)
            return numeric_limits<double>::infinity();
        return 10*log10(pow(statistics.maximum-statistics.minimum,2)/mse);
    }

    inline double normalized_reconstruction_psnr(double normalized_squared_error,uint64_t value_num)
    {
        if(value_num==0)
            return numeric_limits<double>::quiet_NaN();
        if(normalized_squared_error==0)
            return numeric_limits<double>::infinity();
        return -10*log10(normalized_squared_error/value_num);
    }

    template<typename T_data>
    double full_data_range(const vector<uint32_t>&point_set_point_num_vec,const vector<T_data>&point_set_min_data_vec,const vector<T_data>&point_set_max_data_vec)
    {
        if(point_set_point_num_vec.size()!=point_set_min_data_vec.size()||point_set_point_num_vec.size()!=point_set_max_data_vec.size())
            throw runtime_error("点集数量和值域数量不一致");
        double minimum=numeric_limits<double>::infinity(),maximum=-numeric_limits<double>::infinity();
        for(uint32_t point_set_id=0;point_set_id<point_set_point_num_vec.size();point_set_id++)
            if(point_set_point_num_vec[point_set_id]>0)
            {
                minimum=min(minimum,(double)point_set_min_data_vec[point_set_id]);
                maximum=max(maximum,(double)point_set_max_data_vec[point_set_id]);
            }
        if(!isfinite(minimum)||!isfinite(maximum)||maximum<minimum)
            throw runtime_error("无法计算完整数据值域");
        return maximum-minimum;
    }

    template<typename T_data>
    void add_normalized_reconstruction_error(double&normalized_squared_error,uint64_t&normalized_value_num,const reconstruction_statistics_c&statistics,const vector<uint32_t>&point_set_point_num_vec,const vector<T_data>&point_set_min_data_vec,const vector<T_data>&point_set_max_data_vec)
    {
        double data_range=full_data_range(point_set_point_num_vec,point_set_min_data_vec,point_set_max_data_vec);
        if(data_range>0)
            normalized_squared_error+=statistics.squared_error/(data_range*data_range);
        else if(statistics.squared_error!=0)
            throw runtime_error("常量文件未被精确恢复");
        normalized_value_num+=statistics.point_num;
    }

    template<typename T_coord,typename T_data>
    reconstruction_statistics_c param_c<T_coord,T_data>::calculate_reconstruction_statistics(const T_data*original,const T_data*reconstructed,uint32_t value_num,uint32_t point_offset) const
    {
        reconstruction_statistics_c result;
        result.point_num=value_num;
        uint32_t primary_point_set_id=0;
        while(point_set_point_num_vec[primary_point_set_id]==0)
            primary_point_set_id++;
        for(uint32_t value_id=0;value_id<value_num;value_id++)
        {
            uint32_t point_id=point_offset+value_id;
            double error=(double)original[value_id]-reconstructed[value_id];
            result.squared_error+=error*error;
            uint32_t point_set_id=point_set_id_vec.empty()?0:point_set_id_vec[point_id];
            double abs_eb=point_set_abs_eb_vec[point_set_id];
            double error_ratio;
            if(abs_eb>0)
                error_ratio=abs(error)/abs_eb;
            else if(error==0)
                error_ratio=0;
            else
                error_ratio=numeric_limits<double>::infinity();
            if(error_ratio>result.max_error_ratio)
            {
                result.max_error_ratio=error_ratio;
                result.max_error=error;
                result.max_error_abs_eb=abs_eb;
                result.max_error_point_id=point_id;
                result.max_error_point_set_id=point_set_id;
            }
            if(point_set_id==primary_point_set_id)
            {
                result.minimum=min(result.minimum,(double)original[value_id]);
                result.maximum=max(result.maximum,(double)original[value_id]);
            }
        }
        return result;
    }

    template<typename T_coord,typename T_data>
    reconstruction_statistics_c param_c<T_coord,T_data>::calculate_prediction_order_reconstruction_statistics() const
    {
        reconstruction_statistics_c result;
        result.point_num=point_num;
        uint32_t primary_point_set_id=0;
        while(point_set_point_num_vec[primary_point_set_id]==0)
            primary_point_set_id++;
        for(const prediction_structure_c<T_calc>&structure:prediction_structure_vec)
            for(uint32_t prediction_index=0;prediction_index<structure.target_ids.size();prediction_index++)
            {
                uint32_t point_id=(uint32_t)structure.target_ids[prediction_index];
                double error=(double)current_data[point_id]-prediction_order_decompressed_data[structure.prediction_order_offset+prediction_index];
                result.squared_error+=error*error;
                uint32_t point_set_id=point_set_id_vec.empty()?0:point_set_id_vec[point_id];
                double abs_eb=point_set_abs_eb_vec[point_set_id];
                double error_ratio;
                if(abs_eb>0)
                    error_ratio=abs(error)/abs_eb;
                else if(error==0)
                    error_ratio=0;
                else
                    error_ratio=numeric_limits<double>::infinity();
                if(error_ratio>result.max_error_ratio)
                {
                    result.max_error_ratio=error_ratio;
                    result.max_error=error;
                    result.max_error_abs_eb=abs_eb;
                    result.max_error_point_id=point_id;
                    result.max_error_point_set_id=point_set_id;
                }
                if(point_set_id==primary_point_set_id)
                {
                    result.minimum=min(result.minimum,(double)current_data[point_id]);
                    result.maximum=max(result.maximum,(double)current_data[point_id]);
                }
            }
        return result;
    }
}
