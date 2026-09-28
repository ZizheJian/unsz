#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    time_step_order_e param_c<T_coord,T_data>::get_time_step_order(const string &flow) const
    {
        if(flow=="anchor_interpolation_stride_8"||flow=="anchor_interpolation_stride_8_spatial_residual")
            return time_step_order_e::anchor_stride_8;
        if(flow=="endpoint_agreement"||flow=="linear_interpolation"||flow=="linear_interpolation_spatial_residual")
            return time_step_order_e::interpolation;
        return time_step_order_e::chronological;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_time_step_order(time_step_order_e order)
    {
        if(time_step_vec.empty())
            throw runtime_error("至少需要一个time step");
        if(original_data_path_vec.size()!=time_step_vec.size())
            throw runtime_error("time step数量与数据文件数量不一致");
        vector<uint32_t> sorted_id_vec(time_step_vec.size());
        iota(sorted_id_vec.begin(),sorted_id_vec.end(),0);
        sort(sorted_id_vec.begin(),sorted_id_vec.end(),[this](uint32_t a,uint32_t b){return time_step_vec[a]<time_step_vec[b];});
        vector<uint32_t> sorted_time_step_vec;
        vector<string> sorted_data_path_vec;
        sorted_time_step_vec.reserve(time_step_vec.size());
        sorted_data_path_vec.reserve(time_step_vec.size());
        for(uint32_t id:sorted_id_vec)
        {
            sorted_time_step_vec.push_back(time_step_vec[id]);
            sorted_data_path_vec.push_back(original_data_path_vec[id]);
        }
        for(uint32_t i=1;i<sorted_time_step_vec.size();i++)
            if(sorted_time_step_vec[i]==sorted_time_step_vec[i-1])
                throw runtime_error("time step不能重复");
        vector<uint32_t> order_id_vec;
        if(order==time_step_order_e::chronological||sorted_time_step_vec.size()==1)
        {
            order_id_vec.resize(sorted_time_step_vec.size());
            iota(order_id_vec.begin(),order_id_vec.end(),0);
        }
        else if(order==time_step_order_e::interpolation)
        {
            order_id_vec={0,(uint32_t)sorted_time_step_vec.size()-1};
            vector<pair<uint32_t,uint32_t>> interval_vec={{0,(uint32_t)sorted_time_step_vec.size()-1}};
            for(uint32_t interval_id=0;interval_id<interval_vec.size();interval_id++)
            {
                auto [left_id,right_id]=interval_vec[interval_id];
                if(right_id-left_id<=1)
                    continue;
                uint32_t left_time_step=sorted_time_step_vec[left_id],right_time_step=sorted_time_step_vec[right_id];
                double left_coordinate=time_step_coordinate_vec.empty()?left_time_step:time_step_coordinate_vec[left_time_step];
                double right_coordinate=time_step_coordinate_vec.empty()?right_time_step:time_step_coordinate_vec[right_time_step];
                double middle_coordinate=(left_coordinate+right_coordinate)/2;
                uint32_t middle_id=left_id+1;
                double minimum_distance=abs((time_step_coordinate_vec.empty()?sorted_time_step_vec[middle_id]:time_step_coordinate_vec[sorted_time_step_vec[middle_id]])-middle_coordinate);
                for(uint32_t candidate_id=left_id+2;candidate_id<right_id;candidate_id++)
                {
                    uint32_t candidate_time_step=sorted_time_step_vec[candidate_id];
                    double candidate_coordinate=time_step_coordinate_vec.empty()?candidate_time_step:time_step_coordinate_vec[candidate_time_step];
                    double distance=abs(candidate_coordinate-middle_coordinate);
                    if(distance<minimum_distance)
                    {
                        middle_id=candidate_id;
                        minimum_distance=distance;
                    }
                }
                order_id_vec.push_back(middle_id);
                interval_vec.push_back({left_id,middle_id});
                interval_vec.push_back({middle_id,right_id});
            }
        }
        else
        {
            for(uint32_t i=1;i<sorted_time_step_vec.size();i++)
                if(sorted_time_step_vec[i]!=sorted_time_step_vec[i-1]+1)
                    throw runtime_error("anchor_stride_8顺序要求time step连续且不重复");
            if(sorted_time_step_vec.front()!=0)
                throw runtime_error("anchor_stride_8顺序要求time step从0开始");
            uint32_t last_time_step=sorted_time_step_vec.back();
            order_id_vec.push_back(0);
            for(uint64_t time_step=8;time_step<=last_time_step;time_step+=8)
                order_id_vec.push_back((uint32_t)time_step);
            constexpr uint32_t layer_offset_vec[]={4,2,1};
            for(uint32_t offset:layer_offset_vec)
                for(uint64_t time_step=offset;time_step<=last_time_step;time_step+=offset*2)
                    order_id_vec.push_back((uint32_t)time_step);
        }
        if(order_id_vec.size()!=time_step_vec.size())
            throw runtime_error("时间顺序没有覆盖全部time step");
        vector<uint32_t> ordered_time_step_vec;
        vector<string> ordered_data_path_vec;
        ordered_time_step_vec.reserve(order_id_vec.size());
        ordered_data_path_vec.reserve(order_id_vec.size());
        for(uint32_t id:order_id_vec)
        {
            ordered_time_step_vec.push_back(sorted_time_step_vec[id]);
            ordered_data_path_vec.push_back(sorted_data_path_vec[id]);
        }
        time_step_vec=move(ordered_time_step_vec);
        original_data_path_vec=move(ordered_data_path_vec);
        const char *order_name=order==time_step_order_e::chronological?"Chronological":order==time_step_order_e::interpolation?"Interpolation":"Anchor stride 8";
        printf("%s processing order:",order_name);
        for(uint32_t time_step:time_step_vec)
            printf(" %u",time_step);
        printf("\n");
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::load_time_step_coordinates()
    {
        time_step_coordinate_vec.clear();
        if(time_step_coordinate_path.empty())
            return;
        FILE *file=fopen(time_step_coordinate_path.c_str(),"rb");
        if(!file)
            throw runtime_error("找不到time step坐标文件："+time_step_coordinate_path);
        if(fseek(file,0,SEEK_END)!=0)
        {
            fclose(file);
            throw runtime_error("无法定位到time step坐标文件末尾："+time_step_coordinate_path);
        }
        long byte_num_signed=ftell(file);
        if(byte_num_signed<0)
        {
            fclose(file);
            throw runtime_error("无法读取time step坐标文件大小："+time_step_coordinate_path);
        }
        if((uint64_t)byte_num_signed%sizeof(double)!=0)
        {
            fclose(file);
            throw runtime_error("time step坐标文件必须是连续float64数组："+time_step_coordinate_path);
        }
        time_step_coordinate_vec.resize((uint64_t)byte_num_signed/sizeof(double));
        if(fseek(file,0,SEEK_SET)!=0||fread(time_step_coordinate_vec.data(),sizeof(double),time_step_coordinate_vec.size(),file)!=time_step_coordinate_vec.size())
        {
            fclose(file);
            throw runtime_error("读取time step坐标文件失败："+time_step_coordinate_path);
        }
        if(fclose(file)!=0)
            throw runtime_error("关闭time step坐标文件失败："+time_step_coordinate_path);
        if(time_step_coordinate_vec.empty()||any_of(time_step_coordinate_vec.begin(),time_step_coordinate_vec.end(),[](double value){return !isfinite(value);}))
            throw runtime_error("time step坐标文件不能为空且只能包含有限值："+time_step_coordinate_path);
        uint32_t maximum_time_step=*max_element(time_step_vec.begin(),time_step_vec.end());
        if(maximum_time_step>=time_step_coordinate_vec.size())
            throw runtime_error("time step坐标文件没有覆盖time step "+to_string(maximum_time_step));
        if(time_step_coordinate_vec.size()>1)
        {
            if(time_step_coordinate_vec[1]==time_step_coordinate_vec[0])
                throw runtime_error("time step坐标必须严格单调");
            bool increasing=time_step_coordinate_vec[1]>time_step_coordinate_vec[0];
            for(uint64_t i=2;i<time_step_coordinate_vec.size();i++)
                if(time_step_coordinate_vec[i]==time_step_coordinate_vec[i-1]||(time_step_coordinate_vec[i]>time_step_coordinate_vec[i-1])!=increasing)
                    throw runtime_error("time step坐标必须严格单调");
        }
        printf("Loaded %llu time step coordinates from %s\n",(unsigned long long)time_step_coordinate_vec.size(),time_step_coordinate_path.c_str());
    }

    template<typename T_coord, typename T_data>
    void param_c<T_coord,T_data>::init()
    {
        load_time_step_coordinates();
        time_step_order_e label_order=get_time_step_order(label_flow),value_order=get_time_step_order(value_flow);
        if(point_set_config_vec.size()==1)
            prepare_time_step_order(value_order);
        else
            prepare_time_step_order(label_order==value_order?label_order:time_step_order_e::chronological);
        if (dim==0)
            throw runtime_error("坐标维数不能为0");
        FILE *f_coord=fopen(coord_path.c_str(),"rb");
        if (!f_coord)
            throw runtime_error("找不到坐标文件："+coord_path);
        if (fseek(f_coord,0,SEEK_END)!=0)
        {
            fclose(f_coord);
            throw runtime_error("无法定位到坐标文件末尾："+coord_path);
        }
        long coord_file_bytes_signed=ftell(f_coord);
        if (coord_file_bytes_signed<0)
        {
            fclose(f_coord);
            throw runtime_error("无法读取坐标文件大小："+coord_path);
        }
        uint64_t coord_file_bytes=coord_file_bytes_signed;
        uint64_t point_bytes=(uint64_t)dim*sizeof(T_coord);
        if (coord_file_bytes%point_bytes!=0)
            throw runtime_error("坐标文件大小不能被单个点的坐标大小整除");
        uint64_t point_num64=coord_file_bytes/point_bytes;
        if (point_num64>numeric_limits<uint32_t>::max())
            throw runtime_error("点数超过uint32_t的表示范围");
        point_num=(uint32_t)point_num64;
        printf("point_num=%u\n",point_num);
        if(check_original_data_files)
        {
            uint64_t expected_data_file_bytes=(uint64_t)point_num*sizeof(T_data);
            for(auto &data_path:original_data_path_vec)
            {
                error_code error;
                uint64_t data_file_bytes=filesystem::file_size(data_path,error);
                if(error)
                    throw runtime_error("无法读取数据文件大小："+data_path+"："+error.message());
                if(data_file_bytes!=expected_data_file_bytes)
                    throw runtime_error("数据文件大小不正确："+data_path+"，实际="+to_string(data_file_bytes)+" B，预期="+to_string(expected_data_file_bytes)+" B");
            }
        }
        coord_array.resize(point_num,dim);
        if (fseek(f_coord,0,SEEK_SET)!=0)
        {
            fclose(f_coord);
            throw runtime_error("无法定位到坐标文件开头："+coord_path);
        }
        uint64_t coord_value_num=(uint64_t)point_num*dim;
        if (fread(coord_array.data.data(),sizeof(T_coord),coord_value_num,f_coord)!=coord_value_num)
        {
            fclose(f_coord);
            throw runtime_error("读取坐标文件失败："+coord_path);
        }
        if (fclose(f_coord)!=0)
            throw runtime_error("关闭坐标文件失败："+coord_path);

        min_coord.resize(dim,numeric_limits<T_coord>::infinity());
        max_coord.resize(dim,-numeric_limits<T_coord>::infinity());
        for (uint32_t i=0;i<point_num;i++)
            for (uint32_t d=0;d<dim;d++)
            {
                max_coord[d]=max(max_coord[d],coord_array[i][d]);
                min_coord[d]=min(min_coord[d],coord_array[i][d]);
            }
        printf("Coord range: [");
        for (uint32_t d=0;d<dim;d++)
            printf("%s[%f, %f]",d==0?"":", ",(double)min_coord[d],(double)max_coord[d]);
        printf("]\n");
    }
}
