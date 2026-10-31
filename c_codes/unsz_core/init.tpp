#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    slice_order_e param_c<T_coord,T_data>::get_slice_order(const string &pipeline) const
    {
        if(pipeline=="anchor_interpolation"||pipeline=="anchor_interpolation_spatial_residual")
            return slice_order_e::anchor_interpolation;
        if(pipeline=="interpolation"||pipeline=="interpolation_spatial_residual")
            return slice_order_e::interpolation;
        return slice_order_e::sequential;
    }

    template<typename T_coord,typename T_data>
    vector<uint32_t> param_c<T_coord,T_data>::generate_slice_coding_order(slice_order_e order) const
    {
        if(slice_vec.empty())
            throw runtime_error("至少需要一个slice");
        if(original_data_path_vec.size()!=slice_vec.size())
            throw runtime_error("slice数量与数据文件数量不一致");
        vector<uint32_t> sorted_id_vec(slice_vec.size());
        iota(sorted_id_vec.begin(),sorted_id_vec.end(),0);
        sort(sorted_id_vec.begin(),sorted_id_vec.end(),[this](uint32_t a,uint32_t b){return slice_vec[a]<slice_vec[b];});
        vector<uint32_t> sorted_slice_vec;
        sorted_slice_vec.reserve(slice_vec.size());
        for(uint32_t id:sorted_id_vec)
            sorted_slice_vec.push_back(slice_vec[id]);
        for(uint32_t i=1;i<sorted_slice_vec.size();i++)
            if(sorted_slice_vec[i]==sorted_slice_vec[i-1])
                throw runtime_error("slice不能重复");
        auto get_slice_coordinate=[this](uint32_t slice)
        {
            if(slice_coordinate_vec.empty())
                return (T_coord)slice;
            return slice_coordinate_vec[slice];
        };
        vector<uint32_t> order_id_vec;
        if(order==slice_order_e::sequential||sorted_slice_vec.size()==1)
        {
            order_id_vec.resize(sorted_slice_vec.size());
            iota(order_id_vec.begin(),order_id_vec.end(),0);
        }
        else if(order==slice_order_e::interpolation)
        {
            order_id_vec={0,(uint32_t)sorted_slice_vec.size()-1};
            vector<pair<uint32_t,uint32_t>> interval_vec={{0,(uint32_t)sorted_slice_vec.size()-1}};
            for(uint32_t interval_id=0;interval_id<interval_vec.size();interval_id++)
            {
                auto [left_id,right_id]=interval_vec[interval_id];
                if(right_id-left_id<=1)
                    continue;
                uint32_t left_slice=sorted_slice_vec[left_id],right_slice=sorted_slice_vec[right_id];
                T_coord left_coordinate=get_slice_coordinate(left_slice);
                T_coord right_coordinate=get_slice_coordinate(right_slice);
                T_coord middle_coordinate=(left_coordinate+right_coordinate)/2;
                uint32_t middle_id=left_id+1;
                T_coord minimum_distance=abs(get_slice_coordinate(sorted_slice_vec[middle_id])-middle_coordinate);
                for(uint32_t candidate_id=left_id+2;candidate_id<right_id;candidate_id++)
                {
                    uint32_t candidate_slice=sorted_slice_vec[candidate_id];
                    T_coord candidate_coordinate=get_slice_coordinate(candidate_slice);
                    T_coord distance=abs(candidate_coordinate-middle_coordinate);
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
            for(uint32_t i=1;i<sorted_slice_vec.size();i++)
                if(sorted_slice_vec[i]!=sorted_slice_vec[i-1]+1)
                    throw runtime_error("anchor_interpolation顺序要求slice连续且不重复");
            if(sorted_slice_vec.front()!=0)
                throw runtime_error("anchor_interpolation顺序要求slice从0开始");
            uint32_t last_slice=sorted_slice_vec.back();
            order_id_vec.push_back(0);
            for(uint64_t slice=8;slice<=last_slice;slice+=8)
                order_id_vec.push_back((uint32_t)slice);
            constexpr uint32_t layer_offset_vec[]={4,2,1};
            for(uint32_t offset:layer_offset_vec)
                for(uint64_t slice=offset;slice<=last_slice;slice+=offset*2)
                    order_id_vec.push_back((uint32_t)slice);
        }
        if(order_id_vec.size()!=slice_vec.size())
            throw runtime_error("slice顺序没有覆盖全部slice");
        vector<uint32_t> processing_order;
        processing_order.reserve(order_id_vec.size());
        for(uint32_t sorted_id:order_id_vec)
            processing_order.push_back(sorted_id_vec[sorted_id]);
        return processing_order;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_slice_order(slice_order_e order)
    {
        vector<uint32_t> processing_order=generate_slice_coding_order(order);
        auto reorder_file_values=[&](auto&values)
        {
            if(values.empty())
                return;
            if(values.size()!=processing_order.size())
                throw runtime_error("逐文件预计算结果数量与slice数量不一致");
            using value_type=typename decay_t<decltype(values)>::value_type;
            vector<value_type> ordered_values;
            ordered_values.reserve(values.size());
            for(uint32_t id:processing_order)
                ordered_values.push_back(values[id]);
            values=move(ordered_values);
        };
        reorder_file_values(compression_file_label_min_data_vec);
        reorder_file_values(compression_file_label_max_data_vec);
        reorder_file_values(compression_file_label_abs_eb_vec);
        reorder_file_values(compression_file_label_codec_vec);
        vector<uint32_t> ordered_slice_vec;
        vector<string> ordered_data_path_vec;
        ordered_slice_vec.reserve(processing_order.size());
        ordered_data_path_vec.reserve(processing_order.size());
        for(uint32_t id:processing_order)
        {
            ordered_slice_vec.push_back(slice_vec[id]);
            ordered_data_path_vec.push_back(original_data_path_vec[id]);
        }
        slice_vec=move(ordered_slice_vec);
        original_data_path_vec=move(ordered_data_path_vec);
        const char *order_name;
        if(order==slice_order_e::sequential)
            order_name="Chronological";
        else if(order==slice_order_e::interpolation)
            order_name="Interpolation";
        else
            order_name="Anchor stride 8";
        printf("%s processing order:",order_name);
        if(slice_vec.size()<=32)
            for(uint32_t slice:slice_vec)
                printf(" %u",slice);
        else
        {
            for(uint32_t index=0;index<8;index++)
                printf(" %u",slice_vec[index]);
            printf(" ...");
            for(uint32_t index=(uint32_t)slice_vec.size()-8;index<slice_vec.size();index++)
                printf(" %u",slice_vec[index]);
            printf(" (%u total)",(uint32_t)slice_vec.size());
        }
        printf("\n");
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_slice_order_from_pipelines()
    {
        if(current_field_uses_tensor)
        {
            prepare_tensor_slice_order();
            return;
        }
        slice_order_e label_order=get_slice_order(label_pipeline),value_order=get_slice_order(value_pipeline);
        if(label_config_vec.size()==1||label_pipeline=="independent")
            prepare_slice_order(value_order);
        else if(label_order==value_order||value_pipeline=="independent")
            prepare_slice_order(label_order);
        else
            prepare_slice_order(slice_order_e::sequential);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::prepare_tensor_slice_order()
    {
        if(!tensor_pipeline_selected)
            throw runtime_error("尚未选择tensor pipeline");
        const slice_dag_c&plan=get_tensor_slice_dag(selected_tensor_pipeline);
        const vector<uint32_t>&processing_order=plan.processing_order;
        auto reorder_file_values=[&](auto&values)
        {
            if(values.empty())
                return;
            if(values.size()!=processing_order.size())
                throw runtime_error("逐切片预计算结果数量与tensor切片数量不一致");
            using value_type=typename decay_t<decltype(values)>::value_type;
            vector<value_type> ordered_values;
            ordered_values.reserve(values.size());
            for(uint32_t id:processing_order)
                ordered_values.push_back(values[id]);
            values=move(ordered_values);
        };
        reorder_file_values(compression_file_label_min_data_vec);
        reorder_file_values(compression_file_label_max_data_vec);
        reorder_file_values(compression_file_label_abs_eb_vec);
        reorder_file_values(compression_file_label_codec_vec);
        vector<uint32_t> ordered_slice_ids;
        vector<string> ordered_data_paths;
        ordered_slice_ids.reserve(processing_order.size());
        ordered_data_paths.reserve(processing_order.size());
        for(uint32_t slice_id:processing_order)
        {
            ordered_slice_ids.push_back(slice_id);
            ordered_data_paths.push_back(original_data_path_vec[slice_id]);
        }
        slice_vec=move(ordered_slice_ids);
        original_data_path_vec=move(ordered_data_paths);
        printf("Tensor processing order selected: %s, slices=%u\n",tensor_pipeline_name(selected_tensor_pipeline).c_str(),(uint32_t)slice_vec.size());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::load_slice_coordinates()
    {
        slice_coordinate_vec.clear();
        if(slice_coordinate_path.empty())
            return;
        FILE *file=fopen(slice_coordinate_path.c_str(),"rb");
        if(!file)
            throw runtime_error("找不到slice坐标文件："+slice_coordinate_path);
        if(fseek(file,0,SEEK_END)!=0)
        {
            fclose(file);
            throw runtime_error("无法定位到slice坐标文件末尾："+slice_coordinate_path);
        }
        long byte_num_signed=ftell(file);
        if(byte_num_signed<0)
        {
            fclose(file);
            throw runtime_error("无法读取slice坐标文件大小："+slice_coordinate_path);
        }
        if((uint64_t)byte_num_signed%sizeof(double)!=0)
        {
            fclose(file);
            throw runtime_error("slice坐标文件必须是连续float64数组："+slice_coordinate_path);
        }
        vector<double> file_coordinate_vec((uint64_t)byte_num_signed/sizeof(double));
        if(fseek(file,0,SEEK_SET)!=0||fread(file_coordinate_vec.data(),sizeof(double),file_coordinate_vec.size(),file)!=file_coordinate_vec.size())
        {
            fclose(file);
            throw runtime_error("读取slice坐标文件失败："+slice_coordinate_path);
        }
        if(fclose(file)!=0)
            throw runtime_error("关闭slice坐标文件失败："+slice_coordinate_path);
        if(file_coordinate_vec.empty()||any_of(file_coordinate_vec.begin(),file_coordinate_vec.end(),[](double value){return !isfinite(value);}))
            throw runtime_error("slice坐标文件不能为空且只能包含有限值："+slice_coordinate_path);
        slice_coordinate_vec.assign(file_coordinate_vec.begin(),file_coordinate_vec.end());
        if(any_of(slice_coordinate_vec.begin(),slice_coordinate_vec.end(),[](T_coord value){return !isfinite(value);}))
            throw runtime_error("slice坐标无法用当前坐标类型表示："+slice_coordinate_path);
        uint32_t maximum_slice=*max_element(slice_vec.begin(),slice_vec.end());
        if(maximum_slice>=slice_coordinate_vec.size())
            throw runtime_error("slice坐标文件没有覆盖slice "+to_string(maximum_slice));
        if(slice_coordinate_vec.size()>1)
        {
            if(slice_coordinate_vec[1]==slice_coordinate_vec[0])
                throw runtime_error("slice坐标必须严格单调");
            bool increasing=slice_coordinate_vec[1]>slice_coordinate_vec[0];
            for(uint64_t i=2;i<slice_coordinate_vec.size();i++)
                if(slice_coordinate_vec[i]==slice_coordinate_vec[i-1]||(slice_coordinate_vec[i]>slice_coordinate_vec[i-1])!=increasing)
                    throw runtime_error("slice坐标必须严格单调");
        }
        printf("Loaded %llu slice coordinates from %s\n",(unsigned long long)slice_coordinate_vec.size(),slice_coordinate_path.c_str());
    }

    template<typename T_coord, typename T_data>
    void param_c<T_coord,T_data>::init()
    {
        load_slice_coordinates();
        prepare_slice_order(slice_order_e::sequential);
        if (dim==0)
            throw runtime_error("坐标维数不能为0");
        error_code coord_file_error;
        uint64_t coord_file_bytes=filesystem::file_size(coord_path,coord_file_error);
        if(coord_file_error)
            throw runtime_error("无法读取坐标文件大小："+coord_path+"："+coord_file_error.message());
        uint64_t point_bytes=(uint64_t)dim*sizeof(T_coord);
        if (coord_file_bytes%point_bytes!=0)
            throw runtime_error("坐标文件大小不能被单个点的坐标大小整除");
        uint64_t point_num64=coord_file_bytes/point_bytes;
        if(point_num64==0)
            throw runtime_error("坐标文件不能为空："+coord_path);
        if (point_num64>numeric_limits<uint32_t>::max())
            throw runtime_error("点数超过uint32_t的表示范围");
        point_num=(uint32_t)point_num64;
        printf("point_num=%u\n",point_num);
        coord_array.resize(point_num,dim);
        uint64_t coord_value_num=(uint64_t)point_num*dim;
        read_binary_file_exact(coord_path,coord_array.data.data(),coord_value_num);

        min_coord.resize(dim,numeric_limits<T_coord>::infinity());
        max_coord.resize(dim,-numeric_limits<T_coord>::infinity());
        for (uint32_t i=0;i<point_num;i++)
            for (uint32_t d=0;d<dim;d++)
            {
                if(!isfinite(coord_array[i][d]))
                    throw runtime_error("坐标文件包含非有限坐标：点"+to_string(i)+"，维度"+to_string(d));
                max_coord[d]=max(max_coord[d],coord_array[i][d]);
                min_coord[d]=min(min_coord[d],coord_array[i][d]);
            }
        original_coord_array=coord_array;
        original_min_coord=min_coord;
        original_max_coord=max_coord;
        printf("Coord range: [");
        for (uint32_t d=0;d<dim;d++)
            printf("%s[%f, %f]",d==0?"":", ",(double)min_coord[d],(double)max_coord[d]);
        printf("]\n");
    }
}
