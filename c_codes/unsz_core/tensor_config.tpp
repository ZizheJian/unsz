#pragma once

namespace unsz
{
    inline uint32_t require_json_uint32(const json_value_c&object,const string&field,const string&location)
    {
        double value=require_json_number(object,field,location);
        if(value<0||value>numeric_limits<uint32_t>::max()||floor(value)!=value)
            throw runtime_error(location+"的字段'"+field+"'必须是uint32整数");
        return (uint32_t)value;
    }

    inline string resolve_metadata_path(const filesystem::path&metadata_directory,const string&path)
    {
        filesystem::path result(path);
        if(result.is_relative())
            result=metadata_directory/result;
        return result.lexically_normal().string();
    }

    inline vector<double> read_tensor_axis_coordinates(const string&path,uint32_t expected_num,const string&location)
    {
        error_code file_error;
        uint64_t byte_num=filesystem::file_size(path,file_error);
        if(file_error)
            throw runtime_error("无法读取"+location+"文件大小："+path+"："+file_error.message());
        if(byte_num!=(uint64_t)expected_num*sizeof(double))
            throw runtime_error(location+"必须包含"+to_string(expected_num)+"个float64坐标："+path);
        vector<double> coordinates(expected_num);
        read_binary_file_exact(path,coordinates.data(),coordinates.size());
        for(uint32_t index=0;index<coordinates.size();index++)
        {
            if(!isfinite(coordinates[index]))
                throw runtime_error(location+"包含非有限坐标："+path);
            if(index>0&&!(coordinates[index]>coordinates[index-1]))
                throw runtime_error(location+"必须严格递增："+path);
        }
        return coordinates;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::load_tensor_metadata()
    {
        tensor_field_config_map.clear();
        if(tensor_metadata_path.empty())
            return;
        filesystem::path metadata_path(tensor_metadata_path);
        if(metadata_path.is_relative())
            metadata_path=filesystem::path(data_dir)/metadata_path;
        metadata_path=metadata_path.lexically_normal();
        ifstream file(metadata_path);
        if(!file)
            throw runtime_error("找不到tensor metadata文件："+metadata_path.string());
        stringstream buffer;
        buffer<<file.rdbuf();
        if(file.bad())
            throw runtime_error("读取tensor metadata文件失败："+metadata_path.string());
        json_value_c root=json_parser_c(buffer.str()).parse();
        double version=require_json_number(root,"version","tensor metadata根对象");
        if(version!=1)
            throw runtime_error("tensor metadata version目前必须是1");
        reject_unknown_json_fields(root,{"version","fields"},"tensor metadata根对象");
        const json_value_c&fields=require_json_field(root,"fields","tensor metadata根对象");
        if(fields.type!=json_type_e::object_value||fields.object_value.empty())
            throw runtime_error("tensor metadata的fields必须是非空对象");
        filesystem::path metadata_directory=metadata_path.parent_path();
        for(const auto&field_item:fields.object_value)
        {
            const string&name=field_item.first;
            const json_value_c&field=field_item.second;
            string location="fields."+name;
            reject_unknown_json_fields(field,{"shape","axis0_coordinates","axis1_coordinates","active_level_count_file","active_level_count_offset"},location);
            const json_value_c&shape=require_json_field(field,"shape",location);
            if(shape.type!=json_type_e::array_value||shape.array_value.size()!=2)
                throw runtime_error(location+"的shape必须是包含两个正整数的数组");
            tensor_field_config_c config;
            array<string,2> coordinate_keys{{"uniform","uniform"}};
            for(uint32_t axis=0;axis<2;axis++)
            {
                const json_value_c&length_value=shape.array_value[axis];
                if(length_value.type!=json_type_e::number_value)
                    throw runtime_error(location+"的shape必须是包含两个正整数的数组");
                double length=length_value.number_value;
                if(length<1||length>numeric_limits<uint32_t>::max()||floor(length)!=length)
                    throw runtime_error(location+"的shape必须是包含两个正整数的数组");
                config.shape[axis]=(uint32_t)length;
                string coordinate_field="axis"+to_string(axis)+"_coordinates";
                auto coordinate_iterator=field.object_value.find(coordinate_field);
                if(coordinate_iterator!=field.object_value.end())
                {
                    if(coordinate_iterator->second.type!=json_type_e::string_value)
                        throw runtime_error(location+"的"+coordinate_field+"必须是文件路径字符串");
                    string coordinate_path=resolve_metadata_path(metadata_directory,coordinate_iterator->second.string_value);
                    config.axis_coordinates[axis]=read_tensor_axis_coordinates(coordinate_path,config.shape[axis],location+"的"+coordinate_field);
                    coordinate_keys[axis]=coordinate_path;
                }
            }
            config.dependency_plan_key=to_string(config.shape[0])+"x"+to_string(config.shape[1])+"|"+coordinate_keys[0]+"|"+coordinate_keys[1];
            auto active_level_iterator=field.object_value.find("active_level_count_file");
            auto active_level_offset_iterator=field.object_value.find("active_level_count_offset");
            if(active_level_iterator!=field.object_value.end())
            {
                if(active_level_iterator->second.type!=json_type_e::string_value)
                    throw runtime_error(location+"的active_level_count_file必须是文件路径字符串");
                config.active_level_count_path=resolve_metadata_path(metadata_directory,active_level_iterator->second.string_value);
                if(active_level_offset_iterator!=field.object_value.end())
                    config.active_level_count_offset=require_json_uint32(field,"active_level_count_offset",location);
                if(config.active_level_count_offset>1)
                    throw runtime_error(location+"的active_level_count_offset只能是0或1");
                if(config.shape[1]==1)
                    throw runtime_error(location+"只有一个高度切片，不能配置active_level_count_file");
            }
            else if(active_level_offset_iterator!=field.object_value.end())
                throw runtime_error(location+"配置active_level_count_offset时必须同时配置active_level_count_file");
            tensor_field_config_map.emplace(name,move(config));
        }
        for(const string&name:field_name_vec)
            if(tensor_field_config_map.find(name)==tensor_field_config_map.end())
                throw runtime_error("tensor metadata没有field："+name);
        tensor_metadata_path=metadata_path.string();
        printf("Loaded tensor metadata for %llu fields from %s\n",(unsigned long long)tensor_field_config_map.size(),tensor_metadata_path.c_str());
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::activate_tensor_field(const tensor_field_config_c&config)
    {
        current_field_uses_tensor=true;
        current_tensor_shape=config.shape;
        current_tensor_axis_coordinates=config.axis_coordinates;
        current_tensor_dependency_plan_key=config.dependency_plan_key;
        current_active_level_count_path=config.active_level_count_path;
        current_active_level_count_offset=config.active_level_count_offset;
        if(!current_active_level_count_path.empty()&&configured_point_set_config_vec.size()!=2)
            throw runtime_error("结构有效层分类要求配置两个点集："+field_name);
        uint64_t slice_num=(uint64_t)config.shape[0]*config.shape[1];
        if(slice_num>numeric_limits<uint32_t>::max())
            throw runtime_error("field的tensor切片数量超过uint32_t范围："+field_name);
        configured_time_step_vec.resize(slice_num);
        iota(configured_time_step_vec.begin(),configured_time_step_vec.end(),0);
        time_step_vec=configured_time_step_vec;
        if(slice_num==1)
        {
            selected_tensor_flow={};
            tensor_flow_selected=true;
        }
        if(point_num>0)
            load_current_active_level_counts();
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::load_current_active_level_counts()
    {
        current_active_level_counts=nullptr;
        if(current_active_level_count_path.empty())
            return;
        auto cache_iterator=active_level_count_cache.find(current_active_level_count_path);
        if(cache_iterator==active_level_count_cache.end())
        {
            error_code file_error;
            uint64_t byte_num=filesystem::file_size(current_active_level_count_path,file_error);
            if(file_error)
                throw runtime_error("无法读取有效层计数文件大小："+current_active_level_count_path+"："+file_error.message());
            if(byte_num!=(uint64_t)point_num*sizeof(uint32_t))
                throw runtime_error("有效层计数文件必须包含point_num个uint32："+current_active_level_count_path);
            active_level_count_data_c data;
            data.counts.resize(point_num);
            read_binary_file_exact(current_active_level_count_path,data.counts.data(),point_num);
            data.maximum=*max_element(data.counts.begin(),data.counts.end());
            cache_iterator=active_level_count_cache.emplace(current_active_level_count_path,move(data)).first;
        }
        current_active_level_counts=&cache_iterator->second.counts;
        if((uint64_t)cache_iterator->second.maximum+current_active_level_count_offset>current_tensor_shape[1])
            throw runtime_error("有效层计数超过tensor高度："+current_active_level_count_path);
    }
}
