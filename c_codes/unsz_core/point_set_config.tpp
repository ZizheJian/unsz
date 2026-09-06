#pragma once

namespace unsz
{
    inline const json_value_c&require_json_field(const json_value_c &object,const string &field,const string &location)
    {
        if(object.type!=json_type_e::object_value)
            throw runtime_error(location+"必须是JSON对象");
        auto iterator=object.object_value.find(field);
        if(iterator==object.object_value.end())
            throw runtime_error(location+"缺少字段'"+field+"'");
        return iterator->second;
    }

    inline string require_json_string(const json_value_c &object,const string &field,const string &location)
    {
        const json_value_c &value=require_json_field(object,field,location);
        if(value.type!=json_type_e::string_value)
            throw runtime_error(location+"的字段'"+field+"'必须是字符串");
        return value.string_value;
    }

    inline double require_json_number(const json_value_c &object,const string &field,const string &location)
    {
        const json_value_c &value=require_json_field(object,field,location);
        if(value.type!=json_type_e::number_value)
            throw runtime_error(location+"的字段'"+field+"'必须是数值");
        return value.number_value;
    }

    inline void reject_unknown_json_fields(const json_value_c &object,const vector<string> &allowed,const string &location)
    {
        if(object.type!=json_type_e::object_value)
            throw runtime_error(location+"必须是JSON对象");
        for(const auto &item:object.object_value)
            if(find(allowed.begin(),allowed.end(),item.first)==allowed.end())
                throw runtime_error(location+"包含未知字段'"+item.first+"'");
    }

    inline point_set_codec_e parse_point_set_codec(const string &text,const string &location)
    {
        if(text=="inverse_proportion")
            return point_set_codec_e::inverse_proportion;
        if(text=="fitting")
            return point_set_codec_e::fitting;
        if(text=="constant")
            return point_set_codec_e::constant;
        if(text=="hybrid_loo")
            return point_set_codec_e::hybrid_loo;
        throw runtime_error(location+"的codec必须是inverse_proportion、fitting、constant或hybrid_loo："+text);
    }

    inline point_set_error_mode_e parse_point_set_error_mode(const string &text,const string &location)
    {
        if(text=="REL")
            return point_set_error_mode_e::relative;
        if(text=="ABS")
            return point_set_error_mode_e::absolute;
        throw runtime_error(location+"的eb.mode必须是REL或ABS："+text);
    }

    inline point_set_rule_operator_e parse_point_set_rule_operator(const string &text,const string &location)
    {
        if(text=="<")
            return point_set_rule_operator_e::less;
        if(text=="<=")
            return point_set_rule_operator_e::less_equal;
        if(text==">")
            return point_set_rule_operator_e::greater;
        if(text==">=")
            return point_set_rule_operator_e::greater_equal;
        if(text=="==")
            return point_set_rule_operator_e::equal;
        if(text=="!=")
            return point_set_rule_operator_e::not_equal;
        throw runtime_error(location+"的operator必须是<、<=、>、>=、==或!=："+text);
    }

    inline string resolve_config_relative_path(const string &config_path,const string &path)
    {
        if(path.empty() || path.front()=='/')
            return path;
        uint64_t slash_position=config_path.find_last_of('/');
        return slash_position==string::npos?path:config_path.substr(0,slash_position+1)+path;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::configure_point_sets()
    {
        point_set_rule_vec.clear();
        point_set_config_vec.clear();
        point_set_label_path_pattern.clear();
        point_set_classification=point_set_classification_e::ordered_rules;
        if(point_set_config_path.empty())
        {
            if(!got_rel_eb && !got_abs_eb)
                throw runtime_error("单点集模式必须通过-M指定误差模式和误差上限");
            point_set_config_c config;
            config.codec=parse_point_set_codec(predictor,"-predictor");
            config.error_mode=got_rel_eb?point_set_error_mode_e::relative:point_set_error_mode_e::absolute;
            config.eb=got_rel_eb?(double)rel_eb:(double)abs_eb;
            point_set_config_vec.push_back(config);
            return;
        }
        if(got_error_mode)
            printf("警告：指定-point-set-config后忽略命令行中的-M\n");
        if(got_predictor)
            printf("警告：指定-point-set-config后忽略命令行中的-predictor\n");
        ifstream file(point_set_config_path);
        if(!file)
            throw runtime_error("找不到点集配置文件："+point_set_config_path);
        stringstream buffer;
        buffer<<file.rdbuf();
        if(file.bad())
            throw runtime_error("读取点集配置文件失败："+point_set_config_path);
        json_value_c root=json_parser_c(buffer.str()).parse();
        double version=require_json_number(root,"version","点集配置根对象");
        if(version!=1 || floor(version)!=version)
            throw runtime_error("点集配置version目前必须是1");
        string rule_type=require_json_string(root,"rule_type","点集配置根对象");
        bool use_ordered_rules=rule_type=="ordered";
        if(!use_ordered_rules && rule_type!="label_file")
            throw runtime_error("rule_type必须是ordered或label_file："+rule_type);
        reject_unknown_json_fields(root,use_ordered_rules?vector<string>{"version","rule_type","labels"}:vector<string>{"version","rule_type","label_file","labels"},"点集配置根对象");
        const json_value_c &labels=require_json_field(root,"labels","点集配置根对象");
        if(labels.type!=json_type_e::array_value || labels.array_value.empty())
            throw runtime_error("labels必须是包含至少一个label的数组");
        if(labels.array_value.size()>256)
            throw runtime_error("label数量不能超过256");
        for(uint32_t label_id=0;label_id<labels.array_value.size();label_id++)
        {
            const json_value_c &label=labels.array_value[label_id];
            string location="labels["+to_string(label_id)+"]";
            reject_unknown_json_fields(label,use_ordered_rules?vector<string>{"method","eb_mode","eb_value","condition"}:vector<string>{"method","eb_mode","eb_value"},location);
            point_set_config_c config;
            config.codec=parse_point_set_codec(require_json_string(label,"method",location),location);
            auto eb_mode_iterator=label.object_value.find("eb_mode"),eb_value_iterator=label.object_value.find("eb_value");
            if(config.codec==point_set_codec_e::constant)
            {
                if(eb_mode_iterator!=label.object_value.end() || eb_value_iterator!=label.object_value.end())
                    throw runtime_error(location+"使用constant时不能配置eb_mode或eb_value");
            }
            else
            {
                if(eb_mode_iterator==label.object_value.end() || eb_value_iterator==label.object_value.end())
                    throw runtime_error(location+"不是constant，必须同时配置eb_mode和eb_value");
                config.error_mode=parse_point_set_error_mode(require_json_string(label,"eb_mode",location),location);
                config.eb=require_json_number(label,"eb_value",location);
                if(!(config.eb>0) || !isfinite(config.eb))
                    throw runtime_error(location+"的eb_value必须是有限正数");
            }
            point_set_config_vec.push_back(config);
            if(use_ordered_rules)
            {
                bool is_last=label_id+1==labels.array_value.size();
                if(is_last)
                {
                    if(label.object_value.find("condition")!=label.object_value.end())
                        throw runtime_error("最后一个label是默认兜底类型，不能配置condition");
                    continue;
                }
                const json_value_c &condition=require_json_field(label,"condition",location);
                if(condition.type!=json_type_e::object_value)
                    throw runtime_error(location+".condition必须是JSON对象");
                reject_unknown_json_fields(condition,{"operator","value"},location+".condition");
                point_set_rule_c parsed_rule;
                parsed_rule.operator_type=parse_point_set_rule_operator(require_json_string(condition,"operator",location+".condition"),location+".condition");
                parsed_rule.value=require_json_number(condition,"value",location+".condition");
                if(!isfinite(parsed_rule.value))
                    throw runtime_error(location+".condition.value必须是有限数");
                point_set_rule_vec.push_back(parsed_rule);
            }
        }
        if(!use_ordered_rules)
        {
            const json_value_c &label_file=require_json_field(root,"label_file","点集配置根对象");
            reject_unknown_json_fields(label_file,{"path","dtype"},"label_file");
            string dtype=require_json_string(label_file,"dtype","label_file");
            if(dtype!="uint8")
                throw runtime_error("label_file目前只支持dtype=uint8");
            point_set_label_path_pattern=resolve_config_relative_path(point_set_config_path,require_json_string(label_file,"path","label_file"));
            if(point_set_label_path_pattern.find("{time_step}")==string::npos)
                throw runtime_error("label_file.path必须包含{time_step}，以便每个time step使用独立标签");
            point_set_classification=point_set_classification_e::label_file;
        }
    }

    template<typename T_coord,typename T_data>
    vector<uint8_t> param_c<T_coord,T_data>::load_point_set_label_ids()
    {
        string path=point_set_label_path_pattern;
        const string placeholder="{time_step}";
        uint64_t position=0;
        while((position=path.find(placeholder,position))!=string::npos)
        {
            string replacement=to_string(current_time_step);
            path.replace(position,placeholder.size(),replacement);
            position+=replacement.size();
        }
        FILE *file=fopen(path.c_str(),"rb");
        if(!file)
            throw runtime_error("找不到点集标签文件："+path);
        if(fseek(file,0,SEEK_END)!=0)
        {
            fclose(file);
            throw runtime_error("无法定位到点集标签文件末尾："+path);
        }
        long byte_num=ftell(file);
        if(byte_num<0 || (uint64_t)byte_num!=point_num)
        {
            fclose(file);
            throw runtime_error("点集标签文件大小必须等于点数："+path);
        }
        if(fseek(file,0,SEEK_SET)!=0)
        {
            fclose(file);
            throw runtime_error("无法定位到点集标签文件开头："+path);
        }
        vector<uint8_t> label_ids(point_num);
        if(fread(label_ids.data(),1,point_num,file)!=point_num)
        {
            fclose(file);
            throw runtime_error("读取点集标签文件失败："+path);
        }
        if(fclose(file)!=0)
            throw runtime_error("关闭点集标签文件失败："+path);
        for(uint32_t point_id=0;point_id<point_num;point_id++)
            if(label_ids[point_id]>=point_set_config_vec.size())
                throw runtime_error("点集标签文件中的编号越界：文件="+path+"，点="+to_string(point_id)+"，编号="+to_string(label_ids[point_id]));
        return label_ids;
    }
}
