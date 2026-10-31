#pragma once

namespace unsz
{
    template<typename T_data>
    __attribute__((always_inline)) inline uint8_t classify_structural_label(T_data value,uint32_t active_level_count,uint32_t height_index,uint32_t active_level_count_offset)
    {
        bool structurally_active=height_index<(uint64_t)active_level_count+active_level_count_offset;
        bool explicit_anomaly=!isfinite(value)||value==(T_data)-99999;
        return structurally_active&&!explicit_anomaly?0:1;
    }

    template<typename T_coord,typename T_data>
    uint8_t param_c<T_coord,T_data>::classify_label(T_data value,uint32_t point_id,uint32_t slice_id) const
    {
        if(current_active_level_counts)
        {
            if(point_id>=current_active_level_counts->size())
                throw runtime_error("结构标签分类的point id越界");
            return classify_structural_label(value,(*current_active_level_counts)[point_id],slice_id%current_tensor_shape[1],current_active_level_count_offset);
        }
        if(label_rule_vec.size()!=1)
            throw runtime_error("双标签分类要求恰好一条规则");
        const label_rule_c&rule=label_rule_vec[0];
        double x=(double)value;
        bool matches=false;
        switch(rule.operator_type)
        {
            case label_rule_operator_e::less:matches=x<rule.value;break;
            case label_rule_operator_e::less_equal:matches=x<=rule.value;break;
            case label_rule_operator_e::greater:matches=x>rule.value;break;
            case label_rule_operator_e::greater_equal:matches=x>=rule.value;break;
            case label_rule_operator_e::equal:matches=x==rule.value;break;
            case label_rule_operator_e::not_equal:matches=x!=rule.value;break;
        }
        return matches?0:1;
    }

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

    inline label_codec_e parse_label_method(const string &text,const string &location)
    {
        if(text=="predict")
            return label_codec_e::rbf;
        if(text=="constant")
            return label_codec_e::constant;
        throw runtime_error(location+"的method必须是predict或constant："+text);
    }

    inline label_error_mode_e parse_label_error_mode(const string &text,const string &location)
    {
        if(text=="REL")
            return label_error_mode_e::relative;
        if(text=="ABS")
            return label_error_mode_e::absolute;
        if(text=="FIELD_REL")
            return label_error_mode_e::field_relative;
        throw runtime_error(location+"的eb.mode必须是REL、FIELD_REL或ABS："+text);
    }

    inline label_rule_operator_e parse_label_rule_operator(const string &text,const string &location)
    {
        if(text=="<")
            return label_rule_operator_e::less;
        if(text=="<=")
            return label_rule_operator_e::less_equal;
        if(text==">")
            return label_rule_operator_e::greater;
        if(text==">=")
            return label_rule_operator_e::greater_equal;
        if(text=="==")
            return label_rule_operator_e::equal;
        if(text=="!=")
            return label_rule_operator_e::not_equal;
        throw runtime_error(location+"的operator必须是<、<=、>、>=、==或!=："+text);
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::configure_labels()
    {
        label_rule_vec.clear();
        label_config_vec.clear();
        if(label_config_path.empty())
        {
            if(!got_rel_eb && !got_abs_eb)
                throw runtime_error("单标签模式必须通过-M指定误差模式和误差上限");
            label_config_c config;
            config.codec=label_codec_e::rbf;
            config.error_mode=got_rel_eb?label_error_mode_e::relative:label_error_mode_e::absolute;
            config.eb=got_rel_eb?(double)rel_eb:(double)abs_eb;
            label_config_vec.push_back(config);
            return;
        }
        ifstream file(label_config_path);
        if(!file)
            throw runtime_error("找不到标签配置文件："+label_config_path);
        stringstream buffer;
        buffer<<file.rdbuf();
        if(file.bad())
            throw runtime_error("读取标签配置文件失败："+label_config_path);
        json_value_c root=json_parser_c(buffer.str()).parse();
        double version=require_json_number(root,"version","标签配置根对象");
        if(version!=3 || floor(version)!=version)
            throw runtime_error("标签配置version目前必须是3");
        reject_unknown_json_fields(root,{"version","labels"},"标签配置根对象");
        const json_value_c &labels=require_json_field(root,"labels","标签配置根对象");
        if(labels.type!=json_type_e::array_value || labels.array_value.empty())
            throw runtime_error("labels必须是包含至少一个label的数组");
        if(labels.array_value.size()>2)
            throw runtime_error("UNSZ目前只支持一个或两个标签");
        for(uint32_t label_id=0;label_id<labels.array_value.size();label_id++)
        {
            const json_value_c &label=labels.array_value[label_id];
            string location="labels["+to_string(label_id)+"]";
            reject_unknown_json_fields(label,{"method","eb_mode","eb_value","condition"},location);
            label_config_c config;
            string method=require_json_string(label,"method",location);
            config.codec=parse_label_method(method,location);
            auto eb_mode_iterator=label.object_value.find("eb_mode"),eb_value_iterator=label.object_value.find("eb_value");
            if(config.codec==label_codec_e::constant)
            {
                if(eb_mode_iterator!=label.object_value.end() || eb_value_iterator!=label.object_value.end())
                    throw runtime_error(location+"使用constant时不能配置eb_mode或eb_value");
            }
            else
            {
                if(eb_mode_iterator==label.object_value.end() || eb_value_iterator==label.object_value.end())
                    throw runtime_error(location+"不是constant，必须同时配置eb_mode和eb_value");
                config.error_mode=parse_label_error_mode(require_json_string(label,"eb_mode",location),location);
                config.eb=require_json_number(label,"eb_value",location);
                if(!(config.eb>0) || !isfinite(config.eb))
                    throw runtime_error(location+"的eb_value必须是有限正数");
            }
            label_config_vec.push_back(config);
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
            label_rule_c parsed_rule;
            parsed_rule.operator_type=parse_label_rule_operator(require_json_string(condition,"operator",location+".condition"),location+".condition");
            parsed_rule.value=require_json_number(condition,"value",location+".condition");
            if(!isfinite(parsed_rule.value))
                throw runtime_error(location+".condition.value必须是有限数");
            label_rule_vec.push_back(parsed_rule);
        }
    }

    // 把标签的误差上限换算成绝对误差上限：ABS直接使用，REL乘以当前slice中该标签的值域，FIELD_REL乘以整个field中该标签的值域。
    // 压缩端扫描完全部slice之前不知道field值域，此时field_range_known=false，FIELD_REL先按slice值域计算，扫描结束后统一改写。
    template<typename T_coord,typename T_data>
    T_data param_c<T_coord,T_data>::label_abs_eb(uint32_t label_id,bool field_range_known) const
    {
        const label_config_c&config=label_config_vec[label_id];
        if(config.error_mode==label_error_mode_e::absolute)
            return (T_data)config.eb;
        if(config.error_mode==label_error_mode_e::field_relative&&field_range_known)
            return (T_data)(config.eb*(field_label_max_data[label_id]-field_label_min_data[label_id]));
        return (T_data)(config.eb*(label_max_data_vec[label_id]-label_min_data_vec[label_id]));
    }
}
