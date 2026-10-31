#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "parsers.hpp"

inline std::vector<uint32_t> parse_slice_list(std::string expression)
{
    std::vector<uint32_t>result;
    if(expression.size()>=5&&expression.front()=='['&&(expression.back()==']'||expression.back()==')'))
    {
        uint64_t comma_position=expression.find(',');
        if(comma_position==std::string::npos||expression.find(',',comma_position+1)!=std::string::npos)
            throw std::runtime_error("slice区间必须包含且只包含一个逗号："+expression);
        uint32_t left=parse_uint32(expression.substr(1,comma_position-1),"slice左端点");
        uint32_t right=parse_uint32(expression.substr(comma_position+1,expression.size()-comma_position-2),"slice右端点");
        uint64_t end=(uint64_t)right+(expression.back()==']');
        if((uint64_t)left>=end)
            throw std::runtime_error("slice区间为空或端点顺序错误："+expression);
        result.reserve(end-left);
        for(uint64_t slice=left;slice<end;slice++)
            result.push_back((uint32_t)slice);
        return result;
    }
    if(expression.size()<3||expression.front()!='{'||expression.back()!='}')
        throw std::runtime_error("slice必须写成区间'[a,b]'、'[a,b)'或列表'{a1,a2,...}'："+expression);
    if(expression[expression.size()-2]==',')
        throw std::runtime_error("slice列表末尾不能有逗号："+expression);
    std::unordered_set<uint32_t>used_slice_set;
    for(uint64_t begin=1;begin<expression.size()-1;)
    {
        uint64_t end=expression.find(',',begin);
        if(end==std::string::npos)
            end=expression.size()-1;
        uint32_t slice=parse_uint32(expression.substr(begin,end-begin),"slice");
        if(!used_slice_set.insert(slice).second)
            throw std::runtime_error("slice列表中存在重复值："+std::to_string(slice));
        result.push_back(slice);
        begin=end+1;
    }
    if(result.empty())
        throw std::runtime_error("slice列表不能为空");
    return result;
}

inline std::vector<std::string>parse_field_name_list(const std::string&expression)
{
    if(expression.size()<3||expression.front()!='{'||expression.back()!='}')
        throw std::runtime_error("field列表必须写成'{field1,field2,...}'："+expression);
    std::vector<std::string>result;
    std::unordered_set<std::string>used_name_set;
    for(uint64_t begin=1;begin<expression.size()-1;)
    {
        uint64_t end=expression.find(',',begin);
        if(end==std::string::npos)
            end=expression.size()-1;
        std::string name=expression.substr(begin,end-begin);
        if(name.empty())
            throw std::runtime_error("field列表不能包含空名称："+expression);
        if(!used_name_set.insert(name).second)
            throw std::runtime_error("field列表中存在重复名称："+name);
        result.push_back(std::move(name));
        begin=end+1;
    }
    if(result.empty())
        throw std::runtime_error("field列表不能为空");
    return result;
}

inline std::vector<uint32_t>parse_worker_id_list(const std::string&expression)
{
    if(expression.size()<3||expression.front()!='{'||expression.back()!='}')
        throw std::runtime_error("worker列表必须写成'{w1,w2,...}'："+expression);
    std::vector<uint32_t>result;
    for(uint64_t begin=1;begin<expression.size()-1;)
    {
        uint64_t end=expression.find(',',begin);
        if(end==std::string::npos)
            end=expression.size()-1;
        result.push_back(parse_uint32(expression.substr(begin,end-begin),"worker编号"));
        begin=end+1;
    }
    if(result.empty())
        throw std::runtime_error("worker列表不能为空");
    return result;
}
