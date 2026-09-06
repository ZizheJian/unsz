#pragma once

namespace unsz
{
    enum class json_type_e:uint8_t{null_value,bool_value,number_value,string_value,array_value,object_value};

    struct json_value_c
    {
        json_type_e type=json_type_e::null_value;
        bool bool_value=false;
        double number_value=0;
        string string_value;
        vector<json_value_c> array_value;
        map<string,json_value_c> object_value;
    };

    class json_parser_c
    {
        private:
            const string &text;
            uint64_t position=0;

            [[noreturn]] void fail(const string &message) const
            {
                throw runtime_error("JSON第"+to_string(position+1)+"个字符附近格式错误："+message);
            }

            void skip_space()
            {
                while(position<text.size() && isspace((unsigned char)text[position]))
                    position++;
            }

            bool consume(char expected)
            {
                skip_space();
                if(position<text.size() && text[position]==expected)
                {
                    position++;
                    return true;
                }
                return false;
            }

            void expect(char expected)
            {
                if(!consume(expected))
                    fail("缺少字符'"+string(1,expected)+"'");
            }

            uint32_t parse_hex4()
            {
                if(position+4>text.size())
                    fail("不完整的Unicode转义");
                uint32_t value=0;
                for(uint32_t i=0;i<4;i++)
                {
                    char c=text[position++];
                    value*=16;
                    if(c>='0' && c<='9')
                        value+=c-'0';
                    else if(c>='a' && c<='f')
                        value+=c-'a'+10;
                    else if(c>='A' && c<='F')
                        value+=c-'A'+10;
                    else
                        fail("Unicode转义包含非十六进制字符");
                }
                return value;
            }

            void append_utf8(string &result,uint32_t code_point)
            {
                if(code_point<=0x7f)
                    result.push_back((char)code_point);
                else if(code_point<=0x7ff)
                {
                    result.push_back((char)(0xc0|(code_point>>6)));
                    result.push_back((char)(0x80|(code_point&0x3f)));
                }
                else if(code_point<=0xffff)
                {
                    result.push_back((char)(0xe0|(code_point>>12)));
                    result.push_back((char)(0x80|((code_point>>6)&0x3f)));
                    result.push_back((char)(0x80|(code_point&0x3f)));
                }
                else
                {
                    result.push_back((char)(0xf0|(code_point>>18)));
                    result.push_back((char)(0x80|((code_point>>12)&0x3f)));
                    result.push_back((char)(0x80|((code_point>>6)&0x3f)));
                    result.push_back((char)(0x80|(code_point&0x3f)));
                }
            }

            string parse_string()
            {
                skip_space();
                if(position>=text.size() || text[position]!='"')
                    fail("需要字符串");
                position++;
                string result;
                while(position<text.size())
                {
                    unsigned char c=(unsigned char)text[position++];
                    if(c=='"')
                        return result;
                    if(c<0x20)
                        fail("字符串包含控制字符");
                    if(c!='\\')
                    {
                        result.push_back((char)c);
                        continue;
                    }
                    if(position>=text.size())
                        fail("不完整的字符串转义");
                    char escaped=text[position++];
                    if(escaped=='"' || escaped=='\\' || escaped=='/')
                        result.push_back(escaped);
                    else if(escaped=='b')
                        result.push_back('\b');
                    else if(escaped=='f')
                        result.push_back('\f');
                    else if(escaped=='n')
                        result.push_back('\n');
                    else if(escaped=='r')
                        result.push_back('\r');
                    else if(escaped=='t')
                        result.push_back('\t');
                    else if(escaped=='u')
                    {
                        uint32_t code_point=parse_hex4();
                        if(code_point>=0xd800 && code_point<=0xdbff)
                        {
                            if(position+2>text.size() || text[position]!='\\' || text[position+1]!='u')
                                fail("Unicode高代理项后缺少低代理项");
                            position+=2;
                            uint32_t low=parse_hex4();
                            if(low<0xdc00 || low>0xdfff)
                                fail("Unicode低代理项无效");
                            code_point=0x10000+((code_point-0xd800)<<10)+(low-0xdc00);
                        }
                        else if(code_point>=0xdc00 && code_point<=0xdfff)
                            fail("Unicode低代理项前缺少高代理项");
                        append_utf8(result,code_point);
                    }
                    else
                        fail("未知的字符串转义");
                }
                fail("字符串没有结束引号");
            }

            json_value_c parse_number()
            {
                skip_space();
                uint64_t begin=position;
                if(position<text.size() && text[position]=='-')
                    position++;
                if(position>=text.size())
                    fail("不完整的数值");
                if(text[position]=='0')
                    position++;
                else if(text[position]>='1' && text[position]<='9')
                    while(position<text.size() && isdigit((unsigned char)text[position]))
                        position++;
                else
                    fail("无效的数值");
                if(position<text.size() && text[position]=='.')
                {
                    position++;
                    uint64_t decimal_begin=position;
                    while(position<text.size() && isdigit((unsigned char)text[position]))
                        position++;
                    if(position==decimal_begin)
                        fail("小数点后缺少数字");
                }
                if(position<text.size() && (text[position]=='e' || text[position]=='E'))
                {
                    position++;
                    if(position<text.size() && (text[position]=='+' || text[position]=='-'))
                        position++;
                    uint64_t exponent_begin=position;
                    while(position<text.size() && isdigit((unsigned char)text[position]))
                        position++;
                    if(position==exponent_begin)
                        fail("指数部分缺少数字");
                }
                string number_text=text.substr(begin,position-begin);
                char *number_end=nullptr;
                double value=strtod(number_text.c_str(),&number_end);
                if(number_end!=number_text.c_str()+number_text.size() || !isfinite(value))
                    fail("数值超出double范围");
                json_value_c result;
                result.type=json_type_e::number_value;
                result.number_value=value;
                return result;
            }

            json_value_c parse_array()
            {
                json_value_c result;
                result.type=json_type_e::array_value;
                expect('[');
                if(consume(']'))
                    return result;
                while(true)
                {
                    result.array_value.push_back(parse_value());
                    if(consume(']'))
                        return result;
                    expect(',');
                }
            }

            json_value_c parse_object()
            {
                json_value_c result;
                result.type=json_type_e::object_value;
                expect('{');
                if(consume('}'))
                    return result;
                while(true)
                {
                    string key=parse_string();
                    expect(':');
                    if(!result.object_value.emplace(key,parse_value()).second)
                        fail("对象字段重复："+key);
                    if(consume('}'))
                        return result;
                    expect(',');
                }
            }

            json_value_c parse_value()
            {
                skip_space();
                if(position>=text.size())
                    fail("缺少值");
                if(text[position]=='{')
                    return parse_object();
                if(text[position]=='[')
                    return parse_array();
                if(text[position]=='"')
                {
                    json_value_c result;
                    result.type=json_type_e::string_value;
                    result.string_value=parse_string();
                    return result;
                }
                if(text[position]=='-' || isdigit((unsigned char)text[position]))
                    return parse_number();
                for(const pair<const char*,json_type_e> &literal:array<pair<const char*,json_type_e>,3>{{{"true",json_type_e::bool_value},{"false",json_type_e::bool_value},{"null",json_type_e::null_value}}})
                {
                    uint64_t length=strlen(literal.first);
                    if(text.compare(position,length,literal.first)==0)
                    {
                        position+=length;
                        json_value_c result;
                        result.type=literal.second;
                        result.bool_value=literal.first[0]=='t';
                        return result;
                    }
                }
                fail("未知的值");
            }

        public:
            json_parser_c(const string &text_):text(text_){}

            json_value_c parse()
            {
                json_value_c result=parse_value();
                skip_space();
                if(position!=text.size())
                    fail("根值后还有多余内容");
                return result;
            }
    };
}
