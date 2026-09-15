#include <cstdio>
#include <exception>
#include "compress_cli.hpp"

int main(int argc,char**argv)
{
    try
    {
        return unsz::parse_args_and_select_types(argc,argv);
    }
    catch(const std::exception&error)
    {
        fprintf(stderr,"错误：%s\n",error.what());
        return 1;
    }
}
