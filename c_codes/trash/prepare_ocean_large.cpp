#include <netcdf.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void check_nc(int status,const std::string&operation){
    if(status!=NC_NOERR)
        throw std::runtime_error(operation+": "+nc_strerror(status));
}

static int get_dim_id(int file_id,const char*name){
    int dim_id=-1;
    check_nc(nc_inq_dimid(file_id,name,&dim_id),std::string("找不到维度 ")+name);
    return dim_id;
}

static size_t get_dim_size(int file_id,const char*name){
    size_t size=0;
    check_nc(nc_inq_dimlen(file_id,get_dim_id(file_id,name),&size),std::string("无法读取维度 ")+name);
    return size;
}

template<class T>
static void write_binary(const std::filesystem::path&path,const std::vector<T>&data){
    std::ofstream output(path,std::ios::binary);
    if(!output)
        throw std::runtime_error("无法创建文件："+path.string());
    output.write(reinterpret_cast<const char*>(data.data()),static_cast<std::streamsize>(data.size()*sizeof(T)));
    if(!output)
        throw std::runtime_error("写文件失败："+path.string());
}

static void prepare_coords(int file_id,size_t point_num,const std::filesystem::path&output_path){
    std::vector<float> coords(point_num*3),component(point_num);
    const char*names[]={"xCell","yCell","zCell"};
    for(size_t dim=0;dim<3;dim++){
        int var_id=-1;
        check_nc(nc_inq_varid(file_id,names[dim],&var_id),std::string("找不到坐标 ")+names[dim]);
        check_nc(nc_get_var_float(file_id,var_id,component.data()),std::string("无法读取坐标 ")+names[dim]);
        for(size_t point_id=0;point_id<point_num;point_id++)
            coords[point_id*3+dim]=component[point_id];
    }
    write_binary(output_path,coords);
}

static void validate_file_size(const std::filesystem::path&path,uintmax_t expected_size,const std::string&name){
    uintmax_t actual_size=std::filesystem::file_size(path);
    if(actual_size!=expected_size)
        throw std::runtime_error(name+"大小不一致：expected="+std::to_string(expected_size)+",actual="+std::to_string(actual_size));
}

static size_t prepare_connectivity(int file_id,size_t point_num,size_t edge_num,const std::filesystem::path&output_path){
    int var_id=-1;
    check_nc(nc_inq_varid(file_id,"cellsOnEdge",&var_id),"找不到cellsOnEdge");
    std::vector<int32_t> source(edge_num*2);
    check_nc(nc_get_var_int(file_id,var_id,source.data()),"无法读取cellsOnEdge");
    std::vector<int32_t> edges;
    edges.reserve(edge_num*2);
    for(size_t edge_id=0;edge_id<edge_num;edge_id++){
        int32_t first=source[edge_id*2],second=source[edge_id*2+1];
        if(first<=0||second<=0)
            continue;
        first--;
        second--;
        if(static_cast<size_t>(first)>=point_num||static_cast<size_t>(second)>=point_num)
            throw std::runtime_error("cellsOnEdge包含越界cell编号");
        edges.push_back(first);
        edges.push_back(second);
    }
    write_binary(output_path,edges);
    return edges.size()/2;
}

static void prepare_field(int file_id,const std::string&field,int level,size_t point_num,const std::filesystem::path&output_path){
    int var_id=-1;
    check_nc(nc_inq_varid(file_id,field.c_str(),&var_id),"找不到field "+field);
    int dim_num=0;
    check_nc(nc_inq_varndims(file_id,var_id,&dim_num),"无法读取field维数");
    std::vector<int> dim_ids(dim_num);
    check_nc(nc_inq_vardimid(file_id,var_id,dim_ids.data()),"无法读取field维度");
    int time_dim=get_dim_id(file_id,"Time"),cell_dim=get_dim_id(file_id,"nCells");
    if(dim_num!=2&&dim_num!=3)
        throw std::runtime_error("field必须是(Time,nCells)或(Time,nCells,level)");
    if(dim_ids[0]!=time_dim||dim_ids[1]!=cell_dim)
        throw std::runtime_error("field前两个维度不是(Time,nCells)");
    std::vector<size_t> start(dim_num,0),count(dim_num,1);
    count[1]=point_num;
    if(dim_num==3){
        size_t level_num=0;
        check_nc(nc_inq_dimlen(file_id,dim_ids[2],&level_num),"无法读取field层数");
        if(level<0||static_cast<size_t>(level)>=level_num)
            throw std::runtime_error("三维field需要有效的-level，范围为0～"+std::to_string(level_num-1));
        start[2]=static_cast<size_t>(level);
    }
    else if(level>=0)
        throw std::runtime_error("二维field不能指定-level");
    std::vector<float> data(point_num);
    check_nc(nc_get_vara_float(file_id,var_id,start.data(),count.data(),data.data()),"无法读取field数据");
    size_t finite_num=0;
    float min_value=0,max_value=0;
    for(float value:data){
        if(!std::isfinite(value))
            continue;
        if(finite_num==0)
            min_value=max_value=value;
        else{
            min_value=std::min(min_value,value);
            max_value=std::max(max_value,value);
        }
        finite_num++;
    }
    if(finite_num!=point_num)
        throw std::runtime_error("field包含"+std::to_string(point_num-finite_num)+"个非有限值，尚未生成点集标签");
    write_binary(output_path,data);
    std::cout<<"field range = ["<<min_value<<","<<max_value<<"]\n";
}

int main(int argc,char**argv){
    try{
        if(argc!=4&&argc!=6)
            throw std::runtime_error("用法：prepare_ocean_large <source.nc> <output_dir> <field> [-level <id>]");
        std::filesystem::path source_path=argv[1],output_dir=argv[2];
        std::string field=argv[3];
        int level=-1;
        if(argc==6){
            if(std::string(argv[4])!="-level")
                throw std::runtime_error("第4个参数必须是-level");
            level=std::stoi(argv[5]);
        }
        std::filesystem::create_directories(output_dir);
        int file_id=-1;
        check_nc(nc_open(source_path.c_str(),NC_NOWRITE,&file_id),"无法打开NetCDF");
        try{
            size_t point_num=get_dim_size(file_id,"nCells"),edge_num=get_dim_size(file_id,"nEdges");
            std::filesystem::path coord_path=output_dir/"coords.dat",conn_path=output_dir/"conn.dat";
            if(!std::filesystem::exists(coord_path))
                prepare_coords(file_id,point_num,coord_path);
            else
                validate_file_size(coord_path,point_num*3*sizeof(float),"coords.dat");
            size_t valid_edge_num=0;
            if(!std::filesystem::exists(conn_path))
                valid_edge_num=prepare_connectivity(file_id,point_num,edge_num,conn_path);
            else{
                uintmax_t conn_size=std::filesystem::file_size(conn_path);
                if(conn_size%(2*sizeof(int32_t))!=0)
                    throw std::runtime_error("conn.dat大小不是两端点int32记录的整数倍");
                valid_edge_num=conn_size/(2*sizeof(int32_t));
            }
            std::string output_name=field;
            if(level>=0)
                output_name+="_level"+std::to_string(level);
            prepare_field(file_id,field,level,point_num,output_dir/(output_name+".dat.0"));
            check_nc(nc_close(file_id),"无法关闭NetCDF");
            file_id=-1;
            std::cout<<"Ocean-large转换完成：point_num="<<point_num<<",edge_num="<<valid_edge_num<<",field="<<output_name<<"\n";
        }
        catch(...){
            if(file_id>=0)
                nc_close(file_id);
            throw;
        }
    }
    catch(const std::exception&error){
        std::cerr<<"Ocean-large转换失败："<<error.what()<<"\n";
        return 1;
    }
    return 0;
}
