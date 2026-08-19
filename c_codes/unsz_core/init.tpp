#pragma once

namespace unsz
{
    template<typename T_coord, typename T_data>
    void param_c<T_coord,T_data>::init()
    {
        FILE *f_coord=fopen(coord_path.c_str(),"rb");
        if (!f_coord)
            throw runtime_error("coord file not found");
        fclose(f_coord);
        for (auto &data_path:original_data_path_vec)
        {
            FILE *f_data=fopen(data_path.c_str(),"rb");
            if (!f_data)
                throw runtime_error("data file "+data_path+" not found");
            fclose(f_data);
        }

        f_coord=fopen(coord_path.c_str(),"rb");
        fseek(f_coord,0,SEEK_END);
        long coord_file_bytes_signed=ftell(f_coord);
        if (coord_file_bytes_signed<0)
            throw runtime_error("无法读取坐标文件大小");
        uint64_t coord_file_bytes=coord_file_bytes_signed;
        uint64_t point_bytes=(uint64_t)dim*sizeof(T_coord);
        if (coord_file_bytes%point_bytes!=0)
            throw runtime_error("坐标文件大小不能被单个点的坐标大小整除");
        uint64_t point_num64=coord_file_bytes/point_bytes;
        if (point_num64>numeric_limits<uint32_t>::max())
            throw runtime_error("点数超过uint32_t的表示范围");
        point_num=(uint32_t)point_num64;
        printf("point_num=%u\n",point_num);
        coord_array.resize(point_num,dim);
        fseek(f_coord,0,SEEK_SET);
        fread((char*)coord_array.data.data(),sizeof(T_coord),(uint64_t)point_num*dim,f_coord);
        fclose(f_coord);

        min_coord.resize(dim,numeric_limits<T_coord>::infinity());
        max_coord.resize(dim,-numeric_limits<T_coord>::infinity());
        for (uint32_t i=0;i<point_num;i++)
            for (uint32_t d=0;d<dim;d++)
            {
                max_coord[d]=max(max_coord[d],coord_array[i][d]);
                min_coord[d]=min(min_coord[d],coord_array[i][d]);
            }
        printf("Coord range: [[%f, %f], [%f, %f], [%f, %f]]\n",min_coord[0],max_coord[0],min_coord[1],max_coord[1],min_coord[2],max_coord[2]);
    }
}
