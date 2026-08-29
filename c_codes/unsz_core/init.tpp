#pragma once

namespace unsz
{
    template<typename T_coord, typename T_data>
    void param_c<T_coord,T_data>::init()
    {
        if (check_original_data_files)
            for (auto &data_path:original_data_path_vec)
            {
                FILE *f_data=fopen(data_path.c_str(),"rb");
                if (!f_data)
                    throw runtime_error("找不到数据文件："+data_path);
                if (fclose(f_data)!=0)
                    throw runtime_error("关闭数据文件失败："+data_path);
            }

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
