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
        node_num=ftell(f_coord)/(dim*sizeof(T_coord));
        printf("node_num=%d\n",node_num);
        coord_array.resize(node_num,dim);
        fseek(f_coord,0,SEEK_SET);
        fread((char*)coord_array.data.data(),sizeof(T_coord),node_num*dim,f_coord);
        fclose(f_coord);

        min_coord.resize(dim,numeric_limits<T_coord>::infinity());
        max_coord.resize(dim,-numeric_limits<T_coord>::infinity());
        for (int i=0;i<node_num;i++)
            for (int d=0;d<dim;d++)
            {
                max_coord[d]=max(max_coord[d],coord_array[i][d]);
                min_coord[d]=min(min_coord[d],coord_array[i][d]);
            }
        printf("Coord range: [[%f, %f], [%f, %f], [%f, %f]]\n",min_coord[0],max_coord[0],min_coord[1],max_coord[1],min_coord[2],max_coord[2]);
    }
}