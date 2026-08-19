#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::adjust_coord()
    {
        if (cross_time_compress)
            throw runtime_error("暂未实现跨时间步的坐标调整");
        if (dim!=3)
            throw runtime_error("当前硬编码的坐标变换矩阵只支持三维坐标");

        // Katrina attr0使用的变换矩阵：
        // {
        //     {1.12670515e+00,0,0},
        //     {2.70622568e-01,1.28736120e+00,0},
        //     {-1.67488548e-04,-1.83679099e-04,3.29459469e-06}
        // }
        // Ocean temperature使用的变换矩阵：
        // {
        //     {0.57863758,0,0},
        //     {0.27448921,0.38810419,0},
        //     {-0.16613945,0.13620828,1.54695002}
        // }
        T_coord transform_matrix[3][3]={
            {1.12670515e+00,0,0},
            {2.70622568e-01,1.28736120e+00,0},
            {-1.67488548e-04,-1.83679099e-04,3.29459469e-06}
        };

        for (uint32_t point_id=0;point_id<point_num;point_id++)
        {
            T_coord transformed_coord[3]={0,0,0};
            for (uint32_t output_dim=0;output_dim<dim;output_dim++)
                for (uint32_t input_dim=0;input_dim<dim;input_dim++)
                    transformed_coord[output_dim]+=coord_array[point_id][input_dim]*transform_matrix[input_dim][output_dim];
            for (uint32_t d=0;d<dim;d++)
                coord_array[point_id][d]=transformed_coord[d];
        }

        min_coord.assign(dim,numeric_limits<T_coord>::infinity());
        max_coord.assign(dim,-numeric_limits<T_coord>::infinity());
        for (uint32_t point_id=0;point_id<point_num;point_id++)
            for (uint32_t d=0;d<dim;d++)
            {
                min_coord[d]=min(min_coord[d],coord_array[point_id][d]);
                max_coord[d]=max(max_coord[d],coord_array[point_id][d]);
            }
        printf("Adjusted coord range: [[%f, %f], [%f, %f], [%f, %f]]\n",min_coord[0],max_coord[0],min_coord[1],max_coord[1],min_coord[2],max_coord[2]);
    }
}
