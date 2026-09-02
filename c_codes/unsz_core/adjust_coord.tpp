#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::adjust_coord()
    {
        if (cross_time_compress)
            throw runtime_error("暂未实现跨时间步的坐标调整");

        // Katrina attr0使用的变换矩阵：
        // {
        //     {1.12670515e+00,0,0},
        //     {2.70622568e-01,1.28736120e+00,0},
        //     {-1.67488548e-04,-1.83679099e-04,3.29459469e-06}
        // }
        // Katrina attr0使用二次方损失训练10轮得到的候选变换矩阵：
        // {
        //     {1.18623191e+00,0,0},
        //     {-1.26722765e-01,1.25567535e+00,0},
        //     {-1.21165580e-03,-4.67482022e-04,8.54009806e-03}
        // }
        // Katrina attr0使用二次方损失训练20轮得到的候选变换矩阵：
        // {
        //     {1.08447056e+00,0,0},
        //     {2.78612264e-02,1.35023964e+00,0},
        //     {-4.30238779e-04,-1.83334089e-04,2.42928960e-04}
        // }
        // Katrina attr0使用二次方损失、第二次采样率0.05、吸收率0.5训练4轮得到的速度优先矩阵：
        // {
        //     {1.3637873843345396,0,0},
        //     {-0.1070691818966973,1.0616479726221577,0},
        //     {-0.005332455203166994,-0.00019726101388385328,0.038668706401795774}
        // }
        // Ocean temperature使用的变换矩阵：
        // {
        //     {0.57863758,0,0},
        //     {0.27448921,0.38810419,0},
        //     {-0.16613945,0.13620828,1.54695002}
        // }
        vector<T_coord> transform_matrix((uint64_t)dim*dim,0);
        for (uint32_t d=0;d<dim;d++)
            transform_matrix[(uint64_t)d*dim+d]=1;
        if (dim==3)
        {
            T_coord default_transform_matrix[6]={1.320727945153478,-0.050022502016393564,1.119176596028389,-0.0022639755300411725,0.00041386801901295816,0.02477688868808782};
            uint32_t parameter_id=0;
            for (uint32_t row=0;row<dim;row++)
                for (uint32_t col=0;col<=row;col++)
                    transform_matrix[(uint64_t)row*dim+col]=default_transform_matrix[parameter_id++];
        }
        if (!adjust_coord_matrix_parameter_vec.empty())
        {
            uint32_t expected_parameter_num=dim*(dim+1)/2;
            if (adjust_coord_matrix_parameter_vec.size()!=expected_parameter_num)
                throw runtime_error(to_string(dim)+"维坐标变换矩阵必须包含"+to_string(expected_parameter_num)+"个下三角元素");
            uint32_t parameter_id=0;
            for (uint32_t row=0;row<dim;row++)
                for (uint32_t col=0;col<=row;col++)
                    transform_matrix[(uint64_t)row*dim+col]=(T_coord)adjust_coord_matrix_parameter_vec[parameter_id++];
        }

        for (uint32_t point_id=0;point_id<point_num;point_id++)
        {
            vector<T_coord> transformed_coord(dim,0);
            for (uint32_t output_dim=0;output_dim<dim;output_dim++)
                for (uint32_t input_dim=0;input_dim<dim;input_dim++)
                    transformed_coord[output_dim]+=coord_array[point_id][input_dim]*transform_matrix[(uint64_t)input_dim*dim+output_dim];
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
        printf("Adjusted coord range: [");
        for (uint32_t d=0;d<dim;d++)
            printf("%s[%f, %f]",d==0?"":", ",(double)min_coord[d],(double)max_coord[d]);
        printf("]\n");
    }
}
