#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::adjust_coord()
    {
        if(adjust_coord_matrix_parameter_vec.empty())
            throw runtime_error("启用坐标变换时必须先加载坐标变换矩阵");
        uint32_t expected_parameter_num=dim*(dim+1)/2;
        if(adjust_coord_matrix_parameter_vec.size()!=expected_parameter_num)
            throw runtime_error(to_string(dim)+"维坐标变换矩阵必须包含"+to_string(expected_parameter_num)+"个下三角元素");
        vector<T_coord> transform_matrix((uint64_t)dim*dim,0);
        uint32_t parameter_id=0;
        for(uint32_t row=0;row<dim;row++)
            for(uint32_t col=0;col<=row;col++)
                transform_matrix[(uint64_t)row*dim+col]=(T_coord)adjust_coord_matrix_parameter_vec[parameter_id++];

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
