#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::load_adjust_coord_matrix_file()
    {
        if (adjust_coord_matrix_file.empty())
            return;
        ifstream file(adjust_coord_matrix_file);
        if (!file)
            throw runtime_error("找不到坐标变换矩阵文件："+adjust_coord_matrix_file);
        string magic;
        uint32_t version;
        if (!(file>>magic>>version) || magic!="UNSZ_ADJUST_COORD_MATRIX" || version!=2)
            throw runtime_error("坐标变换矩阵文件格式或版本不正确："+adjust_coord_matrix_file);
        bool got_training_type=false,got_field=false,got_coordinate_dim=false,got_matrix=false;
        string matrix_field,key;
        uint32_t matrix_coordinate_dim=0;
        while (file>>key)
        {
            if (key=="training_type")
            {
                if (got_training_type || !(file>>adjust_coord_training_type))
                    throw runtime_error("坐标变换矩阵文件中的training_type不正确");
                got_training_type=true;
            }
            else if (key=="field")
            {
                if (got_field || !(file>>matrix_field))
                    throw runtime_error("坐标变换矩阵文件中的field不正确");
                got_field=true;
            }
            else if (key=="coordinate_dim")
            {
                if (got_coordinate_dim || !(file>>matrix_coordinate_dim))
                    throw runtime_error("坐标变换矩阵文件中的coordinate_dim不正确");
                got_coordinate_dim=true;
            }
            else if (key=="lower_triangular")
            {
                if (got_matrix || !got_coordinate_dim)
                    throw runtime_error("坐标变换矩阵文件中的lower_triangular重复或位于coordinate_dim之前");
                uint32_t parameter_num=matrix_coordinate_dim*(matrix_coordinate_dim+1)/2;
                adjust_coord_matrix_parameter_vec.resize(parameter_num);
                for (uint32_t parameter_id=0;parameter_id<parameter_num;parameter_id++)
                    if (!(file>>adjust_coord_matrix_parameter_vec[parameter_id]) || !isfinite(adjust_coord_matrix_parameter_vec[parameter_id]))
                        throw runtime_error("坐标变换矩阵文件中的矩阵元素数量不足或不是有限数");
                got_matrix=true;
            }
            else
                throw runtime_error("坐标变换矩阵文件包含未知字段："+key);
        }
        if (!got_training_type || !got_field || !got_coordinate_dim || !got_matrix)
            throw runtime_error("坐标变换矩阵文件缺少必要字段："+adjust_coord_matrix_file);
        if(adjust_coord_training_type!="spatial")
            throw runtime_error("坐标变换矩阵训练类型必须是spatial");
        if(matrix_coordinate_dim!=dim)
            throw runtime_error("坐标变换矩阵维数"+to_string(matrix_coordinate_dim)+"与当前coordinate_dim="+to_string(dim)+"不一致");
        if(matrix_field!=field_name)
            throw runtime_error("坐标变换矩阵field="+matrix_field+"与当前field="+field_name+"不一致");
        if_adjust_coord=true;
        printf("Loaded %s adjust coord matrix: field=%s, coordinate_dim=%u, path=%s\n",adjust_coord_training_type.c_str(),matrix_field.c_str(),matrix_coordinate_dim,adjust_coord_matrix_file.c_str());
    }
}
