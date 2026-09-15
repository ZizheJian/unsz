#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::load_adjust_coord_matrix_file()
    {
        if (adjust_coord_matrix_path.empty())
            return;
        ifstream file(adjust_coord_matrix_path);
        if (!file)
            throw runtime_error("找不到坐标变换矩阵文件："+adjust_coord_matrix_path);
        string magic;
        uint32_t version;
        if (!(file>>magic>>version) || magic!="UNSZ_ADJUST_COORD_MATRIX" || version!=1)
            throw runtime_error("坐标变换矩阵文件格式或版本不正确："+adjust_coord_matrix_path);
        bool got_training_type=false,got_dataset=false,got_variable=false,got_dim=false,got_matrix=false;
        string matrix_dataset,matrix_variable,key;
        uint32_t matrix_dim=0;
        while (file>>key)
        {
            if (key=="training_type")
            {
                if (got_training_type || !(file>>adjust_coord_training_type))
                    throw runtime_error("坐标变换矩阵文件中的training_type不正确");
                got_training_type=true;
            }
            else if (key=="dataset")
            {
                if (got_dataset || !(file>>matrix_dataset))
                    throw runtime_error("坐标变换矩阵文件中的dataset不正确");
                got_dataset=true;
            }
            else if (key=="variable")
            {
                if (got_variable || !(file>>matrix_variable))
                    throw runtime_error("坐标变换矩阵文件中的variable不正确");
                got_variable=true;
            }
            else if (key=="dim")
            {
                if (got_dim || !(file>>matrix_dim))
                    throw runtime_error("坐标变换矩阵文件中的dim不正确");
                got_dim=true;
            }
            else if (key=="lower_triangular")
            {
                if (got_matrix || !got_dim)
                    throw runtime_error("坐标变换矩阵文件中的lower_triangular重复或位于dim之前");
                uint32_t parameter_num=matrix_dim*(matrix_dim+1)/2;
                adjust_coord_matrix_parameter_vec.resize(parameter_num);
                for (uint32_t parameter_id=0;parameter_id<parameter_num;parameter_id++)
                    if (!(file>>adjust_coord_matrix_parameter_vec[parameter_id]) || !isfinite(adjust_coord_matrix_parameter_vec[parameter_id]))
                        throw runtime_error("坐标变换矩阵文件中的矩阵元素数量不足或不是有限数");
                got_matrix=true;
            }
            else
                throw runtime_error("坐标变换矩阵文件包含未知字段："+key);
        }
        if (!got_training_type || !got_dataset || !got_variable || !got_dim || !got_matrix)
            throw runtime_error("坐标变换矩阵文件缺少必要字段："+adjust_coord_matrix_path);
        if (adjust_coord_training_type!="spatial" && adjust_coord_training_type!="spatiotemporal")
            throw runtime_error("坐标变换矩阵训练类型必须是spatial或spatiotemporal");
        uint32_t expected_dim=dim+(cross_time_spatial_point_num_experimental>0);
        if(matrix_dim!=expected_dim)
            throw runtime_error("坐标变换矩阵维数"+to_string(matrix_dim)+"与当前流程要求的dim="+to_string(expected_dim)+"不一致");
        if (matrix_variable!=variable_name)
            throw runtime_error("坐标变换矩阵field="+matrix_variable+"与当前field="+variable_name+"不一致");
        if_adjust_coord=true;
        printf("Loaded %s adjust coord matrix: dataset=%s, variable=%s, dim=%u, path=%s\n",adjust_coord_training_type.c_str(),matrix_dataset.c_str(),matrix_variable.c_str(),matrix_dim,adjust_coord_matrix_path.c_str());
    }
}
