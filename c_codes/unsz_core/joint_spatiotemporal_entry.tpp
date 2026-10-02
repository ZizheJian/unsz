#pragma once

namespace unsz
{
template<typename T_coord,typename T_data>
void param_c<T_coord,T_data>::prepare_joint_spatiotemporal_coordinates()
{
    uint32_t spatial_point_num=point_num;
    uint64_t time_step_num=time_step_vec.size();
    uint64_t joint_point_num=(uint64_t)spatial_point_num*time_step_num;
    if(time_step_num<2)
        throw runtime_error("联合时空压缩至少需要两个time step");
    if(joint_point_num>numeric_limits<uint32_t>::max())
        throw runtime_error("联合时空点数超过uint32_t的表示范围");
    uint32_t spatial_dim=dim,joint_dim=dim+1;
    Array2D<T_coord> joint_coord_array((uint32_t)joint_point_num,joint_dim);
    for(uint32_t time_id=0;time_id<time_step_num;time_id++)
    {
        T_coord time_coord=time_step_coordinate_vec.empty()?(T_coord)time_step_vec[time_id]:(T_coord)time_step_coordinate_vec[time_step_vec[time_id]];
        uint64_t point_offset=(uint64_t)time_id*spatial_point_num;
        for(uint32_t spatial_point_id=0;spatial_point_id<spatial_point_num;spatial_point_id++)
        {
            T_coord *joint_coord=joint_coord_array[(uint32_t)(point_offset+spatial_point_id)];
            memcpy(joint_coord,coord_array[spatial_point_id],(uint64_t)spatial_dim*sizeof(T_coord));
            joint_coord[spatial_dim]=time_coord;
        }
    }
    coord_array=move(joint_coord_array);
    point_num=(uint32_t)joint_point_num;
    dim=joint_dim;
    min_coord.assign(dim,numeric_limits<T_coord>::infinity());
    max_coord.assign(dim,-numeric_limits<T_coord>::infinity());
    for(uint32_t point_id=0;point_id<point_num;point_id++)
        for(uint32_t d=0;d<dim;d++)
        {
            min_coord[d]=min(min_coord[d],coord_array[point_id][d]);
            max_coord[d]=max(max_coord[d],coord_array[point_id][d]);
        }
    printf("Prepared joint spatiotemporal coordinates: spatial_points=%u,time_steps=%llu,joint_points=%u,dim=%u\n",spatial_point_num,(unsigned long long)time_step_num,point_num,dim);
}

template<typename T_coord,typename T_data>
template<point_set_codec_e predictor>
void param_c<T_coord,T_data>::compress_joint_spatiotemporal_files()
{
    if(original_data_path_vec.empty()||point_num%original_data_path_vec.size()!=0)
        throw runtime_error("联合时空点数与time step文件数量不一致");
    uint32_t spatial_point_num=point_num/original_data_path_vec.size();
    current_data.resize(point_num);
    for(uint32_t file_id=0;file_id<original_data_path_vec.size();file_id++)
    {
        const string &path=original_data_path_vec[file_id];
        FILE *file=fopen(path.c_str(),"rb");
        if(!file)
            throw runtime_error("找不到原始数据文件："+path);
        T_data *output=current_data.data()+(uint64_t)file_id*spatial_point_num;
        if(fread(output,sizeof(T_data),spatial_point_num,file)!=spatial_point_num)
        {
            fclose(file);
            throw runtime_error("读取原始数据文件失败："+path);
        }
        if(fclose(file)!=0)
            throw runtime_error("关闭原始数据文件失败："+path);
    }
    string compressed_path=data_dir+field_name+".joint.unsz";
    compressed_data_path_vec.push_back(compressed_path);
    printf("Compressing %llu files as one joint spatiotemporal field: %s\n",(unsigned long long)original_data_path_vec.size(),compressed_path.c_str());
    auto compression_start_time=chrono::steady_clock::now();
    one_point_set::spatial::compress<predictor>(*this);
    double compression_time=chrono::duration<double>(chrono::steady_clock::now()-compression_start_time).count();
    uint32_t primary_point_set_id=0;
    while(point_set_point_num_vec[primary_point_set_id]==0)
        primary_point_set_id++;
    double squared_error=0,max_error_ratio=0,max_error=0,max_error_abs_eb=0;
    uint32_t max_error_point_id=0,max_error_point_set_id=0;
    T_data minimum=numeric_limits<T_data>::infinity(),maximum=-numeric_limits<T_data>::infinity();
    for(uint32_t point_id=0;point_id<point_num;point_id++)
    {
        double error=(double)current_data[point_id]-current_decompressed_data[point_id];
        squared_error+=error*error;
        uint32_t point_set_id=point_set_id_vec[point_id];
        double validation_abs_eb=point_set_abs_eb_vec[point_set_id];
        double error_ratio;
        if(validation_abs_eb>0)
            error_ratio=abs(error)/validation_abs_eb;
        else if(error==0)
            error_ratio=0;
        else
            error_ratio=numeric_limits<double>::infinity();
        if(error_ratio>max_error_ratio)
        {
            max_error_ratio=error_ratio;
            max_error=error;
            max_error_abs_eb=validation_abs_eb;
            max_error_point_id=point_id;
            max_error_point_set_id=point_set_id;
        }
        if(point_set_id==primary_point_set_id)
        {
            minimum=min(minimum,current_data[point_id]);
            maximum=max(maximum,current_data[point_id]);
        }
    }
    printf("全部文件最大误差/用户误差上限= %.9f\n",max_error_ratio);
    if(max_error_ratio>1.01)
    {
        uint32_t time_id=max_error_point_id/spatial_point_num,spatial_point_id=max_error_point_id%spatial_point_num;
        throw runtime_error("联合时空压缩误差检查失败：time step="+to_string(time_step_vec[time_id])+"，空间点="+to_string(spatial_point_id)+"，点集="+to_string(max_error_point_set_id)+"，误差="+to_string(max_error)+"，绝对误差上限="+to_string(max_error_abs_eb)+"，误差/上限="+to_string(max_error_ratio));
    }
    FILE *file=fopen(compressed_path.c_str(),"wb");
    if(!file)
        throw runtime_error("无法创建联合时空压缩文件："+compressed_path);
    if(fwrite(current_compressed_bytes.data(),1,current_compressed_bytes.size(),file)!=current_compressed_bytes.size())
    {
        fclose(file);
        throw runtime_error("写入联合时空压缩文件失败："+compressed_path);
    }
    if(fclose(file)!=0)
        throw runtime_error("关闭联合时空压缩文件失败："+compressed_path);
    double cr=(double)point_num*sizeof(T_data)/current_compressed_bytes.size();
    double mse=squared_error/point_num;
    double psnr=mse==0?numeric_limits<double>::infinity():10*log10(pow((double)maximum-minimum,2)/mse);
    printf("total cr= %f, total br= %f, total psnr= %f\n",cr,sizeof(T_data)*8/cr,psnr);
    printf("total compression time= %f s\n",compression_time);
}

template<typename T_coord,typename T_data>
template<point_set_codec_e predictor>
void param_c<T_coord,T_data>::decompress_joint_spatiotemporal_files()
{
    if(original_data_path_vec.empty()||point_num%original_data_path_vec.size()!=0)
        throw runtime_error("联合时空点数与time step文件数量不一致");
    uint32_t spatial_point_num=point_num/original_data_path_vec.size();
    string compressed_path=data_dir+field_name+".joint.unsz";
    FILE *compressed_file=fopen(compressed_path.c_str(),"rb");
    if(!compressed_file)
        throw runtime_error("找不到联合时空压缩文件："+compressed_path);
    if(fseek(compressed_file,0,SEEK_END)!=0)
    {
        fclose(compressed_file);
        throw runtime_error("无法定位到联合时空压缩文件末尾："+compressed_path);
    }
    long compressed_byte_num_signed=ftell(compressed_file);
    if(compressed_byte_num_signed<0)
    {
        fclose(compressed_file);
        throw runtime_error("无法读取联合时空压缩文件大小："+compressed_path);
    }
    current_compressed_bytes.resize((uint64_t)compressed_byte_num_signed);
    if(fseek(compressed_file,0,SEEK_SET)!=0||fread(current_compressed_bytes.data(),1,current_compressed_bytes.size(),compressed_file)!=current_compressed_bytes.size())
    {
        fclose(compressed_file);
        throw runtime_error("读取联合时空压缩文件失败："+compressed_path);
    }
    if(fclose(compressed_file)!=0)
        throw runtime_error("关闭联合时空压缩文件失败："+compressed_path);
    auto decompression_start_time=chrono::steady_clock::now();
    decompress_current_file<false,time_dimension_mode_e::spatial,time_dimension_mode_e::not_applicable,label_value_execution_e::no_label,predictor>();
    double decompression_time=chrono::duration<double>(chrono::steady_clock::now()-decompression_start_time).count();
    double squared_error=0,max_error_ratio=0,max_error=0,max_error_abs_eb=0;
    uint64_t validated_point_num=0;
    uint32_t max_error_joint_point_id=0,max_error_point_set_id=0;
    string max_error_data_path;
    T_data minimum=numeric_limits<T_data>::infinity(),maximum=-numeric_limits<T_data>::infinity();
    for(uint32_t file_id=0;file_id<original_data_path_vec.size();file_id++)
    {
        uint64_t point_offset=(uint64_t)file_id*spatial_point_num;
        string output_path=original_data_path_vec[file_id]+".unsz.bin";
        FILE *output_file=fopen(output_path.c_str(),"wb");
        if(!output_file)
            throw runtime_error("无法创建解压文件："+output_path);
        if(fwrite(current_decompressed_data.data()+point_offset,sizeof(T_data),spatial_point_num,output_file)!=spatial_point_num)
        {
            fclose(output_file);
            throw runtime_error("写入解压文件失败："+output_path);
        }
        if(fclose(output_file)!=0)
            throw runtime_error("关闭解压文件失败："+output_path);
        FILE *original_file=fopen(original_data_path_vec[file_id].c_str(),"rb");
        if(!original_file)
        {
            printf("Decompressed %s（原始文件不存在，跳过误差验证）\n",output_path.c_str());
            continue;
        }
        vector<T_data> original_data(spatial_point_num);
        if(fread(original_data.data(),sizeof(T_data),spatial_point_num,original_file)!=spatial_point_num)
        {
            fclose(original_file);
            throw runtime_error("读取原始文件失败："+original_data_path_vec[file_id]);
        }
        if(fclose(original_file)!=0)
            throw runtime_error("关闭原始文件失败："+original_data_path_vec[file_id]);
        for(uint32_t spatial_point_id=0;spatial_point_id<spatial_point_num;spatial_point_id++)
        {
            uint32_t joint_point_id=(uint32_t)(point_offset+spatial_point_id);
            double error=(double)original_data[spatial_point_id]-current_decompressed_data[joint_point_id];
            squared_error+=error*error;
            uint32_t point_set_id=point_set_id_vec[joint_point_id];
            double validation_abs_eb=point_set_abs_eb_vec[point_set_id];
            double error_ratio;
            if(validation_abs_eb>0)
                error_ratio=abs(error)/validation_abs_eb;
            else if(error==0)
                error_ratio=0;
            else
                error_ratio=numeric_limits<double>::infinity();
            if(error_ratio>max_error_ratio)
            {
                max_error_ratio=error_ratio;
                max_error=error;
                max_error_abs_eb=validation_abs_eb;
                max_error_joint_point_id=joint_point_id;
                max_error_point_set_id=point_set_id;
                max_error_data_path=original_data_path_vec[file_id];
            }
            minimum=min(minimum,original_data[spatial_point_id]);
            maximum=max(maximum,original_data[spatial_point_id]);
        }
        validated_point_num+=spatial_point_num;
        printf("Decompressed %s\n",output_path.c_str());
    }
    printf("全部文件最大误差/用户误差上限= %.9f\n",max_error_ratio);
    if(max_error_ratio>1.01)
    {
        uint32_t time_id=max_error_joint_point_id/spatial_point_num,spatial_point_id=max_error_joint_point_id%spatial_point_num;
        throw runtime_error("联合时空解压误差检查失败：文件="+max_error_data_path+"，time step="+to_string(time_step_vec[time_id])+"，空间点="+to_string(spatial_point_id)+"，点集="+to_string(max_error_point_set_id)+"，误差="+to_string(max_error)+"，绝对误差上限="+to_string(max_error_abs_eb)+"，误差/上限="+to_string(max_error_ratio));
    }
    printf("total decompression time= %f s\n",decompression_time);
    if(validated_point_num>0)
    {
        double mse=squared_error/validated_point_num;
        double psnr=mse==0?numeric_limits<double>::infinity():10*log10(pow((double)maximum-minimum,2)/mse);
        printf("total decompressed psnr= %f\n",psnr);
    }
}
}
