#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::solve_4x4(double A_in[4][4],double b_in[4],double x[4])
    {
        double A[4][5];
        for (uint32_t i=0;i<4;i++)
        {
            for (uint32_t j=0;j<4;j++)
                A[i][j]=A_in[i][j];
            A[i][4]=b_in[i];
        }
        for (uint32_t col=0;col<4;col++)
        {
            uint32_t pivot=col;
            for (uint32_t row=col+1;row<4;row++)
                if (abs(A[row][col])>abs(A[pivot][col]))
                    pivot=row;
            if (abs(A[pivot][col])<1e-12)
                return false;
            if (pivot!=col)
                for (uint32_t i=col;i<5;i++)
                    swap(A[col][i],A[pivot][i]);
            double divisor=A[col][col];
            for (uint32_t i=col;i<5;i++)
                A[col][i]/=divisor;
            for (uint32_t row=0;row<4;row++)
            {
                if (row==col)
                    continue;
                double factor=A[row][col];
                for (uint32_t i=col;i<5;i++)
                    A[row][i]-=factor*A[col][i];
            }
        }
        for (uint32_t i=0;i<4;i++)
            x[i]=A[i][4];
        return true;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::compress()
    {
        if (current_data.size()!=point_num)
            throw runtime_error("当前数据点数与坐标点数不一致");
        if (!got_rel_eb && !got_abs_eb && point_set_rel_eb_config_vec.empty() && point_set_abs_eb_config_vec.empty())
            throw runtime_error("没有指定误差上限");

        point_set_id_vec.resize(point_num);
        uint32_t point_set_num=0;
        for (uint32_t i=0;i<point_num;i++)
        {
            point_set_id_vec[i]=get_point_set_id(current_data[i]);
            point_set_num=max(point_set_num,point_set_id_vec[i]+1);
        }
        if (point_set_num>2)
            throw runtime_error("当前0/1标签修正流只支持最多两个点集");
        point_set_point_num_vec.assign(point_set_num,0);
        point_set_min_data_vec.assign(point_set_num,numeric_limits<T_data>::infinity());
        point_set_max_data_vec.assign(point_set_num,-numeric_limits<T_data>::infinity());
        for (uint32_t i=0;i<point_num;i++)
        {
            uint32_t point_set_id=point_set_id_vec[i];
            point_set_point_num_vec[point_set_id]++;
            point_set_min_data_vec[point_set_id]=min(point_set_min_data_vec[point_set_id],current_data[i]);
            point_set_max_data_vec[point_set_id]=max(point_set_max_data_vec[point_set_id],current_data[i]);
        }
        point_set_abs_eb_vec.resize(point_set_num);
        point_set_codec_vec.resize(point_set_num);
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            if (point_set_point_num_vec[point_set_id]==0)
            {
                point_set_min_data_vec[point_set_id]=0;
                point_set_max_data_vec[point_set_id]=0;
                point_set_abs_eb_vec[point_set_id]=0;
                point_set_codec_vec[point_set_id]=point_set_codec_e::constant;
                continue;
            }
            bool is_constant=point_set_min_data_vec[point_set_id]==point_set_max_data_vec[point_set_id];
            if (is_constant)
                point_set_codec_vec[point_set_id]=point_set_codec_e::constant;
            else if (point_set_id<point_set_codec_config_vec.size())
                point_set_codec_vec[point_set_id]=point_set_codec_config_vec[point_set_id];
            else if (predictor=="inverse_proportion")
                point_set_codec_vec[point_set_id]=point_set_codec_e::inverse_proportion;
            else if (predictor=="fitting")
                point_set_codec_vec[point_set_id]=point_set_codec_e::fitting;
            else
                throw runtime_error("未知的默认数值预测器："+predictor);
            if (!is_constant && point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                throw runtime_error("非常量点集不能使用常量编码");
            if (point_set_id<point_set_abs_eb_config_vec.size())
                point_set_abs_eb_vec[point_set_id]=point_set_abs_eb_config_vec[point_set_id];
            else if (point_set_id<point_set_rel_eb_config_vec.size())
                point_set_abs_eb_vec[point_set_id]=point_set_rel_eb_config_vec[point_set_id]*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]);
            else
                point_set_abs_eb_vec[point_set_id]=got_abs_eb?abs_eb:rel_eb*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]);
            if (!is_constant && !(point_set_abs_eb_vec[point_set_id]>0))
                throw runtime_error("点集"+to_string(point_set_id)+"的绝对误差上限必须大于0");
        }
        uint32_t primary_point_set_id=0;
        while (point_set_point_num_vec[primary_point_set_id]==0)
            primary_point_set_id++;
        current_min_data=point_set_min_data_vec[primary_point_set_id];
        current_max_data=point_set_max_data_vec[primary_point_set_id];
        current_abs_eb=point_set_abs_eb_vec[primary_point_set_id];
        current_valid_node_num=point_set_point_num_vec[0];

        vector<uint32_t> label_symbol_vec;
        vector<uint32_t> label_symbol_original_order_vec(point_num,0);
        vector<uint32_t> actual_label_prediction_order_vec;
        uint64_t label_prediction_error_num=0;
        if (point_set_num>1)
        {
            bool continuous=point_set_predictor=="continuous_actual" || point_set_predictor=="continuous_reflect";
            point_set_prediction_value_vec.assign(point_num,0);
            label_symbol_vec.reserve(point_num);
            actual_label_prediction_order_vec.reserve(point_num);
            for (uint32_t pred_ref_id=0;pred_ref_id<pred_refs_vec.size();pred_ref_id++)
            {
                auto &pred_refs=pred_refs_vec[pred_ref_id];
                auto &ref_weights=pred_ref_weights_vec[pred_ref_id];
                int pred_id=pred_refs.first;
                double predicted_value=continuous?predict_point_set_value(pred_refs.second,ref_weights):0;
                uint32_t predicted_id=continuous?(predicted_value>=0.5):predict_point_set_id(pred_refs.second,ref_weights,point_set_num);
                uint32_t actual_id=point_set_id_vec[pred_id];
                actual_label_prediction_order_vec.push_back(actual_id);
                bool prediction_error=actual_id!=predicted_id;
                label_prediction_error_num+=prediction_error;
                label_symbol_vec.push_back(prediction_error?1:0);
                label_symbol_original_order_vec[pred_id]=prediction_error?1:0;
                if (continuous)
                {
                    if (!prediction_error)
                        point_set_prediction_value_vec[pred_id]=predicted_value;
                    else if (point_set_predictor=="continuous_actual" || predicted_value==0.5)
                        point_set_prediction_value_vec[pred_id]=actual_id;
                    else
                        point_set_prediction_value_vec[pred_id]=1-predicted_value;
                    uint32_t rounded_id=point_set_prediction_value_vec[pred_id]>=0.5;
                    if (rounded_id!=actual_id)
                        throw runtime_error("连续点集标签状态无法恢复真实标签");
                }
            }
        }

        current_decompressed_data.assign(point_num,0);
        vector<vector<int64_t>> point_set_qb_vec(point_set_num);
        uint64_t value_huffman_byte_num=0;
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
            if (point_set_codec_vec[point_set_id]!=point_set_codec_e::constant)
                point_set_qb_vec[point_set_id].reserve(point_set_point_num_vec[point_set_id]);
        vector<vector<int64_t>> round_qb_vec(print_round_statistics?predict_round_num:0);
        uint64_t pred_ref_index=0;
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
            for (uint32_t i=0;i<round_target_ids[round_id].size();i++)
            {
                auto &pred_refs=pred_refs_vec[pred_ref_index];
                auto &ref_weights=pred_ref_weights_vec[pred_ref_index++];
                int pred_id=pred_refs.first;
                uint32_t point_set_id=point_set_id_vec[pred_id];
                if (point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                {
                    current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
                    continue;
                }
                T_data pred=predict_value(pred_id,pred_refs.second,ref_weights,point_set_id,current_decompressed_data);
                int64_t qb=llround(((double)current_data[pred_id]-pred)/(2*point_set_abs_eb_vec[point_set_id]));
                point_set_qb_vec[point_set_id].push_back(qb);
                if (print_round_statistics)
                    round_qb_vec[round_id].push_back(qb);
                current_decompressed_data[pred_id]=pred+qb*2*point_set_abs_eb_vec[point_set_id];
            }
        if (pred_ref_index!=pred_refs_vec.size())
            throw runtime_error("逐轮预测目标数量与预测参考结构不一致");

        vector<unsigned char> container;
        const unsigned char magic[8]={'U','N','S','Z','P','S','0','6'};
        container.insert(container.end(),magic,magic+8);
        append_binary<uint32_t>(container,6);
        append_binary<uint32_t>(container,point_num);
        append_binary<uint32_t>(container,sizeof(T_data));
        append_binary<uint32_t>(container,point_set_num);
        vector<unsigned char> label_position_delta_bytes=point_set_num==1?vector<unsigned char>():binary_one_position_delta_vec(label_symbol_original_order_vec);
        vector<unsigned char> label_zstd_bytes=label_position_delta_bytes.empty()?vector<unsigned char>():zstd_compress(label_position_delta_bytes,zstd_compression_level);
        if (point_set_num>1 && print_label_compression_statistics)
        {
            for (int zstd_level:{1,3,6,9,15,19})
                printf("label_position_delta_zstd_level=%d, bytes=%llu\n",zstd_level,(unsigned long long)zstd_compress(label_position_delta_bytes,zstd_level).size());
            auto print_label_candidate=[&](const char *name,vector<unsigned char> bytes)
            {
                printf("label_candidate=%s, bytes_before_zstd=%llu, bytes_after_zstd=%llu\n",name,
                    (unsigned long long)bytes.size(),(unsigned long long)zstd_size(bytes));
            };
            print_label_candidate("actual_original_byte",binary_to_byte_vec(point_set_id_vec));
            print_label_candidate("actual_original_bit",binary_to_bit_vec(point_set_id_vec));
            print_label_candidate("actual_original_rle",binary_run_length_vec(point_set_id_vec));
            print_label_candidate("actual_prediction_order_byte",binary_to_byte_vec(actual_label_prediction_order_vec));
            print_label_candidate("actual_prediction_order_bit",binary_to_bit_vec(actual_label_prediction_order_vec));
            print_label_candidate("error_byte",binary_to_byte_vec(label_symbol_vec));
            print_label_candidate("error_bit",binary_to_bit_vec(label_symbol_vec));
            print_label_candidate("error_position_delta",binary_one_position_delta_vec(label_symbol_vec));
            print_label_candidate("error_rle",binary_run_length_vec(label_symbol_vec));
            print_label_candidate("error_original_order_bit",binary_to_bit_vec(label_symbol_original_order_vec));
            print_label_candidate("error_original_order_position_delta",binary_one_position_delta_vec(label_symbol_original_order_vec));
            print_label_candidate("error_original_order_rle",binary_run_length_vec(label_symbol_original_order_vec));
            uint64_t round_bit_bytes=0,round_zstd_bytes=0,label_begin=0;
            for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
            {
                uint64_t label_end=label_begin+round_target_ids[round_id].size();
                vector<uint32_t> round_label_vec(label_symbol_vec.begin()+label_begin,label_symbol_vec.begin()+label_end);
                vector<unsigned char> round_bit_vec=binary_to_bit_vec(round_label_vec);
                round_bit_bytes+=round_bit_vec.size();
                round_zstd_bytes+=zstd_size(round_bit_vec);
                label_begin=label_end;
            }
            printf("label_candidate=error_bit_by_round, bytes_before_zstd=%llu, bytes_after_zstd=%llu\n",
                (unsigned long long)round_bit_bytes,(unsigned long long)round_zstd_bytes);
        }
        append_binary<uint64_t>(container,label_zstd_bytes.size());
        container.insert(container.end(),label_zstd_bytes.begin(),label_zstd_bytes.end());
        uint64_t value_zstd_byte_num=0;
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            append_binary<uint8_t>(container,(uint8_t)point_set_codec_vec[point_set_id]);
            append_binary<uint32_t>(container,point_set_point_num_vec[point_set_id]);
            append_binary<T_data>(container,point_set_min_data_vec[point_set_id]);
            append_binary<T_data>(container,point_set_max_data_vec[point_set_id]);
            append_binary<uint64_t>(container,point_set_qb_vec[point_set_id].size());
            vector<unsigned char> qb_huffman_bytes=huffman_encode(point_set_qb_vec[point_set_id]);
            vector<unsigned char> qb_zstd_bytes=qb_huffman_bytes.empty()?vector<unsigned char>():zstd_compress(qb_huffman_bytes,zstd_compression_level);
            if (!qb_huffman_bytes.empty() && print_label_compression_statistics)
                for (int zstd_level:{1,3,6,9,15,19})
                    printf("set%u_zstd_level=%d, bytes=%llu\n",point_set_id,zstd_level,(unsigned long long)zstd_compress(qb_huffman_bytes,zstd_level).size());
            value_huffman_byte_num+=qb_huffman_bytes.size();
            value_zstd_byte_num+=qb_zstd_bytes.size();
            append_binary<uint64_t>(container,qb_zstd_bytes.size());
            container.insert(container.end(),qb_zstd_bytes.begin(),qb_zstd_bytes.end());
            printf("set%u_qb_num=%llu, set%u_huffman_bytes=%llu, set%u_zstd_bytes=%llu\n",point_set_id,
                (unsigned long long)point_set_qb_vec[point_set_id].size(),point_set_id,(unsigned long long)qb_huffman_bytes.size(),
                point_set_id,(unsigned long long)qb_zstd_bytes.size());
        }
        current_zstd_bs_vec=move(container);
        printf("label_prediction_error_ratio=%f, label_position_delta_bytes=%llu, label_zstd_bytes=%llu, zstd_level=%d\n",
            (double)label_prediction_error_num/point_num,(unsigned long long)label_position_delta_bytes.size(),(unsigned long long)label_zstd_bytes.size(),zstd_compression_level);
        printf("value_huffman_bytes=%llu, value_zstd_bytes=%llu, file_bytes=%llu\n",(unsigned long long)value_huffman_byte_num,
            (unsigned long long)value_zstd_byte_num,(unsigned long long)current_zstd_bs_vec.size());

        if (print_round_statistics)
        {
            printf("逐轮量化索引统计（不包含点集标签和文件头）：\n");
            for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
            {
                vector<unsigned char> round_huffman=huffman_encode(round_qb_vec[round_id]);
                vector<unsigned char> round_zstd=round_huffman.empty()?vector<unsigned char>():zstd_compress(round_huffman,zstd_compression_level);
                double squared_error=0;
                for (int pred_id:round_target_ids[round_id])
                {
                    double error=(double)current_data[pred_id]-current_decompressed_data[pred_id];
                    squared_error+=error*error;
                }
                double mse=squared_error/round_target_ids[round_id].size();
                double psnr=mse==0?numeric_limits<double>::infinity():10*log10(pow((double)current_max_data-current_min_data,2)/mse);
                double br=(double)round_zstd.size()*8/round_target_ids[round_id].size();
                printf("Round %u: point_num=%llu, qb_num=%llu, qb_br=%f, psnr=%f\n",round_id,(unsigned long long)round_target_ids[round_id].size(),(unsigned long long)round_qb_vec[round_id].size(),br,psnr);
            }
        }

        double squared_error=0;
        for (uint32_t i=0;i<point_num;i++)
        {
            double error=(double)current_data[i]-current_decompressed_data[i];
            squared_error+=error*error;
            uint32_t point_set_id=point_set_id_vec[i];
            if (abs(error)>point_set_abs_eb_vec[point_set_id]*1.001)
                throw runtime_error("压缩后的误差超过点集"+to_string(point_set_id)+"的误差上限");
        }
        double mse=squared_error/point_num;
        double psnr=mse==0?numeric_limits<double>::infinity():10*log10(pow((double)current_max_data-current_min_data,2)/mse);
        double cr=(double)point_num*sizeof(T_data)/current_zstd_bs_vec.size();
        double br=sizeof(T_data)*8/cr;
        printf("point_set_num=%u",point_set_num);
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            const char *codec_name=point_set_codec_vec[point_set_id]==point_set_codec_e::constant?"constant":point_set_codec_vec[point_set_id]==point_set_codec_e::fitting?"fitting":"inverse_proportion";
            printf(", set%u_num=%u, set%u_codec=%s",point_set_id,point_set_point_num_vec[point_set_id],point_set_id,codec_name);
        }
        printf("\ncr= %f, br= %f, psnr= %f\n",cr,br,psnr);
    }
}
