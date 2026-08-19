#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::predict_and_quantize_inverse_proportion_with_abnormal(int pred_id,vector<int> &ref_ids)
    {
        if (ref_ids.size()==0)
        {
            if (is_abnormal_vec[pred_id])
                current_decompressed_data[pred_id]=(current_min_data+current_max_data)/2;
            else
            {
                qb_vec.push_back(llround(current_decompressed_data[pred_id]/(2*current_abs_eb)));
                current_decompressed_data[pred_id]=qb_vec.back()*2*current_abs_eb;
            }
            return;
        }
        uint32_t ref_num=actual_ref_num;
        if (ref_ids.size()<ref_num)
            ref_num=ref_ids.size();
        vector<T_coord> weight_vec(ref_num);
        T_coord total_weight=0,pred=0;
        for (uint32_t i=0;i<ref_num;i++)
        {
            T_coord dist=get_dist_to_coord(coord_array[pred_id],coord_array[ref_ids[i]],dim);
            weight_vec[i]=pow(dist,-3.61);
            total_weight+=weight_vec[i];
            pred+=weight_vec[i]*current_decompressed_data[ref_ids[i]];
        }
        pred/=total_weight;
        if (is_abnormal_vec[pred_id])
            current_decompressed_data[pred_id]=pred;
        else
        {
            qb_vec.push_back(llround((current_decompressed_data[pred_id]-pred)/(2*current_abs_eb)));
            current_decompressed_data[pred_id]=pred+qb_vec.back()*2*current_abs_eb;
        }
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::predict_and_quantize_inverse_proportion_without_abnormal(int pred_id,vector<int> &ref_ids)
    {   
        if (is_abnormal_vec[pred_id])
            return;
        vector<int> valid_ref_ids;
        for (auto ref_id:ref_ids)
            if (!is_abnormal_vec[ref_id])
                valid_ref_ids.push_back(ref_id);
        if (valid_ref_ids.size()==0)
        {
            qb_vec.push_back(llround(current_decompressed_data[pred_id]/(2*current_abs_eb)));
            current_decompressed_data[pred_id]=qb_vec.back()*2*current_abs_eb;
            return;
        }
        uint32_t ref_num=actual_ref_num;
        if (valid_ref_ids.size()<ref_num)
            ref_num=valid_ref_ids.size();
        vector<T_coord> weight_vec(ref_num);
        T_coord total_weight=0,pred=0;
        for (uint32_t i=0;i<ref_num;i++)
        {
            T_coord dist=get_dist_to_coord(coord_array[pred_id],coord_array[valid_ref_ids[i]],dim);
            weight_vec[i]=pow(dist,-3.61);
            total_weight+=weight_vec[i];
            pred+=weight_vec[i]*current_decompressed_data[valid_ref_ids[i]];
        }
        pred/=total_weight;
        qb_vec.push_back(llround((current_decompressed_data[pred_id]-pred)/(2*current_abs_eb)));
        current_decompressed_data[pred_id]=pred+qb_vec.back()*2*current_abs_eb;
    }

    template<typename T_coord,typename T_data>
    bool param_c<T_coord,T_data>::solve_4x4(double A_in[4][4],double b_in[4],double x[4])
    {
        double A[4][5];
        for(int i=0;i<4;i++)
        {
            for(int j=0;j<4;j++)
                A[i][j]=A_in[i][j];
            A[i][4]=b_in[i];
        }
        double eps=1e-12;
        for(int col=0;col<4;col++)
        {
            int pivot=col;
            for(int r=col+1;r<4;r++)
                if(abs(A[r][col])>abs(A[pivot][col]))
                    pivot=r;
            if(abs(A[pivot][col])<eps)
                return false;
            if(pivot!=col)
                for(int c=col;c<5;c++)
                    swap(A[col][c],A[pivot][c]);
            double div=A[col][col];
            for(int c=col;c<5;c++)
                A[col][c]/=div;

            for(int r=0;r<4;r++)
            {
                if(r==col)
                    continue;
                double factor=A[r][col];
                if(factor==0)
                    continue;
                for(int c=col;c<5;c++)
                    A[r][c]-=factor*A[col][c];
            }
        }
        for(int i=0;i<4;i++)
            x[i]=A[i][4];
        return true;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::predict_and_quantize_fitting_with_abnormal(int pred_id,vector<int> &ref_ids)
    {
        assert(dim==3);
        assert(actual_ref_num>dim);
        if (ref_ids.size()<=dim)
        {
            predict_and_quantize_inverse_proportion_with_abnormal(pred_id,ref_ids);
            return;
        }
        uint32_t ref_num=actual_ref_num;
        if (ref_ids.size()<ref_num)
            ref_num=ref_ids.size();
        double A[4][4]={0},b[4]={0},x[4];
        Array2D<T_coord> temp_coord_array(ref_num,dim);
        for (uint32_t i=0;i<ref_num;i++)
            for (uint32_t d=0;d<dim;d++)
                temp_coord_array[i][d]=(coord_array[ref_ids[i]][d]-coord_array[pred_id][d]);
        for (uint32_t i=0;i<ref_num;i++)
        {
            A[0][0]+=temp_coord_array[i][0]*temp_coord_array[i][0];
            A[0][1]+=temp_coord_array[i][0]*temp_coord_array[i][1];
            A[0][2]+=temp_coord_array[i][0]*temp_coord_array[i][2];
            A[0][3]+=temp_coord_array[i][0];
            A[1][1]+=temp_coord_array[i][1]*temp_coord_array[i][1];
            A[1][2]+=temp_coord_array[i][1]*temp_coord_array[i][2];
            A[1][3]+=temp_coord_array[i][1];
            A[2][2]+=temp_coord_array[i][2]*temp_coord_array[i][2];
            A[2][3]+=temp_coord_array[i][2];
            b[0]+=temp_coord_array[i][0]*current_decompressed_data[ref_ids[i]];
            b[1]+=temp_coord_array[i][1]*current_decompressed_data[ref_ids[i]];
            b[2]+=temp_coord_array[i][2]*current_decompressed_data[ref_ids[i]];
            b[3]+=current_decompressed_data[ref_ids[i]];
        }
        A[1][0]=A[0][1];
        A[2][0]=A[0][2];
        A[2][1]=A[1][2];
        A[3][0]=A[0][3];
        A[3][1]=A[1][3];
        A[3][2]=A[2][3];
        A[3][3]=ref_num;
        if (!solve_4x4(A,b,x))
        {
            predict_and_quantize_inverse_proportion_with_abnormal(pred_id,ref_ids);
            return;
        }
        T_data pred=x[3];
        if (is_abnormal_vec[pred_id])
            current_decompressed_data[pred_id]=pred;
        else
        {
            qb_vec.push_back(llround((current_decompressed_data[pred_id]-pred)/(2*current_abs_eb)));
            current_decompressed_data[pred_id]=pred+qb_vec.back()*2*current_abs_eb;
        }
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::predict_and_quantize_fitting_without_abnormal(int pred_id,vector<int> &ref_ids)
    {
        assert(dim==3);
        assert(actual_ref_num>dim);
        if (is_abnormal_vec[pred_id])
            return;
        vector<int> valid_ref_ids;
        for (auto ref_id:ref_ids)
            if (!is_abnormal_vec[ref_id])
                valid_ref_ids.push_back(ref_id);
        if (valid_ref_ids.size()<=dim)
        {
            predict_and_quantize_inverse_proportion_with_abnormal(pred_id,valid_ref_ids);
            return;
        }
        uint32_t ref_num=actual_ref_num;
        if (valid_ref_ids.size()<ref_num)
            ref_num=valid_ref_ids.size();
        double A[4][4]={0},b[4]={0},x[4];
        Array2D<T_coord> temp_coord_array(ref_num,dim);
        for (uint32_t i=0;i<ref_num;i++)
            for (uint32_t d=0;d<dim;d++)
                temp_coord_array[i][d]=(coord_array[valid_ref_ids[i]][d]-coord_array[pred_id][d]);
        for (uint32_t i=0;i<ref_num;i++)
        {
            A[0][0]+=temp_coord_array[i][0]*temp_coord_array[i][0];
            A[0][1]+=temp_coord_array[i][0]*temp_coord_array[i][1];
            A[0][2]+=temp_coord_array[i][0]*temp_coord_array[i][2];
            A[0][3]+=temp_coord_array[i][0];
            A[1][1]+=temp_coord_array[i][1]*temp_coord_array[i][1];
            A[1][2]+=temp_coord_array[i][1]*temp_coord_array[i][2];
            A[1][3]+=temp_coord_array[i][1];
            A[2][2]+=temp_coord_array[i][2]*temp_coord_array[i][2];
            A[2][3]+=temp_coord_array[i][2];
            b[0]+=temp_coord_array[i][0]*current_decompressed_data[valid_ref_ids[i]];
            b[1]+=temp_coord_array[i][1]*current_decompressed_data[valid_ref_ids[i]];
            b[2]+=temp_coord_array[i][2]*current_decompressed_data[valid_ref_ids[i]];
            b[3]+=current_decompressed_data[valid_ref_ids[i]];
        }
        A[1][0]=A[0][1];
        A[2][0]=A[0][2];
        A[2][1]=A[1][2];
        A[3][0]=A[0][3];
        A[3][1]=A[1][3];
        A[3][2]=A[2][3];
        A[3][3]=ref_num;
        if (!solve_4x4(A,b,x))
        {
            predict_and_quantize_inverse_proportion_with_abnormal(pred_id,valid_ref_ids);
            return;
        }
        T_data pred=x[3];
        qb_vec.push_back(llround((current_decompressed_data[pred_id]-pred)/(2*current_abs_eb)));
        current_decompressed_data[pred_id]=pred+qb_vec.back()*2*current_abs_eb;
    }

    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::compress()
    {
        current_decompressed_data.assign(current_data.begin(),current_data.end());
        qb_vec.clear();
        qb_vec.reserve(point_num);
        is_abnormal_vec.resize(point_num);
        current_min_data=numeric_limits<T_data>::infinity();
        current_max_data=-numeric_limits<T_data>::infinity();
        current_valid_node_num=0;
        for (uint32_t i=0;i<point_num;i++)
        {
            is_abnormal_vec[i]=is_abnormal(current_data[i]);
            if (!is_abnormal_vec[i])
            {
                current_min_data=min(current_min_data,current_data[i]);
                current_max_data=max(current_max_data,current_data[i]);
                current_valid_node_num++;
            }
        }
        if (got_abs_eb)
            current_abs_eb=abs_eb;
        else if (got_rel_eb)
            current_abs_eb=rel_eb*(current_max_data-current_min_data);
        vector<uint64_t> round_qb_begin(predict_round_num),round_qb_end(predict_round_num);
        uint64_t pred_ref_index=0;
        for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
        {
            round_qb_begin[round_id]=qb_vec.size();
            for (uint64_t i=0;i<round_target_ids[round_id].size();i++)
            {
                auto &p=pred_refs_vec[pred_ref_index++];
                int pred_id=p.first;
                vector<int> &ref_ids=p.second;
                T_data current_abs_eb_backup=current_abs_eb;
                current_abs_eb*=pow(1,(double)(current_valid_node_num-1-qb_vec.size())/(current_valid_node_num-1));
                if (predictor=="fitting")
                    if (predict_abnormal_points)
                        predict_and_quantize_fitting_with_abnormal(pred_id,ref_ids);
                    else
                        predict_and_quantize_fitting_without_abnormal(pred_id,ref_ids);
                else if (predictor=="inverse_proportion")
                    if (predict_abnormal_points)
                        predict_and_quantize_inverse_proportion_with_abnormal(pred_id,ref_ids);
                    else
                        predict_and_quantize_inverse_proportion_without_abnormal(pred_id,ref_ids);
                current_abs_eb=current_abs_eb_backup;
            }
            round_qb_end[round_id]=qb_vec.size();
        }
        if (pred_ref_index!=pred_refs_vec.size())
            throw runtime_error("逐轮预测目标数量与预测参考结构不一致");

        if (qb_vec.empty())
            throw runtime_error("没有可编码的量化索引");
        auto encode_qb=[&](int64_t *qb,uint64_t qb_num)
        {
            huffman_encoder_class<int64_t> huffman_encoder;
            huffman_encoder.preprocess_encode(qb,qb_num);
            vector<unsigned char> huffman_bs_vec((uint64_t)point_num*sizeof(int64_t));
            unsigned char *huffman_bs=huffman_bs_vec.data();
            huffman_encoder.save(huffman_bs);
            huffman_encoder.encode(qb,qb_num,huffman_bs);
            uint64_t huffman_bs_len=huffman_bs-huffman_bs_vec.data();
            vector<unsigned char> zstd_bs_vec(ZSTD_compressBound(huffman_bs_len));
            uint64_t zstd_bs_len=ZSTD_compress(zstd_bs_vec.data(),zstd_bs_vec.size(),huffman_bs_vec.data(),huffman_bs_len,3);
            if (ZSTD_isError(zstd_bs_len))
                throw runtime_error("Zstd压缩失败："+string(ZSTD_getErrorName(zstd_bs_len)));
            zstd_bs_vec.resize(zstd_bs_len);
            return zstd_bs_vec;
        };
        current_zstd_bs_vec=encode_qb(qb_vec.data(),qb_vec.size());

        if (print_round_statistics)
        {
            printf("逐轮独立编码统计（每轮包含独立Huffman表）：\n");
            uint64_t round_pred_ref_begin=0;
            for (uint32_t round_id=0;round_id<predict_round_num;round_id++)
            {
                uint64_t round_point_num=round_target_ids[round_id].size();
                uint64_t round_qb_num=round_qb_end[round_id]-round_qb_begin[round_id];
                vector<unsigned char> round_zstd_bs_vec;
                if (round_qb_num>0)
                    round_zstd_bs_vec=encode_qb(qb_vec.data()+round_qb_begin[round_id],round_qb_num);
                double round_squared_error=0,nearest_distance_sum=0;
                uint64_t nearest_distance_num=0,total_ref_num=0;
                for (uint64_t i=0;i<round_point_num;i++)
                {
                    int pred_id=round_target_ids[round_id][i];
                    if (!is_abnormal_vec[pred_id])
                    {
                        double error=(double)current_data[pred_id]-current_decompressed_data[pred_id];
                        round_squared_error+=error*error;
                    }
                    auto &ref_ids=pred_refs_vec[round_pred_ref_begin+i].second;
                    total_ref_num+=ref_ids.size();
                    if (!ref_ids.empty())
                    {
                        nearest_distance_sum+=sqrt((double)get_dist_to_coord(coord_array[pred_id],coord_array[ref_ids[0]],dim));
                        nearest_distance_num++;
                    }
                }
                round_pred_ref_begin+=round_point_num;
                double round_mse=round_squared_error/round_point_num;
                double round_psnr=round_mse==0?numeric_limits<double>::infinity():10*log10(pow((double)current_max_data-current_min_data,2)/round_mse);
                double round_br=(double)round_zstd_bs_vec.size()*8/round_point_num;
                double average_ref_num=(double)total_ref_num/round_point_num;
                double average_nearest_distance=nearest_distance_num==0?numeric_limits<double>::infinity():nearest_distance_sum/nearest_distance_num;
                printf("Round %u: point_num=%llu, qb_num=%llu, br=%f, psnr=%f, average_ref_num=%f, average_nearest_distance=%e\n",
                    round_id,(unsigned long long)round_point_num,(unsigned long long)round_qb_num,round_br,round_psnr,average_ref_num,average_nearest_distance);
            }
        }

        double squared_error=0;
        for (uint32_t i=0;i<point_num;i++)
            if (!is_abnormal_vec[i])
            {
                double error=(double)current_data[i]-current_decompressed_data[i];
                squared_error+=error*error;
            }
        double mse=squared_error/point_num;
        double psnr=10*log10(pow((double)current_max_data-current_min_data,2)/mse);
        float cr=(float)point_num*sizeof(T_data)/current_zstd_bs_vec.size();
        float br=(float)(sizeof(T_data)*8)/cr;
        printf("cr= %f, br= %f, psnr= %f\n",cr,br,psnr);
    }
}
