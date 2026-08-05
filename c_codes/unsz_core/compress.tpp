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
        int ref_num=actual_ref_num;
        if (ref_ids.size()<ref_num)
            ref_num=ref_ids.size();
        vector<T_data> weight_vec(ref_num);
        T_data total_weight=0,pred=0;
        for (int i=0;i<ref_num;i++)
        {
            T_coord dist=get_dist_to_node(coord_array[pred_id],coord_array[ref_ids[i]]);
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
        int ref_num=actual_ref_num;
        if (valid_ref_ids.size()<ref_num)
            ref_num=valid_ref_ids.size();
        vector<T_data> weight_vec(ref_num);
        T_data total_weight=0,pred=0;
        for (int i=0;i<ref_num;i++)
        {
            T_coord dist=get_dist_to_node(coord_array[pred_id],coord_array[valid_ref_ids[i]]);
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
        int ref_num=actual_ref_num;
        if (ref_ids.size()<ref_num)
            ref_num=ref_ids.size();
        double A[4][4]={0},b[4]={0},x[4];
        Array2D<T_coord> temp_coord_array(ref_num,dim);
        for (int i=0;i<ref_num;i++)
            for (int d=0;d<dim;d++)
                temp_coord_array[i][d]=(coord_array[ref_ids[i]][d]-coord_array[pred_id][d]);
        for (int i=0;i<ref_num;i++)
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
        int ref_num=actual_ref_num;
        if (valid_ref_ids.size()<ref_num)
            ref_num=valid_ref_ids.size();
        double A[4][4]={0},b[4]={0},x[4];
        Array2D<T_coord> temp_coord_array(ref_num,dim);
        for (int i=0;i<ref_num;i++)
            for (int d=0;d<dim;d++)
                temp_coord_array[i][d]=(coord_array[valid_ref_ids[i]][d]-coord_array[pred_id][d]);
        for (int i=0;i<ref_num;i++)
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
        for (int i=0;i<point_num;i++)
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
        for (auto &p:pred_refs_vec)
        {
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

        vector<unsigned char> huffman_bs_vec(qb_vec.size()*sizeof(T_data));
        unsigned char *huffman_bs=huffman_bs_vec.data();
        huffman_encoder_class<int64_t> huffman_encoder;
        huffman_encoder.preprocess_encode(qb_vec.data(),qb_vec.size());
        huffman_encoder.save(huffman_bs);
        huffman_encoder.encode(qb_vec.data(),qb_vec.size(),huffman_bs);

        uint64_t huffman_bs_len=huffman_bs-huffman_bs_vec.data();
        uint64_t max_zstd_bs_len=ZSTD_compressBound(huffman_bs_len);
        current_zstd_bs_vec.resize(max_zstd_bs_len);
        unsigned char *current_zstd_bs=current_zstd_bs_vec.data();
        current_zstd_bs+=ZSTD_compress(current_zstd_bs,max_zstd_bs_len,huffman_bs_vec.data(),huffman_bs_len,3);
        current_zstd_bs_vec.resize(current_zstd_bs-current_zstd_bs_vec.data());

        float mse=0;
        for (int i=0;i<point_num;i++)
            if (!is_abnormal_vec[i])
                mse+=pow(current_data[i]-current_decompressed_data[i],2);
        mse/=point_num;
        float psnr=10*log10(pow(current_max_data-current_min_data,2)/mse);
        float cr=(float)point_num*sizeof(T_data)/current_zstd_bs_vec.size();
        float br=(float)(sizeof(T_data)*8)/cr;
        printf("cr= %f, br= %f, psnr= %f\n",cr,br,psnr);
    }
}