#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    void param_c<T_coord,T_data>::decompress()
    {
        if (!got_rel_eb && !got_abs_eb && point_set_rel_eb_config_vec.empty() && point_set_abs_eb_config_vec.empty())
            throw runtime_error("解压时没有指定误差上限");
        vector<unsigned char> &container=current_zstd_bs_vec;
        const unsigned char *cursor=container.data(),*end=container.data()+container.size();
        const unsigned char magic[8]={'U','N','S','Z','P','S','0','6'};
        if ((uint64_t)(end-cursor)<8 || memcmp(cursor,magic,8)!=0)
            throw runtime_error("不是受支持的UNSZ点集文件");
        cursor+=8;
        uint32_t version=read_binary<uint32_t>(cursor,end);
        if (version!=6)
            throw runtime_error("不支持的UNSZ文件版本："+to_string(version));
        uint32_t file_point_num=read_binary<uint32_t>(cursor,end);
        uint32_t data_type_bytes=read_binary<uint32_t>(cursor,end);
        uint32_t point_set_num=read_binary<uint32_t>(cursor,end);
        if (file_point_num!=point_num)
            throw runtime_error("压缩文件点数与坐标点数不一致");
        if (data_type_bytes!=sizeof(T_data))
            throw runtime_error("压缩文件的数据类型大小与解压程序不一致");
        if (point_set_num==0 || point_set_num>point_num)
            throw runtime_error("压缩文件中的点集数量无效");
        if (point_set_num>2)
            throw runtime_error("当前0/1标签修正流只支持最多两个点集");

        uint64_t label_zstd_byte_num=read_binary<uint64_t>(cursor,end);
        if ((uint64_t)(end-cursor)<label_zstd_byte_num)
            throw runtime_error("点集标签流不完整");
        vector<uint32_t> label_symbol_vec;
        if (point_set_num==1)
        {
            if (label_zstd_byte_num!=0)
                throw runtime_error("单点集文件不应包含点集标签流");
        }
        else
        {
            vector<unsigned char> label_zstd_bytes(cursor,cursor+label_zstd_byte_num);
            vector<unsigned char> label_position_delta_bytes=zstd_decompress(label_zstd_bytes);
            label_symbol_vec=decode_binary_one_position_delta_vec(label_position_delta_bytes,point_num);
        }
        cursor+=label_zstd_byte_num;

        point_set_point_num_vec.resize(point_set_num);
        point_set_min_data_vec.resize(point_set_num);
        point_set_max_data_vec.resize(point_set_num);
        point_set_abs_eb_vec.resize(point_set_num);
        point_set_codec_vec.resize(point_set_num);
        vector<vector<int64_t>> point_set_qb_vec(point_set_num);
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
        {
            uint8_t codec=read_binary<uint8_t>(cursor,end);
            if (codec>(uint8_t)point_set_codec_e::constant)
                throw runtime_error("压缩文件包含未知的点集编码方式");
            point_set_codec_vec[point_set_id]=(point_set_codec_e)codec;
            point_set_point_num_vec[point_set_id]=read_binary<uint32_t>(cursor,end);
            point_set_min_data_vec[point_set_id]=read_binary<T_data>(cursor,end);
            point_set_max_data_vec[point_set_id]=read_binary<T_data>(cursor,end);
            uint64_t qb_num=read_binary<uint64_t>(cursor,end);
            uint64_t qb_zstd_byte_num=read_binary<uint64_t>(cursor,end);
            if ((uint64_t)(end-cursor)<qb_zstd_byte_num)
                throw runtime_error("点集"+to_string(point_set_id)+"的数值流不完整");
            if (qb_num==0)
            {
                if (qb_zstd_byte_num!=0)
                    throw runtime_error("空数值流具有非空Zstd字节流");
            }
            else
            {
                vector<unsigned char> qb_zstd_bytes(cursor,cursor+qb_zstd_byte_num);
                vector<unsigned char> qb_huffman_bytes=zstd_decompress(qb_zstd_bytes);
                point_set_qb_vec[point_set_id]=huffman_decode<int64_t>(qb_huffman_bytes.data(),qb_huffman_bytes.size(),qb_num);
            }
            cursor+=qb_zstd_byte_num;
            if (point_set_codec_vec[point_set_id]==point_set_codec_e::constant && qb_num!=0)
                throw runtime_error("常量点集不应包含量化索引");
            if (point_set_id<point_set_abs_eb_config_vec.size())
                point_set_abs_eb_vec[point_set_id]=point_set_abs_eb_config_vec[point_set_id];
            else if (point_set_id<point_set_rel_eb_config_vec.size())
                point_set_abs_eb_vec[point_set_id]=point_set_rel_eb_config_vec[point_set_id]*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]);
            else
                point_set_abs_eb_vec[point_set_id]=got_abs_eb?abs_eb:rel_eb*(point_set_max_data_vec[point_set_id]-point_set_min_data_vec[point_set_id]);
            if (point_set_codec_vec[point_set_id]!=point_set_codec_e::constant && !(point_set_abs_eb_vec[point_set_id]>0))
                throw runtime_error("点集"+to_string(point_set_id)+"的绝对误差上限必须大于0");
        }
        if (cursor!=end)
            throw runtime_error("压缩文件末尾存在未解析数据");

        point_set_id_vec.assign(point_num,0);
        if (point_set_num>1)
        {
            if (point_set_predictor!="inverse_proportion" && point_set_predictor!="majority" && point_set_predictor!="continuous_actual" && point_set_predictor!="continuous_reflect")
                throw runtime_error("未知的点集标签预测器："+point_set_predictor);
            bool continuous=point_set_predictor=="continuous_actual" || point_set_predictor=="continuous_reflect";
            bool use_majority=point_set_predictor=="majority";
            point_set_prediction_value_vec.assign(point_num,0);
            for (uint32_t pred_ref_id=0;pred_ref_id<pred_refs_vec.size();pred_ref_id++)
            {
                auto &pred_refs=pred_refs_vec[pred_ref_id];
                auto &ref_weights=pred_ref_weights_vec[pred_ref_id];
                int pred_id=pred_refs.first;
                double predicted_value=continuous?predict_point_set_value(pred_refs.second,ref_weights):0;
                uint32_t predicted_id=continuous?(predicted_value>=0.5):predict_point_set_id(pred_refs.second,ref_weights,use_majority);
                uint32_t symbol=label_symbol_vec[pred_id];
                if (symbol>1)
                    throw runtime_error("点集标签修正值不是0或1");
                uint32_t actual_id=symbol==0?predicted_id:1-predicted_id;
                if (actual_id>=point_set_num)
                    throw runtime_error("点集标签解码结果越界");
                if (continuous)
                {
                    if (symbol==0)
                        point_set_prediction_value_vec[pred_id]=predicted_value;
                    else if (point_set_predictor=="continuous_actual" || predicted_value==0.5)
                        point_set_prediction_value_vec[pred_id]=actual_id;
                    else
                        point_set_prediction_value_vec[pred_id]=1-predicted_value;
                    actual_id=point_set_prediction_value_vec[pred_id]>=0.5;
                }
                point_set_id_vec[pred_id]=actual_id;
            }
        }
        vector<uint32_t> decoded_point_set_num(point_set_num,0);
        for (uint32_t point_set_id:point_set_id_vec)
            decoded_point_set_num[point_set_id]++;
        if (decoded_point_set_num!=point_set_point_num_vec)
            throw runtime_error("解码后的点集数量与文件头不一致");

        current_decompressed_data.assign(point_num,0);
        vector<uint64_t> qb_index(point_set_num,0);
        for (uint32_t pred_ref_id=0;pred_ref_id<pred_refs_vec.size();pred_ref_id++)
        {
            auto &pred_refs=pred_refs_vec[pred_ref_id];
            auto &ref_weights=pred_ref_weights_vec[pred_ref_id];
            int pred_id=pred_refs.first;
            uint32_t point_set_id=point_set_id_vec[pred_id];
            if (point_set_codec_vec[point_set_id]==point_set_codec_e::constant)
                current_decompressed_data[pred_id]=point_set_min_data_vec[point_set_id];
            else
            {
                if (qb_index[point_set_id]>=point_set_qb_vec[point_set_id].size())
                    throw runtime_error("点集"+to_string(point_set_id)+"的量化索引不足");
                T_data pred=predict_value(pred_id,pred_refs.second,ref_weights,point_set_id,current_decompressed_data);
                int64_t qb=point_set_qb_vec[point_set_id][qb_index[point_set_id]++];
                current_decompressed_data[pred_id]=pred+qb*2*point_set_abs_eb_vec[point_set_id];
            }
        }
        for (uint32_t point_set_id=0;point_set_id<point_set_num;point_set_id++)
            if (qb_index[point_set_id]!=point_set_qb_vec[point_set_id].size())
                throw runtime_error("点集"+to_string(point_set_id)+"存在未使用的量化索引");
        uint32_t primary_point_set_id=0;
        while (point_set_point_num_vec[primary_point_set_id]==0)
            primary_point_set_id++;
        current_min_data=point_set_min_data_vec[primary_point_set_id];
        current_max_data=point_set_max_data_vec[primary_point_set_id];
        current_abs_eb=point_set_abs_eb_vec[primary_point_set_id];
    }
}
