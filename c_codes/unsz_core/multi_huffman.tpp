#pragma once

namespace unsz
{
    template<typename T_coord,typename T_data>
    uint32_t param_c<T_coord,T_data>::get_multi_huffman_group_num()
    {
        if (multi_huffman_mode_experimental=="none")
            return 1;
        if (multi_huffman_mode_experimental=="round")
            return predict_round_num;
        if (multi_huffman_mode_experimental=="partition" || multi_huffman_mode_experimental=="partition_round")
        {
            if (partition_point_ids_vec.empty())
                throw runtime_error("按partition建立多Huffman树需要启用partition");
            uint64_t group_num=partition_point_ids_vec.size();
            if (multi_huffman_mode_experimental=="partition_round")
                group_num*=predict_round_num;
            if (group_num>numeric_limits<uint32_t>::max())
                throw runtime_error("多Huffman树分组数量超过uint32_t范围");
            return (uint32_t)group_num;
        }
        throw runtime_error("未知的多Huffman树模式："+multi_huffman_mode_experimental);
    }

    template<typename T_coord,typename T_data>
    uint32_t param_c<T_coord,T_data>::get_multi_huffman_group_id(uint32_t round_id,int pred_id)
    {
        if (multi_huffman_mode_experimental=="none")
            return 0;
        if (multi_huffman_mode_experimental=="round")
            return round_id;
        uint32_t original_id=reorder_data_by_prediction_order_experimental?prediction_id_to_original_id[pred_id]:pred_id;
        if (original_id>=point_partition_id_vec.size())
            throw runtime_error("多Huffman树无法找到预测目标所属的partition");
        uint32_t block_id=point_partition_id_vec[original_id];
        if (block_id>=partition_point_ids_vec.size())
            throw runtime_error("预测目标所属的多Huffmanpartition编号越界");
        if (multi_huffman_mode_experimental=="partition")
            return block_id;
        if (multi_huffman_mode_experimental=="partition_round")
            return block_id*predict_round_num+round_id;
        throw runtime_error("未知的多Huffman树模式："+multi_huffman_mode_experimental);
    }

    inline vector<unsigned char> encode_multi_huffman_groups(vector<vector<int32_t>> &group_qb_vec,uint64_t &huffman_payload_byte_num,uint64_t &metadata_byte_num)
    {
        uint64_t qb_num=0;
        for (auto &group:group_qb_vec)
            qb_num+=group.size();
        if (qb_num==0)
        {
            huffman_payload_byte_num=0;
            metadata_byte_num=0;
            return {};
        }
        vector<vector<unsigned char>> group_huffman_vec(group_qb_vec.size());
        vector<unsigned char> result;
        append_binary<uint32_t>(result,group_qb_vec.size());
        huffman_payload_byte_num=0;
        for (uint32_t group_id=0;group_id<group_qb_vec.size();group_id++)
        {
            group_huffman_vec[group_id]=huffman_encode(group_qb_vec[group_id]);
            append_binary<uint64_t>(result,group_qb_vec[group_id].size());
            append_binary<uint64_t>(result,group_huffman_vec[group_id].size());
            huffman_payload_byte_num+=group_huffman_vec[group_id].size();
        }
        metadata_byte_num=result.size();
        for (auto &bytes:group_huffman_vec)
            result.insert(result.end(),bytes.begin(),bytes.end());
        return result;
    }

    inline vector<vector<int32_t>> decode_multi_huffman_groups(vector<unsigned char> &bytes,uint32_t expected_group_num,uint64_t expected_qb_num)
    {
        if (expected_qb_num==0)
        {
            if (!bytes.empty())
                throw runtime_error("空量化索引流包含多Huffman树数据");
            return vector<vector<int32_t>>(expected_group_num);
        }
        const unsigned char *cursor=bytes.data(),*end=bytes.data()+bytes.size();
        uint32_t group_num=read_binary<uint32_t>(cursor,end);
        if (group_num!=expected_group_num)
            throw runtime_error("多Huffman树分组数量与当前参数不一致");
        vector<uint64_t> group_qb_num_vec(group_num),group_byte_num_vec(group_num);
        uint64_t total_qb_num=0,total_huffman_byte_num=0;
        for (uint32_t group_id=0;group_id<group_num;group_id++)
        {
            group_qb_num_vec[group_id]=read_binary<uint64_t>(cursor,end);
            group_byte_num_vec[group_id]=read_binary<uint64_t>(cursor,end);
            total_qb_num+=group_qb_num_vec[group_id];
            total_huffman_byte_num+=group_byte_num_vec[group_id];
        }
        if (total_qb_num!=expected_qb_num)
            throw runtime_error("多Huffman树中的量化索引总数不正确");
        if ((uint64_t)(end-cursor)!=total_huffman_byte_num)
            throw runtime_error("多Huffman树字节长度不正确");
        vector<vector<int32_t>> result(group_num);
        for (uint32_t group_id=0;group_id<group_num;group_id++)
        {
            result[group_id]=huffman_decode<int32_t>(cursor,group_byte_num_vec[group_id],group_qb_num_vec[group_id]);
            cursor+=group_byte_num_vec[group_id];
        }
        return result;
    }
}
