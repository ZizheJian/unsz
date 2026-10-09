#include "../c_codes/unsz_core/unsz.hpp"
#include <cassert>

int main()
{
    mt19937 random(142857);
    uint64_t checked=0;
    for(uint32_t trial=0;trial<20000;trial++)
    {
        uint32_t point_num=1+random()%200,time_num=1+random()%8;
        vector<int>target_ids(point_num);
        iota(target_ids.begin(),target_ids.end(),0);
        shuffle(target_ids.begin(),target_ids.end(),random);
        unsz::automatic_flow_window_c window;
        window.region_begin=random()%point_num;
        window.region_end=window.region_begin+1+random()%(point_num-window.region_begin);
        window.start=window.region_begin+random()%(window.region_end-window.region_begin);
        window.quota=1+random()%(point_num+10);
        vector<vector<uint8_t>>labels(time_num,vector<uint8_t>(point_num));
        vector<vector<uint32_t>>candidates(time_num);
        uint32_t expected_width=0;
        for(uint32_t time_index=0;time_index<time_num;time_index++)
        {
            for(uint8_t&label:labels[time_index])
                label=(trial%3==0)?1:(trial%3==1)?0:random()%2;
            uint32_t region_size=window.region_end-window.region_begin,found=0,width=region_size;
            for(uint32_t offset=0;offset<region_size;offset++)
            {
                uint32_t index=window.region_begin+(window.start-window.region_begin+offset)%region_size;
                if(labels[time_index][target_ids[index]]==0&&++found==window.quota)
                {
                    width=offset+1;
                    break;
                }
            }
            uint32_t sparse_width=window.collect_value_candidates(target_ids,labels[time_index],candidates[time_index]);
            assert(sparse_width==width);
            assert(candidates[time_index].size()<=2*window.quota);
            expected_width=max(expected_width,width);
        }
        window.width=expected_width;
        for(uint32_t time_index=0;time_index<time_num;time_index++)
        {
            vector<uint32_t>dense,sparse;
            for(uint32_t index=window.region_begin;index<window.region_end;index++)
                if(window.contains(index)&&labels[time_index][target_ids[index]]==0&&dense.size()<window.quota)
                    dense.push_back(index);
            for(uint32_t index:candidates[time_index])
                if(window.contains(index)&&sparse.size()<window.quota)
                    sparse.push_back(index);
            assert(dense==sparse);
            checked++;
        }
    }
    printf("Sparse window selection matches dense selection: %llu cases\n",(unsigned long long)checked);
}
