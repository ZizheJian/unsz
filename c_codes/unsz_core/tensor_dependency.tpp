#pragma once

namespace unsz
{
    inline uint32_t tensor_slice_id(const array<uint32_t,2>&shape,uint32_t axis0_index,uint32_t axis1_index)
    {
        return axis0_index*shape[1]+axis1_index;
    }

    inline string tensor_axis_mode_name(tensor_axis_mode_e mode)
    {
        if(mode==tensor_axis_mode_e::none)return "none";
        if(mode==tensor_axis_mode_e::constant_extrapolation)return "extrapolation";
        if(mode==tensor_axis_mode_e::anchor_interpolation_stride_8)return "anchor8";
        throw runtime_error("未知tensor轴预测模式");
    }

    inline string tensor_flow_name(const tensor_flow_c&flow)
    {
        string name="axis0_"+tensor_axis_mode_name(flow.axis_modes[0])+"_axis1_"+tensor_axis_mode_name(flow.axis_modes[1]);
        if(flow.axis_modes[0]!=tensor_axis_mode_e::none&&flow.axis_modes[1]!=tensor_axis_mode_e::none)
        {
            name+="_order_"+to_string(flow.axis_order[0])+to_string(flow.axis_order[1]);
            if(flow.schedule==tensor_schedule_e::interleaved_anchor)
                name+="_interleaved_"+to_string(flow.interpolation_axis_order[0])+to_string(flow.interpolation_axis_order[1]);
        }
        name+=flow.use_spatial_residual?"_spatial_residual":"_no_spatial_residual";
        return name;
    }

    inline uint8_t encode_tensor_flow(const tensor_flow_c&flow)
    {
        uint8_t axis0_mode=(uint8_t)flow.axis_modes[0],axis1_mode=(uint8_t)flow.axis_modes[1];
        if(axis0_mode>2||axis1_mode>2||flow.axis_order[0]>1||flow.interpolation_axis_order[0]>1)
            throw runtime_error("不能编码无效tensor flow");
        return axis0_mode|(axis1_mode<<2)|(flow.axis_order[0]<<4)|(flow.interpolation_axis_order[0]<<5)|((uint8_t)flow.schedule<<6)|((uint8_t)flow.use_spatial_residual<<7);
    }

    inline tensor_flow_c decode_tensor_flow(uint8_t encoded)
    {
        tensor_flow_c flow;
        flow.axis_modes={{(tensor_axis_mode_e)(encoded&3),(tensor_axis_mode_e)((encoded>>2)&3)}};
        flow.axis_order={{(uint8_t)((encoded>>4)&1),(uint8_t)(1-((encoded>>4)&1))}};
        flow.interpolation_axis_order={{(uint8_t)((encoded>>5)&1),(uint8_t)(1-((encoded>>5)&1))}};
        flow.schedule=(tensor_schedule_e)((encoded>>6)&1);
        flow.use_spatial_residual=(encoded>>7)&1;
        if((uint8_t)flow.axis_modes[0]>2||(uint8_t)flow.axis_modes[1]>2)
            throw runtime_error("压缩文件包含无效tensor轴模式");
        return flow;
    }

    inline vector<uint32_t> generate_tensor_axis_order(uint32_t length,tensor_axis_mode_e mode)
    {
        if(length==0)
            throw runtime_error("tensor轴长度不能为0");
        vector<uint32_t> order;
        order.reserve(length);
        if(mode!=tensor_axis_mode_e::anchor_interpolation_stride_8||length==1)
        {
            order.resize(length);
            iota(order.begin(),order.end(),0);
            return order;
        }
        order.push_back(0);
        for(uint32_t index=8;index<length;index+=8)
            order.push_back(index);
        constexpr uint32_t offsets[]={4,2,1};
        for(uint32_t offset:offsets)
            for(uint32_t index=offset;index<length;index+=2*offset)
                order.push_back(index);
        if(order.size()!=length)
            throw runtime_error("anchor8轴顺序没有覆盖全部切片");
        return order;
    }

    inline double tensor_axis_coordinate(const array<vector<double>,2>&axis_coordinates,uint8_t axis,uint32_t index)
    {
        return axis_coordinates[axis].empty()?(double)index:axis_coordinates[axis][index];
    }

    inline temporal_dependency_c build_tensor_axis_dependency(uint8_t axis,uint32_t target_position,const vector<uint8_t>&ready_positions,const array<vector<double>,2>&axis_coordinates,const function<uint32_t(uint32_t)>&slice_id_from_position,tensor_axis_mode_e mode)
    {
        temporal_dependency_c dependency;
        int32_t left=-1,right=-1;
        for(int32_t position=(int32_t)target_position-1;position>=0;position--)
            if(ready_positions[position])
            {
                left=position;
                break;
            }
        if(mode==tensor_axis_mode_e::anchor_interpolation_stride_8)
            for(uint32_t position=target_position+1;position<ready_positions.size();position++)
                if(ready_positions[position])
                {
                    right=(int32_t)position;
                    break;
                }
        if(left>=0&&right>=0)
        {
            double left_coordinate=tensor_axis_coordinate(axis_coordinates,axis,(uint32_t)left);
            double right_coordinate=tensor_axis_coordinate(axis_coordinates,axis,(uint32_t)right);
            double target_coordinate=tensor_axis_coordinate(axis_coordinates,axis,target_position);
            if(!(right_coordinate>left_coordinate)||target_coordinate<=left_coordinate||target_coordinate>=right_coordinate)
                throw runtime_error("tensor轴坐标必须严格递增");
            double right_weight=(target_coordinate-left_coordinate)/(right_coordinate-left_coordinate);
            dependency.source_indices={{(int32_t)slice_id_from_position((uint32_t)left),(int32_t)slice_id_from_position((uint32_t)right)}};
            dependency.source_weights={{1-right_weight,right_weight}};
            dependency.source_num=2;
        }
        else
        {
            int32_t source=left>=0?left:right;
            if(source>=0)
            {
                dependency.source_indices[0]=(int32_t)slice_id_from_position((uint32_t)source);
                dependency.source_weights[0]=1;
                dependency.source_num=1;
            }
        }
        return dependency;
    }

    inline void validate_tensor_dependency_plan(const array<uint32_t,2>&shape,const temporal_dependency_plan_c&plan)
    {
        uint64_t slice_num=(uint64_t)shape[0]*shape[1];
        if(slice_num>numeric_limits<uint32_t>::max())
            throw runtime_error("tensor切片数量超过uint32_t范围");
        if(plan.processing_order.size()!=slice_num||plan.dependencies.size()!=slice_num)
            throw runtime_error("tensor依赖计划没有覆盖全部切片");
        vector<int64_t> rank(slice_num,-1);
        for(uint32_t order_index=0;order_index<plan.processing_order.size();order_index++)
        {
            uint32_t slice_id=plan.processing_order[order_index];
            if(slice_id>=slice_num||rank[slice_id]>=0)
                throw runtime_error("tensor依赖计划包含重复或越界切片");
            rank[slice_id]=order_index;
            const temporal_dependency_c&dependency=plan.dependencies[slice_id];
            if(dependency.source_num>2)
                throw runtime_error("tensor切片最多只能有两个参考切片");
            double weight_sum=0;
            for(uint32_t source_index=0;source_index<dependency.source_num;source_index++)
            {
                int32_t source_slice_id=dependency.source_indices[source_index];
                double weight=dependency.source_weights[source_index];
                if(source_slice_id<0||(uint32_t)source_slice_id>=slice_num||rank[source_slice_id]<0||rank[source_slice_id]>=order_index)
                    throw runtime_error("tensor依赖计划引用了尚未重建的切片");
                if(!isfinite(weight))
                    throw runtime_error("tensor预测权重不是有限值");
                weight_sum+=weight;
            }
            if(dependency.source_num>0&&abs(weight_sum-1)>1e-12)
                throw runtime_error("tensor预测权重之和不为1");
        }
    }

    inline temporal_dependency_plan_c generate_tensor_dependency_plan(const array<uint32_t,2>&shape,const array<vector<double>,2>&axis_coordinates,const tensor_flow_c&flow)
    {
        if(shape[0]==0||shape[1]==0)
            throw runtime_error("tensor形状不能包含0");
        for(uint8_t axis=0;axis<2;axis++)
        {
            if(!axis_coordinates[axis].empty()&&axis_coordinates[axis].size()!=shape[axis])
                throw runtime_error("tensor轴坐标数量与形状不一致");
            for(uint32_t index=1;index<axis_coordinates[axis].size();index++)
                if(!(axis_coordinates[axis][index]>axis_coordinates[axis][index-1]))
                    throw runtime_error("tensor轴坐标必须严格递增");
        }
        if(flow.axis_order[0]>1||flow.axis_order[1]>1||flow.axis_order[0]==flow.axis_order[1]||flow.interpolation_axis_order[0]>1||flow.interpolation_axis_order[1]>1||flow.interpolation_axis_order[0]==flow.interpolation_axis_order[1])
            throw runtime_error("tensor轴顺序必须是0和1的排列");
        bool active0=flow.axis_modes[0]!=tensor_axis_mode_e::none,active1=flow.axis_modes[1]!=tensor_axis_mode_e::none;
        if(flow.schedule==tensor_schedule_e::interleaved_anchor&&(!active0||!active1||flow.axis_modes[0]!=tensor_axis_mode_e::anchor_interpolation_stride_8||flow.axis_modes[1]!=tensor_axis_mode_e::anchor_interpolation_stride_8))
            throw runtime_error("交错tensor计划要求两个轴都使用anchor8");
        uint64_t slice_num_64=(uint64_t)shape[0]*shape[1];
        if(slice_num_64>numeric_limits<uint32_t>::max())
            throw runtime_error("tensor切片数量超过uint32_t范围");
        uint32_t slice_num=(uint32_t)slice_num_64;
        temporal_dependency_plan_c plan;
        plan.processing_order.reserve(slice_num);
        plan.dependencies.assign(slice_num,{});
        vector<uint8_t> visited(slice_num,0);
        auto append_slice=[&](uint32_t slice_id,const temporal_dependency_c&dependency)
        {
            if(slice_id>=slice_num||visited[slice_id])
                throw runtime_error("重复生成tensor切片");
            visited[slice_id]=1;
            plan.processing_order.push_back(slice_id);
            plan.dependencies[slice_id]=dependency;
        };
        auto make_slice_id=[&](uint8_t axis,uint32_t position,uint32_t fixed_position)
        {
            return axis==0?tensor_slice_id(shape,position,fixed_position):tensor_slice_id(shape,fixed_position,position);
        };
        auto append_axis_line=[&](uint8_t axis,uint32_t fixed_position,bool include_root)
        {
            tensor_axis_mode_e mode=flow.axis_modes[axis];
            vector<uint32_t> order=generate_tensor_axis_order(shape[axis],mode);
            vector<uint8_t> ready_positions(shape[axis],0);
            uint32_t root_slice_id=make_slice_id(axis,0,fixed_position);
            if(include_root)
                append_slice(root_slice_id,{});
            else if(!visited[root_slice_id])
                throw runtime_error("tensor轴预测的起始切片尚未重建");
            ready_positions[0]=1;
            for(uint32_t order_index=1;order_index<order.size();order_index++)
            {
                uint32_t position=order[order_index];
                auto slice_id_from_position=[&](uint32_t source_position){return make_slice_id(axis,source_position,fixed_position);};
                temporal_dependency_c dependency=build_tensor_axis_dependency(axis,position,ready_positions,axis_coordinates,slice_id_from_position,mode);
                append_slice(make_slice_id(axis,position,fixed_position),dependency);
                ready_positions[position]=1;
            }
        };
        if(!active0&&!active1)
        {
            for(uint32_t axis0_index=0;axis0_index<shape[0];axis0_index++)
                for(uint32_t axis1_index=0;axis1_index<shape[1];axis1_index++)
                    append_slice(tensor_slice_id(shape,axis0_index,axis1_index),{});
        }
        else if(active0!=active1)
        {
            uint8_t active_axis=active0?0:1,other_axis=1-active_axis;
            for(uint32_t fixed_position=0;fixed_position<shape[other_axis];fixed_position++)
                append_slice(make_slice_id(active_axis,0,fixed_position),{});
            for(uint32_t fixed_position=0;fixed_position<shape[other_axis];fixed_position++)
                append_axis_line(active_axis,fixed_position,false);
        }
        else if(flow.schedule==tensor_schedule_e::separable)
        {
            uint8_t first_axis=flow.axis_order[0],second_axis=flow.axis_order[1];
            append_axis_line(first_axis,0,true);
            vector<uint32_t> first_axis_order=generate_tensor_axis_order(shape[first_axis],flow.axis_modes[first_axis]);
            for(uint32_t first_position:first_axis_order)
                append_axis_line(second_axis,first_position,false);
        }
        else
        {
            uint8_t first_axis=flow.axis_order[0],second_axis=flow.axis_order[1];
            array<vector<uint8_t>,2> ready_positions{{vector<uint8_t>(shape[0],0),vector<uint8_t>(shape[1],0)}};
            array<vector<uint32_t>,2> ready_position_lists;
            for(uint8_t axis=0;axis<2;axis++)
            {
                ready_positions[axis][0]=1;
                ready_position_lists[axis].push_back(0);
                for(uint32_t position=8;position<shape[axis];position+=8)
                {
                    ready_positions[axis][position]=1;
                    ready_position_lists[axis].push_back(position);
                }
            }
            append_slice(tensor_slice_id(shape,0,0),{});
            for(uint32_t position_index=1;position_index<ready_position_lists[first_axis].size();position_index++)
            {
                uint32_t position=ready_position_lists[first_axis][position_index],source_position=ready_position_lists[first_axis][position_index-1];
                temporal_dependency_c dependency;
                dependency.source_indices[0]=(int32_t)make_slice_id(first_axis,source_position,0);
                dependency.source_weights[0]=1;
                dependency.source_num=1;
                append_slice(make_slice_id(first_axis,position,0),dependency);
            }
            for(uint32_t first_position:ready_position_lists[first_axis])
                for(uint32_t position_index=1;position_index<ready_position_lists[second_axis].size();position_index++)
                {
                    uint32_t position=ready_position_lists[second_axis][position_index],source_position=ready_position_lists[second_axis][position_index-1];
                    temporal_dependency_c dependency;
                    dependency.source_indices[0]=(int32_t)make_slice_id(second_axis,source_position,first_position);
                    dependency.source_weights[0]=1;
                    dependency.source_num=1;
                    append_slice(make_slice_id(second_axis,position,first_position),dependency);
                }
            constexpr uint32_t offsets[]={4,2,1};
            for(uint32_t offset:offsets)
                for(uint8_t layer_axis_id=0;layer_axis_id<2;layer_axis_id++)
                {
                    uint8_t axis=flow.interpolation_axis_order[layer_axis_id],other_axis=1-axis;
                    vector<uint32_t> targets;
                    for(uint32_t position=offset;position<shape[axis];position+=2*offset)
                        if(!ready_positions[axis][position])
                            targets.push_back(position);
                    for(uint32_t fixed_position:ready_position_lists[other_axis])
                        for(uint32_t position:targets)
                        {
                            auto slice_id_from_position=[&](uint32_t source_position){return make_slice_id(axis,source_position,fixed_position);};
                            temporal_dependency_c dependency=build_tensor_axis_dependency(axis,position,ready_positions[axis],axis_coordinates,slice_id_from_position,tensor_axis_mode_e::anchor_interpolation_stride_8);
                            append_slice(make_slice_id(axis,position,fixed_position),dependency);
                        }
                    for(uint32_t position:targets)
                    {
                        ready_positions[axis][position]=1;
                        ready_position_lists[axis].push_back(position);
                    }
                    sort(ready_position_lists[axis].begin(),ready_position_lists[axis].end());
                }
        }
        validate_tensor_dependency_plan(shape,plan);
        return plan;
    }

    inline vector<tensor_flow_c> generate_tensor_flow_candidates()
    {
        vector<tensor_flow_c> flows;
        flows.push_back({});
        constexpr tensor_axis_mode_e active_modes[]={tensor_axis_mode_e::constant_extrapolation,tensor_axis_mode_e::anchor_interpolation_stride_8};
        for(uint8_t axis=0;axis<2;axis++)
            for(tensor_axis_mode_e mode:active_modes)
                for(bool residual:{false,true})
                {
                    tensor_flow_c flow;
                    flow.axis_modes[axis]=mode;
                    flow.use_spatial_residual=residual;
                    flows.push_back(flow);
                }
        for(tensor_axis_mode_e axis0_mode:active_modes)
            for(tensor_axis_mode_e axis1_mode:active_modes)
                for(uint8_t first_axis=0;first_axis<2;first_axis++)
                    for(bool residual:{false,true})
                    {
                        tensor_flow_c flow;
                        flow.axis_modes={{axis0_mode,axis1_mode}};
                        flow.axis_order={{first_axis,(uint8_t)(1-first_axis)}};
                        flow.use_spatial_residual=residual;
                        flows.push_back(flow);
                    }
        for(uint8_t first_axis=0;first_axis<2;first_axis++)
            for(uint8_t first_interpolation_axis=0;first_interpolation_axis<2;first_interpolation_axis++)
                for(bool residual:{false,true})
                {
                    tensor_flow_c flow;
                    flow.axis_modes={{tensor_axis_mode_e::anchor_interpolation_stride_8,tensor_axis_mode_e::anchor_interpolation_stride_8}};
                    flow.axis_order={{first_axis,(uint8_t)(1-first_axis)}};
                    flow.interpolation_axis_order={{first_interpolation_axis,(uint8_t)(1-first_interpolation_axis)}};
                    flow.schedule=tensor_schedule_e::interleaved_anchor;
                    flow.use_spatial_residual=residual;
                    flows.push_back(flow);
                }
        if(flows.size()!=33)
            throw runtime_error("二维tensor flow候选数量不是33");
        return flows;
    }
}
