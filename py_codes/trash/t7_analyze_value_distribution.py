import os
import numpy as np

data_dir="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean"
variable="temperature"
time_step_num=31
quantile_list=[0,1e-6,1e-5,1e-4,1e-3,1e-2,5e-2,.25,.5,.75,.95,.99,.999,.9999,1]

def main():
    data_list=[]
    print("逐时间步统计：")
    for time_step in range(time_step_num):
        path=os.path.join(data_dir,f"{variable}.dat.{time_step}")
        data=np.fromfile(path,dtype=np.float32)
        data_list.append(data)
        finite_data=data[np.isfinite(data)]
        print(f"  {time_step:2d}: point_num={data.size}, finite={finite_data.size}, nan={np.count_nonzero(np.isnan(data))}, -inf={np.count_nonzero(np.isneginf(data))}, +inf={np.count_nonzero(np.isposinf(data))}, range=[{np.min(finite_data):.9g},{np.max(finite_data):.9g}]")

    data=np.concatenate(data_list)
    finite_data=data[np.isfinite(data)]
    print("\n全部时间步：")
    print(f"  point_num={data.size}, finite={finite_data.size}, nan={np.count_nonzero(np.isnan(data))}, -inf={np.count_nonzero(np.isneginf(data))}, +inf={np.count_nonzero(np.isposinf(data))}")
    print("  分位数：")
    quantile_value=np.quantile(finite_data,quantile_list)
    for quantile,value in zip(quantile_list,quantile_value):
        print(f"    q={quantile:.6g}: {value:.9g}")

    unique_value,unique_count=np.unique(finite_data,return_counts=True)
    print(f"  不同精确值数量：{unique_value.size}")
    frequent_ids=np.argsort(unique_count)[-20:][::-1]
    print("  出现次数最多的精确值：")
    for value_id in frequent_ids:
        print(f"    value={unique_value[value_id]:.9g}, count={unique_count[value_id]}, ratio={unique_count[value_id]/data.size:.9g}")

    endpoint_num=min(20,unique_value.size)
    print("  最小端精确值：")
    for value,count in zip(unique_value[:endpoint_num],unique_count[:endpoint_num]):
        print(f"    value={value:.9g}, count={count}")
    print("  最大端精确值：")
    for value,count in zip(unique_value[-endpoint_num:],unique_count[-endpoint_num:]):
        print(f"    value={value:.9g}, count={count}")

    if unique_value.size>1:
        gap=np.diff(unique_value.astype(np.float64))
        gap_ids=np.argsort(gap)[-20:][::-1]
        print("  最大相邻取值间隔：")
        for gap_id in gap_ids:
            print(f"    [{unique_value[gap_id]:.9g},{unique_value[gap_id+1]:.9g}], gap={gap[gap_id]:.9g}, lower_count={unique_count[gap_id]}, upper_count={unique_count[gap_id+1]}")

if __name__=="__main__":
    main()
