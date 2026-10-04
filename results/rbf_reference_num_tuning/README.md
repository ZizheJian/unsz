# RBF参考点数调优

在五个正式数据上比较七组参数：基线`min=10,max=10`，减少完整RBF规模的`min=max=8/9`，降低双点集同类参考点门槛的`min=7/8/9,max=10`，以及增加RBF候选参考点的`min=10,max=12`。

固定Exponential kernel、shape=0.01、nugget=1e-6、global、MSD1、16个排序候选，以及`inverse_rank_log2(base=0.8,log2_shift=0.5,max_log2_exponent=7)`，其余使用各数据当前正式流程。每组运行完整五点RD曲线，唯一选择指标是相对于`10/10`的同PSNR BD-rate；本轮不以速度作决策。

`10/10`和`8/10`共享同一份10点DAG。`10/12`会重新训练坐标变换并生成独立12点DAG，因为坐标变换训练和DAG结构均依赖最大参考点数。
