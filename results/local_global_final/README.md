# Local与global最终比较

目标是只保留一种排序范围。只使用已经完成流程筛选的Tsunami、LES、E3SM AODDUST、Haiyan zeta和Katrina；Tpost20尚未重新筛选时间流程，不参与本轮决策。

两种方案统一使用RBF10、Exponential、shape=0.01、nugget=1e-6、MSD1、16个排序候选，以及inverse_rank_log2的base=0.8、log2_shift=0.5和max_log2_exponent=7。local固定使用8轮和默认轮大小增长率2；global不分轮。每个scope使用独立DAG缓存并从空缓存开始，坐标变换沿用各正式任务设置。每个field交替local/global运行顺序。

压缩效果使用完整五点RD曲线在共同PSNR区间计算BD-rate。计时均取自程序内部；首次压缩预处理时间包含各自从零生成DAG的过程，核心压缩和解压时间在五个EB点上取几何平均。
