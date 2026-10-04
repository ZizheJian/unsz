# inverse-rank EB参数细调

固定RBF10、Exponential、shape=0.01、nugget=1e-6和inverse_rank_log2的base=0.8。第一阶段以log2_shift=0.5为基准，测试0.25和0.75，并固定max_log2_exponent=7。选定log2_shift后，第二阶段以max_log2_exponent=7为基准测试6和8。

所有候选复用同一field已有的RBF10 DAG，只改变预测目标的EB倍率。每个候选运行完整五点RD曲线，并在共同PSNR区间比较BD-rate。
