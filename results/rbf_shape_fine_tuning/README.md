# Exponential RBF shape细筛

使用当前代码成对比较Exponential的shape=0.01和0.003。五个数据均固定global、RBF10、nugget=1e-6、MSD1、16个排序候选，以及inverse_rank_log2的base=0.8、log2_shift=0.5和max_log2_exponent=7；其余使用各数据正式流程。

同一field的两个shape共享本轮重新生成的DAG。坐标变换训练不依赖RBF kernel或shape，因此只需训练并生成一次DAG；两个候选均在当前二进制下运行完整五点RD曲线，以避免和旧版本计时混用。
