# E3SM label流程细筛

固定E3SM AODDUST的value为anchor_interpolation_stride_8，并在当前融合实现中成对比较label=spatial与label=anchor_interpolation_stride_8。

两者统一使用global、RBF10、Exponential、shape=0.01、nugget=1e-6、MSD1、16个排序候选，以及inverse_rank_log2的base=0.8、log2_shift=0.5和max_log2_exponent=7，并共享同一DAG。实验用于测量时间label替代逐点RBF10空间label后的完整RD与核心速度变化。
