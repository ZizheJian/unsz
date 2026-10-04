# inverse-rank EB base粗筛

固定RBF10、Exponential、shape=0.01、nugget=1e-6以及inverse_rank_log2的log2_shift=0.5和max_log2_exponent=7，先测试base=0.8和0.9，再补测0.75。当前base=0.85直接复用shape粗筛结果。

base越小，DAG前部的EB倍率越小，前部量化越严格；base越接近1，前部与后部的EB差异越弱。
