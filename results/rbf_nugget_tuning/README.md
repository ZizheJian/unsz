# RBF nugget粗筛

固定RBF10、Exponential和shape=0.01，在当前nugget=1e-6两侧各测试一个数量级：1e-7和1e-5。基准1e-6直接复用shape粗筛结果，不重复运行。

1e-7在float中接近单位对角线附近一个ULP，但仍会产生非零对角增量。本轮用于判断更弱或更强正则化的方向，不继续扩展网格。
