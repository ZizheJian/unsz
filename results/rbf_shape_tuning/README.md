# RBF shape粗筛

固定RBF10和nugget=1e-6，为kernel粗筛得到的Exponential与inverse multiquadric分别测试shape=0.01和0.1。shape=0.03直接复用kernel粗筛结果，不重复运行。Exponential初步胜出后继续测试更小的shape=0.003。

所有候选均复用同一field在kernel粗筛阶段生成的DAG；kernel、shape和nugget不影响坐标变换、DAG及参考点，只重新生成RBF权重。
