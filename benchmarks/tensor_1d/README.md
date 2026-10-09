# 一维tensor基线

本入口与UNSZ/UMC源码隔离。每个`field.dat.slice_id`按原始点序作为一个float32一维数组，不读取空间坐标、连接关系或时间/高度预测DAG。官方SZ3、ZFP、PFPL由固定Git子模块提供。

```bash
cmake -S benchmarks/tensor_1d -B build/tensor_1d -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build/tensor_1d --parallel 2 --target benchmark_sz3 benchmark_zfp benchmark_pfpl
sbatch --job-name=ocean_1d_sz3 benchmarks/tensor_1d/run.slurm sz3
sbatch --job-name=ocean_1d_zfp benchmarks/tensor_1d/run.slurm zfp
sbatch --job-name=ocean_1d_pfpl benchmarks/tensor_1d/run.slurm pfpl
```

默认测试完整Ocean-large的95个field、全部36264个切片、REL=1e-3、单线程。各作业申请1个CPU附带的内存、4小时时限；接口预留`--threads`，但此次不启动并行扩展实验。SZ3使用上游默认Interpolation/Lorenzo自动选择；ZFP使用fixed-accuracy；PFPL调用上游ABS编码/解码函数，使用其官方编译选项`-mno-fma -ffp-contract=off`。

每个切片以排除`-99999`后的有限值range计算ABS EB，再传给三者的ABS接口。输入仍完整包含`-99999`，不额外拆分label、不重排、不做后置Zstd。解压后逐点检查全部值，包括sentinel；误差超过1.01×ABS EB的结果标记无效，不能用于正常RD/throughput汇总。

ABS EB为0时必须精确恢复：SZ3使用官方lossless模式，ZFP使用官方reversible模式，PFPL通过其protected-values阈值保留所有normal值及零；PFPL此模式不接受subnormal输入，遇到时明确失败，不静默改变误差要求。PFPL上游单次输入大小限制约2GiB，当前每个切片约14.8MB，不受影响。

计时在C++内部包围一次正式内存编码/解码调用，包括API中的必要元数据、内部缓冲分配等；读取、range/ABS EB计算、调用方重建缓冲分配和误差统计排除。不使用PFPL命令行默认的9次重复测速，也不以进程墙钟计算核心吞吐率。码流保留在内存中并立即解压验证，不写出完整压缩/重建文件。此基线没有可跨切片复用的几何预处理，预处理列为0。

输出位于`results/ocean_large_tensor_1d/<codec>/rel_0.001_t1/`，包括逐切片CSV、field汇总JSON、summary CSV、失败列表、日志及二进制快照。manifest记录输入metadata哈希、各上游提交和二进制SHA；不同版本禁止混合。失败的field仍保留诊断结果，随后继续其他field。

官方并行能力：SZ3支持OpenMP压缩/解压；PFPL提供OpenMP CPU及CUDA GPU实现；ZFP的fixed-accuracy CPU路径支持OpenMP压缩，但解压使用serial执行策略。具体行为以固定子模块源码为准，不把CPU与GPU结果混在同一条曲线中。
