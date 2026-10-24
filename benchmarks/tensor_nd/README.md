# 三维tensor基线

该实验把一个Ocean-large field按`[time,height,point]`组成完整三维float32数组，并调用SZ3三维接口。`point`维保持原始网格编号，因此时间和高度是规则tensor轴，最后一维不表示空间邻近性。

实验同时报告两种误差设置。`global-relative`按完整三维数组的有效值range计算一个ABS bound，这是直接调用tensor compressor时的自然语义。`strict-slice-relative`采用所有slice中最小的非零ABS bound，以满足论文当前每个slice分别按range计算REL bound的协议。如果field包含常量slice，单一码流无法同时用正误差界压缩其他slice并精确保存该slice，程序会明确拒绝`strict-slice-relative`。

```bash
cmake -S benchmarks/tensor_nd -B build/tensor_nd -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build/tensor_nd --parallel 2
sbatch benchmarks/tensor_nd/run.slurm
```

程序内部只计时SZ3内存压缩和解压调用。文件读取、range扫描和逐点验证不计入吞吐率。完整`12×80×3,693,225`字段约13.2 GiB，作业申请64个CPU仅为获得足够内存，SZ3仍以单线程运行。
