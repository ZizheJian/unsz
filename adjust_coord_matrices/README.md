# Adjust coord矩阵

这里保存已经选定、可以直接用于压缩和解压的坐标变换矩阵。训练过程中每轮loss、耗时和中间矩阵仍保存在`adjust_coord_history/*.npz`，二者用途不同。

矩阵文件中的`training_type`明确区分：

- `spatial`：只使用空间坐标训练，矩阵通常是二维或三维。
- `spatiotemporal`：将time step作为额外坐标维度进行联合时空训练，矩阵通常比空间维数多一维。

文本格式为：

```text
UNSZ_ADJUST_COORD_MATRIX 2
training_type spatial
field attr0
coordinate_dim 3
lower_triangular c00 c10 c11 c20 c21 c22
```

`t2_adjust_coord.py`命令行只接受`-data`和`-field`，并通过`t1_start_c_code.py`读取对应数据集JSON中该field的输入路径和实际预测器。已有正式矩阵名时覆盖该文件；尚未启用坐标变换时，先按`<dataset>_<field>_spatial.txt`生成候选矩阵，实测后再决定是否写入JSON配置。例如：

```bash
./unsz_env/bin/python t2_adjust_coord.py -data tsunami -field bathy
```

需要调整loss、采样、优化器、预测器、训练参考点数或输出路径时，在Python中调用`t2_adjust_coord.train_adjust_coord()`；这些参数不属于正式`t2`命令行。

C++压缩和解压使用同一个矩阵文件：

```bash
install/bin/s1_compress ... -adjust-coord-matrix-file adjust_coord_matrices/Katrina_attr0_spatial.txt
install/bin/s1_decompress ... -adjust-coord-matrix-file adjust_coord_matrices/Katrina_attr0_spatial.txt
```

当前统一训练协议使用实际空间预测器的全部参数和`prediction_ggd_bit`离散量化区间码长loss，不使用GGD微分熵loss。训练目标数、训练EB、迭代数和矩阵吸收率可以由数据集JSON中的field task覆盖。Tsunami的六个field已经按RBF10、Matérn-3/2核、shape `0.03`、nugget `1e-6`、无趋势项和10000个训练目标完成重测；改善最多的height也只降低约`0.46%` BR，因此六项全部使用原始坐标，候选矩阵仅保存在结果目录。完整对照结果见`results/tsunami_adjust_coord_rbf10_discrete_entropy/summary.csv`。Katrina最终使用100000个目标、训练REL EB `1e-3`、2轮和吸收率`0.75`；在`78.7245 dB`处的BR为`0.313441`，比可复现的旧二次方loss矩阵低`4.733%`。完整调优记录见`results/katrina_adjust_coord_ggd_tuning/`。

`Katrina_attr0_spatiotemporal_t162_random_hyperrectangle_regions_r4.txt`是全部162步四维联合筛选矩阵。它在REL `1e-3`下同PSNR BR明显差于当前纯空间和上一时刻标签方案，只用于复现否定性实验，不是Katrina正式配置。
