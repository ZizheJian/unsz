# Adjust coord矩阵

这里保存已经选定、可以直接用于压缩和解压的坐标变换矩阵。训练过程中每轮loss、耗时和中间矩阵仍保存在`adjust_coord_history/*.npz`，二者用途不同。

矩阵文件中的`training_type`明确区分：

- `spatial`：只使用空间坐标训练，矩阵通常是二维或三维。
- `spatiotemporal`：将time step作为额外坐标维度进行联合时空训练，矩阵通常比空间维数多一维。

文本格式为：

```text
UNSZ_ADJUST_COORD_MATRIX 1
training_type spatial
dataset Katrina
variable attr0
dim 3
lower_triangular c00 c10 c11 c20 c21 c22
```

`t2_adjust_coord.py`默认把空间矩阵写到该目录；`t5_cross_time.py`默认写入带time step数、采样模式和区域数的联合时空矩阵。也可以分别使用`-matrix-output-path`和`-joint-matrix-output-path`指定其他位置。

C++压缩和解压使用同一个矩阵文件：

```bash
install/bin/s1_compress ... -adjust-coord-matrix-file adjust_coord_matrices/Katrina_attr0_spatial.txt
install/bin/s1_decompress ... -adjust-coord-matrix-file adjust_coord_matrices/Katrina_attr0_spatial.txt
```
