# UNSZ

UNSZ是一个面向非结构点数据的误差有界有损压缩研究原型。它不依赖mesh connectivity，而是根据点坐标建立kd-tree、生成可由压缩端和解压端共同复现的预测顺序与空间参考关系，再对预测残差进行量化、Huffman编码和Zstd压缩。项目目前重点研究坐标距离与数据可预测性的关系、非规则点集中的预测顺序、点集标签以及跨time step预测。

# 1. 使用方法

## 1.1. 输入格式

当前命令行程序使用float32坐标和float32数据。一个数据目录至少包含：

| 文件 | 内容 |
|---|---|
| `coords.dat` | 按点连续保存的坐标，每个点包含`dim`个float32数值 |
| `<variable>.dat.<time_step>` | 对应time step的float32 field，每个点一个数值 |

例如三维、31个time step的temperature数据可以组织为：

```text
Ocean/
├── coords.dat
├── temperature.dat.0
├── temperature.dat.1
├── ...
└── temperature.dat.30
```

所有field文件的点数必须与`coords.dat`一致。UNSZ只使用点坐标，不需要`conn.dat`；二维数据通过`-dim 2`指定。程序读取float32坐标后会根据坐标跨度判断距离计算是否需要升级为double，但不会改变磁盘输入格式。

## 1.2. 安装

需要CMake、支持C++17和AVX2的C++编译器以及Python 3。C++依赖的Zstd和参数解析库均作为submodule提供；Python坐标变换训练需要NumPy和SciPy，绘图脚本还需要PyVista。

克隆项目并初始化submodule：

```bash
git clone --recurse-submodules git@github.com:ZizheJian/unsz.git
cd unsz
```

如果项目已经克隆：

```bash
git submodule update --init --recursive
```

构建并安装压缩与解压程序：

```bash
cmake -S c_codes -B build -DCMAKE_INSTALL_PREFIX="$PWD/install" -DCODE_NAME=s1_compress
cmake --build build --parallel 8
cmake --install build
```

生成的程序位于`install/bin/s1_compress`和`install/bin/s1_decompress`。

Python环境可以这样准备：

```bash
python3 -m venv unsz_env
source unsz_env/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy pyvista
```

## 1.3. 压缩与解压

最基本的REL error bound压缩命令为：

```bash
install/bin/s1_compress -data-dir /path/to/Ocean -variable temperature -time-step '[0,30]' -dim 3 -M REL 1e-3
```

对应的解压命令为：

```bash
install/bin/s1_decompress -data-dir /path/to/Ocean -variable temperature -time-step '[0,30]' -dim 3 -M REL 1e-3
```

ABS error bound使用：

```bash
install/bin/s1_compress -data-dir /path/to/data -variable field -time-step '[0,9]' -dim 3 -M ABS 1e-4
```

`-time-step`支持闭区间`[a,b]`、右开区间`[a,b)`和显式列表`{a1,a2,...}`。例如`'[0,31)'`和`'{0,3,7}'`分别处理0至30以及0、3、7号time step。包含括号或花括号的表达式建议使用引号，避免被shell解释。

压缩文件写到原数据旁边，名称为`<variable>.dat.<time_step>.unsz`；解压结果为`<variable>.dat.<time_step>.unsz.bin`。程序打印总BR、PSNR、最大误差比例以及预处理、核心压缩和核心解压时间。

当前bitstream不保存用户error bound、坐标变换矩阵、预测参数或实验开关。解压时必须提供与压缩时完全相同的参数；参数不一致通常会导致解码失败或误差检查失败。这是当前论文实验采用的设计，不是面向独立分发的自描述格式。

## 1.4. 当前空间预测实验配置

裸命令行使用随机分轮和默认IDW预测。当前正式空间实验还显式启用反向D-only轮内排序：

```bash
install/bin/s1_compress -data-dir /path/to/data -variable field -time-step '[0,9]' -dim 3 -M REL 1e-3 -reverse-mean-distance-only-order-experimental
```

重要参数如下：

| 参数 | 当前默认值 | 含义 |
|---|---:|---|
| `-max-ref-num` | 10 | 在未知点集标签时保留的候选参考点数 |
| `-actual-ref-num` | 10 | 标签预测使用的候选点数，以及数值预测最多使用的同点集参考点数 |
| `-inverse-distance-power` | 3.6 | IDW使用的欧氏距离负指数 |
| `-kd-leaf-size` | 32 | kd-tree叶节点最多保存的点数 |
| `-d-only-cache-refill-threshold-experimental` | 8 | D-only缓存少于该数量时重新查询 |

可以通过`-adjust-coord-matrix`直接提供下三角变换矩阵。三维矩阵按行输入6个非零元素：

```bash
install/bin/s1_compress -data-dir /path/to/data -variable field -time-step '[0,9]' -dim 3 -M REL 1e-3 -adjust-coord-matrix c00 c10 c11 c20 c21 c22 -reverse-mean-distance-only-order-experimental
```

解压时必须提供相同矩阵。仅使用`-adjust-coord`会启用源码中当前硬编码的三维实验矩阵，因此对新数据推荐显式使用`-adjust-coord-matrix`。

## 1.5. 批量实验与脚本

| 脚本 | 功能 |
|---|---|
| `t0_start_single_c_code.py` | 构建并运行一个C++入口，适合单次小实验 |
| `t1_start_c_code.py` | 批量运行UNSZ和UMC，执行压缩、独立解压与误差检查，并将BR、PSNR及三类吞吐率写入数据旁边的CSV |
| `t2_compress.py` | 训练空间坐标变换矩阵，可选择四次方、二次方、对数平方或预测码长近似loss |
| `t3_visualize_katrina_attr0.py` | 生成Katrina坐标变换前后的field图像 |
| `t4_visualize_ocean_temperature.py` | 生成Ocean temperature坐标变换前后的field图像 |
| `t5_cross_time.py` | 比较逐步空间、上一time step、层次时间插值和联合四维预测 |

运行批量实验前需要编辑`t1_start_c_code.py`开头的`data_root`、`umc_root`、`dataset_list`、`rel_eb_list`和`unsz_extra_arguments`。UMC需要事先单独构建；脚本不会构建UMC。之后运行：

```bash
python3 t1_start_c_code.py
```

历史消融脚本、Slurm文件和原始结果集中保存在`py_codes/trash/`，索引见`py_codes/trash/README.md`。它们用于复现实验，不代表当前推荐接口。

# 2. 方案描述

## 2.1. 整体流程

UNSZ将公共网格预处理与逐field文件压缩分开：

1. 读取坐标，必要时将坐标和距离计算从float升级为double。
2. 可选地使用离线学习的线性矩阵变换坐标，使变换后距离更接近数据值的可预测性。
3. 建立公共kd-tree，将全部点划分为多轮，确定轮内预测顺序，并为每个目标寻找已经重建的空间参考点。
4. 对每个time step独立读取数据、划分点集、计算各点集值域和error bound。
5. 根据已重建的同点集参考点执行预测，立即量化预测残差并用重建值覆盖目标，保证压缩端与解压端具有相同的误差传播路径。
6. 将点集标签和各点集量化索引分别无损压缩，最后组装成一个`.unsz`容器。

坐标、预测轮和候选参考关系可以被同一网格上的所有field和time step复用。默认不会把公共预测结构写入每个压缩文件；解压端根据坐标重新生成。`-prediction-structure <path>`可以把它保存为本地缓存，但当前缓存文件较大，不适合作为必须传输的压缩metadata。

## 2.2. Adjust coord

原始欧氏距离不一定能表示两个点之间的数据可预测性。例如Katrina的z轴被严重拉伸，几何最近点未必是数值上最合适的参考点。Adjust coord离线学习下三角矩阵`C`，使点对满足：

```text
|v1-v2|≈||(c1-c2)C||₂
```

Python版本从多个训练文件中均匀选择文件，先采样坐标与有效数据，再为采样目标寻找同一文件内的有效近邻，最后迭代更新矩阵。当前保留四种可替换目标：

| loss | 拟合关系 |
|---|---|
| `fourth_power` | 最小化`(dv²-distance²)²` |
| `second_power` | 最小化`(|dv|-distance)²` |
| `log_distance` | 最小化带abs EB常数项的对数距离差平方 |
| `prediction_ggd_bit` | 根据IDW预测误差的广义高斯分布近似码长 |

当前主线采用二次方loss、少量外层轮次和早停；Gauss–Newton是速度优先训练器。Python负责训练矩阵，C++正式压缩器只应用给定矩阵，不在运行时重新优化。

## 2.3. kd-tree、预测轮与参考点

默认将全部点分为10轮，前期轮点数较少，后期轮点数近似翻倍。一个点只能引用解压顺序中已经恢复的点，因此整个预测关系构成有向无环图。

当前正式实验使用反向D-only顺序：在反向剔除过程中，只根据坐标计算目标到可用参考集合的平均平方距离`D`，并在空间邻近的候选目标之间选择。最终反转剔除顺序，使较难建立可靠局部参考的点更早恢复、较容易预测的点留到后面。该顺序只依赖坐标，可以离线生成并复用于所有time step。

kd-tree目前采用32点叶节点、tree-order连续坐标、共享树拓扑以及连续参考边数组。目标`i`的参考点位于`pred_ref_ids[pred_ref_begin[i]:pred_ref_begin[i+1]]`，对应权重保存在连续权重数组中，避免大量小`vector`和指针跳转。

实验性的global方案不先随机分轮，而是对全部点生成一条D-only顺序，再按照原有各轮点数切分。它在LES和Tsunami上明显改善BR，但在Katrina上略有退化，因此尚未成为默认方案。

## 2.4. 点集与标签

不同类型的数据点不一定应该互相预测。当前Katrina实现将`value<-1000`的fill value与正常值划分为两个点集：

- 正常点只使用正常参考点预测。
- fill value点只使用fill value参考点预测；当前全部为`-99999`时自动使用常量编码，没有数值预测误差。
- 额外预测每个点属于哪个点集，只保存标签预测错误位置，并使用Zstd压缩。

每个time step独立生成标签，因此异常点位置可以随时间变化。单点集数据不生成标签流。当前容器最多支持两个点集；推广到用户定义的有序多点集是下一项主要工作。

REL error bound根据各点集自身的最大值和最小值转换为ABS error bound，避免fill value扩大正常数据的值域。容器记录各点集值域，但用户仍需在解压时提供相同的REL或ABS error bound。

## 2.5. 数值预测、量化与无损编码

当前通用预测器是inverse-distance weighting：

```text
prediction=Σ(valueᵢ·distanceᵢ^-3.6)/Σ(distanceᵢ^-3.6)
```

预测只使用已经重建且属于当前点集的参考点。压缩端立即用量化重建值替换原值，解压端因而可以严格复现后续预测。当前默认按照轮间公比0.8逐轮、逐点放宽error bound，并在最后达到用户指定上限；这一经验参数仍需与预测轮数共同消融。

量化索引使用int32_t，随后采用Huffman编码和Zstd level 3。按空间block或预测轮拆分多棵Huffman tree的收益较小且不稳定，因此当前仍使用单树方案。

项目还实现了局部仿射拟合、IDW/拟合LOO选择、连续混合、IDW残差拟合和block自适应选择。它们可以改善部分平滑标量field，但目前明显慢于纯IDW，仍属于研究候选。

## 2.6. 跨time step预测

当前实现了三类跨时间方案：

1. `previous`：从第二个time step开始，优先使用上一重建time step中同一空间位置的值。
2. `interpolation`：先空间压缩首尾time step，再按层次顺序恢复区间中点，使用两端重建值进行时间线性插值；若点集类型不一致则回退空间预测。
3. 联合四维：把全部time step展平成四维点集，通过四维adjust coord自动平衡空间距离和时间距离，再执行统一空间近邻预测。

层次时间插值和联合四维方案目前均为实验路径。联合四维数据的准备、矩阵训练和对比由`t5_cross_time.py`完成。

# 3. 当前结果

以下BR改善均在RD曲线的共同PSNR范围内插值得到，不是比较相同REL error bound。结果仍属于论文准备阶段，正式投稿前会在冻结版本上统一重测。

## 3.1. 空间预测的rate-distortion

| 数据 | 方法与比较 | 同PSNR BR变化 |
|---|---|---:|
| Ocean，9个field×31步 | 4-block自适应UNSZ相对UMC | 平均`-15.843%`，中位`-8.105%` |
| LES velocity_x/y/z | random UNSZ相对UMC | `-26.07%～-28.20%` |
| LES velocity_x/y/z | global UNSZ相对UMC | `-36.53%～-38.66%` |
| Tsunami四个field | global UNSZ相对UMC | `-15.019%～+2.808%` |

Ocean收益主要集中在salinity、temperature和zTop，速度field通常约改善`5%～10%`。4-block自适应预测器的RD较好，但核心压缩约慢`4.2～5.3`倍、解压约慢`6.6～7.0`倍，因此尚未替换纯IDW正式配置。

Adjust coord在坐标距离明显失真的数据上尤其重要。Katrina当前二次方loss矩阵相对旧四次方100轮矩阵降低`12.182%` BD-rate；Ocean temperature的新二次方早停矩阵相对不变换降低`2.722%`，说明不同field的收益差异很大。

## 3.2. global排序

| 数据 | global相对random的BD-rate |
|---|---:|
| LES velocity_x | `-12.218%` |
| LES velocity_y | `-12.292%` |
| LES velocity_z | `-12.395%` |
| Tsunami momentum_x | `-5.585%` |
| Tsunami momentum_y | `-6.422%` |
| Tsunami velocity_x | `-0.987%` |
| Tsunami velocity_y | `-5.696%` |
| Katrina attr0，time step 0 | `+1.683%` |

global首次预处理比random慢约`23%～32%`，但LES核心压缩快约`4.7%～5.9%`、解压快约`5.9%～7.8%`。它不是通用赢家，下一步需要根据少量训练数据在random和global之间离线选择。

## 3.3. 跨time step结果

Ocean velocityZonal全部31步的层次时间插值，相对其他方案的BD-rate为：

| 基线 | BR变化 |
|---|---:|
| 逐time step空间预测 | `-61.170%` |
| 上一time step同位置预测 | `-29.065%` |
| 4区域联合四维预测 | `-22.455%` |

层次时间插值是当前最强的跨时间基线。是否仍存在联合四维方案不可替代的数据，需要在更多具有不同时间相关性的field和数据集上验证。

联合四维预测可以在adjust后的坐标上独立分区。velocityZonal、31步、REL`1e-3`下，64分区相对不分区使BR降低`0.498%`、预处理加快`17.14%`、压缩加快`14.75%`；128分区使BR降低`0.373%`、预处理加快`19.27%`、压缩加快`15.85%`。这为后续一分区一任务的并行实现提供了基础。

## 3.4. 串行吞吐率

以下为同一独占节点上的核心吞吐率范围，不包含文件I/O、误差统计和结果重排：

| 数据 | 方法 | 压缩MB/s | 解压MB/s |
|---|---|---:|---:|
| Katrina attr0，162步 | UNSZ | `70.03～72.66` | `67.24～69.99` |
| Katrina attr0，162步 | UMC | `105.70～112.90` | `226.55～308.94` |
| Ocean temperature，31步 | UNSZ | `63.59～70.34` | `78.58～107.35` |
| Ocean temperature，31步 | UMC | `71.92～91.20` | `102.59～188.21` |

权重预计算和逐文件点集路径优化已经把Katrina上的压缩速度差距从旧版本约`3.67×`缩小到约`1.58×`，但UMC的解压仍明显更快。连续参考数组在不改变码流的情况下，使Ocean/Katrina总压缩分别加快`2.69%/1.47%`，解压分别加快`0.31%/3.65%`。

# 4. TODO

## 4.1. 投稿前主线

- [ ] 使用少量训练文件的预测误差或估计码长，在random和global预测顺序之间离线选择；不能只根据坐标覆盖率判断。
- [ ] 寻找更多具有不同网格结构、坐标尺度、点集分布和时间相关性的非结构数据，验证UNSZ、adjust coord和跨时间结论的普适性。
- [ ] 将当前两点集接口推广为用户定义的有序多点集分类函数。`n`个点集使用`n-1`个条件标签通道，每个点集可以独立选择codec和error bound。
- [ ] 冻结串行参数、bitstream和实验脚本，在统一版本上重新生成五点RD、吞吐率以及各阶段流长度。
- [ ] 实现adjust后独立分区的OpenMP版本：逻辑分区数不少于线程数，线程动态领取分区，每个分区保持独立预测DAG和码流片段。
- [ ] 完成single-core throughput、strong/weak scaling和端到端压缩—传输—解压实验，再决定DCC或IPDPS投稿路线。

## 4.2. 算法与实现候选

- [ ] 在更多数据上验证Gauss–Newton adjust coord训练，并研究低成本的“是否需要adjust”判断。
- [ ] 比较分块常矩阵、坐标相关低阶矩阵和预测码长近似loss；新增参数和metadata必须计入压缩成本。
- [ ] 联合消融预测轮数、每轮点数比例和逐轮EB比例，确定能够跨数据使用的固定参数。
- [ ] 为block自适应预测增加父块或全field回退与置信门槛，减少采样选择产生负收益的概率。
- [ ] 根据预测DAG中的影响力分配error bound，使会被大量后续点引用的参考点使用更小误差。
- [ ] 固定相同坐标、相同距离和浮点tie时的点编号tie-break，并验证重复压缩产生相同预测结构、文件SHA、BR和PSNR。
- [ ] 推导Huffman输出缓冲区的严格上界，检查空量化索引和极端符号范围。

详细参数、完整消融结果和已经否定的方案见`TODO_DETAIL.md`；面向日常决策的精简状态见`TODO.md`。
