# UNSZ

UNSZ是一个主要面向多time step点云数据的误差有界有损压缩研究原型，不同time step应共享同一套公共点坐标。只有一个time step的数据同样可以压缩。UNSZ不依赖mesh connectivity，而是根据公共点坐标建立kd-tree、生成可由压缩端和解压端共同复现的预测结构，再对预测残差进行量化、Huffman编码和Zstd压缩。

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

所有field文件的点数必须与`coords.dat`一致。二维数据通过`-dim 2`指定。程序读取float32坐标后会根据坐标跨度判断距离计算是否需要升级为double，但不会改变磁盘输入格式。

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

默认不会把公共预测结构写入每个压缩文件，解压端会根据坐标重新生成。使用`-prediction-structure <path>`可以把预测结构保存为本地缓存：路径不存在时生成并写入，路径存在时直接读取。当前缓存文件较大，不适合作为必须随压缩数据传输的元数据；压缩和解压必须使用由相同坐标变换和预测参数生成的缓存。

## 1.4. 自定义有序点集

不传`-point-set-config`时，程序把全部数据作为一个点集，使用命令行`-predictor`和`-M`来指定使用的预测器和误差上限。传入JSON配置文件时，文件可以定义一个或多个点集；命令行中即使同时出现`-predictor`或`-M`也会被忽略。压缩和解压必须传入相同的配置文件：

```bash
install/bin/s1_compress -data-dir /path/to/Katrina -variable attr0 -time-step '[0,161]' -dim 3 -point-set-config point_set_configs/Katrina_attr0_rel_1e-2.json
install/bin/s1_decompress -data-dir /path/to/Katrina -variable attr0 -time-step '[0,161]' -dim 3 -point-set-config point_set_configs/Katrina_attr0_rel_1e-2.json
```

`rule_type`为`ordered`时，`labels`数组下标就是点集编号。程序按顺序测试除最后一个label外的`condition`，第一条成立的条件决定label；全部不成立时属于最后一个label。比较符直接使用`<`、`<=`、`>`、`>=`、`==`或`!=`。下面是Katrina的两点集配置，其中正常值使用自身值域上的REL EB，fill value点集使用常量编码：

```json
{
  "version":1,
  "rule_type":"ordered",
  "labels":[
    {"method":"inverse_proportion","eb_mode":"REL","eb_value":0.01,"condition":{"operator":">=","value":-1000}},
    {"method":"constant"}
  ]
}
```

最后一个label是默认兜底类型，因此省略`condition`；配置文件只有一个label时同样不写`condition`。每个非常量label必须显式配置`eb_mode`和`eb_value`，模式为`REL`或`ABS`；`constant`省略这两个字段。可用method为`inverse_proportion`、`fitting`、`hybrid_loo`和`constant`。

分类逻辑很复杂时，用户可以为每个time step提供一个`uint8`点集编号文件，再把`rule_type`设为`label_file`。路径必须包含`{time_step}`，相对路径以JSON文件所在目录为基准：

```json
"rule_type":"label_file",
"label_file":{"path":"point_set_id.dat.{time_step}","dtype":"uint8"}
```

每个标签文件必须恰好包含`point_num`个字节，编号范围为`0`到`labels`数量减1。`label_file`模式下各label不写`condition`。完整示例位于`point_set_configs/`。由于EB不写入bitstream，解压时仍必须使用与压缩时相同的配置。

## 1.5. 批量实验与脚本

| 脚本 | 功能 |
|---|---|
| `t0_start_single_c_code.py` | 构建并运行一个C++入口，适合单次小实验 |
| `t1_start_c_code.py` | 批量运行UNSZ和UMC，执行压缩、独立解压与误差检查，并将BR、PSNR及三类吞吐率写入数据旁边的CSV |
| `t2_adjust_coord.py` | 训练空间坐标变换矩阵，可选择四次方、二次方、对数平方或预测码长近似loss |
| `t3_visualize.py` | 使用当前矩阵生成Katrina attr0与Ocean temperature坐标变换前后的field图像 |
| `t5_cross_time.py` | 比较逐步空间、上一time step、层次时间插值和联合四维预测 |

运行批量实验前需要编辑`t1_start_c_code.py`开头的`data_root`、`umc_root`、`dataset_list`、`rel_eb_list`和`unsz_extra_arguments`。UMC需要事先单独构建；脚本不会构建UMC。之后运行：

```bash
python3 t1_start_c_code.py
```

`t1_start_c_code.py`会为每个数据集创建本次运行专用的临时空间预测结构：第一个REL EB生成缓存，后续REL EB及对应解压直接读取，数据集测试结束后自动删除，从而复用预处理并避免误读其他参数生成的旧缓存。

训练单time step空间坐标变换矩阵可以使用：

```bash
./unsz_env/bin/python t2_adjust_coord.py -data-dir /path/to/Katrina -variable attr0 -time-step-num 162 -loss second_power -optimizer least_squares -iteration-num 4 -matrix-history-path adjust_coord_history/katrina.npz -matrix-output-path adjust_coord_matrices/Katrina_attr0_spatial.txt
```

`.npz`历史文件保存每轮矩阵和优化统计；文本矩阵文件只保存最终选定矩阵，供C++直接读取。`t5_cross_time.py`使用`-joint-matrix-output-path`保存`spatiotemporal`矩阵，默认文件名还包含time step数、采样模式和区域数，避免和普通空间矩阵混淆。

统一可视化程序默认生成Katrina attr0与Ocean temperature的变换前、变换后图像，也可以只选择一个数据集：

```bash
./unsz_env/bin/python t3_visualize.py
./unsz_env/bin/python t3_visualize.py -dataset katrina
./unsz_env/bin/python t3_visualize.py -dataset ocean-temperature
```

Katrina原坐标图中的`z*=0.001`只是为了避免图像被严重拉长的显示缩放，不是PyVista自动执行的坐标变换，也不参与压缩或adjust coord训练。

历史消融脚本、Slurm文件和原始结果集中保存在`py_codes/trash/`，索引见`py_codes/trash/README.md`。它们用于复现实验，不代表当前推荐接口。

# 2. 方案描述

本章各优化表中的“状态”只表示当前决策：`已采纳`表示已经进入当前主线；`实验性保留`表示已有实现但不默认启用；`需要更多样本`表示现有数据不足以决定；`待实现`表示方案已提出但尚无实现；`待验证`表示实现基本完成但缺少正确性或复现检查；`暂缓`表示优先级较低；`放弃`表示现有证据不支持继续投入。实验结果和原因单独写在“已有结论”列。

## 2.1. 整体流程

UNSZ目前包含三种单线程、不分块的压缩流程：

1. **所有文件逐个进行空间压缩。** 读取公共坐标，可选地训练并应用空间变换矩阵，然后在变换后的坐标上建立kd-tree并生成预测DAG。每个time step分别划分label，`n`个label依次使用`n-1`个递进二分类通道，按照DAG预测当前点是否属于对应label，只记录预测错误的位置，对位置作差分后将每个通道分别用Zstd压缩。随后分别计算各label的error bound，只在同label的已重建点之间完成数值预测和量化。在变换矩阵和其他预测参数不变时，空间变换矩阵、kd-tree和DAG可以被同一网格上的所有field和time step复用，但不同field可能需要不同的变换矩阵。
2. **所有文件进行联合时空压缩。** 将时间维度视为一个新的空间维度，把所有time step视为一个数据，生成时空变换矩阵、kd-tree和DAG。
3. **一部分文件空间压缩，其余文件进行时间插值。** 当前实现先完整空间压缩第一个和最后一个time step，再按照类似SZ3的层次插值顺序依次处理各区间的中点。处理每个非端点time step时，仍按照公共空间DAG预测其中的所有点。对于每个点，如果同一空间位置上左右两个已重建时间端点的label都与目标label相同，则优先使用时间线性插值；只要任一端点的label与目标label不同，该点就使用当前time step内的空间参考点预测。

每预测一个点就立即量化预测残差并用重建值覆盖目标，最后使用无损压缩。

逐文件空间压缩和时间插值流程可以复用同一份公共空间预测结构。联合时空压缩的预测结构还取决于参与压缩的time step及其时间坐标，需要单独生成。

## 2.2. 当前空间预测实验配置

裸命令行使用随机分轮和默认IDW预测。当前正式空间实验还显式启用反向平均平方距离排序，也就是只用目标点到可用参考点的平均平方距离`D`评价预测难度，后文简称D-only：

```bash
install/bin/s1_compress -data-dir /path/to/data -variable field -time-step '[0,9]' -dim 3 -M REL 1e-3 -reverse-mean-distance-only-order-experimental
```

重要参数如下：

| 参数 | 当前默认值 | 含义 |
|---|---:|---|
| `-max-ref-num` | 10 | 在未知点集标签时保留的候选参考点数 |
| `-actual-ref-num` | 10 | 标签预测使用的候选点数，以及数值预测最多使用的同点集参考点数 |
| `-predict-round-num` | 8 | 预测轮数；五点筛选中，8轮+逐轮EB公比0.8在4个代表field均优于原10轮默认值 |
| `-round-size-growth-ratio-experimental` | 2 | 前两轮等大，此后每轮目标点数相对上一轮的增长比例 |
| `-inverse-distance-power` | 3.6 | IDW使用的欧氏距离负指数 |
| `-kd-leaf-size` | 32 | kd-tree叶节点最多保存的点数 |
| `-d-only-cache-refill-threshold-experimental` | 8 | 排序所用的有效参考点缓存少于8个时重新查询kd-tree |

可以通过`-adjust-coord-matrix`直接提供下三角变换矩阵。三维矩阵按行输入6个非零元素：

```bash
install/bin/s1_compress -data-dir /path/to/data -variable field -time-step '[0,9]' -dim 3 -M REL 1e-3 -adjust-coord-matrix c00 c10 c11 c20 c21 c22 -reverse-mean-distance-only-order-experimental
```

解压时必须提供相同矩阵。仅使用`-adjust-coord`会启用源码中当前硬编码的三维实验矩阵，因此对新数据推荐显式使用`-adjust-coord-matrix`或`-adjust-coord-matrix-file`。

也可以使用独立矩阵文件，避免在源码或命令行中复制矩阵元素：

```bash
install/bin/s1_compress -data-dir /path/to/Katrina -variable attr0 -time-step '[0,9]' -dim 3 -M REL 1e-3 -adjust-coord-matrix-file adjust_coord_matrices/Katrina_attr0_spatial.txt -reverse-mean-distance-only-order-experimental
install/bin/s1_decompress -data-dir /path/to/Katrina -variable attr0 -time-step '[0,9]' -dim 3 -M REL 1e-3 -adjust-coord-matrix-file adjust_coord_matrices/Katrina_attr0_spatial.txt -reverse-mean-distance-only-order-experimental
```

矩阵文件记录数据集、field、维数、下三角元素和训练类型。`training_type=spatial`表示只使用空间坐标训练；`training_type=spatiotemporal`表示把time step作为额外坐标维度进行联合时空训练。当前选定矩阵集中保存在`adjust_coord_matrices/`，格式和文件索引见其中的`README.md`。

## 2.3. Adjust coord

UNSZ根据点之间的距离选择参考点，但原始坐标中的距离不一定能反映数据是否容易互相预测。例如Katrina的z轴被严重拉伸，坐标上最近的点不一定是数值上最接近的点。我们也不能在压缩每个文件时直接查看原始数据，再据此决定预测关系：解压端没有原始数据，想恢复数据必须先知道预测关系，而想重建这种预测关系又必须先知道原始数据。

Adjust coord用少量训练文件提前学习一个下三角矩阵`C`，以后只根据公共坐标和这个矩阵生成预测关系。坐标按照`c'=cC`变换，训练希望两个点的数值差与它们在新坐标中的距离接近：

```text
|v1-v2|≈||(c1-c2)C||₂
```

这个过程是迭代式的，但当前只迭代4轮。

### 2.3.1. 采样

训练不需要读取所有time step，而是在整个时间范围内等间隔选择`sample_file_num`个文件，默认选择3个。例如Katrina共有162个文件时，会读取第27、80和134个文件。

每轮训练要做两次采样。第一次采样决定用哪些点建立近邻树，主要控制建树和查询的规模；第二次采样只从树上的点中挑选一部分作为目标，主要控制交给优化器的点对数量。具体过程如下：

1. 从全部坐标中无放回地随机选择`ceil(point_num×first_sample_ratio)`个点，默认使用25%。选出的编号会排序，因此没有重复且保持原始点编号顺序；所有训练文件使用同一批坐标。第一次采样不能太稀疏，否则树上的“最近点”会离目标太远，无法代表完整点云中的局部关系。
2. 对每个训练文件分别删除这批点中的异常值。当前Python版本只通过`is_abnormal()`把点分为正常和异常两类，并且不让异常点参与训练。由于异常点的位置会随time step改变，每个文件最终留下的点可能不同。
3. 从每个文件剩余的点中再次无放回地随机选择一部分作为目标，也就是训练点对的第一个点。目标总数最多为`ceil(第一次采样点数×训练文件数×second_sample_ratio)`，默认取10%，并尽量平均分给各训练文件。
4. 对每个文件单独建立一棵`cKDTree`。树上的坐标先经过当前`C`变换；查询时先取`actual_ref_num+1`个近邻，删除目标自身，再保留最近的`actual_ref_num`个点作为参考点，也就是点对的第二个点。当前默认保留4个参考点，因此每个目标产生4个训练点对。
5. 对每个点对记录数值差`Δv`和未经变换的原始坐标差`Δc`。近邻查询使用当前`C`，但训练新矩阵时仍从原始`Δc`开始计算，避免把坐标重复变换。

得到新的`C`后，下一轮会重新进行两次采样、异常点检查、建树和查询。这样近邻关系会随着`C`更新，而不是四轮训练始终使用第一轮找到的点对。随机种子固定，因此相同参数可以复现实验。

采样相关方案如下：

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 调整两次<br>采样率 | 待验证 | 降低第一次采样率会减少建树、近邻查询的时间和内存，但树中点变稀疏，查询到的近邻会比完整点云中的真实近邻更远，可能使训练矩阵失真。第二次采样率控制作为训练目标的点数；减半可使优化阶段近似加速两倍，但会轻微增加BR。由于当前目标数量按第一次采样点数乘第二次采样率计算，降低第一次采样率也会同时减少优化点对 | 分开扫描两种采样率；若只研究树的稀疏程度，应调整第二次采样率以保持训练目标数不变 |
| 随机选定矩形<br>区域作为样本 | 待验证 | 空间训练中，区域代表性不足：LES velocity_x的2/4区域约改善`0.30%`，但1/8区域分别退化`27.88%/0.49%`；定额紧凑L∞区域的同PSNR BR也比随机散点高约`0.20%`。不过，类似的随机四维矩形区域采样曾在Ocean velocityZonal的31步联合时空训练中起效：4区域方案相对上一时间步同点预测改善`8.920%` BD-rate。该结果没有单独分离区域采样本身的贡献，而且旧实现还在`M=CCᵀ`空间平均区域矩阵，因此需要重新确认 | 保留实验开关；在相同训练点对预算下比较随机散点和矩形区域，并使用当前直接平均`C`的后处理方式复测空间与时空训练 |
| 训练时使用<br>4个近邻 | 已采纳，可以再验证 | 比较了3、4、5和10个训练近邻，4个近邻在LES和Tsunami的多个field上整体最稳定。3个近邻偶尔收益很大，但出现轻微退化的field更多；10个近邻的结果明显依赖field，BR变化范围达到`-13.704%～+10.460%`；5个近邻也没有表现出相对4个近邻的整体优势 | 默认使用4个训练近邻；保留其他近邻数的候选矩阵及筛选脚本。压缩预测仍默认使用10个参考点 |

### 2.3.2. 训练

汇总所有样本点对的数值差和原始坐标差后，程序训练一个公共的下三角矩阵`C`。当前默认使用平方loss。令第`i`个点对的数值差为`Δv_i=v_i1-v_i2`，原始坐标差为行向量`Δc_i=c_i1-c_i2`，则目标函数为：

```text
L(C)=Σ_i(|Δv_i|-d_i(C))²=Σ_i(|Δv_i|-||Δc_iC||₂)²
```

除了首轮以外，训练会从历史`C`开始。

训练目标和优化器相关方案如下：

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 使用<br>二次方loss | 已采纳 | Katrina相同随机种子的20轮对照中，二次方loss保存点的最佳同PSNR BR为`0.215432`，优于对数平方的`0.219716`和四次方的`0.236681` | 作为默认loss |
| 新矩阵吸收率<br>设为`0.5` | 已采纳 | 比较`0.3/0.5/0.7/1.0`后，`0.5`在第4轮得到最低BR；更大的吸收率会更早进入优化方向不明确的阶段，`1.0`明显不稳定 | 默认使用`0.5`；`0.7`只作为速度候选 |
| 外层训练<br>迭代4轮 | 已采纳 | 吸收率为`0.5`时，第4轮BR为`0.211122`，第5轮回升到`0.214531`；真实BR通常早于训练loss停止改善 | 默认迭代4轮；通用早停条件仍需更多数据验证 |
| 直接优化度量矩阵<br>`M=CCᵀ` | 放弃 | 当前直接优化`C`会自动得到半正定度量`CCᵀ`，列符号歧义也已通过统一正对角线解决。直接优化`M`时，平方loss仍包含`sqrt(ΔcMΔcᵀ)`，不会变成简单的线性最小二乘，反而还要额外施加半正定约束或进行投影 | 不实现 |
| Gauss–Newton<br>快速训练 | 需要更多样本 | Katrina约`1.2 s`，相对least-squares约有`0.17%` RD代价 | 在LES、Tsunami和新数据验证 |
| 直接近似预测残差码长的<br>训练目标 | 部分采纳 | Katrina只比二次方矩阵改善约`0.38%`；Ocean temperature、velocityZonal、zTop在time step 0/15/30相对原坐标分别改善`0.54%～1.29%/7.02%～8.67%/2.37%～4.52%` | Ocean这三个field采用码长loss矩阵；其他field继续按实际BR筛选 |
| 使用<br>四次方loss | 放弃 | Katrina相同随机种子的20轮对照中，四次方loss保存点的最佳同PSNR BR为`0.236681`，弱于二次方loss的`0.215432` | 保留可切换实现作为历史基线，不再作为主线 |
| 外层训练<br>迭代100轮 | 放弃 | 各种loss的训练值继续下降时，真实BR都可能已经开始退化；旧四次方loss第100轮矩阵也比当前配置差`12.182%` BD-rate | 不再用固定100轮或训练loss最低点选择矩阵 |
| 对数平方loss<br>作为主线 | 放弃 | 收敛慢且真实BR弱于二次方loss | 保留可切换实现，不再继续调参 |
| 二次差分<br>loss | 放弃 | 定义依赖近似共线三点，适用范围和预期收益都较弱 | 不继续实现 |

### 2.3.3. 后处理

一次训练结束后会得到原始下三角矩阵`C_new`，随后按以下顺序处理：

1. **检查优化结果。** 程序首先确认`C_new`的所有元素和loss残差均为有限值。出现NaN或Inf说明优化已经失效，继续建树会传播错误，因此立即终止。
2. **统一列符号。** 把`C_new`中负对角元素对应的整列取反。同一个度量可以由列符号不同的多个`C`表示，因为任意对角元素为`±1`的`D`都满足`(CD)(CD)ᵀ=CCᵀ`；统一为正对角线可以避免等价因子在后续平均时相互抵消，而且列取反本身不改变任何距离。
3. **分离矩阵形状和整体倍率。** 将统一符号后的矩阵写成`C_raw=sC`，其中`||C||_F²=dim`。倍率`s`表示数值差相对于坐标差的总体尺度，归一化矩阵`C`表示真正影响近邻顺序和IDW权重的度量形状；两者在后续步骤中分别缓冲。
4. **缓冲。** 归一化后的变换矩阵和历史上的归一化矩阵加权算数平均，默认新矩阵的权重是0.5，随后再次归一化。倍率`s`和历史倍率加权几何平均，默认新倍率的权重是0.9；实验表明这个倍率权重不会影响最终归一化矩阵、BR或训练时间。

矩阵后处理相关方案如下：

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 在度量矩阵`M=CCᵀ`上<br>混合新旧训练结果 | 放弃 | `M`混合表示平均平方距离，不符合当前“平均新旧变换后坐标”的缓冲定义。使用相同吸收率`0.5`和4轮时，Katrina全部162步同PSNR比较中，直接混合`C`为`BR=0.231460,PSNR=65.192228`，混合`M`为`BR=0.240020,PSNR=65.198746` | 保留`-matrix-mix-space metric`仅用于复现；把仍在平均区域`M`的联合时空区域训练改成统一符号后平均`C` |
| 修正<br>过小特征值 | 放弃 | 在Katrina上启用和关闭修正得到的四轮历史矩阵最大只相差`3.33e-16`，全部162步的BR和PSNR完全相同；修正还需要构造`M=CCᵀ`并进行特征值分解。关闭后只需统一下三角因子的列符号，四轮累计矩阵处理时间约由`1.75 ms`降至`0.23 ms` | 默认不修正；保留`-small-eigenvalue-correction on`仅用于复现 |
| 分离迭代矩阵形状<br>和整体倍率 | 已采纳，待验证 | 原方案每轮归一化`C`并丢弃整体倍率，新方案分别缓冲归一化矩阵`C`和倍率`s`。平方loss下，固定矩阵形状后的倍率只是一个容易求解的一维最小二乘问题，优化器能迅速重新找到合适倍率；近邻选择又只由归一化矩阵形状决定，因此两种方案在Katrina上的最终归一化矩阵最大只相差`4.44e-16`，内层评估次数同为`6/7/6/6`，训练耗时均约14秒，BR和PSNR也几乎不变。新方案不会无故丢弃倍率，算法定义更合理，因此暂时采纳 | 默认使用`separate`；在其他数据尺度和loss上继续验证，保留`discard`用于复现原方案 |
| 分离矩阵形状和倍率时<br>调整`s`的采纳率 | 不再调优 | 平方loss下，固定矩阵形状后只剩一个一维尺度最小二乘问题，优化器可以迅速重新得到合适倍率。采纳率`0.7/0.9/1.0`得到相同的归一化矩阵和`6/7/6/6`轮内层评估次数，对BR与时间均无可见影响 | 默认固定为`0.9`，不再把它作为性能参数扫描 |

### 2.3.4. 适用范围与矩阵形式

当前默认方案为一个field训练一个公共常矩阵。下面这些方案改变的是“何时使用adjust coord”“哪些数据共用矩阵”或“矩阵本身采用什么形式”，不属于采样、loss或后处理参数。

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 自动判断是否需要<br>adjust coord | 待实现 | 不同field的adjust收益差异明显 | 寻找只需少量样本的低成本判据 |
| 让adjust coord支持<br>多个用户定义点集 | 待实现 | C++已支持多个点集和`n-1`个标签通道；Python仍只训练`is_abnormal()`筛出的单个有效点集。constant点集的距离loss会错误地把矩阵推向零，标签通道与数值流也可能需要不同几何 | 读取同一JSON配置；为每个非constant数值点集和每个递进标签通道分别训练候选矩阵，再比较独立矩阵、只共享预测顺序以及完全共享矩阵的BR与预处理代价 |
| 让不同压缩partition使用<br>不同的常变换矩阵 | 待实现 | 当前每个field只使用一个公共线性矩阵 | 保证partition之间互不引用，再比较局部收益、边界损失与矩阵元数据成本 |
| 以坐标为自变量的<br>变换矩阵 | 暂缓 | 令下三角矩阵的每个非零元素都是坐标的一次函数，并以`cC(c)`变换每个点；三维时参数量由常矩阵的6个增加到24个 | 只有公共常矩阵明显不足时再继续 |
| 为每个Ocean field独立训练<br>早停空间矩阵 | 部分采纳 | 二次方loss初筛中temperature、velocityY、velocityZonal的BR分别改变`-0.90%/-3.38%/-5.57%`，其余6个field改变`+0.25%～+25.59%` | velocityY保留二次方矩阵；temperature和velocityZonal已由更好的码长loss矩阵替换 |
| 分区域训练矩阵后<br>仍进行全局预测 | 实验作废 | 曾把Ocean zTop划成2/4个区域并应用不同矩阵，随后合并坐标建立全局kd-tree；矩阵接缝会产生没有意义的跨区域距离 | 该实现不代表互不依赖的压缩partition，结果不能用于判断局部矩阵是否有效 |
| 预先把各坐标维度<br>scale到相同范围 | 放弃 | Katrina重新训练后相对原坐标adjust退化`22.848%` BD-rate | 不在adjust前人为等范围缩放 |

## 2.4. 生成DAG

预测前需要根据公共坐标生成一套所有time step都能复用的有向无环图（DAG）。整体分为四步：

1. 使用经过adjust coord变换的坐标生成kd-tree，为后续的目标搜索和参考点搜索提供空间索引。
2. 将全部点随机分配到若干预测轮。越早的轮越先解码，也更难获得丰富的参考点。
3. 在每一轮内排列预测目标，使解码次序同时兼顾可预测性和空间局部性。
4. 按最终次序逐点选择已经恢复的参考点，并添加“参考点→预测目标”的有向边，形成最终DAG。

默认流程是先随机分轮、再在每轮内排序；实验方案也可以先对全部点生成一个全局顺序，再按照各轮点数切分。因此，分轮和目标排序放在同一小节讨论。

### 2.4.1. 生成kd-tree

程序在变换后的坐标上建立一棵公共kd-tree。树同时用于搜索空间邻近的候选目标和最近参考点，但对外保存的目标与参考点编号仍是原始点编号。公共坐标不随time step变化，因此树的只读拓扑、包围盒和编号映射可以复用。

当前kd-tree的每个叶节点最多保存32个点，并按树内点编号连续存储坐标。需要不同有效点集合时，多棵逻辑树共享只读拓扑，只分别维护自己的有效点计数，从而避免复制整棵树。

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 优化kd-tree的<br>查询与内存布局 | 已采纳 | 使用容量受控的栈内小数组维护近邻、每个叶节点最多保存32个点、按树内编号连续存储坐标，并让多棵逻辑树共享只读拓扑。32点叶节点在Ocean和Katrina上是最稳定的共同选择；tree-order坐标使两者的总预处理均缩短约`4.1%`，同时不改变预测结构和BR | 作为当前默认实现 |
| 进一步优化<br>kd-tree查询 | 待实现 | 当前查询仍使用递归，并在完整树中区分有效目标；尚未测试显式栈和只包含当前目标点的紧凑树 | 只有kd-tree查询重新成为瓶颈时再实现并比较遍历时间 |

### 2.4.2. 分轮和预测目标排序

默认将全部点随机分为8轮，前两轮点数相同，此后每轮目标数近似翻倍。`-round-size-growth-ratio-experimental`可以修改增长比例，同时保证所有点恰好属于一轮。

分轮后，程序在每一轮内部使用反向D-only策略排列目标。反向剔除时，只根据坐标计算目标到当前可用参考集合的平均平方距离`D`，并在空间邻近的候选目标中选择；最后反转剔除顺序，使较难建立可靠局部参考的点较早恢复、较容易预测的点留到后面。该顺序只依赖坐标，可以离线生成并复用于所有time step。

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 仅根据平均平方距离`D`<br>生成公共预测顺序 | 已采纳 | 不需要参考数据，能跨field复用并由坐标重建，但压缩能力弱于参考数据辅助顺序 | 继续作为无参考数据或分布漂移时的默认方案 |
| 先生成全局顺序<br>再按照轮大小切分 | 效果优先候选 | 相对先随机分轮，LES三个field改善`12.22%～12.40%` BD-rate，Tsunami四个field改善`0.99%～6.42%`，Katrina退化`1.683%`；首次预处理慢约`23%～32%`，但核心压缩和解压通常略快 | 保留`-global-order-before-round-split-experimental`；不设为通用默认，等待离线选择方案 |
| 利用参考数据搜索<br>公共预测顺序 | 效果优先候选 | 三组参考/测试划分及全部held-out time step上，Katrina和Ocean temperature的总BD-rate分别稳定为`-3.57%～-4.02%`和`-27.45%～-27.71%`；3个参考文件效果最好，2个已获得大部分收益 | 测试更多field和分布漂移；预测结构作为field级离线结果保存 |
| 参考数据辅助排序中<br>复用未变化目标的预测误差 | 已采纳 | 某个目标的有效近邻列表没有变化时，直接复用它在多个参考文件上的动态预测误差。Katrina和Ocean temperature的近邻列表不变率为`70.27%/71.55%`，分数计算量减少`63.24%/64.40%`，首次预处理加速`1.54/1.49`倍；顺序和BR不变 | 保留统计输出，在更多数据上观察命中率 |
| 参考数据辅助排序中<br>跨文件复用几何权重 | 已采纳 | 多个参考文件使用公共坐标，因此同一目标和候选参考点之间的距离及未归一化IDW权重只计算一次。各文件仍独立筛选同点集参考点并重新归一化。在预测误差复用的基础上，Katrina和Ocean temperature继续加速`1.18/1.15`倍；相对最初实现累计加速`1.82/1.72`倍，预测结构SHA不变 | 作为参考数据辅助排序的默认实现 |
| 参考数据辅助排序中<br>缓存近邻编号和几何权重 | 已采纳 | 近邻缓存由10点减少到9或8点但尚不需要重新查询时，同时复用剩余近邻的编号和未归一化权重，无需重新计算距离的幂。在前两项优化基础上，Katrina和Ocean temperature相对最初实现累计加速`1.98/2.04`倍，预测结构SHA不变 | 仅在参考数据辅助排序中分配约`点数×11×8 B`的额外内存 |
| 参考数据辅助排序中<br>改用对数量化误差评分 | 放弃 | 相对平均绝对误差评分最多只改善Katrina `0.287%`、Ocean temperature `0.083%` | 活动C++路径已删除，只保留历史脚本和结果 |
| 参考数据辅助排序中<br>改用静态预测误差评分 | 放弃 | 首次预处理约快`2.18～2.27`倍，但相对动态分数在Katrina和Ocean temperature增加BD-rate `6.336%/42.165%` | 不以预测质量换取这部分速度 |
| 在先随机分轮和<br>先全局排序之间离线选择 | 待实现 | 先随机分轮能维持各轮空间覆盖；先全局排序再切分在LES和Tsunami较好、Katrina略差 | 用少量参考文件的预测误差或估计码长进行确定性选择 |
| 坐标或距离相同时<br>按点编号打破并列 | 待验证 | 主要路径已有点编号比较，但尚未完成跨机器复现检查 | 比较预测结构和压缩文件SHA以及BR、PSNR |
| 优先让相邻目标<br>复用相同参考点 | 放弃 | Ocean temperature的BR退化`14.57%` | 只保留历史实现 |
| 联合使用`D`、`S2`或<br>方向集中度排列目标 | 暂缓 | 两个数据上的完整或小规模实验均未超过只使用`D` | 等更多数据后再决定是否重测 |

### 2.4.3. 生成DAG

分轮和排序给出最终解码次序后，程序按这个次序逐点选择参考点。每个目标只能引用已经恢复的点；一旦当前目标的参考点确定，它就立即成为后续目标可用的参考点，因此同一轮内排在后面的目标也可以依赖本轮较早的目标。

若目标`i`使用参考点`j`，程序就添加一条`j→i`的边。每条边都从较早恢复的点指向较晚恢复的点，所以图中不可能出现环。最终参考点编号连续保存在`pred_ref_ids`中，目标`i`的区间为`pred_ref_ids[pred_ref_begin[i]:pred_ref_begin[i+1]]`；对应权重也使用连续数组保存，避免每个目标单独分配一个小`vector`。

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 每选择完整一轮后<br>再更新可用参考点 | 放弃 | 该方案禁止同轮内依赖，Katrina attr0前3步BR恶化约`20.3%`，速度没有明显改善 | 实现保存在`c_codes/trash` |
| 使用连续数组保存<br>参考边和权重 | 已采纳 | 减少小对象、内存分配和指针跳转，并保持预测结构和BR不变 | 作为当前默认存储方式 |
| 将公共预测DAG<br>缓存到本地文件 | 已采纳，可选 | 第一个REL EB生成结构，后续REL EB及解压直接加载，避免重复预处理；缓存较大，不适合作为必须传输的压缩元数据 | 每次数据集测试使用独立临时缓存，测试结束后删除 |

## 2.5. 数值预测、量化与无损编码

当前通用预测器是inverse-distance weighting：

```text
squared_distanceᵢ=||coord_target-coordᵢ||₂²
prediction=Σ(valueᵢ·squared_distanceᵢ^-1.8)/Σ(squared_distanceᵢ^-1.8)
```

C++的kd-tree查询返回并保存的是欧氏距离的平方，因此程序实际对`squared_distance`使用`-1.8`次方，即`-inverse_distance_power/2`。这在数学上等价于对欧氏距离使用`-3.6`次方。

预测只使用已经重建且属于当前点集的参考点。压缩端立即用量化重建值替换原值，解压端因而可以严格复现后续预测。当前默认共有8个预测轮。若轮编号为`r=0,...,7`，第`r`轮开头使用的error bound是用户指定值的`0.8^(7-r)`倍：第一轮从约`0.21`倍开始，后续轮逐渐放宽。在同一轮内，error bound也按照等比数列平滑增加到下一轮的起始值；最后一轮使用完整的用户error bound。公比`0.8`仍是经验参数，需要和预测轮数一起验证。

量化索引使用int32_t，随后采用Huffman编码和Zstd level 3。串行无partition路径使用全局Huffman树和Zstd frame；启用独立partition时，每个partition对各点集分别建立Huffman树和Zstd frame，以支持并行压缩、解压和随机定位。

<table width="100%">
  <colgroup>
    <col width="1%">
    <col width="29%">
    <col width="42%">
    <col width="28%">
  </colgroup>
  <thead>
    <tr>
      <th width="1%">优化方案</th>
      <th width="29%">状态</th>
      <th width="42%">已有结论</th>
      <th width="28%">下一步</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>距离权重为<code>d^-3.6</code>、最多使用10个参考点、有效缓存少于8点时重新查询</td>
      <td>已采纳</td>
      <td>跨Katrina、Ocean、LES和Tsunami比更小参考集合稳定</td>
      <td>作为当前空间默认IDW预测配置</td>
    </tr>
    <tr>
      <td>Huffman+Zstd level 3</td>
      <td>已采纳</td>
      <td>当前仍是速度和BR较均衡的无损组合</td>
      <td>作为当前默认无损路径</td>
    </tr>
    <tr>
      <td>联合调整轮数、轮大小和逐轮EB</td>
      <td>已采纳</td>
      <td><code>8轮+EB公比0.8</code>相对原<code>10轮+0.8</code>在4个代表field均改善；轮大小增长比例的field最优值为Katrina/Ocean temperature/Ocean velocityZonal/Tsunami <code>2/4/3.5/1.75</code>，不存在共同最优值</td>
      <td>默认保持8轮、轮大小增长比例2和EB公比0.8；更完整的联合调参留到更多数据后</td>
    </tr>
    <tr>
      <td>根据一个重构点会影响多少后续预测来分配EB</td>
      <td>实验性保留</td>
      <td>8轮五点曲线中，按一次权重影响力直接映射EB的4-field平均BD-rate为<code>-0.214%</code>，按二次权重传播方差影响力为<code>+0.024%</code>；核心压缩约慢<code>1.6～2.1</code>倍</td>
      <td>一次公式优于二次公式但收益很小；若继续，先把单点集影响力缓存为公共预处理</td>
    </tr>
    <tr>
      <td>使用局部仿射拟合代替IDW</td>
      <td>研究候选</td>
      <td>Ocean temperature的BD-rate改善<code>35.202%</code>，但Ocean layerThickness和Katrina分别退化<code>16.175%/17.753%</code>；单文件核心压缩约比IDW慢<code>2.7～6.8</code>倍</td>
      <td>不作为通用默认；保留给适合局部线性模型的平滑field</td>
    </tr>
    <tr>
      <td>根据参考点的留一误差选择IDW或局部拟合</td>
      <td>暂缓</td>
      <td>依次把每个参考点当作未知值，并用其余参考点比较两种预测器的累计绝对误差；解压端可复现选择而无须保存selector。早期三数据中均不差于纯IDW，但核心压缩约慢<code>15～49</code>倍</td>
      <td>只在找到更廉价的留一误差近似后继续</td>
    </tr>
    <tr>
      <td>根据留一误差连续混合IDW与距离加权拟合</td>
      <td>研究候选</td>
      <td>由两种预测器的留一误差确定连续权重，不保存selector；Ocean 9-field的1/4-partition平均BR分别改善<code>6.507%/8.351%</code>，但收益集中在少数平滑标量field，代表组合的核心压缩和解压约慢<code>3～4</code>倍</td>
      <td>研究置信回退和中间计算复用，再决定是否纳入效果优先配置</td>
    </tr>
    <tr>
      <td>按空间partition采样并选择预测器</td>
      <td>实验性保留</td>
      <td>每个partition抽样估计候选预测器的码长并保存所选配置；相对固定配置的平均收益可达<code>3.523%～7.627%</code>，但中位收益很小且核心压缩约慢<code>4.2～5.3</code>倍、解压约慢<code>6.6～7.0</code>倍</td>
      <td>先改善采样决策；固定门槛回退在Ocean 9-field平均/中位仅改善<code>0.050%/0.029%</code>，不足以设为默认</td>
    </tr>
    <tr>
      <td>先用IDW预测再拟合IDW残差</td>
      <td>放弃</td>
      <td>在1/4-partition共18个代表组合中没有一次胜出</td>
      <td>停止完整RD测试</td>
    </tr>
    <tr>
      <td>为局部拟合加入ridge正则项</td>
      <td>放弃</td>
      <td>与IDW残差拟合相同，在18个代表组合中没有一次胜出</td>
      <td>停止完整RD测试</td>
    </tr>
    <tr>
      <td>根据训练数据减少相邻量化bin切换</td>
      <td>暂缓</td>
      <td>空间邻近顺序已间接考虑连续性；激进参考复用在Ocean退化<code>14.57%</code>，而数值接近本身不代表Huffman+Zstd码流接近</td>
      <td>只有预测质量优化基本结束后，才考虑直接减少相同bin之间的切换或估计条件码长</td>
    </tr>
    <tr>
      <td>按预测轮或空间分块分别建立多棵Huffman树</td>
      <td>放弃</td>
      <td>正收益field较少且收益不稳定</td>
      <td>当前不再继续投入</td>
    </tr>
    <tr>
      <td>rANS等其他无损编码</td>
      <td>暂缓</td>
      <td>Huffman+Zstd level 3仍是速度和BR的合理折中</td>
      <td>先明确论文更重视速度还是BR</td>
    </tr>
    <tr>
      <td>推导Huffman输出缓冲区上界并检查特殊输入</td>
      <td>待实现</td>
      <td>当前实际数据运行正常，但缺少严格上界证明</td>
      <td>检查空量化索引和极端符号范围</td>
    </tr>
    <tr>
      <td>将<code>d^-3.5</code>、8个参考点、cache-6设为通用默认</td>
      <td>放弃</td>
      <td>在LES三个速度field增加BR <code>2.34%～3.02%</code></td>
      <td>只保留为数据相关或速度优先候选</td>
    </tr>
    <tr>
      <td>将4或5个参考点设为通用默认</td>
      <td>放弃</td>
      <td>Ocean temperature分别退化约<code>18.08%/10.63%</code></td>
      <td>不再作为通用配置测试</td>
    </tr>
  </tbody>
</table>

## 2.6. 点集与标签

不同类型的数据点不一定应该互相预测。当前Katrina实现将`value<-1000`的fill value与正常值划分为两个点集：

- 正常点只使用正常参考点预测。
- fill value点只使用fill value参考点预测；当前全部为`-99999`时自动使用常量编码，没有数值预测误差。
- 额外预测每个点属于哪个点集，只保存标签预测错误位置，并使用Zstd压缩。

每个time step独立生成标签，因此异常点位置可以随时间变化。单点集数据不生成标签流。若总共有`n`个点集，只需使用`n-1`个递进二分类标签通道；已经归入较早点集的点不再产生后续通道的修正bit，但仍作为“不属于当前类型”的参考点参与标签预测。实际只有一或两个点集时继续生成原v7码流，三个及以上点集使用v8多通道容器。

REL error bound根据各点集自身的最大值和最小值转换为ABS error bound，避免fill value扩大正常数据的值域。使用JSON配置时，每个非常量label必须独立指定method以及REL或ABS error bound，不存在沿用命令行全局EB的模式；不使用JSON的单label数据才通过命令行`-predictor`和`-M`配置。容器记录实际method和各点集值域，但用户配置的EB仍需在解压时保持一致。

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 硬0/1预测错误bit、原始编号差分和Zstd | 已采纳 | 是当前Katrina标签BR与速度最好的已测组合 | 用作每个条件二分类标签通道 |
| 用户定义有序多点集及其完整压缩方式 | 已采纳 | 已实现运行期JSON有序条件、外部逐点label文件、`n-1`个递进二分类标签通道及逐label method/REL/ABS EB；一或两个实际label使用v7，三个及以上使用v8 | 在新的多异常值数据上统计各标签通道开销 |
| 标签预测与编码 | 需要更多样本 | 当前各条件二分类通道分别保存预测错误bitstream并使用Zstd level 3 | 在具有三种以上真实类型的新数据上比较标签BR与预测参考数 |
| 连续空间标签传播 | 放弃 | 将预测值作为连续标签参与后续预测没有降低最终标签开销 | 继续使用每点硬0/1修正 |
| 按位打包、逐字节、游程、按轮或按预测顺序编码标签 | 放弃 | 均未超过当前原始编号错误位置差分方案；预测顺序标签还同时降低速度 | 不再重复这些二分类实验 |

## 2.7. 跨time step预测

当前实现了三类跨时间方案：

1. `previous`：从第二个time step开始，优先使用上一重建time step中同一空间位置的值。
2. `interpolation`：先空间压缩首尾time step，再按层次顺序恢复区间中点，使用两端重建值进行时间线性插值；若点集类型不一致则回退空间预测。
3. 联合四维：把全部time step展平成四维点集，通过四维adjust coord自动平衡空间距离和时间距离，再执行统一空间近邻预测。

层次时间插值和联合四维方案目前均为实验路径。联合四维数据的准备、矩阵训练和对比由`t5_cross_time.py`完成。

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 非因果层次时间插值 | 实验性保留 | Ocean velocityZonal上比上一time step预测改善`29.065%` BD-rate | 在更多多time step数据上验证 |
| 纯空间、纯时间与联合四维比较 | 需要更多样本 | Ocean上层次时间插值当前最好 | 获得更多数据后判断联合方案是否有不可替代收益 |
| 分段层次插值 | 待实现 | 当前整段首尾作为端点 | 测试固定关键帧间隔，权衡随机访问、误差传播和BR |
| 因果线性外推 | 待实现 | 尚无实验结果 | 比较`2*v[t-1]-v[t-2]`与非因果插值 |
| 根据解压端可复现的条件在空间预测和时间预测之间切换 | 待实现 | 点集不一致时已有回退，普通预测失败尚无无需额外标记的判断方法 | 设计压缩端和解压端均可复现的判断 |
| 将adjust后的四维时空点集切成互不引用的独立分区 | 需要更多样本 | velocityZonal上64/128分区同时改善速度且未损失BR | 在更多field和EB上复查边界效应 |
| 让多个time step共用Huffman频率统计或Zstd字典 | 暂缓 | 尚不清楚树头或跨时间重复模式占多少BR | 只有占用可观时才测试 |
| 为Ocean field寻找共享adjust矩阵分组 | 放弃 | temperature专属矩阵有收益，但共享规则无法跨数据集复用 | 每个field独立训练矩阵 |
| 跨field数值预测 | 暂缓 | 既有研究与当前判断均表明field间相关性通常弱于field内相关性 | 除非新数据表现出很强field相关性，否则不修改格式 |

## 2.8. 并行设计

当前OpenMP原型已经支持adjust后独立partition：每个partition只引用本partition内的数据，因而具有独立预测DAG。kd-tree拓扑、包围盒、tree-order坐标和ID映射只保存一份；线程共享这些只读结构，并分别修改互不重叠的partition状态。预测结构生成、标签预测、数值预测、量化、Huffman、Zstd和重建均按partition并行。v9容器使用`partition_payload_begin[P+1]`定位各独立payload，最终只按partition编号串行拼接字节。

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| OpenMP独立partition | 已实现原型 | 预测结构、标签、数值、Huffman、Zstd和重建均已并行；Katrina的1/4线程v9码流与重构SHA一致 | 在大型数据上测试扩展性并减少小partition调度开销 |
| 每partition独立无损码流 | 已实现原型 | Katrina第0步4分区BR为`0.200938`；1/4线程核心压缩为`0.0356/0.0287 s`、解压为`0.0304/0.0252 s` | 在更多field和更多partition上统计树头、frame开销与BR变化 |
| 强扩展和弱扩展实验 | 暂缓 | 依赖并行版本完成 | 在独占节点测试增加线程或同步增加数据规模时的加速效率 |
| 端到端性能 | 暂缓 | 已有单核核心吞吐率，UMC解压仍更快 | 并行版本完成后统一统计压缩、传输和解压 |

# 3. 当前结果

以下BR改善均在RD曲线的共同PSNR范围内插值得到，不是比较相同REL error bound。结果仍属于论文准备阶段，正式投稿前会在冻结版本上统一重测。

## 3.1. 空间预测的rate-distortion

UNSZ中的不少方案形成明确的压缩效果—速度权衡，强行冻结一个折中默认值会掩盖算法边界。论文实验将分别冻结一套“压缩效果优先”和一套“速度优先”的可复现配置，同时给出两者之间的Pareto关系及关键消融；不会针对测试field事后挑选最优配置。先生成全局顺序再分轮、基于训练数据排列公共顺序和复杂预测器属于效果优先候选，先随机分轮、只使用坐标的D-only顺序、纯IDW和快速adjust训练属于速度优先候选。最终吞吐率会分开报告一次性公共预处理、adjust训练、逐文件核心压缩和解压，避免把离线成本与逐time step成本混在一起。

| 数据 | 方法与比较 | 同PSNR BR变化 |
|---|---|---:|
| Ocean，9个field×31步 | 分成4个空间块并在每块选择预测器的UNSZ相对UMC | 平均`-15.843%`，中位`-8.105%` |
| LES velocity_x/y/z | 先随机分轮的UNSZ相对UMC | `-26.07%～-28.20%` |
| LES velocity_x/y/z | 先生成全局顺序再分轮的UNSZ相对UMC | `-36.53%～-38.66%` |
| Tsunami四个field | 先生成全局顺序再分轮的UNSZ相对UMC | `-15.019%～+2.808%` |

Ocean收益主要集中在salinity、temperature和zTop，速度field通常约改善`5%～10%`。按4个空间块分别选择预测器的方案RD较好，但核心压缩约慢`4.2～5.3`倍、解压约慢`6.6～7.0`倍，因此尚未替换纯IDW正式配置。

Adjust coord在坐标距离明显失真的数据上尤其重要。Katrina当前二次方loss矩阵相对旧四次方100轮矩阵降低`12.182%` BD-rate；Ocean temperature的新二次方早停矩阵相对不变换降低`2.722%`，说明不同field的收益差异很大。

Katrina前5个time step上的等坐标范围消融进一步比较了三种策略。各维范围以x轴范围为目标，scale系数为`[1,1.40891484,0.00466570]`：

| 坐标策略 | REL=0.01 BR | PSNR | 相对原坐标adjust的BD-rate |
|---|---:|---:|---:|
| 原坐标直接adjust | `0.204324` | `59.4311` | 基准 |
| 仅等范围scale | `0.227640` | `60.3536` | `+2.350%` |
| 等范围scale后直接使用原adjust矩阵 | `0.262361` | `60.4821` | `+12.876%` |
| 等范围scale后重新训练adjust矩阵 | `0.282496` | `60.2860` | `+22.848%` |

因此，显示时让各坐标轴具有相近跨度不能替代针对可预测性学习的adjust coord；在Katrina上预先等范围scale还会干扰后续训练。该小实验的脚本和五点RD结果分别为`py_codes/trash/t64_benchmark_katrina_coord_scale.py`和`py_codes/trash/results/katrina_coord_scale_rd.csv`。

## 3.2. 先生成全局顺序再分轮

| 数据 | 相对先随机分轮的BD-rate |
|---|---:|
| LES velocity_x | `-12.218%` |
| LES velocity_y | `-12.292%` |
| LES velocity_z | `-12.395%` |
| Tsunami momentum_x | `-5.585%` |
| Tsunami momentum_y | `-6.422%` |
| Tsunami velocity_x | `-0.987%` |
| Tsunami velocity_y | `-5.696%` |
| Katrina attr0，time step 0 | `+1.683%` |

先生成全局顺序再分轮的首次预处理比先随机分轮慢约`23%～32%`，但LES核心压缩快约`4.7%～5.9%`、解压快约`5.9%～7.8%`。它不是通用赢家，下一步需要根据少量训练数据在两种分轮方法之间离线选择。

## 3.3. 利用参考数据搜索公共预测顺序

参考数据辅助顺序保留随机分轮和反向“好点后置”框架，但用多个参考time step上动态IDW预测的平均归一化绝对误差代替纯几何D。用于搜索顺序和用于评估的time step严格分离，生成的预测结构随后供所有测试文件复用。

| field | 参考time step | 测试time step | 相对仅使用`D`排序的BD-rate | 仅使用`D`的首次预处理 | 参考数据辅助搜索的首次预处理 |
|---|---|---|---:|---:|---:|
| Katrina attr0 | 三组，每组3步 | 每组其余159步 | `-3.571%～-4.024%` | `4.65 s` | `9.50～9.68 s` |
| Ocean temperature | 三组，每组3步 | 每组其余28步 | `-27.451%～-27.713%` | `2.19 s` | `4.94～5.08 s` |

Katrina三组参考数据顺序分别改善154/159、159/159和155/159个held-out time step，最差单步退化`0.880%`；Ocean三组均改善28/28步，最差单步仍改善`25.917%`。相同实验中的“先全局排序再分轮”在Katrina总BD-rate退化`0.964%～1.041%`，在Ocean改善`12.236%～12.272%`，说明参考数据辅助顺序并非仅仅复现全局几何排序。仅使用平均平方距离`D`的优势仍是无需读取参考数据、可以跨field共用、没有参考/测试划分和分布漂移风险、可由坐标重新生成，以及首次预处理更快。参考数据辅助顺序一旦保存，逐文件核心压缩没有明显时间代价；但它通常是field相关的，而且解压端必须获得保存好的公共预测结构。

缓存动态预测误差、让多个参考文件共用一次距离权重计算、再把原始IDW权重与缓存近邻编号一起保存后，Katrina/Ocean temperature单次首次预处理已从`9.532/4.879 s`降至`4.803/2.387 s`，累计加速`1.98/2.04`倍。相同版本的D-only为`4.316/2.015 s`，参考数据辅助搜索目前只慢约`11.3%/18.5%`；这些数字来自共享登录节点单次运行，主要用于判断优化方向。新旧预测结构SHA完全一致。

固定相同held-out集合后，Katrina使用1/2/3个参考文件的总BD-rate依次为`-2.546%/-3.354%/-4.024%`，首次预处理为`6.51/7.94/9.54 s`；Ocean temperature依次为`-26.246%/-27.267%/-27.613%`，首次预处理为`3.21/4.20/5.02 s`。三个文件仍得到最低BR，适合作为效果优先配置；两个文件已分别获得三文件约83.3%和98.7%的收益，可作为更快的离线搜索候选。

## 3.4. 跨time step结果

Ocean velocityZonal全部31步的层次时间插值，相对其他方案的BD-rate为：

| 基线 | BR变化 |
|---|---:|
| 逐time step空间预测 | `-61.170%` |
| 上一time step同位置预测 | `-29.065%` |
| 4区域联合四维预测 | `-22.455%` |

层次时间插值是当前最强的跨时间基线。是否仍存在联合四维方案不可替代的数据，需要在更多具有不同时间相关性的field和数据集上验证。

联合四维预测可以在adjust后的坐标上独立分区。velocityZonal、31步、REL`1e-3`下，64分区相对不分区使BR降低`0.498%`、预处理加快`17.14%`、压缩加快`14.75%`；128分区使BR降低`0.373%`、预处理加快`19.27%`、压缩加快`15.85%`。这为后续一分区一任务的并行实现提供了基础。

## 3.5. 串行吞吐率

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

- [ ] Katrina和Ocean temperature的三组参考/测试划分、全部held-out time step及1/2/3个参考文件已经验证；继续测试更多field和分布漂移。
- [ ] 使用少量参考文件的预测误差或估计码长，在“先随机分轮”和“先生成全局顺序再分轮”之间离线选择；不能只根据坐标覆盖率判断。
- [ ] 寻找更多具有不同网格结构、坐标尺度、点集分布和时间相关性的非结构数据，验证UNSZ、adjust coord和跨时间结论的普适性。
- [x] 用户定义的有序多点集、`n-1`个条件标签通道及逐点集codec/error bound已经形成压缩—解压闭环。
- [ ] 冻结串行参数、bitstream和实验脚本，在统一版本上重新生成五点RD、吞吐率以及各阶段流长度。
- [x] 完成adjust后独立partition的OpenMP原型：预测结构、标签、数值、Huffman、Zstd和重建均已按partition并行，v9格式保存独立payload及偏移表。
- [ ] 完成single-core throughput、strong/weak scaling和端到端压缩—传输—解压实验，再决定DCC或IPDPS投稿路线。

详细参数、完整消融结果和已经否定的方案见`TODO_DETAIL.md`；面向日常决策的精简状态见`TODO.md`。
