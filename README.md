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

## 1.4. 自定义有序点集

点集分类和逐点集配置集中在`c_codes/unsz_core/user_point_set.hpp`。用户按优先级编写`classify0`到`classify<n-2>`，并将函数依次放进`get_user_point_set_classifier_vec()`；第一个返回`true`的函数决定点集，全部返回`false`的数据属于最后一个点集。`get_user_point_set_config_vec()`为全部`n`个点集分别指定codec和EB：

```cpp
template<typename T_data>
bool classify0(T_data value){return value>=0;}

template<typename T_data>
bool classify1(T_data value){return value>=-1000;}

template<typename T_data>
vector<point_set_classifier_t<T_data>> get_user_point_set_classifier_vec(){return {classify0<T_data>,classify1<T_data>};}

inline vector<point_set_config_c> get_user_point_set_config_vec()
{
    return {
        {point_set_codec_e::inverse_proportion,point_set_error_mode_e::relative,0.01},
        {point_set_codec_e::fitting,point_set_error_mode_e::absolute,0.1},
        {point_set_codec_e::constant,point_set_error_mode_e::use_global,0}
    };
}
```

`use_default`表示沿用命令行`-predictor`，`use_global`表示沿用命令行`-M`。逐点集REL EB根据该点集自己的值域计算；如果实际出现的所有非常量点集都有独立EB，命令行可以省略`-M`。修改后需要重新构建压缩器和解压器；因为EB不写入文件，两端必须使用同一份配置。默认文件保持Katrina的两点集规则：`value>=-1000`属于正常值点集，其余数据属于fill-value点集。

## 1.5. 当前空间预测实验配置

裸命令行使用随机分轮和默认IDW预测。当前正式空间实验还显式启用反向平均平方距离排序，也就是只用目标点到可用参考点的平均平方距离`D`评价预测难度，后文简称D-only：

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
| `-d-only-cache-refill-threshold-experimental` | 8 | 排序所用的有效参考点缓存少于8个时重新查询kd-tree |

可以通过`-adjust-coord-matrix`直接提供下三角变换矩阵。三维矩阵按行输入6个非零元素：

```bash
install/bin/s1_compress -data-dir /path/to/data -variable field -time-step '[0,9]' -dim 3 -M REL 1e-3 -adjust-coord-matrix c00 c10 c11 c20 c21 c22 -reverse-mean-distance-only-order-experimental
```

解压时必须提供相同矩阵。仅使用`-adjust-coord`会启用源码中当前硬编码的三维实验矩阵，因此对新数据推荐显式使用`-adjust-coord-matrix`或`-adjust-coord-matrix-file`。

也可以使用独立矩阵文件，避免在源码或命令行中复制矩阵元素：

```bash
install/bin/s1_compress -data-dir /path/to/Katrina -variable attr0 -time-step '[0,9]' -dim 3 -M REL 1e-3 -adjust-coord-matrix-file adjust_coord_matrices/Katrina_attr0_spatial.txt -reverse-mean-distance-only-order-experimental
install/bin/s1_decompress -data-dir /path/to/Katrina -variable attr0 -time-step '[0,9]' -dim 3 -M REL 1e-3 -adjust-coord-matrix-file adjust_coord_matrices/Katrina_attr0_spatial.txt
```

矩阵文件记录数据集、field、维数、下三角元素和训练类型。`training_type=spatial`表示只使用空间坐标训练；`training_type=spatiotemporal`表示把time step作为额外坐标维度进行联合时空训练。当前选定矩阵集中保存在`adjust_coord_matrices/`，格式和文件索引见其中的`README.md`。

## 1.6. 批量实验与脚本

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

UNSZ将公共网格预处理与逐field文件压缩分开：

1. 读取坐标，必要时将坐标和距离计算从float升级为double。
2. 可选地使用离线学习的线性矩阵变换坐标，使变换后距离更接近数据值的可预测性。
3. 建立公共kd-tree，将全部点划分为多轮，确定轮内预测顺序，并为每个目标寻找已经重建的空间参考点。
4. 对每个time step独立读取数据、划分点集、计算各点集值域和error bound。
5. 根据已重建的同点集参考点执行预测，立即量化预测残差并用重建值覆盖目标，保证压缩端与解压端具有相同的误差传播路径。
6. 将点集标签和各点集量化索引分别无损压缩，最后组装成一个`.unsz`容器。

坐标、预测轮和候选参考关系可以被同一网格上的所有field和time step复用。默认不会把公共预测结构写入每个压缩文件；解压端根据坐标重新生成。`-prediction-structure <path>`可以把它保存为本地缓存，但当前缓存文件较大，不适合作为必须传输的压缩元数据。

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

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 二次方loss、吸收率`0.5`、第4轮矩阵 | 已采纳 | Katrina全部162步相对旧四次方100轮矩阵改善`12.182%` BD-rate | 作为当前效果优先训练配置 |
| Gauss–Newton快速训练 | 需要更多样本 | Katrina约`1.2 s`，相对least-squares约有`0.17%` RD代价 | 在LES、Tsunami和新数据验证 |
| 自动判断是否需要adjust | 待实现 | 不同field的adjust收益差异明显 | 寻找只需少量样本的低成本判据 |
| 让不同空间分块使用不同的常变换矩阵 | 待实现 | 当前每个field只使用一个公共线性矩阵 | 比较局部收益与矩阵元数据成本 |
| 以坐标为自变量的变换矩阵 | 暂缓 | 令下三角矩阵的每个非零元素都是坐标的一次函数，并以`cC(c)`变换每个点；三维时参数量由常矩阵的6个增加到24个 | 只有公共常矩阵明显不足时再继续 |
| 直接近似预测残差码长的训练目标 | 暂缓 | 二次方loss是当前主线，已有预测码长近似目标只有小幅收益 | 在更多数据上复查，并计入训练成本 |
| 减少训练样本和轮数 | 实验性保留 | 第二次采样率减半可近似加速两倍，但会轻微增加BR | 在目标矩阵和BR稳定范围内确定速度优先配置 |
| 四次方loss训练100轮 | 放弃 | 训练loss继续下降时真实BR已经退化，Katrina比当前矩阵差`12.182%` BD-rate | 只保留为历史基线 |
| 对数平方loss作为主线 | 放弃 | 收敛慢且真实BR弱于二次方loss | 保留可切换实现，不再继续调参 |
| 预先把各坐标维度scale到相同范围 | 放弃 | Katrina重新训练后相对原坐标adjust退化`22.848%` BD-rate | 不在adjust前人为等范围缩放 |
| 二次差分loss | 放弃 | 定义依赖近似共线三点，适用范围和预期收益都较弱 | 不继续实现 |

## 2.3. kd-tree、预测轮与参考点

默认将全部点分为10轮，前期轮点数较少，后期轮点数近似翻倍。一个点只能引用解压顺序中已经恢复的点，因此整个预测关系构成有向无环图。

当前正式实验使用反向D-only顺序：在反向剔除过程中，只根据坐标计算目标到可用参考集合的平均平方距离`D`，并在空间邻近的候选目标之间选择。最终反转剔除顺序，使较难建立可靠局部参考的点更早恢复、较容易预测的点留到后面。该顺序只依赖坐标，可以离线生成并复用于所有time step。

kd-tree目前采用32点叶节点、按树内点编号连续存储坐标、共享树拓扑以及连续参考边数组。目标`i`的参考点位于`pred_ref_ids[pred_ref_begin[i]:pred_ref_begin[i+1]]`，对应权重保存在连续权重数组中，避免大量小`vector`和指针跳转。

实验性的“先生成全局顺序再分轮”方案不先随机分配每轮目标，而是对全部点生成一条D-only顺序，再按照原有各轮点数切分。它在LES和Tsunami上明显改善BR，但在Katrina上略有退化，因此尚未成为默认方案。

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 基于训练数据的公共预测顺序 | 需要更多样本 | 在未参与训练的Katrina/Ocean temperature测试集上，BD-rate为`-5.435%/-25.542%` | 测多组划分、训练文件数、全部time step和更多field |
| 在“先随机分轮”和“先生成全局顺序再分轮”之间离线选择 | 待实现 | 先随机分轮能维持空间覆盖；先全局排序再按每轮点数切分在LES/Tsunami较好、Katrina较差 | 用少量训练文件的预测误差或估计码长做确定性选择 |
| 仅根据平均平方距离`D`生成公共预测顺序 | 已采纳 | 不需要训练数据，能跨field复用并由坐标重建，但当前压缩能力弱于基于训练数据的顺序 | 继续作为无训练数据或分布漂移时的默认方案 |
| 坐标或距离相同时按点编号固定顺序 | 待验证 | 主要路径已有点编号比较，但尚未做跨机器复现检查 | 验证预测结构、压缩文件SHA、BR和PSNR完全一致 |
| 继续优化kd-tree的查询与内存布局 | 暂缓 | 32点叶、按树内编号连续存储坐标和共享只读树结构已经实施 | 只有预处理重新成为瓶颈时再尝试显式栈和只含目标点的紧凑树 |
| 选完整轮后再更新可用的参考点 | 放弃 | Katrina attr0前3步BR恶化约`20.3%`，速度没有明显改善 | 实现保存在`c_codes/trash` |
| 优先让相邻目标复用相同参考点 | 放弃 | Ocean temperature退化`14.57%` | 只保留历史实现 |
| 同时使用平均参考距离`D`、参考点分布`S2`或方向集中度排列目标 | 暂缓 | 两个数据上的完整或小规模实验均未超过只使用`D` | 保留开关，等更多数据再决定是否重测 |

## 2.4. 点集与标签

不同类型的数据点不一定应该互相预测。当前Katrina实现将`value<-1000`的fill value与正常值划分为两个点集：

- 正常点只使用正常参考点预测。
- fill value点只使用fill value参考点预测；当前全部为`-99999`时自动使用常量编码，没有数值预测误差。
- 额外预测每个点属于哪个点集，只保存标签预测错误位置，并使用Zstd压缩。

每个time step独立生成标签，因此异常点位置可以随时间变化。单点集数据不生成标签流。任意数量的有序点集使用`n-1`个递进二分类标签通道；已经归入较早点集的点不再产生后续通道的修正bit，但仍作为“不属于当前类型”的参考点参与标签预测。实际只有一或两个点集时继续生成原v7码流，三个及以上点集使用v8多通道容器。

REL error bound根据各点集自身的最大值和最小值转换为ABS error bound，避免fill value扩大正常数据的值域。每个点集可以独立选择codec以及全局、REL或ABS error bound；容器记录实际codec和各点集值域，但用户配置的EB仍需在解压时保持一致。

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 硬0/1预测错误bit、原始编号差分和Zstd | 已采纳 | 是当前Katrina标签BR与速度最好的已测组合 | 用作每个条件二分类标签通道 |
| 用户定义有序多点集及其完整压缩方式 | 已采纳 | 已实现分类函数、`n-1`个递进二分类标签通道及逐点集codec/REL/ABS EB；默认二点集码流逐字节不变，三点集压缩—解压已闭环 | 在新的多异常值数据上统计各标签通道开销 |
| 标签预测与编码 | 需要更多样本 | 当前各条件二分类通道分别保存预测错误bitstream并使用Zstd level 3 | 在具有三种以上真实类型的新数据上比较标签BR与预测参考数 |
| 连续空间标签传播 | 放弃 | 将预测值作为连续标签参与后续预测没有降低最终标签开销 | 继续使用每点硬0/1修正 |
| 按位打包、逐字节、游程、按轮或按预测顺序编码标签 | 放弃 | 均未超过当前原始编号错误位置差分方案；预测顺序标签还同时降低速度 | 不再重复这些二分类实验 |

## 2.5. 数值预测、量化与无损编码

当前通用预测器是inverse-distance weighting：

```text
prediction=Σ(valueᵢ·distanceᵢ^-3.6)/Σ(distanceᵢ^-3.6)
```

预测只使用已经重建且属于当前点集的参考点。压缩端立即用量化重建值替换原值，解压端因而可以严格复现后续预测。当前默认按照轮间公比0.8逐轮、逐点放宽error bound，并在最后达到用户指定上限；这一经验参数仍需与预测轮数共同消融。

量化索引使用int32_t，随后采用Huffman编码和Zstd level 3。按空间partition或预测轮拆分多棵Huffman tree的收益较小且不稳定，因此当前仍使用单树方案。

项目还实现了局部仿射拟合、根据留一预测误差在IDW与拟合预测之间选择、连续混合、先用IDW预测再拟合其残差，以及按空间分块采样并选择预测器。它们可以改善部分平滑标量field，但目前明显慢于纯IDW，仍属于研究候选。

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| 距离权重为`d^-3.6`、最多使用10个参考点、有效缓存少于8点时重新查询 | 已采纳 | 跨Katrina、Ocean、LES和Tsunami比更小参考集合稳定 | 作为当前空间默认IDW预测配置 |
| Huffman+Zstd level 3 | 已采纳 | 当前仍是速度和BR较均衡的无损组合 | 作为当前默认无损路径 |
| 联合调整轮数、轮大小和逐轮EB | 需要更多样本 | 当前10轮、经验性轮大小和公比`0.8`的EB | 在更多数据上寻找固定通用配置 |
| 根据一个重构点会影响多少后续预测来分配EB | 待实现 | 被大量后续点引用的参考点，其误差可能传播到更多目标 | 让影响更多后续预测的参考点使用更小EB |
| 按空间分块选择预测器，并在选择不可靠时回退 | 实验性保留 | 标量field可能获益，部分速度field会被错误选择拖累 | 加入回退到父分块或整个field配置的置信门槛 |
| 在反距离加权预测和局部拟合预测之间选择或混合 | 暂缓 | 部分平滑field改善BR，但比纯IDW慢`3～4`倍 | 只在找到廉价、可由解压端复现的选择规则后继续 |
| 排列压缩顺序，使相邻量化索引更接近 | 待实现 | 当前顺序已经兼顾空间邻近，但尚未直接优化估计码长 | 比较BR收益是否足以覆盖排序成本 |
| 按预测轮或空间分块分别建立多棵Huffman树 | 放弃 | 正收益field较少且收益不稳定 | 当前不再继续投入 |
| rANS等其他无损编码 | 暂缓 | Huffman+Zstd level 3仍是速度和BR的合理折中 | 先明确论文更重视速度还是BR |
| 推导Huffman输出缓冲区上界并检查特殊输入 | 待实现 | 当前实际数据运行正常，但缺少严格上界证明 | 检查空量化索引和极端符号范围 |
| 距离权重为`d^-3.5`、8个参考点、有效缓存少于6点时重查，作为通用默认 | 放弃 | 在LES三个速度field增加BR `2.34%～3.02%` | 只保留为数据相关或速度优先候选 |
| 4或5个参考点作为通用默认 | 放弃 | Ocean temperature分别退化约`18.08%/10.63%` | 不再作为通用配置测试 |
| 先用IDW预测再拟合其残差，以及带ridge正则项的局部拟合 | 放弃 | 18个代表组合中没有一次胜出 | 停止完整RD测试 |

## 2.6. 跨time step预测

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

## 2.7. 并行设计

当前OpenMP原型已经支持adjust后独立partition：每个partition只引用本partition内的数据，因而具有独立预测DAG。kd-tree拓扑、包围盒、tree-order坐标和ID映射只保存一份；线程共享这些只读结构，并分别修改互不重叠的partition状态。预测结构生成、标签预测、数值预测、量化和重建已经按partition并行，当前Huffman、Zstd和容器组装仍使用原来的全局串行码流。

| 优化方案 | 状态 | 已有结论 | 下一步 |
|---|---|---|---|
| OpenMP独立partition | 部分实现 | 已并行预测结构生成、标签预测、数值预测、量化和重建；Katrina的1/4线程预测结构、码流和重构SHA一致 | 在大型数据上测试扩展性并减少小partition调度开销 |
| 并行码流布局 | 待实现 | 当前Huffman、Zstd和容器组装仍为全局串行路径 | 为每个partition保存独立片段和`partition_payload_begin`偏移表 |
| 强扩展和弱扩展实验 | 暂缓 | 依赖并行版本完成 | 在独占节点测试增加线程或同步增加数据规模时的加速效率 |
| 端到端性能 | 暂缓 | 已有单核核心吞吐率，UMC解压仍更快 | 并行版本完成后统一统计压缩、传输和解压 |

# 3. 当前结果

以下BR改善均在RD曲线的共同PSNR范围内插值得到，不是比较相同REL error bound。结果仍属于论文准备阶段，正式投稿前会在冻结版本上统一重测。

## 3.1. 空间预测的rate-distortion

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

## 3.3. 基于训练数据的公共预测顺序

基于训练数据的顺序保留随机分轮和反向“好点后置”框架，但用多个训练time step上动态IDW预测的平均归一化绝对误差代替纯几何D。训练和测试time step严格分离，生成的预测结构随后供所有测试文件复用。

| field | 训练time step | 测试time step | 相对仅使用`D`排序的BD-rate | 仅使用`D`的首次预处理 | 基于训练数据的首次预处理 |
|---|---|---|---:|---:|---:|
| Katrina attr0 | `40/80/120` | `0～4` | `-5.435%` | `4.63 s` | `8.42 s` |
| Ocean temperature | `8/16/24` | `0～4` | `-25.542%` | `2.82 s` | `4.67 s` |

在当前两个未参与训练的测试集上，仅使用平均平方距离`D`排序尚未表现出压缩能力优势；其优势是无需训练数据、可以跨field共用、没有训练划分和分布漂移风险、可由坐标重新生成，以及首次预处理更快。基于训练数据的顺序一旦保存，逐文件核心压缩没有明显时间代价；但它通常是field相关的，而且解压端不能只根据坐标重建，必须提供保存好的公共预测结构。当前结果仍需通过多组训练/测试划分和更多数据验证。

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

- [ ] 用多组互不重叠的训练/测试time step和更多field验证基于训练数据的公共预测顺序，并比较训练文件数量和分布漂移。
- [ ] 使用少量训练文件的预测误差或估计码长，在“先随机分轮”和“先生成全局顺序再分轮”之间离线选择；不能只根据坐标覆盖率判断。
- [ ] 寻找更多具有不同网格结构、坐标尺度、点集分布和时间相关性的非结构数据，验证UNSZ、adjust coord和跨时间结论的普适性。
- [x] 用户定义的有序多点集、`n-1`个条件标签通道及逐点集codec/error bound已经形成压缩—解压闭环。
- [ ] 冻结串行参数、bitstream和实验脚本，在统一版本上重新生成五点RD、吞吐率以及各阶段流长度。
- [ ] 完成adjust后独立partition的OpenMP版本：预测结构和数值阶段已经并行，下一步加入partition独立Huffman/Zstd片段和偏移表，并在大型数据上测试扩展性。
- [ ] 完成single-core throughput、strong/weak scaling和端到端压缩—传输—解压实验，再决定DCC或IPDPS投稿路线。

详细参数、完整消融结果和已经否定的方案见`TODO_DETAIL.md`；面向日常决策的精简状态见`TODO.md`。
