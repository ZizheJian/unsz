# UNSZ

UNSZ是一个主要面向多time step点云数据的误差有界有损压缩研究原型，不同time step应共享同一套公共点坐标。只有一个time step的数据同样可以压缩。UNSZ不依赖mesh connectivity，而是根据公共点坐标建立kd-tree、生成可由压缩端和解压端共同复现的预测DAG，再对预测残差进行量化、Huffman编码和Zstd压缩。

# 1. 使用方法

## 1.1. 输入格式

当前命令行程序使用float32坐标和float32数据。一个数据目录至少包含：

| 文件 | 内容 |
|---|---|
| `coords.dat` | 按点连续保存的坐标，每个点包含`dim`个float32数值 |
| `<field>.dat.<time_step>` | 对应time step的float32 field，每个点一个数值 |

例如三维、31个time step的temperature数据可以组织为：

```text
Ocean/
├── coords.dat
├── temperature.dat.0
├── temperature.dat.1
├── ...
└── temperature.dat.30
```

所有field文件的点数必须与`coords.dat`一致。二维数据通过`-coordinate-dim 2`指定。程序使用float32坐标和距离计算，不会改变磁盘输入格式。

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
install/bin/s1_compress -data-dir /path/to/Ocean -field temperature -time-step '[0,30]' -coordinate-dim 3 -M REL 1e-3
```

对应的解压命令为：

```bash
install/bin/s1_decompress -data-dir /path/to/Ocean -field temperature -time-step '[0,30]' -coordinate-dim 3 -M REL 1e-3
```

ABS error bound使用：

```bash
install/bin/s1_compress -data-dir /path/to/data -field field -time-step '[0,9]' -coordinate-dim 3 -M ABS 1e-4
```

`-time-step`支持闭区间`[a,b]`、右开区间`[a,b)`和显式列表`{a1,a2,...}`。例如`[0,31)`和`{0,3,7}`分别处理0至30以及0、3、7号time step。包含括号或花括号的表达式建议使用引号，避免被shell解释。

压缩文件写到原数据旁边，名称为`<field>.dat.<time_step>.unsz`；解压结果为`<field>.dat.<time_step>.unsz.bin`。程序打印总BR、PSNR、最大误差比例、压缩端与解压端各自的预处理时间，以及核心压缩和核心解压时间。

当前bitstream不保存用户error bound或坐标变换结果。解压时必须提供与压缩时相同的error bound或点集配置；空间DAG由压缩端和解压端分别现场生成，不提供DAG缓存命令行选项。

空间预测固定使用10点Exponential RBF、global反向MSD顺序和统一的EB schedule。时间flow与坐标变换由压缩器自动选择，并把最终flow写入bitstream。`-value-flow`、`-label-flow`和`-coordinate-transform`只保留为诊断自动决策的实验覆盖；正式配置不传这些选项。`-partition-num`或`-partition-size`可以和`-thread-num`一起显式设置。

## 1.4. 自定义有序点集

不传`-point-set-config`时，程序把全部数据作为一个点集，并用`-M`指定误差上限。`-M`和`-point-set-config`必须且只能选择一个。使用`-point-set-config`时，JSON定义一个或两个点集的分类条件、编码方法和误差上限；所有`method="predict"`的点集统一使用内置RBF。压缩和解压必须传入相同的点集配置：

UNSZ目前只支持一个或两个点集。压缩入口依次执行`main`→`unsz::parse_args_and_select_types`→`unsz::parse_args`→初始化并选择`T_coord/T_data`→`unsz::compress`，再按配置中的点集数量进入`unsz::one_point_set::compress`或`unsz::two_point_sets::compress`。空间预测器固定为RBF，各入口只按点集数量和最终flow分派。

```bash
install/bin/s1_compress -data-dir /path/to/Katrina -field attr0 -time-step '[0,161]' -coordinate-dim 3 -point-set-config point_set_configs/Katrina_attr0_rbf_rel_1e-2.json
install/bin/s1_decompress -data-dir /path/to/Katrina -field attr0 -time-step '[0,161]' -coordinate-dim 3 -point-set-config point_set_configs/Katrina_attr0_rbf_rel_1e-2.json
```

`labels`数组下标就是点集编号。程序按顺序测试除最后一个label外的`condition`，第一条成立的条件决定label；全部不成立时属于最后一个label。比较符直接使用`<`、`<=`、`>`、`>=`、`==`或`!=`。下面是Katrina的两点集配置，其中点集1使用自身值域上的REL EB，点集2由固定fill value组成并使用常量编码：

```json
{
  "version":3,
  "labels":[
    {"method":"predict","eb_mode":"REL","eb_value":0.01,"condition":{"operator":">=","value":-1000}},
    {"method":"constant"}
  ]
}
```

最后一个label是默认兜底类型，因此省略`condition`（同时省略eb信息是因为`method="constant"`，不需要eb）；配置文件只有一个label时同样不写`condition`。每个`method="predict"`的label必须显式配置`eb_mode`和`eb_value`，模式为`REL`或`ABS`；`method="constant"`省略这两个字段。点集JSON中的`method`只能是`predict`或`constant`，不配置预测器参数。

点集分类只读取当前field的数值并按上述条件执行，不接受额外标签文件。由于EB不写入bitstream，解压时仍必须使用与压缩时相同的配置。

## 1.5. 批量实验与脚本

| 脚本 | 功能 |
|---|---|
| `t1_start_c_code.py` | 正式实验通用入口，只接受一个JSON位置参数：`configs/`中的JSON名称或任意JSON路径，例如`./unsz_env/bin/python t1_start_c_code.py katrina`或`./unsz_env/bin/python t1_start_c_code.py /path/to/test.json`。包含UNSZ任务时，入口默认先执行CMake配置、增量构建和安装；临时复用现有安装时可添加`--skip-build`。随后直接构造并执行UNSZ或项目内`third_party/umc`命令，不再调用第二层Python实验入口，结果直接写入CSV。UMC固定为官方commit，`third_party/build_umc.sh`联网核对后安装`third_party/umc_replacements`中的计时文件并构建。 |
| `py_codes/parse_unsz_output.py`、`py_codes/parse_umc_output.py` | 只解析并核对压缩器输出，不负责调度或启动实验。 |
| `t3_visualize.py` | 读取C++压缩器生成的`*.unsz.coord_transform.txt`，生成Katrina attr0坐标变换前后的field图像，以及Tsunami最佳与最差field的原网格图像 |
| `py_codes/prepare_datasets/prepare_<dataset>.py` | 分别把Tpost20、E3SM-h0、E3SM-h4、ADCIRC Haiyan、Tsunami、TAPS和Ocean-large发布数据转换为UNSZ/UMC共用的平坦二进制基准输入；共享工具位于同目录的`prepare_dataset_common.py` |
| `py_codes/run_spatial_field_benchmark.slurm` | 在Slurm节点上运行环境变量`CONFIG`明确指定的完整task JSON，不在运行时拆分或改写JSON |
| `py_codes/run_configurations.slurm` | 以Slurm array依次把当前9个正式数据集JSON交给`t1_start_c_code.py`。 |
| `py_codes/plot_dataset_rd.py` | 为一个数据集的各field分别绘制UNSZ/UMC空间RD与核心吞吐率，并计算同PSNR下的平均BR变化 |

JSON最外层是task list；同一个JSON可以包含不同数据目录的任务。每个task都必须自行记录描述实验用途的唯一`name`、绝对`data_dir`和CSV文件路径`csv_path`，即使相邻task使用同一数据目录也不从外层继承。相对`csv_path`以项目根目录为基准；一个task包含多个field时，路径使用`{field}`作为field名称占位符。同一数据、同一field需要测试不同固定参数时，应写成名称不同的多个task。

`field_list`和`compressor_list`是所有task共有的穷举维度。单点集task使用`rel_eb_list`穷举误差上限；多点集task不再重复提供`rel_eb_list`，而是在`unsz.point_set_config_path_list`中依次列出待测的完整点集JSON，相对路径以项目根目录为基准。每份点集JSON是一次完整的UNSZ误差配置，不同点集可以使用不同EB。若同一task还测试UMC，t1收集全部点集JSON中出现的REL EB、去重后分别运行UMC；UMC不模拟一个文件内不同点集使用不同EB，而是生成用于同PSNR曲线比较的普通REL EB基线。UNSZ结果使用点集JSON路径标识，UMC和单点集UNSZ结果使用REL EB标识。正式运行的`unsz`对象通常为空；多点集任务只写`point_set_config_path_list`，启用partition时可再写`partition_num`或`partition_size`以及`thread_num`。预测器、DAG、EB schedule、flow与坐标变换参数由程序固定或自动选择，不出现在正式JSON中。诊断自动决策时仍可在临时配置中用`flow`或`coordinate_transform`强制覆盖。

```json
[
  {
    "name":"les_velocity",
    "data_dir":"/path/to/LES",
    "csv_path":"results/current/fields/les_{field}.csv",
    "field_list":["velocity_x","velocity_y"],
    "rel_eb_list":[0.01,0.001],
    "compressor_list":["unsz","umc"],
    "time_step_num":1,
    "coordinate_dim":3,
    "unsz":{},
    "umc":{
      "from_edge":0,
      "reorder":1,
      "method":"NBP-IDW",
      "neighbor_num":4
    }
  }
]
```

RBF权重在DAG生成后统一以批量LDLT和Schur补求解。固定参数为10个参考点、Exponential核、shape 0.01、nugget 1e-6、global反向MSD顺序、16个排序候选，以及base 0.8、log2 shift 0.5、最大指数7的inverse-rank EB schedule。两个点集的参考点不全同类时退化为最近的同类参考点。task设置`"cleanup_after_each_test":true`时，每组结果成功写入CSV后删除UNSZ生成的`.unsz`和`.unsz.bin`，以及UMC生成的`.umc`和`.umc.out`。

UMC任务在独立解压完成后始终逐点读取原始field和磁盘上的`.umc.out`。检查使用NumPy分块计算每个time step的数据范围、最大误差和总MSE，验证逐点REL误差界，并核对磁盘PSNR与UMC压缩端和解压端报告一致；该检查在UMC程序计时之外执行，不计入核心压缩或解压时间。

统一可视化程序默认生成Katrina attr0与Ocean temperature的变换前后图像，以及Tsunami velocity_x和height的原网格图像，也可以只选择一个数据集：

```bash
./unsz_env/bin/python t3_visualize.py
./unsz_env/bin/python t3_visualize.py -dataset katrina
./unsz_env/bin/python t3_visualize.py -dataset tsunami-velocity-x
./unsz_env/bin/python t3_visualize.py -dataset tsunami-height
```

# 2. 方案描述

## 2.1. 整体流程

UNSZ目前实现了以下压缩流程：

<table class="fixed-layout-table" width="100%">
  <colgroup>
    <col width="25%">
    <col width="10%">
    <col width="65%">
  </colgroup>
  <thead>
    <tr>
      <th width="25%">流程</th>
      <th width="10%">当前状态</th>
      <th width="65%">含义</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>纯空间</td>
      <td>可选</td>
      <td>读取公共坐标，可选地训练并应用空间变换矩阵。然后在变换后的坐标上建立kd-tree并生成预测DAG，最后在每个文件上完成数值预测、量化和无损压缩。</td>
    </tr>
    <tr>
      <td>时间零阶外插预测</td>
      <td>可选</td>
      <td>先用纯空间方案压缩第一个文件，再不断用上一个文件的重建值预测下一个文件。</td>
    </tr>
    <tr>
      <td>时间一阶内插预测</td>
      <td>可选</td>
      <td>先用纯空间方案压缩第一个和最后一个文件，再按照类似SZ3的层次顺序处理其余文件。若提供time step坐标文件，层次中点按实际坐标选择，插值权重也按实际坐标计算；否则把time step编号当作等间距坐标。</td>
    </tr>
    <tr>
      <td>时间零阶外插预测＋空间补偿</td>
      <td>可选</td>
      <td>先用上一文件的重建值预测当前值，再使用当前文件中已经重建的空间邻点预测时间残差，最后量化剩余残差。</td>
    </tr>
    <tr>
      <td>时间一阶内插预测＋空间补偿</td>
      <td>可选</td>
      <td>先用左右两个已重建time step线性预测当前值，再使用当前文件中已经重建的空间邻点预测时间残差，最后量化剩余残差。</td>
    </tr>
    <tr>
      <td>固定步长锚点内插</td>
      <td>可选</td>
      <td>以固定stride依次处理锚点，再在每个锚点区间内按一阶内插顺序递归处理其余time step。标签侧的锚点使用零阶外插、区间内部使用端点一致性并在不一致时进行空间补充；数值侧的锚点使用零阶外插、区间内部使用一阶内插，并可选择是否继续使用空间预测压缩时间残差。</td>
    </tr>
    <tr>
      <td>时间一阶内插预测中，最后一个文件使用来自第一个文件零阶外插预测</td>
      <td>尚未测试</td>
      <td></td>
    </tr>
    <tr>
      <td>空间预测＋时间补偿</td>
      <td>尚未测试</td>
      <td>先使用空间预测来得到预测值，然后使用某种时间预测来压缩预测误差。</td>
    </tr>
  </tbody>
</table>

`configs/`中的task JSON不是由C++直接读取的。对于单点集field，`unsz.flow.value`指定该field采用纯空间、时间零阶外插、时间一阶内插或带空间补偿的时间预测；对于双点集field，`unsz.flow.label`和`unsz.flow.value`分别指定标签与数值流程。`t1_start_c_code.py`展开流程组合，把它们转换为命令行参数，并用同一组流程参数分别启动压缩器和解压器。C++再根据实际time step数量进入相应的专用函数：只有1步时固定使用纯空间；有2步时，一阶内插退化为零阶外插，“一阶内插＋空间补偿”相应退化为“零阶外插＋空间补偿”；至少有3步时才会进入一阶内插函数。普通流程使用编译期固定的预测模式；锚点流程只在每个文件开始前选择该文件对应的固定模式，逐点循环中同样没有流程判断。

固定步长锚点内插直接作为两个stride固定为8的flow记录。标签侧写成`anchor_interpolation_stride_8`；数值侧写成`anchor_interpolation_stride_8`或`anchor_interpolation_stride_8_spatial_residual`。例如标签、数值都使用空间补充或修正时，分别写成`"label":["anchor_interpolation_stride_8"]`和`"value":["anchor_interpolation_stride_8_spatial_residual"]`。处理顺序固定为：先处理`0`；再以8为步长处理`8,16,24,...`，建立anchor；随后依次处理偏移为4、2和1的三层，即`4,12,20,...`、`2,6,10,...`和`1,3,5,...`。首个time step使用纯空间，anchor使用零阶外插；其余位置在左右端点都已经重构时使用端点一致性标签预测或一阶数值内插。末尾位置缺少右端点时先使用左侧零阶外插，随后仍可作为更细一层内插的端点；末尾time step不提前作为anchor。只启用其中一侧的锚点flow时，另一侧可以选择普通flow。程序不从flow名称解析任意stride；以后若要尝试其他stride，应把它作为新的完整flow实现和测试。

时间遍历顺序与标签、数值预测公式分开处理。程序统一提供顺序遍历、递归内插和anchor8三种time step顺序；双点集的标签flow和数值flow必须使用同一种顺序。每次遍历开始前会生成time step依赖DAG，直接记录各目标的标签和值参考节点以及各参考节点的剩余出度；目标处理完成后删除已消费的边，某类历史的剩余出度归零时立即释放对应的完整数组，不再扫描或永久保留全部既有time step。

多time step正式压缩自动选择label和value flow。程序在预处理计时开始前读取全部time step，准确计算各点集的ABS EB和点数；该输入统计时间不计入预处理。空间DAG和权重生成后，程序均匀选择最多32个time step，在每个独立partition的DAG上确定4段公共连续采样窗口，并仅用这些time step估算候选码流后按总time step数缩放。程序先合并这些评估步骤及全部候选的直接时间参考，只从该并集中稀疏读取样本值；候选估算复用正式空间预测、时间依赖、量化、Huffman和Zstd函数，覆盖纯空间、零阶外插及anchor8流程。每个partition固定采样512个第一点集目标。候选所需的数据读取不计入预处理时间，分类、候选预测和编码计入。最终选中的label/value flow写入压缩文件；解压端从文件头恢复它们。单time step只有纯空间流程，不执行自动选择。

时间一阶内插可通过`-time-step-coordinate-file <path>`读取一维float64原始数组，第`i`个数是time step `i`的实际坐标。压缩和解压必须读取同一文件；不传该参数时保持原有的等间距假设。对应task使用`time_step_coordinate_file`记录文件名并同时传给两端，普通等时间间隔数据不需要记录这一项。

## 2.2. 生成坐标变换矩阵

坐标变换已并入C++预处理，不再使用独立的Python训练程序或预先生成的下三角矩阵。task中的`coordinate_transform`支持：

- `"off"`：不训练也不应用坐标变换。
- `"on"`：训练并始终应用坐标变换。
- `"auto"`：根据坐标flatness自动决定是否训练和应用坐标变换。

训练固定使用最多3个均匀分布的time step。每个训练文件先以固定随机种子保留30%的点建立kd-tree，再在这些点中总共选取最多3000个目标；每个目标使用10个近邻形成训练点对。当前只学习各坐标轴的对角缩放，不混合坐标轴，也不执行第二轮重新采样。

对每个训练点对，令数值差为`Δv`、原始坐标差为`Δc`，训练对角向量`s`使以下损失最小：

$$
L_{\mathrm{diag}}=\min_{s,a}\sum_i\left(|\Delta v_i|-a\lVert\Delta c_i\odot s\rVert_2\right)^2.
$$

训练使用Gauss–Newton法；求解后把`s`归一化，使其平方和等于坐标维数。`auto`模式首先计算原始坐标协方差矩阵的flatness：

$$
f=\sqrt{\frac{\lambda_{\min}}{\lambda_{\max}}}.
$$

当原始flatness不大于`0.05`时确定启用并继续训练；当它不小于`0.75`时直接拒绝，不读取训练数据也不训练。其余情况完成训练，并用`DCD`直接计算变换后协方差，其中`C`是原始协方差，`D`是训练出的对角矩阵。仅当

$$
\frac{f_{\mathrm{transformed}}}{f_{\mathrm{original}}}\ge 1.075
$$

时才应用变换，否则关闭。flatness计算、训练和坐标变换都计入预处理时间；坐标和训练数据的文件读取不计时。变换后的坐标用于建树、生成DAG和计算预测权重；数据值本身不变。

压缩器把本次训练结果写到数据目录中的`<field>.unsz.coord_transform.txt`。文件以明文记录维数、是否启用和每个坐标轴的缩放值；解压器读取同一文件，不重新训练。例：

```text
UNSZ_COORDINATE_TRANSFORM 3
dimension 3
enabled 1
diagonal 1.31 0.74 0.08
```

`t3_visualize.py`也直接读取该文件，因此可视化和实际压缩使用完全相同的C++训练结果。

## 2.3. 生成DAG

预测前需要根据公共坐标生成一套所有time step都能复用的有向无环图（DAG）。整体分为三步：

1. 使用经过坐标变换的坐标生成kd-tree。
2. 对所有点生成一个预测目标的顺序。
3. 根据这个顺序，对每个预测目标寻找合适的参考点。

### 2.3.1. 生成kd-tree

程序在变换后的坐标上建立一棵公共kd-tree。这个树包含所有点，无论它的标签是什么。

当前kd-tree的每个叶节点可以保存至多32个点，并按树内点编号连续存储坐标。树还维护一个有效点计数，可以用来表达各种含义，比如“在指定block、指定标签的点中已经被压缩的点的数量”，有效点计数含义不同的树可以共用树的结构，只要维护自己的有效点计数数组就行了。

<table class="fixed-layout-table" width="100%">
  <colgroup>
    <col width="20%">
    <col width="10%">
    <col width="42%">
    <col width="28%">
  </colgroup>
  <thead>
    <tr>
      <th width="20%">优化方案</th>
      <th width="10%">状态</th>
      <th width="42%">已有结论</th>
      <th width="28%">下一步</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>树上每个叶节点可以保存多个点</td>
      <td>已采纳</td>
      <td>不改变DAG和压缩效果，但是可以加速</td>
      <td></td>
    </tr>
    <tr>
      <td>进一步优化kd-tree查询</td>
      <td>待实现</td>
      <td>当前查询仍使用递归，并在完整树中区分有效目标；尚未测试显式栈和只包含当前目标点的紧凑树</td>
      <td>只有kd-tree查询重新成为瓶颈时再实现并比较遍历时间</td>
    </tr>
  </tbody>
</table>

### 2.3.2. 预测目标排序

当前预测目标排序固定为反向MSD，不再把排序方向或评分公式作为可调配置。先假定所有点均可作为参考点，每次只在上一个被剔除点附近的固定数量候选点中，选择MSD最小的点剔除；完成后反转剔除序列，得到正向预测顺序。MSD定义为：

$$
\operatorname{MSD}(c)=\frac{1}{k}\sum_{i=1}^{k}\lVert c-c_i\rVert_2^2,
$$

其中$c$是候选目标坐标，$c_i$是当前仍可用的评分参考点坐标。该分数只依赖坐标和已确定的剔除状态，因此压缩端和解压端可以独立生成相同DAG，无需保存预测顺序。

当前仍显式配置与反向MSD本身相关的参数：`ordering_scope`决定全局排序或各轮内排序，`ordering_candidate_num`决定每次检查的附近候选数，`ordering_score_max_ref_num`和`ordering_score_min_ref_num`决定MSD评分使用的参考点数量。Tsunami和Katrina当前均使用1点MSD和16个排序候选点。

生成顺序时会缓存每个候选的MSD参考点，只在缓存中的可用点数低于下限时重新查询kd-tree。全局反向1点MSD还会自动复用排序候选查询的几何信息：只有能严格证明候选集内某点是全局最近可用参考点时才省略额外查询，因此不改变DAG。

正向排序、最近参考距离、`S2`、方向集中度、联合评分和参考数据辅助排序等实现及CLI均已删除。这些方案的历史结果仅保留在`TODO_DETAIL.md`中；如果以后需要消融实验，再从Git历史恢复或重写。

### 2.3.3. 生成DAG

得到预测目标的顺序后，程序按这个次序逐点选择参考点。每个目标只能引用已经恢复的点。

全局反向剔除且排序候选数不少于预测参考点数时，不必为最终参考点再做一轮kd-tree查询：剔除当前点之后，下一次以它为中心查询的附近未剔除点，正好就是正向预测时它能引用的点。程序从该查询结果保存最近的参考点；距离并列时，取16个候选再截取前10个可能与单独查询10点的参考编号略有不同，当前不为此重新查询。local先分轮后排序，仍按正向轮次重新查询最终参考点。独立partition不再维护另一套算法：每个partition把自己的全局点编号和包围盒交给同一个global或local生成函数。

<table class="fixed-layout-table" width="100%">
  <colgroup>
    <col width="20%">
    <col width="10%">
    <col width="42%">
    <col width="28%">
  </colgroup>
  <thead>
    <tr>
      <th width="20%">优化方案</th>
      <th width="10%">状态</th>
      <th width="42%">已有结论</th>
      <th width="28%">下一步</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>在先分轮后排序的方案中，每选择完整一轮后再更新可用参考点</td>
      <td>放弃</td>
      <td>该方案禁止同轮内依赖，Katrina attr0前3步BR恶化约<code>20.3%</code>，速度没有明显改善</td>
      <td></td>
    </tr>
    <tr>
      <td>使用连续数组保存参考边和权重</td>
      <td>已采纳</td>
      <td>减少小对象、内存分配和指针跳转，并保持预测DAG和BR不变</td>
      <td></td>
    </tr>
    <tr>
      <td>把公共预测DAG压缩为适合传输的元数据</td>
      <td>待实现</td>
      <td>正式程序不保存DAG；压缩和解压分别根据坐标重新生成。这里尚未实现适合随数据传输的紧凑DAG格式。</td>
      <td></td>
    </tr>
  </tbody>
</table>

## 2.4. 数值预测、量化与无损编码

当前数值预测固定使用RBF。以下统一把预测目标的坐标和值记为`c`和`v`，把第`i`个参考点的坐标和值记为`cᵢ`和`vᵢ`。RBF先根据坐标计算目标与各参考点、参考点彼此之间的相似度，再求出一组与这些相似度匹配的权重，最后对参考值加权求和。首先用目标到参考点的均方距离统一缩放当前局部邻域：

$$
s^2=\frac{1}{k}\sum_i\lVert c-c_i\rVert_2^2.
$$

当前采用Exponential核。用`κ(c₁,c₂)`表示坐标`c₁`和`c₂`之间的相似度：

$$
z(c_1,c_2)=\frac{\varepsilon\lVert c_1-c_2\rVert_2}{s},\qquad
\kappa(c_1,c_2)=\exp\left(-z(c_1,c_2)\right).
$$

其中固定使用`ε=0.01`。

随后求解权重`w₁,...,wₖ`和一个辅助量`μ`，使每个参考点`cᵢ`都满足：

$$
\begin{aligned}
\sum_jw_j\kappa(c_i,c_j)+\lambda w_i+\mu&=\kappa(c_i,c),&&i=1,\ldots,k,\\
\sum_jw_j&=1.
\end{aligned}
$$

固定使用`λ=10^{-6}`；最后一个约束保证常量field仍能被精确预测。得到权重后：

$$
\hat v_{\mathrm{RBF}}=\sum_iw_iv_i.
$$

这就是RBF的完整计算。UNSZ固定使用0阶常数趋势：辅助量`μ`和约束`Σwᵢ=1`保证常量field被精确预测，不实现或配置更高阶趋势。

<table class="fixed-layout-table" width="100%">
  <colgroup>
    <col width="20%">
    <col width="10%">
    <col width="42%">
    <col width="28%">
  </colgroup>
  <thead>
    <tr>
      <th width="20%">优化方案</th>
      <th width="10%">状态</th>
      <th width="42%">已有结论</th>
      <th width="28%">下一步</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>在IDW预测器中，权重反比于<code>d^2</code>、最多使用4个参考点、最少使用3个</td>
      <td>可选</td>
      <td>IDW预测器允许使用非常少量的参考点，几乎不影响BR，但是大幅加速。</td>
      <td></td>
    </tr>
    <tr>
      <td>局部线性拟合预测器</td>
      <td>待验证</td>
      <td>压缩比IDW慢很多，但是在部分field上相对于IDW明显提升压缩比。但当时还没有引入时间预测和RBF。另外，虽然4个参考点足以计算出参数，但并不稳定，需要更多参考点。</td>
      <td>需要和各种新的预测方法比较。</td>
    </tr>
    <tr>
      <td>局部二次拟合预测器</td>
      <td>待验证</td>
      <td>明显比局部线性拟合更好，但压缩更慢。</td>
      <td>需要和各种新的预测方法比较。</td>
    </tr>
    <tr>
      <td>RBF预测器</td>
      <td>已采纳</td>
      <td>RBF通常需要比IDW更多的参考点。Tsunami momentum_x当前选择原始坐标上的10点Matérn-3/2 RBF；旧的固定adjust矩阵对比已经归档，不再用于说明当前配置。</td>
      <td></td>
    </tr>
    <tr>
      <td>批量向量化RBF权重生成</td>
      <td>已采纳</td>
      <td>RBF现在只使用一种正式预处理流程：生成DAG时连续保存参考点坐标和目标距离，DAG完成后按32个目标一批，以SoA布局计算核矩阵、LDLT分解和Schur补。参考点数未达到固定批宽但仍满足RBF下限的少量边界目标使用标量LDLT；不再保留逐目标即时求解流程或general/ldlt配置开关。同节点交替测试中，Tsunami和Katrina的参考生成阶段相对旧逐目标流程分别缩短33.4%和30.0%，完整首次预处理分别缩短8.2%和6.4%。</td>
      <td>批宽16/32/64的同节点中位权重时间分别为Tsunami 0.090815/0.079755/0.081280秒、Katrina 0.351170/0.316769/0.320254秒，因此固定为32。紧凑下三角矩阵使权重阶段反而慢约4%～6%，已回退。批量浮点运算会造成很小的码流变化，但独立解压误差界检查通过。</td>
    </tr>
    <tr>
      <td>提前计算归一化权重</td>
      <td>已采纳</td>
      <td>对于加权平均预测器（包括IDW和RBF），在预处理阶段直接计算出归一化后的权重。压缩时直接加权和，可以省掉一次除法。如果是有标签的数据，通常来说参考点都是同类型的，此时依旧可以加权和。但如果有不同类型的参考点，就得加权平均，也不会更慢。</td>
      <td></td>
    </tr>
    <tr>
      <td>调整局部排序的轮数、相邻轮的数据量比例和逐轮EB</td>
      <td>已完成</td>
      <td>局部排序仍需要先分轮再分别排序；两种范围均使用按预测位置连续变化的EB倍率</td>
      <td>局部排序保留8轮和相邻轮的数据量比例2；局部与全局统一使用<code>inverse_rank_log2</code></td>
    </tr>
    <tr>
      <td>根据一个重构点会影响多少后续预测来分配EB</td>
      <td>放弃</td>
      <td>几乎无收益。理论上应该不会让压缩变慢，但是会让预处理变慢。</td>
      <td></td>
    </tr>
    <tr>
      <td>多点集数据中，RBF参考点不足时复制最近的同类点</td>
      <td>待验证</td>
      <td>空间DAG中的参考点可能属于不同点集，而预测数值时只能使用与目标相同点集的参考点。如果凑不齐RBF所需数量，就不再用少量参考点做不稳定的加权平均，而是直接复制最近的同类参考点；如果一个同类参考点也没有，则预测为0。Katrina使用这种处理后效果更好</td>
      <td>单点集数据不会触发该规则，需要在更多多点集数据上测试</td>
    </tr>
    <tr>
      <td>为每个目标在IDW和局部拟合之间自动选择</td>
      <td>放弃</td>
      <td>在每次预测中，程序依次把每个参考点当作测试点，用其余参考点分别检查IDW和局部拟合，再选择测试误差较小的方法预测真正的目标。这样不需要保存选择结果，但每预测一个点都要重复计算很多次，速度太慢</td>
      <td></td>
    </tr>
    <tr>
      <td>使用IDW控制局部拟合中的拟合误差权重</td>
      <td>放弃</td>
      <td>在局部拟合中，各参考点的loss不再等权重地相加，而是按照IDW权重相加。它只对少数平滑field有效，而且比局部拟合更慢</td>
      <td></td>
    </tr>
    <tr>
      <td>让不同空间区域选择不同预测器</td>
      <td>待验证</td>
      <td>在单线程中，是指把空间划分成不同区域，每个区域用各自的预测器。DAG依旧是同一个，所以不同区域可能存在相互依赖。在多线程中，是指将每个线程负责的partition再次分为更小的区域。旧实验让各个小区域相互独立，没有什么效果。</td>
      <td>需要以正确的方式重新实现并验证</td>
    </tr>
    <tr>
      <td>先用一个预测器预测，再用另一个预测器预测之前的误差。</td>
      <td>弃用</td>
      <td>理论上来说，当两个预测器都是空间预测器的时候，不太可能有效果，这和“时间预测＋空间补偿”不同。</td>
      <td></td>
    </tr>
    <tr>
      <td>限制局部拟合参数</td>
      <td>弃用</td>
      <td>这种限制没有改善测试结果</td>
      <td></td>
    </tr>
    <tr>
      <td>为不同预测轮分别建立Huffman编码</td>
      <td>放弃</td>
      <td>不同field上的结果不稳定，总体不值得保留</td>
      <td></td>
    </tr>
    <tr>
      <td>用rANS等方法替换Huffman和Zstd</td>
      <td>暂缓</td>
      <td>当前Huffman和Zstd已经能兼顾速度与BR，暂时没有证据表明更换无损编码器是主要优化方向</td>
      <td></td>
    </tr>
    <tr>
      <td>保证Huffman输出缓冲区在任何输入下都足够大</td>
      <td>暂缓</td>
      <td>当前实验数据都能正常运行，但代码还没有证明缓冲区可以容纳所有极端输入</td>
      <td></td>
    </tr>
    <tr>
      <td>离散预测残差的顺序</td>
      <td>待验证</td>
      <td>数据处理顺序往往不同于原始数据的顺序，离散预测残差传给huffman的顺序可以是这两者中的一个。</td>
      <td></td>
    </tr>
    <tr>
      <td>qb的写出顺序</td>
      <td>已固定</td>
      <td>
        目前的经验是：<br>
        对于标签，始终按照原始顺序，写出每个错误位置相对于上一个错误位置的编号差。<br>
        对于数值，固定按照处理顺序写出。按照原始顺序进行时间预测时，qb自然采用原始顺序；使用空间预测或时间预测＋空间修正时，qb采用DAG顺序。原本用于强制按原始点编号存储数值qb的命令行选项已经删除。
      </td>
      <td></td>
    </tr>
  </tbody>
</table>

## 2.5. 点集与标签

当前Katrina实现使用两个点集：

- 点集1满足`value>=-1000`，只使用点集1中的参考点预测数值。
- 点集2满足`value<-1000`，当前值全部为`-99999`，因此使用常量编码。
- 额外预测每个点属于点集1还是点集2，只保存标签预测错误位置，并使用Zstd压缩。

点集1和点集2只是两个对等的类别，不预设哪个点集是主体，也不把另一个称为异常点集。标签存储方式不依赖两个点集各自所占比例。每个time step独立生成标签，因此点集分布可以随时间变化；单点集数据不生成标签流，两点集数据只生成一条二分类标签通道。

REL error bound根据各点集自身的最大值和最小值转换为ABS error bound，避免fill value扩大正常数据的值域。使用JSON配置时，每个非常量label必须独立指定method以及REL或ABS error bound，不存在沿用命令行全局EB的模式；不使用JSON的单label数据才通过命令行`-M`配置。容器记录实际method和各点集值域，但用户配置的EB仍需在解压时保持一致。

双点集始终使用融合路径。当标签和数值共用time step顺序与DAG时，在一次DAG遍历中逐目标同步处理：先恢复当前目标的标签，随后立即根据该标签预测、量化并重建数值。压缩输出同时报告精确的label stream、value stream和共享元数据字节数及BR，因此无需再执行只供筛选使用的两遍staged流程。未实现的标签×数值融合组合在进入对应组合函数后明确报错，不在上层分派前用白名单拦截。

具体来说，当前实现了零阶外插、一阶内插、大步长零阶外插+一阶内插三种时间顺序，每一种都兼容这种顺序下的时间预测、时间预测+空间补偿、以及纯空间预测。

<table class="fixed-layout-table" width="100%">
  <colgroup>
    <col width="20%">
    <col width="10%">
    <col width="42%">
    <col width="28%">
  </colgroup>
  <thead>
    <tr>
      <th width="20%">优化方案</th>
      <th width="10%">状态</th>
      <th width="42%">已有结论</th>
      <th width="28%">下一步</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>在值域{0,1}中预测标签，每预测一个就纠正到正确的结果</td>
      <td>已采纳</td>
      <td>比允许标签的重建值位于[0,1]要好</td>
      <td></td>
    </tr>
    <tr>
      <td>记录预测错误的标签相对于上一个错误的位置，并使用zstd压缩</td>
      <td>已采纳</td>
      <td>比直接把所有正确/错误标记用huffman+zstd压缩要好</td>
      <td></td>
    </tr>
    <tr>
      <td>标签和数值压缩共用DAG</td>
      <td>可选</td>
      <td>可以加速。如果使用的预测器也相同的话，还能进一步复用权重并加速。</td>
      <td></td>
    </tr>
    <tr>
      <td>对于使用时间预测的标签或数据，使用原始顺序而不是DAG</td>
      <td>可选</td>
      <td>加速效果较好。前提是使用时间预测。</td>
      <td></td>
    </tr>
    <tr>
      <td>压缩和解压时，可以使用不同处理顺序。</td>
      <td>数值qb方案已放弃</td>
      <td>在已知DAG时，数值qb可以在原始顺序和DAG顺序之间转换，但转换没有稳定收益，现已固定按处理顺序存储。标签仍可根据具体时间流程选择原始顺序或DAG顺序处理，这与数值qb存储选项无关。</td>
      <td></td>
    </tr>
    <tr>
      <td>按预测顺序编码标签</td>
      <td>待验证，但优先级较低</td>
      <td>在katrina上，压缩比和速度都下降了。</td>
      <td></td>
    </tr>
  </tbody>
</table>

## 2.6. 跨time step预测

2.1节中已经涉及了很多关于跨time step预测的内容，这里仅作补充。另外，很多预测方案对标签无效，因为标签只有0和1。

<table class="fixed-layout-table" width="100%">
  <colgroup>
    <col width="20%">
    <col width="10%">
    <col width="42%">
    <col width="28%">
  </colgroup>
  <thead>
    <tr>
      <th width="20%">优化方案</th>
      <th width="10%">当前状态</th>
      <th width="42%">已有结论</th>
      <th width="28%">下一步</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>纯时间高阶预测</td>
      <td>放弃</td>
      <td>在katrina上预测错误率增加，br增加。</td>
      <td></td>
    </tr>
    <tr>
      <td>时间零阶外插预测中，准备开始处理下一个文件前，只交换当前文件的数据和上一个文件的数据的地址，而不是数据</td>
      <td>已采纳</td>
      <td>每个文件完成输出和误差验证后交换当前与历史重构数据、点集编号vector的所有权，删除逐time step全量复制。不会影响压缩效果，但是可以加速。</td>
      <td></td>
    </tr>
    <tr>
      <td>数值上的一阶外插</td>
      <td>待实现</td>
      <td>已经确认标签上的一阶外插无效。但数值上的一阶外插没有试过。</td>
      <td></td>
    </tr>
  </tbody>
</table>

## 2.7. 并行设计

当前OpenMP实现已经支持adjust后的独立partition。程序先按坐标递归中位数切分，只生成顶层partition点集，不建立整场kd-tree；随后每个worker为自己的partition建立局部kd-tree，并独立生成预测顺序、参考关系、权重和EB倍率。每个partition只引用本partition内的数据，单点集以及融合双点集的标签预测、数值预测、量化、Huffman、Zstd和重建均按partition并行。partition本身始终互相独立。v15容器统一使用`payload_begin[P+1]`定位各预测结构的独立payload，并在文件头记录最终label/value flow；不分区时P=1，分区时每个partition对应一个payload，最终只按结构编号串行拼接字节。REL模式先根据当前完整time step的field range换算统一ABS EB，再由该time step内所有partition共享。正式程序不缓存DAG。

并行实验不使用“多线程UNSZ对串行UMC”作为主要速度结论。单线程实验始终比较一核UNSZ与一核UMC。高速版要求核心压缩与核心解压的耗时之和不超过UMC的`1.1倍`，高压缩比版要求该总和不超过`2倍`；在同一节点上交替运行双方的五个EB，并以逐EB总耗时倍率的几何平均作判断。压缩和解压分项继续报告以定位热点，但不分别设置门槛；暂不把可由全部time step复用的公共预处理计入该限制。对于包含多个独立time step的数据，使用相同数量的进程把不同文件分配给UNSZ和UMC，比较双方相同资源下的总吞吐率；独立进程也避免依赖UMC接口是否线程安全。对于单个大field，单独报告UNSZ从1线程到多线程的strong/weak scaling；UMC官方当前没有通用的单field并行入口，因此只列其官方串行结果，不把二者直接解释为并行算法优劣。如果论文需要同核数的单field比较，再为UMC增加与UNSZ相同partition数量的外部分区，并同时报告两种方法因partition边界造成的BR变化。

<table class="fixed-layout-table" width="100%">
  <colgroup>
    <col width="20%">
    <col width="10%">
    <col width="42%">
    <col width="28%">
  </colgroup>
  <thead>
    <tr>
      <th width="20%">优化方案</th>
      <th width="10%">状态</th>
      <th width="42%">已有结论</th>
      <th width="28%">下一步</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>OpenMP独立partition</td>
      <td>已实现原型</td>
      <td>预测DAG、标签、数值、Huffman、Zstd和重建均已并行；Katrina的1/4线程v10码流均能独立往返</td>
      <td>在大型数据上测试扩展性并减少小partition调度开销</td>
    </tr>
    <tr>
      <td>每partition独立无损码流</td>
      <td>已实现原型</td>
      <td>Katrina第0步4分区BR为<code>0.200938</code>；1/4线程核心压缩为<code>0.0356/0.0287 s</code>、解压为<code>0.0304/0.0252 s</code></td>
      <td>在更多field和更多partition上统计树头、frame开销与BR变化</td>
    </tr>
    <tr>
      <td>强扩展和弱扩展实验</td>
      <td>暂缓</td>
      <td>依赖并行版本完成</td>
      <td>在独占节点测试增加线程或同步增加数据规模时的加速效率</td>
    </tr>
    <tr>
      <td>端到端性能</td>
      <td>暂缓</td>
      <td>已有单核核心吞吐率，UMC解压仍更快</td>
      <td>并行版本完成后统一统计压缩、传输和解压</td>
    </tr>
    <tr>
      <td>相同资源下的UNSZ/UMC并行比较</td>
      <td>实验口径已确定</td>
      <td>多文件时双方都按文件并行；单个大field当前只比较串行性能，并把UNSZ内部并行作为独立扩展性实验</td>
      <td>若必须比较单field并行吞吐率，为UMC实现相同partition数量的外部分区并同时计入BR代价</td>
    </tr>
  </tbody>
</table>

# 3. 数据集与当前结果

## 3.1. 数据集

当前已经下载并转换为UNSZ输入格式的数据共有10个。

| 数据集 | field数 | 单个field的总大小 | 每步点数 | time step数 |
|---|---:|---:|---:|---:|
| E3SM-h0（弃用） | 8 | 0.09 MB | 21600 | 1 |
| Tsunami | 6 | 0.46 MB | 113885 | 1 |
| ADCIRC-Haiyan-1.5km depth | 1 | 8.75 MB | 2187832 | 1 |
| Tpost20 | 1 | 10.49 MB | 131072 | 20 |
| Ocean | 9 | 29.16 MB | 235160 | 31 |
| LES | 3 | 50.95 MB | 12737274 | 1 |
| E3SM-h4 | 2 | 62.21 MB | 21600 | 720 |
| TAPS | 7 | 64.01 MB | 16002271 | 1 |
| ADCIRC-Haiyan-1.5km zeta | 1 | 210.03 MB | 2187832 | 24 |
| Katrina | 1 | 270.63 MB | 417642 | 162 |
| Ocean-large单快照restart | 24 | 详情见下方说明 | | |
| Ocean-large月度输出 | 91 | 详情见下方说明 | | |

单个field的总大小是该field全部`<field>.dat.<time_step>`文件的大小之和，不包含坐标、connectivity、标签以及各种格式的metadata。ADCIRC-Haiyan-1.5km的depth和zeta因time step数不同而分成两行。

Ocean-large有两套不同的输入。第一套此前被误称为“初始状态”，实际主文件由`restart.0001-01-04_00.00.00.nc`重命名而来：模拟开始时间是`0001-01-01`，文件时刻是`0001-01-04`，并记录了`config_do_restart=YES`。因此它应称为单快照restart，而不是初始状态。它只有一个真实time step；月度输出则有12个真实time step。

### Ocean-large单快照restart

这套单快照restart数据共有24个模拟数据field，全部只有一个真实time step：

| 存储位置 | 垂直层数 | field数 | field |
|---|---:|---:|---|
| cell | 1 | 11 | `temperatureSurfaceValue`、`salinitySurfaceValue`、两个surface velocity、四个SSH gradient、`boundaryLayerDepth`、`seaIcePressure`、`atmosphericPressure` |
| cell | 80 | 3 | `temperature`、`salinity`、`layerThickness` |
| cell | 81 | 1 | `vertNonLocalFluxTemp` |
| edge | 1 | 1 | `normalBarotropicVelocity` |
| edge | 80 | 1 | `normalVelocity` |
| forcing文件中的cell | 1 | 7 | 两个wind stress、`seaSurfacePressure`、两个piston velocity、两个surface restoring value |

前五行来自restart主文件，共17个field；最后一行来自独立forcing文件，共7个field。因此这套单快照数据共有`17+7=24`个可测试field。

目前只转换了`temperatureSurfaceValue`和`temperature`。前者作为一个普通单步field测试；后者曾把80个垂直层临时当成80个伪time step，用于探索层间预测，但这些垂直层不是真实时间。

### Ocean-large完整版

月度输出由12个NetCDF文件组成，每个文件对应一个真实time step。同名变量跨12个文件组成一个12步field。每个文件中有58个单层cell变量和33个80层cell变量，此外还有网格、edge和区域统计等变量。

UMC公开代码读取91个长度均为`nCells`的float32 field，每个field有12个真实time step。目前怀疑UMC使用了58个单层cell变量，并取33个80层cell变量的第0层，正好得到91个单层field。这个判断尚未确认，因为公开仓库没有`varnames_91.txt`和NetCDF转换程序。

## 3.2. 方案

下表逐field记录`configs/`中各task的当前方案，并按照上一表中的单个field总大小递增排列；同一数据集内的field按名称字母序排列。通常取UMC在REL `1e-3`处的PSNR作为代表点，在现有UNSZ与UMC曲线上对正值的对数按PSNR插值；`CR/UMC`大于1表示UNSZ压缩率更高。`总时间/UMC`是在两条曲线的共同PSNR区间内对压缩与解压总时间之比的对数积分后取指数，小于1表示UNSZ更快。

<table class="fixed-layout-table" width="100%">
  <colgroup>
    <col width="10%">
    <col width="5%">
    <col width="10%">
    <col width="5%">
    <col width="10%">
    <col width="10%">
    <col width="10%">
  </colgroup>
  <thead>
    <tr>
      <th width="10%">数据与field</th>
      <th width="5%">多点集</th>
      <th width="10%">时空方案</th>
      <th width="5%">坐标变换</th>
      <th width="5%">评价目标时参考点数量</th>
      <th width="10%">代表PSNR</th>
      <th width="10%">CR/UMC</th>
      <th width="10%">总时间/UMC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Tsunami bathy</td>
      <td>否</td>
      <td>纯空间</td>
      <td>否</td>
      <td>1</td>
      <td><code>64.90</code></td>
      <td><code>1.053×</code></td>
      <td>—</td>
    </tr>
    <tr>
      <td>Tsunami height</td>
      <td>否</td>
      <td>纯空间</td>
      <td>否</td>
      <td>1</td>
      <td><code>65.46</code></td>
      <td><code>0.988×</code></td>
      <td>—</td>
    </tr>
    <tr>
      <td>Tsunami momentum_x</td>
      <td>否</td>
      <td>纯空间</td>
      <td>否</td>
      <td>1</td>
      <td><code>66.49</code></td>
      <td><code>1.394×</code></td>
      <td>—</td>
    </tr>
    <tr>
      <td>Tsunami momentum_y</td>
      <td>否</td>
      <td>纯空间</td>
      <td>否</td>
      <td>1</td>
      <td><code>66.48</code></td>
      <td><code>1.316×</code></td>
      <td>—</td>
    </tr>
    <tr>
      <td>Tsunami velocity_x</td>
      <td>否</td>
      <td>纯空间</td>
      <td>否</td>
      <td>1</td>
      <td><code>67.03</code></td>
      <td><code>1.522×</code></td>
      <td>—</td>
    </tr>
    <tr>
      <td>Tsunami velocity_y</td>
      <td>否</td>
      <td>纯空间</td>
      <td>否</td>
      <td>1</td>
      <td><code>67.16</code></td>
      <td><code>1.449×</code></td>
      <td>—</td>
    </tr>
    <tr>
      <td>E3SM-h4 AODDUST</td>
      <td>是</td>
      <td>自动选择</td>
      <td>否</td>
      <td>1</td>
      <td><code>77.30</code></td>
      <td><code>7.798×</code></td>
      <td><code>0.753×</code></td>
    </tr>
    <tr>
      <td>Katrina attr0</td>
      <td>是</td>
      <td>自动选择</td>
      <td>是</td>
      <td>1</td>
      <td><code>81.47</code></td>
      <td><code>2.534×</code></td>
      <td>—</td>
    </tr>
  </tbody>
</table>

Tsunami的`CR/UMC`来自既有1点MSD五点曲线；切换正式配置后尚未统一复测五个REL EB的总时间，因此时间栏暂不填写。E3SM-h4 AODDUST曾固定使用stride为8的锚点内插；相对标签纯空间方案，完整五点曲线的BD-rate增加<code>3.442%</code>，核心压缩与解压总时间减少<code>35.895%</code>。正式配置现改为自动选择flow，表中的既有压缩率和时间待统一复测后更新。

以下是所有实验通用的其他参数。如果发现其中有些参数在不同数据上有不同的最优值，并且有很大提升，应该移动到上表。

<table class="fixed-layout-table" width="100%">
  <colgroup>
    <col width="30%">
    <col width="70%">
  </colgroup>
  <thead>
    <tr>
      <th width="30%">参数</th>
      <th width="70%">值</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>坐标变换</td>
      <td>C++内置的一轮对角缩放训练；自动flatness判据，第一轮采样比例<code>0.3</code>，第二轮采样<code>3000</code>点，最多使用<code>3</code>个文件</td>
    </tr>
    <tr>
      <td>预测器</td>
      <td>RBF，10参考点，exponential核，<code>rbf_shape=0.01</code>，<code>rbf_nugget=1e-6</code>，0阶常数趋势</td>
    </tr>
    <tr>
      <td>多点集、空间预测时，RBF没有足够同类参考点的退化方案</td>
      <td>用最近的一个有效参考点</td>
    </tr>
    <tr>
      <td>排序规模</td>
      <td>global</td>
    </tr>
    <tr>
      <td>排序标准</td>
      <td>MSD</td>
    </tr>
    <tr>
      <td>预测目标候选点</td>
      <td>16</td>
    </tr>
    <tr>
      <td>EB</td>
      <td><code>inverse_rank_log2</code>，底数<code>0.8</code>，shift<code>0.5</code>，最大log2指数<code>7</code></td>
    </tr>
  </tbody>
</table>

## 3.3. 实验口径

`configs/`中的task JSON是当前运行配置的权威记录，3.2节只汇总已经选定或正在调优的方案。主要baseline为UMC；SZ3和ZFP留到正式实验阶段补充。

单点集RD曲线固定使用REL `1e-1、1e-2、1e-3、1e-4、1e-5`五点；多点集RD曲线由`point_set_config_path_list`中的完整误差配置构成，每个点集可以使用不同EB。UNSZ与UMC曲线不要求使用相同EB，而是在共同PSNR范围内对`log(BR)`和`log(time)`作分段线性插值后比较；BD-rate为`log(BR)`差的区间积分均值取指数，因而不受额外插入EB采样点影响。UMC合并多个time step的MSE时所需的数据范围，在每个task的每个field开始实验前计算一次，并传给该field的全部UMC实验，不随REL EB重复读取输入。速度统计压缩与解压的程序内部核心时间之和；压缩端与解压端的完整预处理时间分别记录。正式实验中两个压缩器都不保存预测DAG，压缩端和解压端分别现场重新生成各自的预测结构；UNSZ压缩预处理包含坐标变换训练与应用、DAG生成和RBF权重计算、自动flow选择，并分别输出这三段的程序内部计时。解压预处理读取可复用的坐标变换参数并包含变换应用、DAG生成和RBF权重计算。坐标、connectivity、坐标变换训练数据、自动flow采样数据及坐标变换参数文件的读取不计时，输入正确性验证也不计时。坐标变换参数是field级可复用预处理结果，不计入每个压缩码流的大小。正式倍率必须来自同一节点上的UNSZ与UMC交替运行。

UMC统一使用其正式的`reorder=1`（DPFS重排）。NBP和NBP-INP采用同一计时口径：压缩时间为field重排与核心压缩之和，解压时间为核心解压与恢复原始顺序之和；读取输入、写出结果、读取原始field进行误差验证和误差统计均不计入。ADCIRC-Haiyan-1.5km的connectivity包含两个连通分量，而UMC的DPFS只遍历0号点所在的连通分量，无法生成覆盖全部点的有效重排。我们不修改UMC算法，也不改用`reorder=0`；因此Haiyan汇报为UMC无法运行，不计算`CR/UMC`和`总时间/UMC`。

数据转换入口见`py_codes/prepare_datasets/prepare_*.py`。当前仓库只保留Tsunami和Katrina的详细实验结果；其他数据集保留可重新运行的正式配置和转换程序。

# 4. TODO

## 4.1. 投稿前主线

- [ ] 冻结bitstream、配置和实验脚本后，统一生成正式五点BR–PSNR、核心压缩＋解压时间和各阶段码流长度；在此之前不再无目的地重复全数据复测。
- [ ] 把`results/current`整理为与3.1及`configs/`中的task JSON一致的结果集；Tsunami最终使用16候选。
- [ ] 优先优化仍未达到目标的field，包括Tsunami height、ADCIRC-Haiyan depth和Ocean-large垂直temperature；方案固定后再迁移到其他field验证泛化。
- [ ] 测试一阶内插中“最后一个文件由第一个文件作零阶外插预测”，并实现“空间预测＋时间补偿”。
- UMC-INP的batch固定为README示例使用的10，不再作为task配置或实验维度。
- [ ] 研究DAG和RBF权重的紧凑持久化，明确公共metadata需要在多少field或time step之间摊销。
- [ ] 确认UMC论文Ocean-large的`varnames_91.txt`，核实是否使用58个单高度cell变量及33个多高度cell变量的第0层，再把已经取得的12个月度NetCDF转换为91×12输入；单快照restart不能冒充论文月度输出。
- [ ] 完成single-core throughput、strong/weak scaling以及端到端压缩—传输—解压实验，再决定投稿路线。

详细历史实验和已经否定的方案见`TODO_DETAIL.md`；日常决策状态见`TODO.md`。
