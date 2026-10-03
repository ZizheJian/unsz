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
install/bin/s1_compress -data-dir /path/to/Ocean -field temperature -time-step '[0,30]' -coordinate-dim 3 -M REL 1e-3 -predictor idw -prediction-min-ref-num 1 -prediction-max-ref-num 4 -ordering-candidate-num 16 -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1
```

对应的解压命令为：

```bash
install/bin/s1_decompress -data-dir /path/to/Ocean -field temperature -time-step '[0,30]' -coordinate-dim 3 -M REL 1e-3 -predictor idw -prediction-min-ref-num 1 -prediction-max-ref-num 4 -ordering-candidate-num 16 -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1
```

ABS error bound使用：

```bash
install/bin/s1_compress -data-dir /path/to/data -field field -time-step '[0,9]' -coordinate-dim 3 -M ABS 1e-4 -predictor idw -prediction-min-ref-num 1 -prediction-max-ref-num 4 -ordering-candidate-num 16 -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1
```

`-time-step`支持闭区间`[a,b]`、右开区间`[a,b)`和显式列表`{a1,a2,...}`。例如`[0,31)`和`{0,3,7}`分别处理0至30以及0、3、7号time step。包含括号或花括号的表达式建议使用引号，避免被shell解释。

普通流程把压缩文件写到原数据旁边，名称为`<field>.dat.<time_step>.unsz`；解压结果为`<field>.dat.<time_step>.unsz.bin`。联合时空流程把全部time step写入单个`<field>.joint.unsz`，解压时仍拆成各个`<field>.dat.<time_step>.unsz.bin`。程序打印总BR、PSNR、最大误差比例、压缩端与解压端各自的预处理时间，以及核心压缩和核心解压时间。

当前bitstream不保存用户error bound、坐标变换矩阵、预测参数或实验开关。解压时必须提供与压缩时完全相同的参数；参数不一致通常会导致解码失败或误差检查失败。

空间预测DAG默认不缓存，压缩和解压分别现场生成。显式传入`-dag-cache <path>`后才启用缓存：路径不存在时，压缩器或解压器生成DAG并写入该路径；路径存在时直接读取。缓存必须由相同坐标变换和预测参数生成，同一field测试不同配置时必须为它们指定不同路径。

## 1.4. 自定义有序点集

不传`-point-set-config`时，程序把全部数据作为一个点集，使用命令行`-predictor`和`-M`显式指定预测器和误差上限。`-M`和`-point-set-config`必须且只能选择一个。使用`-point-set-config`时，JSON定义一个或多个点集的分类条件、编码方法和误差上限；只要其中存在`method="predict"`的点集，就必须用`-predictor`统一指定它们共用的空间预测器。压缩和解压必须传入相同的点集配置和预测器参数：

UNSZ目前只支持一个或两个点集。压缩入口依次执行`main`→`unsz::parse_args_and_select_types`→`unsz::parse_args`→初始化并选择`T_coord/T_data`→`unsz::compress`，再按配置中的点集数量进入`unsz::one_point_set::compress`或`unsz::two_point_sets::compress`。单点集入口继续按整体流程和预测器进入`unsz::one_point_set::<流程>::<idw或rbf>::compress`；六种流程与两个预测器组合成12个独立入口。

```bash
install/bin/s1_compress -data-dir /path/to/Katrina -field attr0 -time-step '[0,161]' -coordinate-dim 3 -point-set-config point_set_configs/Katrina_attr0_rbf_rel_1e-2.json -predictor rbf -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-candidate-num 16 -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1
install/bin/s1_decompress -data-dir /path/to/Katrina -field attr0 -time-step '[0,161]' -coordinate-dim 3 -point-set-config point_set_configs/Katrina_attr0_rbf_rel_1e-2.json -predictor rbf -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-candidate-num 16 -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1
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

最后一个label是默认兜底类型，因此省略`condition`（同时省略eb信息是因为`method="constant"`，不需要eb）；配置文件只有一个label时同样不写`condition`。每个`method="predict"`的label必须显式配置`eb_mode`和`eb_value`，模式为`REL`或`ABS`；`method="constant"`省略这两个字段。点集JSON中的`method`只能是`predict`或`constant`；IDW或RBF及其参数只在主配置的`predictor`中记录，不在点集JSON中重复指定。

点集分类只读取当前field的数值并按上述条件执行，不接受额外标签文件。由于EB不写入bitstream，解压时仍必须使用与压缩时相同的配置。

## 1.5. 批量实验与脚本

| 脚本 | 功能 |
|---|---|
| `t1_start_c_code.py` | 正式实验通用入口，只接受一个JSON位置参数：`configs/`中的JSON名称或任意JSON路径，例如`./unsz_env/bin/python t1_start_c_code.py tsunami`或`./unsz_env/bin/python t1_start_c_code.py /path/to/test.json`。包含UNSZ任务时，入口默认先执行CMake配置、增量构建和安装；临时复用现有安装时可添加`--skip-build`。随后直接构造并执行UNSZ或UMC命令，不再调用第二层Python实验入口，结果直接写入CSV。入口不会编译UMC，UMC需要提前手动编译。 |
| `py_codes/parse_unsz_output.py`、`py_codes/parse_umc_output.py` | 只解析并核对压缩器输出，不负责调度或启动实验。 |
| `t2_adjust_coord.py` | 训练坐标变换矩阵。命令行与`t1_start_c_code.py`一样只接受一个JSON名称或路径，并依次训练每个task的全部field；Python函数`train_adjust_coord()`直接接收已经选定field的配置，仍可调整loss、采样、优化器、预测器、训练参考点数和输出路径。 |
| `t3_visualize.py` | 生成Katrina attr0与Ocean temperature坐标变换前后的field图像，以及Tsunami最佳与最差field的原网格图像 |
| `py_codes/prepare_datasets/prepare_<dataset>.py` | 分别把Tpost20、E3SM-h0、E3SM-h4、ADCIRC Haiyan、Tsunami、TAPS和Ocean-large发布数据转换为UNSZ/UMC共用的平坦二进制基准输入；共享工具位于同目录的`prepare_dataset_common.py` |
| `py_codes/run_spatial_field_benchmark.slurm` | 在Slurm节点上运行环境变量`CONFIG`明确指定的完整task JSON，不在运行时拆分或改写JSON |
| `py_codes/run_configurations.slurm` | 以Slurm array依次把当前9个正式数据集JSON交给`t1_start_c_code.py`。 |
| `py_codes/plot_dataset_rd.py` | 为一个数据集的各field分别绘制UNSZ/UMC空间RD与核心吞吐率，并计算同PSNR下的平均BR变化 |

JSON最外层是task list；同一个JSON可以包含不同数据目录的任务。每个task都必须自行记录描述实验用途的唯一`name`、绝对`data_dir`和CSV文件路径`csv_path`，即使相邻task使用同一数据目录也不从外层继承。相对`csv_path`以项目根目录为基准；一个task包含多个field时，路径使用`{field}`作为field名称占位符。同一数据、同一field需要测试不同固定参数时，应写成名称不同的多个task。

`field_list`和`compressor_list`是所有task共有的穷举维度。单点集task使用`rel_eb_list`穷举误差上限；多点集task不再重复提供`rel_eb_list`，而是在`unsz.point_set_config_path_list`中依次列出待测的完整点集JSON，相对路径以项目根目录为基准。每份点集JSON是一次完整的UNSZ误差配置，不同点集可以使用不同EB。若同一task还测试UMC，t1收集全部点集JSON中出现的REL EB、去重后分别运行UMC；UMC不模拟一个文件内不同点集使用不同EB，而是生成用于同PSNR曲线比较的普通REL EB基线。UNSZ结果使用点集JSON路径标识，UMC和单点集UNSZ结果使用REL EB标识。UNSZ的`flow`是一个dict：`label`和`value`分别列出待测的标签与数值流程，二者做笛卡尔积；单点集只写`value`。UMC不使用UNSZ流程，因此下面的单点集task会运行`2个field×2个REL EB×2种压缩器=8`组实验。数据目录、time step数和坐标维数等公共信息直接写在task中；只有UNSZ使用的流程、预测、排序、坐标变换和EB参数放在`unsz`中，只有UMC使用的参数放在`umc`中。t1直接把选中的点集JSON传给压缩器和解压器，不临时生成或修改它。其他参数均为单值，不因写成list而自动展开。正式配置不接受绕过结构化字段的额外命令行参数列表。

```json
[
  {
    "name":"tsunami_height_tuning",
    "data_dir":"/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Tsunami",
    "csv_path":"results/current/fields/tsunami_{field}.csv",
    "field_list":["height","momentum_x"],
    "rel_eb_list":[0.01,0.001],
    "compressor_list":["unsz","umc"],
    "time_step_num":1,
    "coordinate_dim":2,
    "unsz":{
      "flow":{"value":["spatial"]},
      "predictor":{"name":"rbf","rbf_kernel":"matern32","rbf_shape":0.03,"rbf_nugget":1e-6},
      "ordering_scope":"global",
      "prediction_min_ref_num":10,
      "prediction_max_ref_num":10,
      "ordering_candidate_num":16,
      "ordering_score_max_ref_num":1,
      "ordering_score_min_ref_num":1,
      "eb_schedule":{"name":"inverse_rank_log2","base":0.85,"log2_shift":0.5,"max_log2_exponent":7}
    },
    "umc":{
      "from_edge":1,
      "reorder":1,
      "method":"NBP-IDW",
      "neighbor_num":4
    }
  }
]
```

`predictor`集中记录预测器自身的参数：IDW使用`name`和`idw_power`；RBF使用`name`、`rbf_kernel`、`rbf_shape`和`rbf_nugget`。RBF权重在DAG生成后统一以批量LDLT和Schur补求解，不再提供求解器配置。RBF在同点集参考点不足或权重求解失败时固定退化为最近的同点集参考点，不再提供IDW fallback。`prediction_max_ref_num`既是DAG为每个目标最多保存的参考点数，也是主预测器最多使用的同点集参考点数；同点集参考点少于`prediction_min_ref_num`时，RBF触发最近点fallback。当前正式RBF配置的上下限相同，IDW的下限为1。预测目标固定使用反向MSD排序，不再把排序标准作为配置维度；MSD评分使用的参考点数量由独立的`ordering_score_max_ref_num`和`ordering_score_min_ref_num`控制。`ordering_scope="local"`时先把点分轮，再在各轮内部排序，并通过`prediction_round_num`明确记录轮数；`ordering_scope="global"`时只生成一条完整顺序，不接受`prediction_round_num`，也不能使用名称中含`round`的逐轮EB序列。global可以使用`none`、`point`或连续的`inverse_rank_log2`。`ordering_candidate_num`必须显式记录每次选择下一个预测目标时检查的候选数，不根据预测器或时空流程推断默认值。RBF固定使用0阶常数约束。例如要比较不同`rbf_shape`，应写成两个名称不同的task，而不是把`rbf_shape`写成list。`eb_schedule`记录EB曲线的`name`和`base`；连续的`inverse_rank_log2`还记录`log2_shift`和`max_log2_exponent`。程序不再生成或接受额外的`key`；压缩和解压文件继续采用原始field文件名，CSV写入`csv_path`，并用task名称、压缩器、标签流程、数值流程、标签/数值执行方式、REL EB和点集配置路径共同区分每一行，其中不适用的标识写为`not_applicable`。需要缓存DAG时，在`unsz`中显式设置`dag_cache_path`；相对路径以`data_dir`为基准，并可使用`{field}`占位符。没有该参数时，压缩器和解压器分别现场生成DAG且不写入缓存。同一field测试不同DAG配置时必须显式使用不同路径，避免误读不兼容缓存。task设置`"cleanup_after_each_test":true`时，每组结果成功写入CSV后删除UNSZ生成的`.unsz`、`.unsz.bin`和联合时空`.joint.unsz`，以及UMC生成的`.umc`和`.umc.out`；点集JSON与显式指定的DAG缓存不会被删除。

UMC任务在独立解压完成后始终逐点读取原始field和磁盘上的`.umc.out`。检查使用NumPy分块计算每个time step的数据范围、最大误差和总MSE，验证逐点REL误差界，并核对磁盘PSNR与UMC压缩端和解压端报告一致；该检查在UMC程序计时之外执行，不计入核心压缩或解压时间。

统一可视化程序默认生成Katrina attr0与Ocean temperature的变换前后图像，以及Tsunami velocity_x和height的原网格图像，也可以只选择一个数据集：

```bash
./unsz_env/bin/python t3_visualize.py
./unsz_env/bin/python t3_visualize.py -dataset katrina
./unsz_env/bin/python t3_visualize.py -dataset ocean-temperature
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
      <td>联合时空压缩</td>
      <td>几乎放弃</td>
      <td>C++读取一次空间坐标，在末尾增加时间坐标并为每个time step复制一份；全局点编号为<code>time_id×N+spatial_id</code>。随后把各文件的值按相同编号连接起来，作为一个高维点云调用对应的一点集或两点集纯空间预测；解压单个联合码流后再按每步<code>N</code>个点拆回原文件。在所有已有的数据上，总能找到比它好的方案，而且预处理耗时较长，因此不进入常规自动选择。</td>
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

时间遍历顺序与标签、数值预测公式分开处理。程序统一提供顺序遍历、递归内插和anchor8三种time step顺序；标签flow和数值flow分别选择自己的顺序。每次遍历开始前会生成time step依赖DAG，直接记录各目标的标签和值参考节点以及各参考节点的剩余出度；目标处理完成后删除已消费的边，某类历史的剩余出度归零时立即释放对应的完整数组，不再扫描或永久保留全部既有time step。`staged`依次应用两份顺序并分别生成依赖DAG，允许二者不同。`fused`保留两份独立顺序：相同时使用现有同步实现，不同时进入专门的不同顺序fused入口；该入口目前明确报告尚未实现，以后可以在不改变配置结构的情况下加入交错调度。

时间一阶内插可通过`-time-step-coordinate-file <path>`读取一维float64原始数组，第`i`个数是time step `i`的实际坐标。压缩和解压必须读取同一文件；不传该参数时保持原有的等间距假设。对应task使用`time_step_coordinate_file`记录文件名并同时传给两端，普通等时间间隔数据不需要记录这一项。

联合时空流程使用`"value_flow":"joint_spatiotemporal"`。C++直接从`coords.dat`获得每个time step的空间点数，将空间坐标复制到各个time step并增加一个时间坐标维度，再把所有time step作为一个统一点云生成DAG；不再要求用户重复传入空间点数。输入仍是原始空间坐标和逐time step的field文件，不需要由Python预先生成升维坐标或拼接field；时间坐标由`time_step_coordinate_file`提供，不设置时使用time step编号。单点集和两点集都把升维后的数据交给普通纯空间压缩；两点集`fused`生成正式码流，`staged`只在内存中估计标签流与数值流的总大小。联合时空只接受显式ABS误差上限。启用partition时，程序在升维坐标上切分统一点云，每个partition独立生成DAG、权重和payload，最后只拼接payload与offset表。普通非联合时空流程仍可使用REL：每个time step根据完整空间field的range换算一次ABS误差上限，该time step内的所有partition共享换算结果。

## 2.2. 生成坐标变换矩阵

UNSZ根据点之间的距离选择参考点，但原始坐标中的距离不一定能反映数据是否容易互相预测。例如Katrina的z轴被严重拉伸，坐标上最近的点不一定是数值上最接近的点。Adjust coord用少量训练文件提前学习一个下三角矩阵`C`，以后只根据公共坐标和这个矩阵生成预测关系和计算距离。需要注意的是，并不是所有数据都应该生成坐标变换矩阵，并且没有一个优化目标能够精确指出哪个矩阵更好。

待验证的简化方案是只学习或扫描各坐标轴的缩放比例，不使用轴间混合，并在相同预测器、排序和PSNR下比较BR。不能先认定Katrina现有矩阵等价于纯z缩放：当前矩阵中z到x、y的系数约为`-0.00534/-0.00258`，乘上原始z跨度后，对x、y距离的影响可能与这两轴自身跨度同量级。轴向缩放若仍用于生成顺序、选择参考点和计算RBF权重，压缩端与解压端必须使用相同的缩放比例。

正式训练入口只指定一份task JSON。例如：

```bash
./unsz_env/bin/python t2_adjust_coord.py katrina
```

`t2`通过`t1_start_c_code.load_json_tasks()`读取明确指定的JSON，不会扫描`configs/`并根据dataset猜测配置文件。它依次使用每个task的数据路径、维数、time step数和最终数值预测器。正式训练所需的loss、优化器、采样数量、迭代数、矩阵混合参数、训练REL EB和输出矩阵文件名都必须在task中显式记录；缺少字段时直接报错，不从其他参数推导，也不使用隐藏默认值。命令行不提供单独的训练参数，避免正式矩阵使用不同协议。需要只训练部分field时，应提供只包含这些task或field的JSON；做消融时可以调用`t2_adjust_coord.train_adjust_coord()`，通过函数参数明确覆盖训练配置。更底层的`adjust_coord()`保留完整参数接口。

坐标按照`c'=cC`变换，训练希望两个点的数值差与它们在新坐标中的距离接近：

```text
|v1-v2|≈||(c1-c2)C||₂
```

这个过程是迭代式的。迭代轮数由task中的`adjust_coord_iteration_num`明确给出；Katrina的码长loss在第3轮后明显退化，因此当前配置使用2轮。

训练不需要读取所有time step，而是在整个时间范围内等间隔选择task中`adjust_coord_sample_file_num`指定数量的文件。Katrina当前明确设置为3，因此162个文件中会读取第27、80和134个文件。随后，每轮迭代中依次执行采样、训练和后处理。

### 2.2.1. 采样

每轮迭代要做两次采样。第一次采样决定用哪些点建立近邻树，主要控制建树和查询的规模；第二次采样只从树上的点中挑选一部分作为目标，主要控制交给优化器的点对数量。具体过程如下：

1. 从全部坐标中无放回地随机选择`ceil(point_num×first_sample_ratio)`个点；比例由task中的`adjust_coord_first_sample_ratio`明确给出，Katrina当前使用25%。选出的编号会排序，因此没有重复且保持原始点编号顺序。第一次采样不能太稀疏，否则树上的“最近点”会离目标太远，无法代表完整点云中的局部关系。
2. 对每个训练文件分别删除这批点中的异常值。当前Python版本只通过`is_abnormal()`把点分为正常和异常两类，并且不让异常点参与训练。由于异常点的位置会随time step改变，每个文件最终留下的点可能不同。
3. 从每个文件剩余的点中再次无放回地随机选择目标，也就是训练点对的第一个点。目标数由task中的`adjust_coord_second_sample_num`明确给出，并尽量平均分给各训练文件，从而不让点数或time step数量改变优化器的输入规模；Katrina当前使用100000个目标。Python函数可以通过`second_sample_num`覆盖该固定数量；只有显式设为0时才改用`ceil(第一次采样点数×训练文件数×second_sample_ratio)`。
4. 对每个文件单独建立一棵`cKDTree`。树上的坐标先经过当前`C`变换；查询时选择`prediction_max_ref_num`个近邻作为点对的第二个点。正式`t2`从对应task读取实际预测器的参考点数，因此当前Tsunami配置中每个目标使用10个近邻；Python函数仍可显式覆盖该数值。
5. 对每个点对记录数值差`Δv`和未经变换的原始坐标差`Δc`。

采样相关方案如下：

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
      <td>第一次采样密度和第二次目标数量</td>
      <td>待验证</td>
      <td>降低第一次采样率会减少建树、近邻查询的时间和内存，但树中点变稀疏，查询到的近邻会比完整点云中的真实近邻更远，可能使训练矩阵失真，Katrina当前使用25%。第二次采样只控制作为训练目标的点数，基本上决定adjust coord的速度。Katrina码长loss从10000增加到31324、60000和100000个目标时实际BR持续改善，当前配置固定为100000；其他field在建立正式训练配置时也必须显式记录该值。</td>
      <td>新field先使用10000个目标；只有实际压缩验证表明扩样有效时再增加</td>
    </tr>
    <tr>
      <td>随机选定矩形区域作为样本</td>
      <td>放弃</td>
      <td>没有表现出稳定优势。在LES velocity_x上，固定采样点数并估算区域大小时，使用2或4个区域只让BR降低约<code>0.30%</code>，使用1或8个区域却分别使BR增加<code>27.88%/0.49%</code>。旧Ocean velocityZonal实验中的四维4区域方案曾比零阶外插降低<code>8.920%</code> BD-rate，但该实验同时改用了联合四维预测，无法证明收益来自区域采样；联合四维方案本身也已被更好的时间预测方案取代。因此现有结果不支持继续使用矩形区域采样</td>
      <td>不再作为训练候选；只在trash中保留历史实现和结果用于复现</td>
    </tr>
    <tr>
      <td>训练使用与实际压缩相同的参考点数</td>
      <td>已采纳</td>
      <td>训练与实际压缩使用相同的预测器参数，包括参考点数。此前固定使用4个训练近邻的结果只保留为历史消融，不再作为正式矩阵</td>
      <td>每次修改field的预测参考点数后重新训练并复测矩阵</td>
    </tr>
  </tbody>
</table>

### 2.2.2. 训练

汇总所有样本点对的数值差和原始坐标差后，程序训练一个公共的下三角矩阵`C`。训练loss由task中的`adjust_coord_loss`明确指定；Katrina当前使用离散量化区间码长loss `prediction_ggd_bit`。程序从task JSON读取该field最终采用的IDW或RBF预测器及其参考点数、`idw_power`、`rbf_kernel`、`rbf_shape`和`rbf_nugget`，再在真实数据上生成预测误差。二次方loss定义为：

$$ L(C)=\sum_i\left(\lvert\Delta v_i\rvert-\lVert\Delta c_iC\rVert_2\right)^2 $$

除了首轮以外，训练会以历史`C`作为初值。

训练目标和优化器相关方案如下：

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
      <td>量化区间码长loss</td>
      <td>Katrina已采纳</td>
      <td>这是当前<code>prediction_ggd_bit</code>的实现。使用预测器生成误差，闭式估计最大似然的GGD尺度参数（固定中心为0、形状参数为0.5），通过CDF之差估计每个误差附近的量化区间概率，并最小化累计估计码长。该方法依赖EB，而且没有先生成离散量化索引，因此只是量化码长的连续近似。Katrina使用100000个目标、训练REL EB <code>1e-3</code>和第2轮矩阵后，同PSNR BR比可复现的旧二次方loss矩阵降低<code>4.733%</code>。</td>
      <td>在其他启用坐标变换的field上分别验证</td>
    </tr>
    <tr>
      <td>GGD微分熵loss</td>
      <td>已实现，待比较</td>
      <td>这是最初设想的码长loss。设预测误差为<code>e_i(C)</code>，固定GGD中心为0、形状参数<code>β=0.5</code>，先闭式估计尺度参数：$$ \alpha(C)=\left(\frac{\beta}{n}\sum_i\lvert e_i(C)\rvert^\beta\right)^{1/\beta}. $$ 然后直接最小化微分熵：$$ h(C)=\frac{1/\beta+\ln\left(2\alpha(C)\Gamma(1/\beta)/\beta\right)}{\ln 2}. $$ 当<code>β=0.5</code>时，$$ h(C)=\frac{2+\ln(4\alpha(C))}{\ln 2}. $$ 它不量化预测误差，也不依赖EB。当前实现对IDW和零次多项式RBF提供解析梯度，并使用L-BFGS；在函数接口中传入<code>loss_function_name="prediction_ggd_differential_entropy"</code>即可选择，若希望`t2`命令行采用它，则在对应field task中记录<code>adjust_coord_loss</code>和<code>adjust_coord_optimizer="lbfgs"</code>。</td>
      <td>与量化区间码长loss及二次方loss比较BR和训练时间</td>
    </tr>
    <tr>
      <td>二次方loss</td>
      <td>待验证</td>
      <td>$$ L(C)=\sum_i\left(\lvert\Delta v_i\rvert-\lVert\Delta c_iC\rVert_2\right)^2 $$
      比四次loss和对数loss要好（在稳定性和能够达到的最优压缩效果上），但疑似不如码长loss。另外速度也疑似比码长loss慢。
      </td>
      <td>需要和码长loss比较</td>
    </tr>
    <tr>
      <td>新矩阵吸收率</td>
      <td>按field选择</td>
      <td>默认值仍为<code>0.5</code>。Katrina在100000样本、训练REL EB <code>1e-3</code>的第2轮比较<code>0.25/0.5/0.75/1.0</code>后选择<code>0.75</code></td>
      <td>新field先使用默认值，只在实际BR对吸收率敏感时调节</td>
    </tr>
    <tr>
      <td>外层训练迭代数</td>
      <td>按field选择</td>
      <td>默认值为4轮。Katrina码长loss逐轮实测后选择第2轮；第3和第4轮即使训练loss变化不大，实际BR也明显退化</td>
      <td>保存并验证每一轮矩阵，不按训练loss自动选择</td>
    </tr>
    <tr>
      <td>直接优化<code>M=CCᵀ</code></td>
      <td>放弃</td>
      <td>当前直接优化<code>C</code>会自动得到半正定的<code>CCᵀ</code>，列符号歧义也已通过统一正对角线解决。直接优化<code>M</code>时，平方loss仍包含<code>sqrt(ΔcMΔcᵀ)</code>，不会变成简单的线性最小二乘，反而还要额外施加半正定约束或进行投影</td>
      <td></td>
    </tr>
    <tr>
      <td>手写Gauss–Newton快速迭代</td>
      <td>需要更新并重新验证</td>
      <td>当前版本针对的是二次方loss，有明显的加速效果，只带来非常微小的br损失。</td>
      <td>需要实现针对码长loss的版本，并验证</td>
    </tr>
    <tr>
      <td>二次差分loss</td>
      <td>放弃</td>
      <td>定义依赖近似共线三点，适用范围和预期收益都较弱</td>
      <td></td>
    </tr>
    <tr>
      <td>随坐标变化的变换矩阵</td>
      <td>待验证</td>
      <td></td>
      <td></td>
    </tr>
  </tbody>
</table>

### 2.2.3. 后处理

一次训练结束后会得到原始下三角矩阵`C_new`，随后按以下顺序处理：

1. **检查优化结果。** 程序首先确认`C_new`的所有元素和loss残差均为有限值。出现NaN或Inf说明优化已经失效，继续建树会传播错误，因此立即终止。
2. **统一列符号。** 把`C_new`中负对角元素对应的整列取反。同一个度量可以由列符号不同的多个`C`表示，因为任意对角元素为`±1`的`D`都满足`(CD)(CD)ᵀ=CCᵀ`；统一为正对角线可以避免等价因子在后续平均时相互抵消，而且列取反本身不改变任何距离。
3. **分离矩阵形状和整体倍率。** 将统一符号后的矩阵写成`C_new=sC_norm`，其中`||C_norm||_F²=dim`。倍率`s`可以理解为数值差相对于坐标差的总体尺度，归一化矩阵`C_norm`表示真正影响近邻顺序和IDW权重的度量形状。
4. **缓冲。** 归一化后的变换矩阵和历史上的归一化矩阵加权算数平均，默认新矩阵的权重是0.5，随后再次归一化。倍率`s`和历史倍率加权几何平均，默认新倍率的权重是0.9，实验表明这个倍率权重不会影响最终归一化矩阵、BR或训练时间。

矩阵后处理相关方案如下：

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
      <td>在<code>M=CCᵀ</code>上混合新旧训练结果</td>
      <td>放弃</td>
      <td><code>M</code>混合表示平均平方距离，不符合当前“平均新旧变换后坐标”的缓冲定义。另外在katrina上效果略差。</td>
      <td></td>
    </tr>
    <tr>
      <td>分离矩阵形状和整体倍率</td>
      <td>已采纳，待验证</td>
      <td>原方案每轮归一化<code>C</code>并丢弃整体倍率，新方案分别缓冲归一化矩阵<code>C</code>和倍率<code>s</code>。平方loss下，固定矩阵形状后的倍率只是一个容易求解的一维最小二乘问题，优化器能迅速重新找到合适倍率；近邻选择又只由归一化矩阵形状决定，因此两种方案在Katrina上完全一样，报错耗时和结果。只是新方案不会无故丢弃倍率，算法定义更合理，因此暂时采纳</td>
      <td>在其他数据和码长loss上验证</td>
    </tr>
    <tr>
      <td><code>s</code>的采纳率</td>
      <td>暂时固定</td>
      <td>平方loss下，固定矩阵形状后只剩一个一维尺度最小二乘问题，优化器可以迅速重新得到合适倍率。采纳率<code>0.7/0.9/1.0</code>得到相同的归一化矩阵和<code>6/7/6/6</code>轮内层评估次数，对BR与时间均无可见影响</td>
      <td>默认固定为<code>0.9</code>。如果发现“分离矩阵形状和整体倍率”在其他数据上也没有性能影响，则这一项也不需要调整了。</td>
    </tr>
  </tbody>
</table>

### 2.2.4. 其他优化方案

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
      <td>让adjust coord训练读取用户定义多点集配置</td>
      <td>待实现</td>
      <td>C++压缩端已经支持多个点集、各点集独立的预测方法和误差界以及<code>n-1</code>个标签通道；Python训练端没有读取点集JSON，仍只把<code>is_abnormal()</code>筛出的所有正常值视为一个点集，并为每个文件使用一个公共误差界</td>
      <td>让训练端读取压缩所用的同一份JSON。非constant点集分别按自身预测方法和误差界计算码长loss，再合并成一个公共矩阵的目标；constant点集不产生数值码长loss。标签码流是否计入目标需单独验证</td>
    </tr>
    <tr>
      <td>预先把各坐标维度scale到相同范围</td>
      <td>放弃</td>
      <td>Katrina重新训练后相对原坐标adjust退化<code>22.848%</code> BD-rate</td>
      <td></td>
    </tr>
  </tbody>
</table>

### 2.2.5. 使用训练好的矩阵

推荐通过矩阵文件使用训练结果。压缩和解压必须提供同一个文件：

```bash
install/bin/s1_compress -data-dir /path/to/Katrina -field attr0 -time-step '[0,9]' -coordinate-dim 3 -M REL 1e-3 -predictor idw -prediction-min-ref-num 1 -prediction-max-ref-num 4 -ordering-candidate-num 16 -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -adjust-coord-matrix-file adjust_coord_matrices/Katrina_attr0_spatial.txt
install/bin/s1_decompress -data-dir /path/to/Katrina -field attr0 -time-step '[0,9]' -coordinate-dim 3 -M REL 1e-3 -predictor idw -prediction-min-ref-num 1 -prediction-max-ref-num 4 -ordering-candidate-num 16 -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -adjust-coord-matrix-file adjust_coord_matrices/Katrina_attr0_spatial.txt
```

矩阵文件记录数据集、field、维数、下三角元素和训练类型。`training_type=spatial`表示只使用空间坐标训练；`training_type=spatiotemporal`表示把time step作为额外坐标维度进行联合时空训练。当前选定矩阵集中保存在`adjust_coord_matrices/`，格式和文件索引见其中的`README.md`。

## 2.3. 生成DAG

预测前需要根据公共坐标生成一套所有time step都能复用的有向无环图（DAG）。整体分为三步：

1. 使用经过adjust coord变换的坐标生成kd-tree。
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
      <td>当前本地DAG cache已经可以保存和读取完整DAG，但文件很大；这里尚未实现的是把它压缩到适合随数据传输的大小。</td>
      <td></td>
    </tr>
  </tbody>
</table>

## 2.4. 数值预测、量化与无损编码

当前数值预测主要使用IDW（权重反比例与距离的某次方）和RBF。以下统一把预测目标的坐标和值记为`c`和`v`，把第`i`个参考点的坐标和值记为`cᵢ`和`vᵢ`。IDW定义为：

$$
d_i^2=\lVert c-c_i\rVert_2^2,\qquad
\hat v_{\mathrm{IDW}}=\frac{\sum_i v_i(d_i^2)^{-p/2}}{\sum_i(d_i^2)^{-p/2}}.
$$

其中$p$由`-idw-power`指定。程序保存的是欧氏距离的平方，因此$(d_i^2)^{-p/2}$等价于使用$d_i^{-p}$；3.3.1节的高速IDW配置固定使用$p=2$。

RBF的思路是：先根据坐标计算目标与各参考点、参考点彼此之间的相似度，再求出一组与这些相似度匹配的权重，最后对参考值加权求和。首先用目标到参考点的均方距离统一缩放当前局部邻域：

$$
s^2=\frac{1}{k}\sum_i\lVert c-c_i\rVert_2^2.
$$

当前采用Matérn-3/2核。用`κ(c₁,c₂)`表示坐标`c₁`和`c₂`之间的相似度：

$$
z(c_1,c_2)=\frac{\sqrt{3}\varepsilon\lVert c_1-c_2\rVert_2}{s},\qquad
\kappa(c_1,c_2)=\left(1+z(c_1,c_2)\right)\exp\left(-z(c_1,c_2)\right).
$$

其中`ε`是`rbf_shape`。另外也存在Gaussian、Exponential、Matérn-5/2、Inverse Multiquadric，不同的核对权重生成的速度的影响不大，因为耗时主要在下一步解方程中。

随后求解权重`w₁,...,wₖ`和一个辅助量`μ`，使每个参考点`cᵢ`都满足：

$$
\begin{aligned}
\sum_jw_j\kappa(c_i,c_j)+\lambda w_i+\mu&=\kappa(c_i,c),&&i=1,\ldots,k,\\
\sum_jw_j&=1.
\end{aligned}
$$

`λ`是`rbf_nugget`；最后一个约束保证常量field仍能被精确预测。得到权重后：

$$
\hat v_{\mathrm{RBF}}=\sum_iw_iv_i.
$$

这就是RBF的完整计算。UNSZ固定使用0阶常数趋势：辅助量`μ`和约束`Σwᵢ=1`保证常量field被精确预测，不实现或配置更高阶趋势。在相同参考点数量的情况下，RBF预测器的速度其实和IDW一样，因为最终的压缩还是加权平均。但RBF需要更多的参考点才能预测，并且预处理阶段生成权重需要更长的时间。另一方面来说，在参考点较多时，RBF提供更准确的预测。

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
      <td>局部排序仍需要先分轮再分别排序，默认使用<code>8轮+EB公比0.8</code>；全局排序已经没有预测意义上的轮，不再沿用分段EB倍率</td>
      <td>局部排序保留8轮、相邻轮的数据量比例2和EB公比0.8；全局排序使用按预测位置连续变化的EB倍率</td>
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

REL error bound根据各点集自身的最大值和最小值转换为ABS error bound，避免fill value扩大正常数据的值域。使用JSON配置时，每个非常量label必须独立指定method以及REL或ABS error bound，不存在沿用命令行全局EB的模式；不使用JSON的单label数据才通过命令行`-predictor`和`-M`配置。容器记录实际method和各点集值域，但用户配置的EB仍需在解压时保持一致。

`fused`用于正式速度路径。当标签和数值共用time step顺序与DAG时，在一次DAG遍历中逐目标同步处理：先恢复当前目标的标签，随后立即根据该标签预测、量化并重建数值。未实现的标签×数值融合组合在进入对应组合函数后明确报错，不在上层分派前用白名单拦截。

`staged`只用于调试压缩效果，不用于比较速度。它执行真正的两遍处理：第一遍按标签flow自己的time step顺序完成全部标签；第二遍再按数值flow自己的time step顺序完成全部数值。程序根据第一遍的标签payload和第二遍的数值payload估算每个time step组合后的字节数，不生成压缩文件，因此BR可用于比较流程效果但不能随后解压。其计时包含两遍完整调试开销，不与fused或UMC比较。标签和数值可以使用不同的文件遍历顺序。当前staged调试路径只支持单个预测结构。

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

当前OpenMP实现已经支持adjust后的独立partition。程序先按坐标递归中位数切分，只生成顶层partition点集，不建立整场kd-tree；随后每个worker为自己的partition建立局部kd-tree，并独立生成预测顺序、参考关系、权重和EB倍率。每个partition只引用本partition内的数据，单点集以及fused双点集的标签预测、数值预测、量化、Huffman、Zstd和重建均按partition并行。`global`和`local`表示每个partition内部是否分轮；partition本身始终互相独立。v11容器统一使用`payload_begin[P+1]`定位各预测结构的独立payload；不分区时P=1，分区时每个partition对应一个payload，最终只按结构编号串行拼接字节。普通REL模式先根据当前完整time step的field range换算统一ABS EB，再由该time step内所有partition共享；联合时空目前只接受ABS。DAG缓存与staged调试路径暂不支持partition。

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
      <td>标签纯空间＋数值anchor8内插</td>
      <td>否</td>
      <td>1</td>
      <td><code>77.30</code></td>
      <td><code>7.798×</code></td>
      <td><code>0.753×</code></td>
    </tr>
    <tr>
      <td>Katrina attr0</td>
      <td>是</td>
      <td>标签零阶外插＋数值零阶外插＋空间修正</td>
      <td>是</td>
      <td>1</td>
      <td><code>81.47</code></td>
      <td><code>2.534×</code></td>
      <td>—</td>
    </tr>
  </tbody>
</table>

Tsunami的`CR/UMC`来自既有1点MSD五点曲线；切换正式配置后尚未统一复测五个REL EB的总时间，因此时间栏暂不填写。E3SM-h4 AODDUST使用10点Matérn-3/2 RBF、1点MSD和16个排序候选点；标签使用纯空间预测，数值使用stride为8的固定锚点内插。五点曲线的独立解压和逐点误差界检查均通过，UNSZ相对UMC的BD-rate为<code>-87.162%</code>。

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
      <td>adjust coord配置</td>
      <td>码长loss、100000个目标、训练REL EB <code>1e-3</code>、2轮和吸收率<code>0.75</code></td>
    </tr>
    <tr>
      <td>预测器</td>
      <td>RBF，10参考点，Matérn-3/2核，<code>rbf_shape=0.03</code>，<code>rbf_nugget=1e-6</code>，0阶常数趋势</td>
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
      <td>底数0.85，shift0.5</td>
    </tr>
  </tbody>
</table>

## 3.3. 实验口径

`configs/`中的task JSON是当前运行配置的权威记录，3.2节只汇总已经选定或正在调优的方案。主要baseline为UMC；SZ3和ZFP留到正式实验阶段补充。

单点集RD曲线固定使用REL `1e-1、1e-2、1e-3、1e-4、1e-5`五点；多点集RD曲线由`point_set_config_path_list`中的完整误差配置构成，每个点集可以使用不同EB。UNSZ与UMC曲线不要求使用相同EB，而是在共同PSNR范围内对`log(BR)`和`log(time)`作分段线性插值后比较；BD-rate为`log(BR)`差的区间积分均值取指数，因而不受额外插入EB采样点影响。UMC合并多个time step的MSE时所需的数据范围，在每个task的每个field开始实验前计算一次，并传给该field的全部UMC实验，不随REL EB重复读取输入。速度统计压缩与解压的程序内部核心时间之和；压缩端与解压端的预处理时间分别记录，其中包括各自的DAG加载或生成、RBF权重生成等操作，输入正确性验证和坐标变换时间单独排除。正式倍率必须来自同一节点上的UNSZ与UMC交替运行。

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
