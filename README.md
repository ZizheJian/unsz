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

需要CMake、支持C++17和AVX2的C++编译器以及Python 3。C++依赖的Zstd和参数解析库均作为submodule提供；基准结果处理需要NumPy和Matplotlib，坐标可视化脚本还需要PyVista。

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

压缩文件写到原数据旁边，名称为`<field>.dat.<time_step>.unsz`；解压结果为`<field>.dat.<time_step>.unsz.bin`。程序打印总BR、PSNR、最大误差比例、压缩端与解压端各自的预处理时间，以及核心压缩和核心解压时间。仅做内存误差验证和性能实验时，解压命令可添加`-skip-decompressed-output`，此时仍完成解压与逐点误差验证，但不写`.unsz.bin`；默认行为仍然写出结果。

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
| `t3_visualize.py` | 读取C++压缩器生成的`*.unsz.coord_transform.txt`，生成Katrina坐标变换前后的field可视化 |
| `py_codes/prepare_datasets/prepare_<dataset>.py` | 分别把Tpost20、E3SM-h4、ADCIRC Haiyan、TAPS和Ocean-large发布数据转换为平坦二进制基准输入；`prepare_ocean_large_tensor.py`另外生成完整Ocean-large的`[time,height,point]`切片和tensor metadata，共享工具位于同目录的`prepare_dataset_common.py` |
| `py_codes/prepare_datasets/prepare_ocean_large_tensor.slurm` | 申请16个CPU及其附带内存执行约499GiB的完整Ocean-large tensor转换；这是大型I/O任务，提交前应再次确认输出目录配额 |
| `py_codes/run_spatial_field_benchmark.slurm` | 在Slurm节点上运行环境变量`CONFIG`明确指定的完整task JSON，不在运行时拆分或改写JSON |
| `py_codes/run_configurations.slurm` | 以Slurm array依次把当前8个正式数据集JSON交给`t1_start_c_code.py`。 |
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

统一可视化程序读取C++坐标变换结果，生成Katrina attr0与Ocean temperature的变换前后图像：

```bash
./unsz_env/bin/python t3_visualize.py
./unsz_env/bin/python t3_visualize.py -dataset katrina
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

多time step正式压缩自动选择label和value flow。程序在预处理计时开始前读取全部time step，准确计算各点集的ABS EB和点数；该输入统计时间不计入预处理。空间DAG和权重生成后，程序均匀选择最多32个time step，把全局2048目标预算按prediction structure均分，并在每个structure内布置公共连续采样窗口；只用这些time step估算候选码流后按总time step数缩放。程序先合并这些评估步骤及全部候选的直接时间参考，只从该并集中稀疏读取样本值；候选估算复用正式空间预测、时间依赖、量化、Huffman和Zstd函数，覆盖纯空间、零阶外插及anchor8流程。候选内已量化的采样目标使用重建值继续后续空间和时间预测；REL EB宽于`1e-3`时统一临时使用`1e-3`决策，正式压缩仍使用用户指定EB。候选所需的数据读取不计入预处理时间，分类、候选预测和编码计入。最终选中的label/value flow写入压缩文件；解压端从文件头恢复它们。单time step只有纯空间流程，不执行自动选择。

Katrina四partition测试中，全局2048目标预算相对每partition各2048目标把自动选择时间从`0.358 s`降到`0.173 s`，选中的flow、最终BR和PSNR不变。

双点集的公共窗口仍按第一类点数量扩展，但只保存各评估切片最终选中的value目标、原有label目标及必要空间参考点，不保存窗口中间未被使用的第二类点。最终目标并集最多为`(32+1)×2048=67584`个，参考点另计。该修改保留原有选点和候选估计结果；完整Ocean-large density的目标候选从435864降至7523，含参考点从615724降至20823，自动flow内部耗时为0.952秒。该field在单partition、单线程下完成全部960切片，压缩/解压峰值分别为8.85/8.86GiB；按当前Anvil每CPU附带1896MiB内存，需要申请8个CPU，但程序仍只使用1个线程。

完整Ocean-large的`results/ocean_large_tensor_full/run_rel_1e-3.slurm`使用8个CPU附带内存，时限20小时，程序固定单partition、单线程。启动前检查density、layerThickness及vertVelocityTop内存验证的二进制SHA、逐点误差和内存余量；任一缺失或失败就停止。新结果保存在`results/ocean_large_tensor_full/sparse_window/`，不覆盖旧实验，manifest记录代码提交和二进制SHA。

## 2.2. 二维tensor切片DAG

二维tensor预测已经接入正式压缩、解压、自动flow选择和V17码流。输入逻辑上是`[axis0,axis1,unstructured_point]`；程序先把前两个轴的切片展平为一维`slice_id=axis0_index*axis1_size+axis1_index`，再生成切片级DAG。每个切片仍只执行一次已有的一维tensor baseline预测，随后可选地执行非结构空间残差预测，不新增“二维数值预测器”。切片依赖直接保存最多两个参考`slice_id`及其权重，因此非均匀高度轴按真实坐标间距计算权重，不会按展平后的编号计算。

结构化flow最多有33种候选：纯空间1种；只启用一个轴时，轴、外插/anchor8和是否使用空间残差组成8种；两个轴都启用时，两轴模式、先后顺序和空间残差组成16种；两个轴都使用anchor8时，另有锚点建立顺序、逐层内插顺序和空间残差组成的8种交错候选。长度为1的轴不会生成或评估无意义的预测，因此`12×1`field只保留与原一维流程等价的5种候选。空间残差不改变切片依赖，所以成对候选共享依赖计划：完整二维只保存17份不同计划，`12×1`只保存3份。自动选择器使用与一维流程相同的采样目标、正式量化、Huffman和Zstd路径比较这些候选；选中的二维flow、tensor形状和切片依赖解释写入码流，解压端重新生成同一依赖计划并检查每个切片的文件头。同一进程中形状和轴坐标文件相同的field共享候选切片依赖计划，因此完整Ocean-large只分别生成`12×1`、`12×80`和`12×81`三套。

二维自动flow仍固定最多评估32个切片，但不会把二维切片展平后直接等距抽样。程序把切片划分为原点、`axis1=0`边、`axis0=0`边和二维内部四个区域；原点全部测试，两条边各至少测试3个切片，其余预算分配给二维内部。边界在等宽分箱内确定性随机抽样；内部使用二维Latin hypercube，使两个轴的边际位置分别均匀覆盖，再以确定性随机排列配对。所有候选共享同一批样本，四个区域分别按“区域切片数/样本数”加权。主随机种子固定，field名称只用于派生可复现的scramble，因此重复运行结果完全一致，也不会针对实验结果挑选种子。

压缩和解压使用`-tensor-metadata <path>`读取field级tensor定义；该选项与`-time-step`、`-time-step-coordinate-file`互斥。task JSON在最外层记录`"tensor_metadata":"tensor_metadata.json"`，同一个field list可以包含`[12,1]`、`[12,80]`和`[12,81]`等不同形状。metadata版本1示例：

```json
{
  "version":1,
  "fields":{
    "timeMonthly_avg_velocityMeridional":{
      "shape":[12,80],
      "axis0_coordinates":"time_12.f64",
      "axis1_coordinates":"height_center_80.f64",
      "active_level_count_file":"active_level_count.u32",
      "active_level_count_offset":0
    }
  }
}
```

坐标文件是严格递增的float64原始数组；有效层计数文件为每个水平点一个uint32。中心层使用offset 0，界面层使用offset 1。配置有效层后，程序以`height_index<maxLevelCell+offset`判断结构有效性，并把非有限值或显式`-99999`标记归入第二点集；有效区域中的有限0仍是普通数值。完整Ocean-large转换器会在写文件前检查有效有限值是否与`-99999`冲突。该结构标签和现有普通数值点集规则相互隔离，未配置有效层的已有数据行为不变。

时间一阶内插可通过`-time-step-coordinate-file <path>`读取一维float64原始数组，第`i`个数是time step `i`的实际坐标。压缩和解压必须读取同一文件；不传该参数时保持原有的等间距假设。对应task使用`time_step_coordinate_file`记录文件名并同时传给两端，普通等时间间隔数据不需要记录这一项。

## 2.3. 生成坐标变换矩阵

坐标变换已并入C++预处理，不再使用独立的Python训练程序或预先生成的下三角矩阵。task中的`coordinate_transform`支持：

- `"off"`：不训练也不应用坐标变换。
- `"on"`：训练并始终应用坐标变换。
- `"auto"`：根据坐标flatness自动决定是否训练和应用坐标变换。

训练固定使用最多3个均匀分布的time step。每个训练文件先以固定随机种子保留30%的点建立kd-tree，再在这些点中总共选取最多3000个目标；每个目标使用10个近邻形成训练点对。当前只学习各坐标轴的对角缩放，不混合坐标轴，也不执行第二轮重新采样。

对REL任务，每个采样`field×time step`直接复用压缩开始前预计算的对应可预测点集value range来归一化`Δv`，避免重复扫描数据，也避免不同time step的绝对幅度直接决定训练权重。令归一化数值差为`Δv`、原始坐标差为`Δc`。各采样group始终使用`p=3`聚合，使困难group不会被普通group平均掉，并且只执行一轮训练：

$$
L_{\mathrm{diag}}=\min_s\sum_t\left[\frac{1}{N_t}\sum_{i\in t}\left(|\Delta v_i|-\lVert\Delta c_i\odot s\rVert_2\right)^2\right]^p.
$$

训练使用Gauss–Newton法；求解后把`s`归一化，使其平方和等于坐标维数。`auto`模式首先计算原始坐标协方差矩阵的flatness：

$$
f=\sqrt{\frac{\lambda_{\min}}{\lambda_{\max}}}.
$$

当原始flatness不大于`0.05`时确定启用并继续训练；当它不小于`0.75`时直接拒绝，不读取训练数据也不训练。其余情况完成训练，并用`DCD`直接计算变换后协方差，其中`C`是原始协方差，`D`是训练出的对角矩阵。仅当

$$
\frac{f_{\mathrm{transformed}}}{f_{\mathrm{original}}}\ge 1.21
$$

时才应用变换，否则关闭。该阈值统一用于单field和多field；它会拒绝收益不足以抵消额外DAG成本的弱变换。flatness计算、训练和坐标变换都计入预处理时间；坐标和训练数据的文件读取不计时。变换后的坐标用于建树、生成DAG和计算预测权重；数据值本身不变。

多个field共享同一组坐标时可用`-field-list`一次传入。每个field独立训练并判断坐标变换；各field复用只依赖原始坐标的训练采样几何。未启用坐标变换的field共享原始坐标上的空间DAG和RBF权重，启用变换的field生成自己的空间DAG和权重。REL训练按照当前field的每个采样time step分别使用可预测点集value range归一化，每个训练文件形成一个独立group；ABS任务不归一化。单field与多field使用相同的flatness判断、固定`p=3`的group loss和矩阵训练流程。

两级并行模式额外传入`-field-worker-num N`和与`-field-list`等长的`-field-worker-map {w0,w1,...}`。每个编号指定对应field所属的worker。所有worker位于同一进程，未启用坐标变换的field通过`std::call_once`按需构建并只读共享唯一的原始坐标DAG；如果没有field需要它，这个公共DAG不会生成。每个worker内部继续使用`-thread-num`个OpenMP线程处理`-partition-num`个空间partition。正式Ocean-large两级并行入口位于`benchmarks/two_level/`；field映射由`benchmarks/common.py`精确求解，先使各worker的最大slice数最小，再使最大field数最小。

点集配置的`eb_mode`除`REL`和`ABS`外还支持`FIELD_REL`：它用当前field全部slice中该点集的整体range换算同一个ABS EB，而`REL`按每个slice自身的range换算。压缩端在计时前的点集预扫描中得到整体range，解压端在预处理阶段读取全部slice文件头中的点集min/max（文件读取不计时），因此码流格式不变。`FIELD_REL`与`REL`共用同一套预测、自动flow决策上限和坐标变换训练，只是误差上限不同，多个field仍可在一个进程中共享空间DAG。

压缩器把本次训练结果写到数据目录中的`<field>.unsz.coord_transform.txt`。文件以明文记录维数、是否启用和每个坐标轴的缩放值；解压器读取同一文件，不重新训练。例：

```text
UNSZ_COORDINATE_TRANSFORM 3
dimension 3
enabled 1
diagonal 1.31 0.74 0.08
```

`t3_visualize.py`也直接读取该文件，因此可视化和实际压缩使用完全相同的C++训练结果。

## 2.4. 生成DAG

预测前需要根据公共坐标生成一套所有time step都能复用的有向无环图（DAG）。整体分为三步：

1. 使用经过坐标变换的坐标生成kd-tree。
2. 对所有点生成一个预测目标的顺序。
3. 根据这个顺序，对每个预测目标寻找合适的参考点。

### 2.4.1. 生成kd-tree

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

### 2.4.2. 预测目标排序

当前预测目标排序固定为反向MSD，不再把排序方向或评分公式作为可调配置。先假定所有点均可作为参考点，每次只在上一个被剔除点附近的固定数量候选点中，选择MSD最小的点剔除；完成后反转剔除序列，得到正向预测顺序。MSD定义为：

$$
\operatorname{MSD}(c)=\frac{1}{k}\sum_{i=1}^{k}\lVert c-c_i\rVert_2^2,
$$

其中$c$是候选目标坐标，$c_i$是当前仍可用的评分参考点坐标。该分数只依赖坐标和已确定的剔除状态，因此压缩端和解压端可以独立生成相同DAG，无需保存预测顺序。

反向MSD参数已固定在程序内部：全局排序、16个附近候选和1个评分参考点。正式JSON和命令行不再暴露这些调参项。

生成顺序时会缓存每个候选的MSD参考点，只在缓存中的可用点数低于下限时重新查询kd-tree。全局反向1点MSD还会自动复用排序候选查询的几何信息：只有能严格证明候选集内某点是全局最近可用参考点时才省略额外查询，因此不改变DAG。

正向排序、最近参考距离、`S2`、方向集中度、联合评分和参考数据辅助排序等实现及CLI均已删除；被删除实现和历史结论需要时从Git历史恢复。

### 2.4.3. 生成DAG

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

## 2.5. 数值预测、量化与无损编码

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

DAG只保存参考编号和最终权重，不长期保存每条边的参考坐标及目标距离。权重生成按32个目标成批从公共坐标数组重新读取所需坐标。LES单partition测试中，该布局把峰值RSS从`3.84 GB`降到`1.85 GB`，完整预处理时间仅从`108.64 s`变为`108.91 s`，码流与误差结果不变。

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
      <td>RBF预测器</td>
      <td>已采纳</td>
      <td>唯一正式数值预测器，固定使用10个参考点、Exponential核、shape 0.01、nugget 1e-6和0阶常数趋势。</td>
      <td></td>
    </tr>
    <tr>
      <td>批量向量化RBF权重生成</td>
      <td>已采纳</td>
      <td>DAG只长期保存目标和参考编号；权重阶段按32个目标一批，从坐标数组重新读取所需坐标并计算距离，再以SoA布局完成核矩阵、LDLT和Schur补，不再保存每条边的完整临时几何。</td>
      <td>固定批宽为32；不足10个参考点的开头目标使用最近参考点退化方案。批量浮点路径已通过独立解压误差检查。</td>
    </tr>
    <tr>
      <td>提前计算归一化权重</td>
      <td>已采纳</td>
      <td>预处理阶段直接生成RBF权重，压缩和解压热循环只执行固定10点加权和。多点集参考点不全同类时使用最近同类参考点退化方案。</td>
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

## 2.6. 点集与标签

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

## 2.7. 跨time step预测

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

## 2.8. 并行设计

当前OpenMP实现已经支持adjust后的独立partition。程序先按坐标递归中位数切分，只生成顶层partition点集，不建立整场kd-tree；随后每个worker为自己的partition建立局部kd-tree，并独立生成预测顺序、参考关系、权重和EB倍率。每个partition只引用本partition内的数据，单点集以及融合双点集的标签预测、数值预测、量化、Huffman、Zstd和重建均按partition并行。partition本身始终互相独立。V17容器统一使用`payload_begin[P+1]`定位各预测结构的独立payload，并在文件头记录一维label/value flow或二维tensor flow、field级有效点集配置及tensor形状；不分区时P=1，分区时每个partition对应一个payload，最终只按结构编号串行拼接字节。配置为双点集、但整个field只出现一个点集时会自动降级为单点集并省略标签流。REL模式先根据当前完整切片的field range换算统一ABS EB，再由该切片内所有partition共享。正式程序不缓存DAG。

核心partition循环保留OpenMP动态调度：LES p16测试中，静态调度使压缩从`0.1129 s`变为`0.1293 s`，解压仅从`0.0542 s`变为`0.0532 s`。解压恢复原始点序使用“原始编号→预测位置”逆映射，从而连续写输出；LES p16的该阶段中位时间从`0.0347 s`降到`0.0250 s`。

partition会改变空间DAG边界和每段无损编码上下文，因此不保证BR逐partition数单调或完全相同。LES的p1/p4/p8/p16 BR分别为`1.1980/1.1295/1.1508/1.1735`；当前将这一幅度视为可接受的partition代价，不为它引入自动partition调参。

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
      <td>已实现</td>
      <td>预测DAG、标签、数值、Huffman、Zstd、重建和恢复原始点序均按独立partition并行；串行与并行统一使用V17容器。</td>
      <td>补充独占节点上的正式扩展性实验</td>
    </tr>
    <tr>
      <td>每partition独立无损码流</td>
      <td>已实现</td>
      <td>LES的p1/p4/p8/p16 BR分别为<code>1.1980/1.1295/1.1508/1.1735</code>；partition会同时改变DAG边界和无损编码上下文。</td>
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

下表记录当前正式实验使用的数据。完整版Ocean-large采用95个field，其中58个field的tensor形状为`[12,1]`，33个为`[12,80]`，4个为`[12,81]`。

| 数据集 | field数 | 单个field的总大小 | 每步点数 | time step数 |
|---|---:|---:|---:|---:|
| ADCIRC-Haiyan-1.5km depth | 1 | 8.75 MB | 2187832 | 1 |
| Tpost20 | 1 | 10.49 MB | 131072 | 20 |
| Ocean | 9 | 29.16 MB | 235160 | 31 |
| LES | 3 | 50.95 MB | 12737274 | 1 |
| E3SM-h4 | 2 | 62.21 MB | 21600 | 720 |
| TAPS | 7 | 64.01 MB | 16002271 | 1 |
| ADCIRC-Haiyan-1.5km zeta | 1 | 210.03 MB | 2187832 | 24 |
| Katrina | 1 | 270.63 MB | 417642 | 162 |
| Ocean-large单快照restart | 24 | 详情见下方说明 | | |
| Ocean-large月度输出 | 95 | 177.27 MB/14.18 GB/14.36 GB | 3693225 | 12个月×1/80/81层 |

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

月度输出由12个NetCDF文件组成，每个文件对应一个真实time step。同名变量跨12个文件组成一个field。完整cell数据包含58个单层field、33个80个中心层field和4个81个界面层field。转换器按`slice_id=time_index*height_num+height_index`生成切片文件，因此分别形成`[12,1]`、`[12,80]`和`[12,81]`三类tensor形状；共36264个切片、535724445600字节原始float32数据。

### Ocean-large旧91-field单高度口径

UMC公开代码读取91个长度均为`nCells`的float32 field，每个field有12个真实time step。当前可复现实验定义使用58个单层cell变量，并取33个80层cell变量的第0层，正好得到91个单层field；公开仓库没有论文实验使用的`varnames_91.txt`和NetCDF转换程序，因此不能声称该列表与论文逐项相同。

这91个field在UNSZ中使用同一个field list和同一份双点集配置。`timeMonthly_avg_bruntVaisalaFreqML`中的71个缺失值被统一转换为常量标记`-99999`，该field保留“RBF普通值＋常量异常值”两个点集；其余90个field没有该标记，预扫描后自动折叠为空标签流的单点集。所有field因此可以在同一进程中复用坐标采样几何和未变换坐标上的空间DAG。

完整垂直版本由`prepare_ocean_large_tensor.py`转换。深于`maxLevelCell`的结构无效层、NetCDF非有限值和fill value统一写为`-99999`；分类时仍以`maxLevelCell`决定结构有效区域，不能把合法0误判为无效。转换器以一个field的一个月为原子提交单元：先写`.tmp`切片，全部高度切片通过尺寸检查后再替换正式文件，并在`conversion_progress.json`保存统计；中断后重跑会跳过已经完整提交的组。完整95个field、36264个切片和约499GiB数据已经转换完成。二维入口、最多33种flow和结构标签通过了`[2,2]`真实依赖、普通一维V17回归以及三个完整代表field测试：`ssh[12,1]`选择时间anchor8加空间残差，`velocityMeridional[12,80]`选择高度anchor8加空间残差，`vertVelocityTop[12,81]`选择先时间后高度的双轴anchor8且不使用空间残差。最后一项实际BR为`0.532053`，相比旧展平采样误选的纯高度外插BR `0.629549`降低`15.49%`，同时PSNR从`77.5281 dB`提高到`78.3002 dB`；逐点最大误差不超过用户EB。

旧91-field单高度五点结果保存在`results/ocean_large_curves/metrics_*.json`，图位于同目录的`figures/`。五点共同PSNR区间内，UNSZ相对UMC的压缩率几何优势为`2.112×`，核心压缩＋解压速度为`1.882×`，计入两端预处理后速度为`1.687×`。这组结果用于复现UMC式单高度输入，不代表当前95-field完整tensor实验。`run_benchmark.slurm`从头运行UMC与UNSZ；`run_unsz.slurm`复用正式UMC指标并只更新UNSZ，设置`EXPONENTS="1 3"`可选择EB，设置`FORCE=true`可覆盖已经完成的结果。

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

E3SM-h4 AODDUST曾固定使用stride为8的锚点内插；正式配置现改为自动选择flow，表中的既有压缩率和时间待统一复测后更新。

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
      <td>C++内置对角缩放训练；REL按time step range归一化并自动处理困难时间模式，自动flatness判据，第一轮采样比例<code>0.3</code>，第二轮采样<code>3000</code>点，最多使用<code>3</code>个文件</td>
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

`configs/`中的task JSON是当前运行配置的权威记录，3.2节只汇总已经选定或正在调优的方案。结构感知baseline为UMC；SZ3、ZFP和PFPL的一维baseline入口见[benchmarks/tensor_1d/README.md](benchmarks/tensor_1d/README.md)。它们将每个切片按原始点序作为一维float32数组，使用有效值range换算出的ABS EB，保留sentinel并验证全部点，不使用时间、高度或空间结构。当前完整Ocean-large基线固定单线程，UNSZ、UMC、SZ3、ZFP和PFPL均已完成五个REL对应的ABS误差点；源码使用固定Git子模块，内部API计时与输入读取、range计算、误差验证分离。

最新完整Ocean-large对比图见[三组RD](results/ocean_large_tensor_1d/comparison_figures/grouped_rd.pdf)、[核心throughput](results/ocean_large_tensor_1d/comparison_figures/grouped_releb_throughput.pdf)和[REL=1e-3总耗时](results/ocean_large_tensor_1d/comparison_figures/total_releb_1e-3_time.pdf)。按`12×1/12×80/12×81`分组，PSNR先转NMSE按点数平均，throughput不计预处理，总耗时包含两端预处理及全部95个field（不剔除UMC零EB异常慢field）。五个压缩器均使用完整五点；ZFP超界实测点保留并明确标记。结果来自不同作业，不能将时间差直接作为同节点配对加速比。

UNSZ/串行UMC其余四个EB的可复现入口分别是`sbatch results/ocean_large_tensor_full/run_remaining_eb.slurm`和`sbatch results/ocean_large_tensor_full_umc/run_remaining_eb.slurm`。各入口提交一个四任务数组，对应REL `1e-1、1e-2、1e-4、1e-5`；每任务申请8个CPU附带的内存额度，程序仍单线程、UNSZ单partition，时限分别为12/22小时。同一压缩器的数组并发限制为1，防止共享输入目录中的临时码流互相覆盖；UNSZ与UMC可同时运行，结果分目录保存。启动时核对REL=1e-3基准的二进制SHA、metadata、点集规则与UMC计时替换文件，不重编译、不改变UMC参数。manifest记录启动器、实验驱动与相关补丁的SHA；逐点验证、共享DAG及计时口径沿用已有实验。

PT不纳入正式性能对比：UMC论文报告PT的公开Python实现速度极慢。此外，官方C++实现（`rcrcarissa/UnstructuredLossyCompression`，提交`79dd92f4213c4d62020ed0f782671abf4e1993f7`）在36节点、50三角形的小网格ABS测试中，平滑场解压报`Unexpected end of bitstream`，并存在硬编码解码数量。未修改其算法；已移除弃用的子模块、测试入口及临时结果。

单点集RD曲线固定使用REL `1e-1、1e-2、1e-3、1e-4、1e-5`五点；多点集RD曲线由`point_set_config_path_list`中的完整误差配置构成，每个点集可以使用不同EB。UNSZ与UMC曲线不要求使用相同EB，而是在共同PSNR范围内对`log(BR)`和`log(time)`作分段线性插值后比较；BD-rate为`log(BR)`差的区间积分均值取指数，因而不受额外插入EB采样点影响。UMC合并多个time step的MSE时所需的数据范围，在每个task的每个field开始实验前计算一次，并传给该field的全部UMC实验，不随REL EB重复读取输入。速度统计压缩与解压的程序内部核心时间之和；压缩端与解压端的完整预处理时间分别记录。正式实验中两个压缩器都不保存预测DAG，压缩端和解压端分别现场重新生成各自的预测结构；UNSZ压缩预处理包含坐标变换训练与应用、DAG生成和RBF权重计算、自动flow选择、时间依赖计划生成以及最终flow的时间依赖和历史容器初始化。解压预处理读取可复用的坐标变换参数，并包含码流头解析、时间依赖计划和历史容器初始化、坐标变换应用、DAG生成和RBF权重计算。坐标、connectivity、坐标变换训练数据、自动flow采样数据、压缩码流头及坐标变换参数文件的读取不计时，输入正确性验证、逐点误差统计和输出文件写入也不计时。坐标变换参数是field级可复用预处理结果，不计入每个压缩码流的大小。正式倍率必须来自同一节点上的UNSZ与UMC交替运行。

UMC统一使用其正式的`reorder=1`（DPFS重排）。NBP和NBP-INP采用同一计时口径：压缩时间为field重排与核心压缩之和，解压时间为核心解压与恢复原始顺序之和；读取输入、写出结果、读取原始field进行误差验证和误差统计均不计入。ADCIRC-Haiyan-1.5km的connectivity包含两个连通分量，而UMC的DPFS只遍历0号点所在的连通分量，无法生成覆盖全部点的有效重排。我们不修改UMC算法，也不改用`reorder=0`；因此Haiyan汇报为UMC无法运行，不计算`CR/UMC`和`总时间/UMC`。

数据转换入口见`py_codes/prepare_datasets/prepare_*.py`。常规结果以`configs/`和`results/current`为准；旧91-field单高度Ocean-large结果位于`results/ocean_large_curves`，完整95-field tensor结果分别位于`results/ocean_large_tensor_full`、`results/ocean_large_tensor_full_umc`和`results/ocean_large_tensor_1d`。

# 4. TODO

## 4.1. 投稿前主线

- [x] 完成完整95-field Ocean-large的五点BR–PSNR、核心压缩/解压时间、预处理时间和分组图，并保存实验manifest与脚本SHA。
- [ ] 常规数据的25个field和五个EB结果已经齐全；投稿前用最终程序统一复测一次，并保存运行日志、计算节点、程序版本和输入版本。
- [ ] 整理`results/current`，使其目录和`configs/`中的任务名称一一对应；Ocean-large结果继续单独维护。
- [ ] 优先优化仍未达到目标的正式field，方案固定后再迁移到其他field验证泛化。
- [ ] 测试一阶内插中“最后一个文件由第一个文件作零阶外插预测”，并实现“空间预测＋时间补偿”。
- UMC-INP的batch固定为README示例使用的10，不再作为task配置或实验维度。
- [ ] 研究DAG和RBF权重的紧凑持久化，明确公共metadata需要在多少field或time step之间摊销。
- [ ] 取得UMC论文实验原始的`varnames_91.txt`以核对当前58个单层变量＋33个多层变量第0层的91-field定义；12个月度NetCDF已经按当前定义转换，单快照restart不作为月度输出使用。
- [x] 完成完整Ocean-large的single-core throughput与REL=1e-3总耗时实验。
- [ ] 复测当前strong-scaling结果，并完成weak scaling以及端到端压缩—传输—解压实验。
- [ ] 基于3DAC官方单GPU代码测量若干训练iteration或epoch，估算ScanNet完整20轮及多个量化步长的总训练成本；该结果用于Related Work讨论，不作为正式baseline。

日常审阅以本README为准；被删除实现、历史实验和一次性脚本从Git历史恢复。
