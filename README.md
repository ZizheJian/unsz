# UNSZ

UNSZ是面向非结构化科学数据的误差有界有损压缩器。它只需要每个点的坐标，不需要mesh connectivity：压缩端和解压端用同样的坐标各自生成预测DAG，再对预测残差做量化、Huffman编码和Zstd压缩。数据可以包含多个slice（时间步，或二维tensor的切片），所有slice共享同一组点坐标。

算法设计和实验结论以论文为准（`论文/`）。本文件只说明如何构建和运行程序、如何复现实验，以及代码的组织方式。

# 1. 构建

需要CMake、支持C++17和AVX2的C++编译器以及Python 3。C++依赖（Zstd和参数解析库）和各基线压缩器都以submodule提供。

```bash
git clone --recurse-submodules git@github.com:ZizheJian/unsz.git
cd unsz
cmake -S c_codes -B build -DCMAKE_INSTALL_PREFIX="$PWD/install" -DCODE_NAME=s1_compress
cmake --build build --parallel 8
cmake --install build
```

已有克隆时先执行`git submodule update --init --recursive`。生成的程序为`install/bin/s1_compress`和`install/bin/s1_decompress`。

Python环境用于批量实验、结果处理和画图：

```bash
python3 -m venv unsz_env
source unsz_env/bin/activate
python -m pip install numpy scipy matplotlib pyvista
python -m pip install -e third_party/paperfig
```

# 2. 使用方法

## 2.1. 输入格式

程序读取float32坐标和float32数据。一个数据目录至少包含：

| 文件 | 内容 |
|---|---|
| `coords.dat` | 按点连续保存的坐标，每个点`dim`个float32 |
| `<field>.dat.<slice>` | 一个slice的float32数据，每个点一个数值，点的顺序与`coords.dat`相同 |

例如三维、31个slice的temperature数据：

```text
Ocean/
├── coords.dat
├── temperature.dat.0
├── ...
└── temperature.dat.30
```

## 2.2. 压缩与解压

```bash
install/bin/s1_compress -data-dir /path/to/Ocean -field temperature -slice '[0,30]' -M REL 1e-3
install/bin/s1_decompress -data-dir /path/to/Ocean -field temperature -slice '[0,30]' -M REL 1e-3
```

码流写为`<field>.dat.<slice>.unsz`，解压结果写为`<field>.dat.<slice>.unsz.bin`，默认放在数据目录，可用`-output-dir`改到别处。程序打印总BR、PSNR、最大误差与误差上限之比，以及两端的预处理时间和codec时间。码流不保存误差上限，解压时必须给出与压缩时相同的`-M`或标签配置。

| 选项 | 含义 |
|---|---|
| `-data-dir`、`-output-dir` | 数据目录；输出目录，默认与数据目录相同 |
| `-field`、`-field-list` | 一个field，或`{a,b,c}`形式的多个共享坐标的field |
| `-slice` | 闭区间`[a,b]`、右开区间`[a,b)`或列表`{a1,a2,...}`，建议加引号 |
| `-slice-coordinate-file` | 非等间距slice的实际坐标，float64原始数组，两端必须相同 |
| `-tensor-metadata` | 二维tensor的field定义（2.4节），与`-slice`互斥 |
| `-coordinate-dim` | 坐标维数，默认3 |
| `-M <REL或ABS> <eb>` | 没有异常值时的误差上限，与`-label-config`二选一 |
| `-label-config` | 标签配置JSON（2.3节） |
| `-partition-num`、`-thread-num` | 把点划分为2的幂个独立partition，并用OpenMP线程并行处理 |
| `-field-worker-num`、`-field-worker-map` | 把多个field分给多个worker并行处理（2.5节） |
| `-value-pipeline`、`-label-pipeline`、`-coordinate-rescaling` | 覆盖自动选择的cross-slice pipeline和坐标缩放，只用于诊断和消融实验。数值pipeline可选`independent`、`extrapolation`、`interpolation`、`anchor_interpolation`，后三者可加后缀`_spatial_residual`；标签pipeline可选`independent`、`extrapolation`、`interpolation_spatial_residual`或`anchor_interpolation`；坐标缩放可选`off`、`on`或`auto` |
| `-compression-stage-time`、`-decompression-stage-time` | 打印逐阶段耗时 |
| `-skip-decompressed-output` | 解压时只做逐点误差验证，不写结果文件 |

## 2.3. 标签配置

不传`-label-config`时，全部数据都是有效值，误差上限由`-M`指定。数据含有fill value等异常值时，用标签配置把点分成有效值（标签0）和异常值（标签1），例如Katrina：

```json
{
  "version":3,
  "labels":[
    {"method":"predict","eb_mode":"REL","eb_value":0.01,"condition":{"operator":">=","value":-1000}},
    {"method":"constant"}
  ]
}
```

`labels`的下标就是标签编号。程序依次测试除最后一个以外的`condition`，第一条成立的条件决定标签，全都不成立时属于最后一个标签，因此最后一个省略`condition`。比较符为`<`、`<=`、`>`、`>=`、`==`或`!=`。`method`为`predict`（RBF预测）或`constant`（常量编码）。`method`为`predict`的标签必须给出`eb_mode`和`eb_value`：`ABS`直接作为绝对误差上限，`REL`乘以该slice中这个标签的值域，`FIELD_REL`乘以整个field中这个标签的值域。配置了两个标签、但某个field只出现一个标签时，程序自动按单标签压缩。

## 2.4. 二维tensor数据

Ocean-large这类`[time,height,point]`数据用`-tensor-metadata`描述。第一个轴的下标为`i`、第二个轴的下标为`j`的切片保存在`<field>.dat.<i*shape[1]+j>`。同一个field list可以混合不同形状：

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

坐标文件是严格递增的float64数组。有效层计数文件为每个水平点一个uint32，中心层用offset 0，界面层用offset 1。配置有效层后，`height_index<maxLevelCell+offset`以外的点、非有限值和`-99999`都归入标签1（异常值）。

## 2.5. 多field与两级并行

共享坐标的多个field可以用`-field-list`在一个进程中压缩，不做坐标缩放的field共用同一个空间DAG。再加上`-field-worker-num N`和与field list等长的`-field-worker-map {w0,w1,...}`，就把field分给N个worker并行处理，每个worker内部仍按`-partition-num`和`-thread-num`并行。Ocean-large的worker分配由`benchmarks/common.py`求解。

# 3. 实验

## 3.1. 中等规模数据集

`configs/`中每个JSON描述一个数据集的实验，`t1_start_c_code.py`读取它，构建并运行UNSZ和UMC，把结果写入CSV：

```bash
./unsz_env/bin/python t1_start_c_code.py katrina
```

JSON最外层是task list。每个task必须有唯一的`name`、绝对路径`data_dir`和结果路径`csv_path`（多个field时用`{field}`占位）。`field_list`和`compressor_list`是穷举维度。没有异常值的task用`rel_eb_list`列出误差上限，有异常值的task在`unsz.label_config_path_list`中列出完整的标签配置。UMC对每个标签配置中出现的REL EB分别运行，作为同PSNR比较的基线。

```json
[
  {
    "name":"les_velocity",
    "data_dir":"/path/to/LES",
    "csv_path":"results/current/fields/les_{field}.csv",
    "field_list":["velocity_x","velocity_y"],
    "rel_eb_list":[0.1,0.01,0.001,0.0001,0.00001],
    "compressor_list":["unsz","umc"],
    "slice_num":1,
    "coordinate_dim":3,
    "unsz":{},
    "umc":{"from_edge":0,"reorder":1,"method":"NBP-IDW","neighbor_num":4}
  }
]
```

| 脚本 | 作用 |
|---|---|
| `t1_start_c_code.py` | 实验入口，默认先增量构建，复用已有安装时加`--skip-build` |
| `py_codes/run_configurations.slurm`、`py_codes/run_spatial_field_benchmark.slurm` | 在计算节点上运行一个或多个`configs/`中的JSON |
| `py_codes/parse_unsz_output.py`、`py_codes/parse_umc_output.py` | 解析压缩器输出 |
| `py_codes/summarize_current_results.py` | 逐field计算UNSZ相对UMC的压缩比和耗时倍率，写入`results/current/summary.csv` |
| `py_codes/plot_current_benchmarks.py` | 画中等规模数据集的率失真图和耗时图 |
| `py_codes/prepare_datasets/` | 把各数据集的发布格式转换为2.1节的输入格式 |
| `t3_visualize.py` | 可视化Katrina和Ocean坐标缩放前后的field |

UMC使用`third_party/umc`中的官方代码，`third_party/build_umc.sh`替换其中的计时文件后构建，固定使用`reorder=1`。ADCIRC-Haiyan的mesh有两个连通分量，UMC的重排只遍历一个分量，因此报告为UMC无法运行。

## 3.2. Ocean-large

Ocean-large的实验入口在`benchmarks/`下，各目录的README说明运行方式：

| 目录 | 内容 |
|---|---|
| `benchmarks/two_level/` | UNSZ的field worker加partition两级并行 |
| `benchmarks/umc_field_parallel/` | UMC按field并行，每个worker运行串行UMC |
| `benchmarks/slice_parallel/` | SZ3-1D和PFPL按slice并行 |
| `benchmarks/field_relative/` | 按field误差上限比较UNSZ、SZ3-ND和QoZ-ND |
| `benchmarks/scaling/` | 不同核数下的扩展性 |
| `benchmarks/tensor_1d/` | 一维基线压缩器的构建与单线程运行 |

`py_codes/plot_ocean_large.py`读取这些结果并画论文中的Ocean-large图。

# 4. 代码结构

C++代码在`c_codes/`。`s1_compress.cpp`和`compress_cli.cpp`是压缩器入口，`s1_decompress.cpp`是解压器入口，两者共用`cli_common.hpp`中的命令行解析和field worker调度。核心实现在`c_codes/unsz_core/`，`unsz.hpp`定义全部数据结构，开头有代码术语与论文术语的对照表：

| 文件 | 内容 |
|---|---|
| `compress_entry.tpp` | 压缩的整体流程：逐field预处理，再逐slice压缩 |
| `coordinate_rescaling.tpp` | 坐标缩放的训练与启用判断 |
| `tree.tpp`、`partition.tpp` | kd-tree和空间partition |
| `generate_prediction_structure_and_weights.tpp` | 反向MSD消去，生成预测顺序和参考点 |
| `prediction_weights.tpp` | 批量求解RBF权重 |
| `eb_schedule.tpp` | 按预测位置调整误差上限 |
| `automatic_pipeline_selection.tpp` | 自动选择跨slice预测方案 |
| `prediction_framework.tpp` | 每个slice的预测方式，以及把它转换成模板参数的分发函数 |
| `init.tpp`、`tensor_dependency.tpp`、`tensor_prediction.tpp` | slice的处理顺序、跨slice参考关系和跨slice预测 |
| `label_config.tpp` | 标签配置与误差上限换算 |
| `compress.tpp`、`decompress.tpp` | 逐点预测、量化和重建 |
| `spatial_prediction.tpp` | 用RBF权重做空间预测 |
| `compression_format.tpp` | 码流的编码、拼装和解析 |

修改C++代码后，用逐字节回归测试确认正式路径的输出不变。它用参考二进制和新二进制分别压缩并解压15个固定用例，比较全部码流和解压结果的SHA256。输出文件默认写到`$SCRATCH/unsz_regression`，可用`--work-dir`改到别处：

```bash
./unsz_env/bin/python tests/regression/run_regression.py --binary-dir build/regression_new/install/bin --manifest results/regression/candidate.json --compare results/regression/reference.json
```

`results/regression/reference.json`记录的是清理代码前的参考输出。
