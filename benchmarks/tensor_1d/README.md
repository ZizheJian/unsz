# 一维tensor基线

本入口与UNSZ/UMC源码隔离。每个`field.dat.slice_id`按原始点序作为一个float32一维数组，不读取空间坐标、连接关系或时间/高度预测DAG。官方SZ3、ZFP、PFPL由固定Git子模块提供。

```bash
cmake -S benchmarks/tensor_1d -B build/tensor_1d -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build/tensor_1d --parallel 2 --target benchmark_sz3 benchmark_zfp benchmark_pfpl
sbatch --job-name=ocean_1d_sz3 benchmarks/tensor_1d/run.slurm sz3
sbatch --job-name=ocean_1d_zfp benchmarks/tensor_1d/run.slurm zfp
sbatch --job-name=ocean_1d_pfpl benchmarks/tensor_1d/run.slurm pfpl
```

默认测试完整Ocean-large的95个field、全部36264个切片、REL=1e-3、单线程。各作业申请1个CPU附带的内存、4小时时限；接口预留`--threads`，但此次不启动并行扩展实验。SZ3使用上游默认Interpolation/Lorenzo自动选择；ZFP使用fixed-accuracy；PFPL调用上游ABS编码/解码函数，使用其官方编译选项`-mno-fma -ffp-contract=off`。

当前单线程实验运行SZ3、ZFP和PFPL。ZFP纳入单线程RD与throughput比较；因官方ABS模式缺少CPU并行解压，暂不纳入原生并行扩展比较。

补跑另外四个REL EB（1e-1、1e-2、1e-4、1e-5）：`sbatch --job-name=ocean_1d_sz3_4eb benchmarks/tensor_1d/run_remaining_eb.slurm sz3`，PFPL/ZFP分别替换名称和最后的codec参数。每个codec的四个数组任务依次运行，每任务单核，二进制SHA必须与已完成1e-3版本相同。PFPL可用`--time=03:00:00`覆盖默认每任务4小时限制。各EB保存在独立目录，不覆盖1e-3。ZFP误差超界仍保留BR、PSNR和时间，并以`valid=false`及非零退出码明确标记；一个数组任务的超界不会阻止其他EB启动。非零退出码不能直接解释为结果缺失，必须检查完成field和slice数量。

每个切片以排除`-99999`后的有限值range计算ABS EB，再传给三者的ABS接口。输入仍完整包含`-99999`，不额外拆分label、不重排、不做后置Zstd。解压后逐点检查全部值，包括sentinel；误差超过1.01×ABS EB的结果标记无效。按当前实验约定，ZFP的失败结果仍保留在实测RD、throughput和时间图中，并明确标记超界，不能宣称满足ABS误差保证。

ABS EB为0时必须精确恢复：SZ3使用官方lossless模式，ZFP使用官方reversible模式，PFPL通过其protected-values阈值保留所有normal值及零；PFPL此模式不接受subnormal输入，遇到时明确失败，不静默改变误差要求。PFPL上游单次输入大小限制约2GiB，当前每个切片约14.8MB，不受影响。

计时在C++内部包围一次正式内存编码/解码调用，包括API中的必要元数据、内部缓冲分配等；读取、range/ABS EB计算、调用方重建缓冲分配和误差统计排除。不使用PFPL命令行默认的9次重复测速，也不以进程墙钟计算核心吞吐率。码流保留在内存中并立即解压验证，不写出完整压缩/重建文件。此基线没有可跨切片复用的几何预处理，预处理列为0。

输出位于`results/ocean_large_tensor_1d/<codec>/rel_0.001_t1/`，包括逐切片CSV、field汇总JSON、summary CSV、失败列表、日志及二进制快照。manifest记录输入metadata哈希、各上游提交和二进制SHA；不同版本禁止混合。失败的field仍保留诊断结果，随后继续其他field。

Git保存各field汇总JSON、summary CSV、版本manifest、诊断结论和最终图表。逐切片`*.slices.csv`、可重新生成的切片路径清单`*.inputs.txt`、日志及二进制快照保留在本地但不上传；重新运行相应codec/EB会生成这些文件。`tests/zfp_double_failed_fields.py`的完整切片诊断依赖本地逐切片CSV和路径清单，不能仅凭仓库汇总表运行。

官方并行能力：SZ3支持OpenMP压缩/解压；PFPL提供OpenMP CPU及CUDA GPU实现；ZFP的fixed-accuracy CPU路径支持OpenMP压缩，但解压使用serial执行策略。具体行为以固定子模块源码为准，不把CPU与GPU结果混在同一条曲线中。

## ZFP float误差诊断

完整Ocean-large实验发现部分field不满足ABS误差界，失败行保留并在经验RD/throughput图中标记，不能视作有效的bounded-error结果。独立诊断程序`tests/zfp_error_diagnostic.cpp`直接调用固定上游版本的API，并检查编码与解码使用同一误差参数；保留/去掉码流头得到相同结果。已用四点块复现两种情况：

- 大动态范围：`[-99999,1e-4,2e-4,3e-4]`在float fixed-accuracy、ABS EB=1e-7时恢复为`[-99999,0,0,0]`，约3000×EB。这是块内共用指数造成的精度限制，与上游FAQ关于sentinel的说明一致。
- 极小normal值：`[0,0,0,1.8797537051e-30]`、ABS EB约1.87975e-33时，四个值均恢复为约-6.31089e-30。当前`src/template/encodef.c`先以float计算`2^(30-emax)`作为缩放因子；emax=-98时该因子为`2^128`，溢出为Inf。该真实切片最坏块误差为4357.29×EB，不是subnormal输入，也不是sentinel污染。

两类真实失败块改用double fixed-accuracy或float reversible均通过；这只是诊断，没有改变正在运行的float正式基线，也不能据此保证double版本对全部field均正确。

```bash
g++ -O3 -DNDEBUG -march=native -std=c++17 -I third_party/zfp/include tests/zfp_error_diagnostic.cpp build/tensor_1d/zfp/lib64/libzfp.a -fopenmp -lm -o /tmp/unsz_zfp_error_diagnostic
/tmp/unsz_zfp_error_diagnostic
```

可传入一个真实float32切片路径；程序按相同规则计算ABS EB、找到最大误差所在的四点块，再比较float、double、reversible和不保存码流头的路径。不修改输入文件或上游代码。

后续整切片double诊断：从当前已完成的31个失败field各选择原float误差比最大的一个切片，使用原ABS EB执行double编码/解码，并转回float检查全部点。18个切片通过严格1×EB检查，13个仍失败；这不代表18个field的其他切片已通过。剩余失败均包含正常值超界，而不是仅sentinel恢复误差；其中`temperatureShortWaveTendency`仍约1003.19×EB，`activeTracerSurfaceFluxTendency_temperatureSurfaceFluxTendency`约276.77×EB。因此不能直接把float正式基线替换为double并宣称正确。

复现：编译上述诊断程序后执行`python tests/zfp_double_failed_fields.py`。结果保存在`results/ocean_large_tensor_1d/zfp_double_worst_slices.json`，不覆盖正式float结果；这只是正确性实验，不报告吞吐率。并行外壳尚未实现。

### 剩余13个切片的进一步定位

`tests/zfp_double_investigate.py`对13个double仍失败的完整切片定位最坏四点块，并比较原EB、EB/2、EB/4、EB/8、reversible、仅诊断时将sentinel改零，以及1e-300容差强制编码到64位精度上限。另对每个完整切片测试EB/2，最终误差仍按原EB判断。

- 10个属于默认`ZFP_ROUND_NEVER`粗量化路径：最坏块不含sentinel，所有原值幅度小于EB，但解码可将一个小负数变成正数，误差达1.219～1.398×EB。独立四点块重现相同结果；EB/2可使这10个完整切片通过。更直接的验证是仅使用官方`ZFP_ROUND_FIRST`构建选项、保持原EB：这10个完整切片全部通过，最大误差范围0.451～0.666×EB。这里是低精度系数量化及重构误差，不是float写回舍入或sentinel污染。
- 3个属于sentinel造成的动态范围精度上限：`activeTracerSurfaceFluxTendency_temperatureSurfaceFluxTendency`约276.77×EB，`temperatureShortWaveTendency`约1003.19×EB，`activeTracerNonLocalTendency_salinityNonLocalTendency`约19.73×EB。最坏块均混有-99999；即使容差收紧到1e-300也不改善。double lossy将块映射到64位整数，-99999对应emax=17，此时整数映射步长约2^(17-62)=2^-45；远小于此步长的有效值不能保留。最坏块将sentinel临时改零后均通过原EB，reversible也全部精确恢复。这些改零测试仅用于定位，不是正式输入变更，也没有实现mask适配。

诊断编译在`/tmp`，不覆盖正式二进制和结果，不修改上游源码：

```bash
python tests/zfp_double_investigate.py
cmake -S third_party/zfp -B /tmp/unsz_zfp_round_first_diag -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DBUILD_UTILITIES=OFF -DBUILD_TESTING=OFF -DZFP_WITH_OPENMP=OFF -DZFP_ROUNDING_MODE=ZFP_ROUND_FIRST -DCMAKE_C_COMPILER=gcc
cmake --build /tmp/unsz_zfp_round_first_diag --target zfp --parallel 2
g++ -O3 -DNDEBUG -march=native -std=c++17 -I third_party/zfp/include tests/zfp_error_diagnostic.cpp /tmp/unsz_zfp_round_first_diag/lib64/libzfp.a -lm -o /tmp/unsz_zfp_round_first_diagnostic
python tests/zfp_double_investigate.py --binary /tmp/unsz_zfp_round_first_diagnostic --output results/ocean_large_tensor_1d/zfp_double_round_first_investigation.json
```

结果只覆盖这些代表切片，不能据此声称所有field、所有切片均通过，也没有测量上述适配的速度代价。

## 绘图

`unsz_env/bin/python benchmarks/tensor_1d/plot.py`核对三个codec各五个EB的95个field、36264个切片、输入规模、metadata哈希和每个codec的二进制一致性，输出到`results/ocean_large_tensor_1d/figures/`。默认按`12×1/12×80/12×81`分为58/33/4个field的三组，生成三面板的`grouped_rd`、`grouped_releb_throughput`及各组独立PNG/PDF；REL=1e-3耗时只看全部95个field的总压缩/解压分段柱。可用`--metadata`指定metadata路径，但文件SHA必须与实验记录一致。

每组BR使用总压缩字节数×8除以总点数；各field统一PSNR先转为`NMSE=10^(-PSNR/10)`，按点数加权平均，再用`-10log10(mean NMSE)`得到`Aggregated normalized PSNR`。组内每个field点数相同，因此BR与NMSE均等价于field算术平均，不能直接平均PSNR。精确恢复field贡献零NMSE，不剔除其点数、码流或耗时。RD的BR为log scale；throughput从0开始，使用总输入字节数除以总核心时间，不平均单field吞吐率。所有field均参与汇总，ZFP任一field超界时在对应组点上标叉号，总时间柱使用交叉阴影，不宣称超界结果满足ABS误差保证。

这三种一维codec的几何预处理均为0。PNG/PDF之外保存逐field指标、全量汇总指标、`grouped_metrics.csv`和包含全部源CSV哈希、形状分组及公式的绘图manifest。加`--per-field`可重新生成完整95-field RD/throughput及分field耗时图，默认不再重画，但保留已有完整图用于补充材料。单元测试：`unsz_env/bin/python -m unittest discover -s tests -p test_tensor_1d_plot.py`。

`unsz_env/bin/python benchmarks/tensor_1d/plot.py --include-structured`将当前UNSZ、UMC完整95-field的REL=1e-3结果叠加到相同三组，输出到独立的`comparison_figures/`，不覆盖三baseline图。UNSZ/UMC目前只绘制一个点，不补值、不连接成完整曲线；通过`--unsz-results/--umc-results`可以指定结果目录，必须与baseline的每个field输入字节数和切片数一致，并满足单partition、单线程、REL=1e-3配置。UNSZ的REL从记录的point-set配置读取并核对，码流大小由记录BR换算，保留其六位小数精度，不伪造精确字节数；两个结构感知压缩器的逐field最大误差比没有保存到CSV，指标表以NaN表示未知，不伪造数字，重建正确性已经由各自实验入口验证。

五codec总耗时图包含压缩预处理、压缩、解压预处理、解压四段，所有field均计入。UNSZ共享原坐标空间DAG只计算一次；当前UMC串行启动器按field分别预处理。吞吐率仍只使用核心时间。结果来自独立作业而非同节点交替测速，因此不把图中的时间差直接宣称为受控的同节点加速比。记录和绘图不修改UMC源码、参数，也不剔除其零EB异常慢field。
