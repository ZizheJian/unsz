# UNSZ详细实验状态与TODO

面向用户的简要结论和待办见`TODO.md`；本文件保留实验参数、具体数字、历史方案和实现细节，供后续复查。

标记：`[x]`已完成，`[ ]`待完成。已弃用实现保存在`c_codes/trash`，这里主要记录结论和后续决策，避免重复保存完整实验日志。

已完成的历史Python实验统一归档在`py_codes/trash`，对应CSV统一归档在`py_codes/trash/results`。下文只写文件名的旧实验脚本和根目录CSV均指这两个目录；当前入口仍保留在项目根目录。

## 已完成：更多数据与field首轮筛选（2026-09-06）

- [x] 为Tpost20、E3SM-h0、E3SM-h4、ADCIRC Haiyan、Tsunami、TAPS和Ocean-large分别提供独立Python转换程序，共享工具位于`third_party/prepare_dataset_common/prepare_dataset_common.py`。AODVIS中的481个NaN转换为`-99999`；所有转换文件均检查长度和值域。旧Ocean-large C++转换器归档于`c_codes/trash/prepare_ocean_large.cpp`。
- [x] 每个field使用REL `1e-1～1e-5`五点曲线，分别测试原坐标和按当前二次方loss、4训练近邻、4外层迭代重新训练的空间矩阵；各EB都执行独立磁盘解压和误差检查。旧入口`screen_more_fields.py`及其Slurm array已归档，配置按数据集拆分到`configs/`并由`t1_start_c_code.py`统一加载。
- [x] E3SM-h0原坐标UNSZ在8个field上相对UMC降低`11.325%～18.479%` BD-rate。矩阵只让TAUX/PRECC/FLNS/CLDTOT/U10改善`0.534%～0.683%`，AODVIS近似不变，并使PS/TREFHT分别退化`5.505%/7.048%`。Tsunami bathy接近持平；height和Haiyan depth的原坐标UNSZ相对UMC分别退化`3.927%/1.665%`，矩阵没有挽回。
- [x] TAPS六个新增field没有可用于UMC的connectivity，只比较矩阵和原坐标。矩阵在C11H22、VELOCITY_X、CO2、RMS_T和TKE分别降低`13.186%/9.731%/5.300%/3.345%/3.014%` BD-rate，PRESSURE增加`0.354%`，平均/中位变化为`-5.704%/-4.322%`。核心压缩均值为`23.17/23.16 MB/s`，解压为`31.14/31.31 MB/s`，矩阵不增加逐文件成本。
- [x] Slurm数组`20450907`全部成功。Haiyan任务用时`1:26`、峰值`0.92 GB`；TAPS任务用时`11:51～13:01`、峰值`4.56～4.64 GB`。TAPS每个field的adjust训练约`21～25 s`，两套DAG的轮内排序各约`152～168 s`，是主要离线成本。
- [ ] 新数据进一步确认adjust coord必须允许field级关闭或选择不同目标。后续选择一个对当前UNSZ较弱但规模适中的多time step数据作为优化对象；最终固定方案仍需满足逐文件核心压缩不超过UMC两倍的约束。

## 已完成：adjust coord损失与训练加速（2026-08-28）

- [x] 已取消尚未开始的Slurm任务`20194817`，队列查询为空；这次暂停没有产生需要续跑的进程或不完整结果。
- [x] Slurm任务`20194962`完成吸收率短实验。脚本调用`t26_benchmark_adjust_coord_weight.py`，固定二次方损失、`first_sample_ratio=0.25`、`second_sample_ratio=0.10`、3个训练文件、5个外层迭代，并比较`new_matrix_weight=0.3/0.5/0.7/1.0`。每轮保存矩阵，再以原四次方100轮矩阵的`PSNR=59.868332 dB`为目标计算同PSNR BR。
- [x] 较大weight确实使最佳轮次提前，但不是越大越好：`weight=0.3`第5轮为`BR=0.212661,7.70 s`；`0.5`第4轮达到当前最低`BR=0.211122,7.89 s`，第5轮回升到`0.214531`；`0.7`第3轮为`BR=0.213701,4.01 s`，第4轮开始退化且第5轮用满`nfev=100`；`1.0`第2轮为`0.219097`，第3轮骤升到`0.383301`。这验证了更大weight会更快进入方向不明确的阶段，`0.5`是当前效果候选，`0.7`是速度候选，`1.0`应排除。
- [x] Slurm任务`20195099`完成采样与训练文件数缩减实验。第二次采样率从`0.10`降到`0.05`后，点对数从156620降到78310，第4轮训练时间从`7.89 s`降到`4.01 s`，同PSNR BR从`0.211122`升到`0.213638`，即训练约加速1.96倍、BR增加1.19%，可作为速度优先候选。训练文件数从3降到2或1后，点对数分别为104415和52210，第4轮时间为`5.24/2.98 s`，BR为`0.234319/0.215681`；2文件明显退化，1文件虽接近但仍比3文件高2.16%，暂时保留3个训练文件。
- [x] 新增同PSNR下的分流长度统计。3文件、第二次采样`0.10`、`weight=0.5`从第1轮到第4轮时，标签Zstd流由12391增至17040字节，而数值Zstd流由25809降至15755字节；总BR改善来自数值流下降超过标签流上升。第二次采样`0.05`第4轮对应15728/17461字节，说明它的标签几何稍好但数值拟合稍差。
- [x] Slurm任务`20195226`完成对数平方loss的`max_nfev=20/30`实验，固定第一次采样`0.25`、第二次采样`0.10`、`weight=0.5`和3个训练文件。20次上限的第1轮训练约`3.37 s`、同PSNR BR为`0.217749`，之后第2～5轮为`0.219103/0.232927/0.226503/0.231452`；30次上限第1轮实际使用21次求值、约`2.98 s`、BR为`0.218407`，之后也整体退化。20和30的5轮总训练约`12.59/16.86 s`，远低于旧对数平方20轮的`424.43 s`。更充分求解代理目标没有改善真实BR，暂不实现L-BFGS；对数平方loss保留为可切换基线，但当前效果弱于二次方。
- [x] Slurm任务`20195305`在Katrina全部162个time step、5个REL EB上完成矩阵RD验证，统一使用反向D-only顺序和默认`round_point_ratio=0.8`，15组结果均通过独立解压PSNR检查。相对原四次方100轮矩阵，二次方效果优先矩阵`second=0.10,weight=0.5,iteration=4`的BD-rate为`-12.181966%`，速度优先矩阵`second=0.05,weight=0.5,iteration=4`为`-8.840362%`。效果矩阵的标签流由1558073降至858195字节，减少44.9%；速度矩阵进一步降至796681字节，但数值流和PSNR更差。默认Katrina矩阵已切换到效果优先矩阵，旧四次方矩阵和速度优先矩阵保留在源码注释中。
- [x] 默认矩阵切换后完成Release编译、安装和Katrina第0步磁盘闭环验证。`REL=0.01`、反向D-only顺序下压缩端和解压端PSNR均为`59.487443 dB`，最大误差为用户EB的`0.999226`倍，BR为`0.209059`。
- [x] Slurm任务`20196003`和`20196215`完成预测误差估计码长loss实验。目标使用当前C++一致的10个反比例参考点，将预测误差按各文件abs EB归一化，以固定广义高斯形状参数`β=0.5`拟合每个文件的尺度参数，并最小化量化区间概率对应的估计`-log2(P)`。第一次采样`0.25`、第二次采样`0.10`、3个训练文件、`weight=0.5`、`max_nfev=20`时，第4轮训练约`13.28 s`。全部162步、5个EB的BD-rate相对旧四次方矩阵为`-12.509730%`，相对当前二次方默认矩阵为`-0.377952%`；第3轮相对当前默认反而为`+0.260986%`。第4轮标签流由当前默认的858195增至874275字节，但数值流下降得更多。收益只有约0.38%且训练慢约1.68倍，暂存为效果候选，不立即替换默认矩阵。
- [x] Python训练入口新增`-actual-ref-num`，adjust coord训练默认值设为4；C++数值与标签预测的`actual_ref_num`仍默认是10，两者含义不同。新的NPZ历史同时记录参考点数、距离指数和广义高斯形状参数，旧实验文件和结论保持不变。
- [x] Slurm任务`20196322`和`20196424`完成二次方loss加速消融。新增保留原`least_squares`的可切换分块Gauss–Newton和标量loss＋解析梯度L-BFGS；两者均不构造完整的点对数×6 Jacobian。为与当前二次方默认矩阵严格对齐，本轮统一使用5个训练参考点。
- [x] 同一节点上，LS第4轮训练为`4.20 s,BR=0.211122`；Gauss–Newton为`1.24 s,BR=0.210496`，训练加速3.38倍。全部162步、5个EB验证后，Gauss–Newton相对旧四次方矩阵的BD-rate为`-12.060380%`，相对LS默认矩阵为`+0.174034%`。它以约0.17%的RD代价把训练降到约1.2秒，已显著低于反向D-only等公共预处理时间，保留为速度优先优化器。
- [x] L-BFGS第4轮训练为`6.17 s,BR=0.224652`，比LS更慢且更差。L-BFGS配合第二次采样`0.075`时局部3文件结果为`5.60 s,BR=0.209570`，但全部162步验证相对LS矩阵的BD-rate为`+1.397636%`，局部优势未泛化。L-BFGS配合`weight=0.6`并在第3轮停止为`4.35 s,BR=0.212258`，也没有优势；L-BFGS路径不采用。
- [x] Slurm任务`20196481`和`20196537`在Gauss–Newton下公平测试第二次采样率`0.075`和`weight=0.6/3轮`。`0.075`第4轮训练约`1.02 s`，全部162步验证相对标准Gauss–Newton的BD-rate为`+0.066792%`、相对LS矩阵为`+0.243358%`；它以约0.07%的额外RD代价进一步减少约18%的训练时间，保留为激进速度模式。`weight=0.6`第3轮为`0.87 s,BR=0.212801`，效果不如保持`weight=0.5`训练4轮，不采用。
- [ ] Gauss–Newton速度模式需要在LES、Tsunami和新数据集上验证稳定性；在此之前保留`least_squares`作为效果基线。由于Gauss–Newton总训练仅约1.2秒，文件级并行预计最多再节省零点几秒，暂不实施。
- 已完成结果保存在`adjust_coord_history/katrina_history_benchmark.csv`、`adjust_coord_history/katrina_efficiency_benchmark.csv`、`adjust_coord_history/katrina_weight_short_benchmark.csv`、`adjust_coord_history/katrina_reduction_benchmark.csv`、`adjust_coord_history/katrina_log_nfev_benchmark.csv`、`adjust_coord_history/katrina_prediction_ggd_bit_benchmark.csv`、`adjust_coord_history/katrina_second_speed_benchmark.csv`、`adjust_coord_history/katrina_gauss_parameter_benchmark.csv`和`adjust_coord_history/katrina_adjust_coord_rd.csv`；矩阵历史保存在`adjust_coord_history/katrina_*_20.npz`、`adjust_coord_history/weight_short_*.npz`、`adjust_coord_history/reduction_*.npz`、`adjust_coord_history/log_max_nfev_*.npz`、`adjust_coord_history/prediction_ggd_bit_5.npz`、`adjust_coord_history/second_speed_*.npz`及`adjust_coord_history/second_gauss_*.npz`。

## 当前Ocean多field的同PSNR判断（2026-08-28）

- 现有CSV覆盖Ocean的9个field、每个31个time step，但每条曲线只有`REL=1e-2/1e-3/1e-4`三个点。以UMC中间点约65 dB为目标，在相邻UNSZ点之间对`log(BR)`线性插值；temperature改用最新cache-8五点曲线。得到的UNSZ相对UMC BR降低约为：layerThickness `2.9%`、salinity `45.7%`、temperature `16.0%`、velocityMeridional `9.0%`、velocityX `7.1%`、velocityY `6.4%`、velocityZ `11.9%`、velocityZonal `5.6%`、zTop `22.1%`。中位数约`9.0%`，算术平均约`14.1%`，后者被salinity明显拉高。
- 三个误差点之间相隔约15～20 dB，插值误差不可忽略；除temperature外的结果来自cache-8成为默认前的精确D-only顺序。多点叶节点不改变RD，cache-8在temperature/Katrina只造成`-0.273%/-0.142%` BD-rate变化，但其他field仍需正式复测。因此上述数字只用于确定后续方向，不作为论文结果。
- 从旧日志汇总`REL=0.01`的数值流，Zstd相对Huffman输出的字节减少率为：layerThickness `52.1%`、salinity `86.2%`、temperature `65.2%`、velocityMeridional `43.7%`、velocityX `33.1%`、velocityY `34.0%`、velocityZ `48.5%`、velocityZonal `30.0%`、zTop `64.5%`。Huffman输出仍含有很强序列冗余，尤其是标量field；因此量化索引顺序和参考复用路由仍可能降低最终BR，不能只比较零阶熵或Huffman长度。
- 多数Ocean速度field只有约6%～12%的优势。接下来先改善串行RD并补齐可信曲线，而不是立即开始并行；kd-tree和内存布局类工作只影响速度，应从BR主线中分开。
- 约`9%`的Ocean典型field中位BR优势没有采用后来退化的`-reverse-reference-reuse-order-experimental`。原批量入口使用的是`-reverse-mean-distance-only-order-experimental`；参考复用优先仅由`t40_benchmark_reference_reuse_order.py`单独测试并一直保留为非默认实验。因此`9%`估计不需要为此回滚，但它仍受三点曲线稀疏、除temperature外未用cache-8正式复测等既有不确定性影响。

## 预测公式、混合预测与量化索引连续性初测（2026-08-28）

- [x] 新增`-inverse-distance-power`，压缩和解压均由用户提供相同指数；当时默认保留`3.61`，后续因`3.6`结果几乎相同而简化为`3.6`。新增`-predictor inverse_proportion|fitting|hybrid_loo`，原IDW与局部拟合路径均保留。`hybrid_loo`对同点集重构参考值做leave-one-out，用绝对误差较小的预测器预测当前点，不保存selector bit；局部线性PRESS误差由hat matrix计算，避免为每个参考点重新求解一次。
- [x] `t39_benchmark_predictor_formula.py`在当前反向D-only、cache-8、`round_point_ratio=0.8`下，对单个时间步建立三点闭环RD曲线。相对`d^-3.61/10点`，Ocean temperature的`d^-2/10点`、纯拟合、混合预测BD-rate分别为`-10.209%/-35.202%/-34.363%`；Ocean layerThickness的最好IDW候选为`d^-5/4点`的`-5.692%`，纯拟合为`+16.175%`，混合为`-0.080%`；Katrina的`d^-3.61/4点`为`-0.424%`，纯拟合为`+17.753%`，混合为`-2.431%`。参数和codec明显依赖field，当前不能替换通用默认值。
- [x] 混合预测器已通过独立解压检查，但代价过高。单文件IDW/纯拟合/混合的核心时间在Ocean temperature约为`0.022/0.15/1.07 s`，Katrina约为`0.035/0.093/0.53 s`。PRESS已将最初逐参考点重复拟合版本大约加速2倍，但仍不适合作为当前默认；下一步比较更便宜的几何条件数selector、只抽查少量LOO参考点，以及按field固定codec。
- [x] 新增可选`-hybrid-selector-path`，按原始点编号导出`0=IDW,1=局部拟合,255=未参与混合选择`，不进入正式压缩格式。`t42_analyze_hybrid_selector_spatial.py`在每个数据各3个代表时间步上统计selector空间分布：temperature/layerThickness/Katrina的最近点同selector比例平均约为`78.8%/70.0%/74.8%`，扣除全局类别比例后的调整一致性约为`0.45/0.40/0.49`，说明存在中等强度空间聚集。
- [x] 使用递归中位数空间块模拟“块内抽样后固定预测器”。每块64点、抽样16点时，temperature/layerThickness/Katrina的selector分类准确率平均为`79.2%/71.9%/73.7%`；每块256点、抽样16点时为`75.8%/68.1%/68.6%`。对应全局固定多数预测器仅为`74.0%/54.1%/54.2%`，因此分块对layerThickness和Katrina尤其有意义。块64抽样16仍需对25%的点运行LOO，块256只需6.25%但准确率下降；下一步补块128并直接测最终BR和速度，不能只根据selector分类准确率决定。
- [x] selector跨time step并非完全固定。temperature三个代表步两两一致约`81%～82%`，layerThickness约`77%～79%`，Katrina约`63%～69%`。因此temperature可以优先测试离线固定field/块codec；Katrina更适合每个文件动态块抽样，不能直接复用一个time step的selector图。
- [x] 新增实验参数`-partition-size`和更直观的`-partition-num`。后者目前接受2的幂，直接使用全局kd-tree深度`log2(block_num)`处的节点作为分区根。各分区对应连续的`tree_id`区间，共享同一棵树的只读拓扑、包围盒、tree-order坐标和ID映射，不再复制块坐标或为每块重建局部树；参考树和目标树只各自保存可变的有效状态。每块拥有完全独立的预测轮、D-only顺序和参考结构；数值与标签预测都不允许跨块引用，并在加载预测DAG后逐边检查。块内仍对每个点独立执行LOO，没有使用块级codec投票。
- [x] 独立块D-only已经接入与全局路径相同的cache-8近邻缓存、连续有效标记和可配置排序候选数。Ocean temperature两块的首次预处理由`4.70 s`降到`1.95 s`，与全局`1.97 s`持平；layerThickness为全局/2块/4块`1.92/1.91/1.87 s`，Katrina为`4.28/3.96/3.67 s`。核心压缩和解压时间基本不变。cache-8会像全局默认一样轻微改变D评分和顺序，因此重新完成了三点RD而不是沿用精确查询结果。
- [x] cache-8后的最终结果是：Ocean temperature、Ocean layerThickness和Katrina第0步中，2块相对全局LOO的BD-rate分别为`+1.245%/-0.340%/+1.036%`，4块分别为`+4.040%/+0.332%/+3.320%`。两块在三个数据上只有约`-0.3%～+1.3%`的RD变化，是当前最稳妥的独立并行分区候选；四块在temperature和Katrina约损失3%～4%，需要更多数据或更大问题规模再决定。后续从公共结构文件加载的时间接近。
- [x] 先前块上限`256/1024/4096`在Ocean对应约1024/256/64块、Katrina对应约2048/512/128块，BD-rate明显恶化；这只能说明大量小块边界代价很高，不能用于否定2～4个粗分区。当前结论是：少量完全独立大块值得作为未来一线程一块的并行候选，更大数据再逐步测试8、16或更多块。块内抽样固定codec属于另一项selector加速实验，不应和切断跨块参考同时改变。结果见`independent_block_loo_benchmark.csv`和`t43_benchmark_independent_block_loo.py`。
- [x] 新增`-reverse-reference-reuse-order-experimental`，在反向搜索的10个局部候选中首先最大化与上一目标的参考点重叠，再以D和路由距离打破平局；原D-only方案未删除。`t40_benchmark_reference_reuse_order.py`三点闭环结果为Katrina`-0.799%` BD-rate、Ocean temperature`+14.570%`。Katrina的Huffman流略增但Zstd流下降，说明符号连续性确实增强；Ocean的预测和Zstd均变差，说明“复用优先”过于激进，只保留为实验路径。
- [x] 重新训练Ocean temperature早停坐标矩阵。二次损失、3个训练文件、第一次/第二次采样`0.25/0.10`、吸收率`0.5`、4轮得到`[[0.66138232,0,0],[0.14103727,0.59977193,0],[0.04748699,0.05206220,1.47580150]]`，训练约18.4秒。全部31步五点闭环曲线相对不变换为`-2.721590%` BD-rate，重叠PSNR为`[47.545,65.267] dB`；旧100轮矩阵无收益不能再用于断言temperature无需adjust coord。
- 实验逐点结果保存在`predictor_formula_benchmark.csv`、`reference_reuse_order_benchmark.csv`、`hybrid_selector_spatial.csv`和`adjust_coord_history/ocean_temperature_early_adjust_rd.csv`，矩阵历史保存在`adjust_coord_history/ocean_temperature_second_early_4.npz`。

## 新增的降BR实验计划（2026-08-28）

- [x] 完成Ocean全部9个field第0步、三档EB、10个参考点的整数IDW指数扫描，共162组压缩—独立解压均通过PSNR一致性检查。相对`d^-3.61`，每个field最好的整数及BD-rate为：layerThickness `d^-5,-4.197%`，salinity `d^-3,-4.405%`，temperature `d^-2,-10.209%`，velocityMeridional `d^-4,+0.585%`，velocityX `d^-4,+0.041%`，velocityY `d^-4,+0.069%`，velocityZ `d^-3,+0.640%`，velocityZonal `d^-4,-0.222%`，zTop `d^-3,-2.401%`。负整数胜出次数为`d^-2`一次、`d^-3`三次、`d^-4`四次、`d^-5`一次，但四个field的所有整数都不如`3.61`。
- [x] 不存在一个跨Ocean field始终更好的整数指数。相对`3.61`，`d^-3`的跨field平均/中位BD-rate为`-0.874%/+0.670%`，`d^-4`为`+1.504%/+0.585%`；按field允许保留`3.61`并从整数中取oracle，平均改善约`2.38%`，中位数为0，说明单靠field级指数选择收益有限且集中在少数标量field。当时继续保留`3.61`，后续默认值简化为近似等价的`3.6`。
- [x] 使用time step 15和30复查第0步选出的field候选。layerThickness的`d^-5`分别改善`4.009%/3.228%`，salinity的`d^-3`改善`5.207%/4.894%`，temperature的`d^-2`改善`10.302%/10.028%`，zTop的`d^-3`改善`2.854%/2.566%`，与第0步方向一致。velocityMeridional/X/Z的整数候选在两个新增时间步仍略差于`3.61`；velocityY和velocityZonal只有约`±0.2%`波动，不值得按field固定整数。结果见`ocean_idw_power_time_benchmark.csv`、`ocean_idw_power_time_summary.csv`和`t45_benchmark_ocean_idw_power_time.py`。
- [x] 完成4个时间稳定标量field的`0.1`间隔细扫。time step 0/15/30平均最优为：temperature `d^-2.3,-11.004%`，salinity `d^-2.6,-4.953%`，zTop `d^-2.9,-2.687%`。layerThickness在`4.5～5.5`仍随指数增大而改善，因此继续扩展到`5.6～7.5`，最终`d^-7.4`平均改善`5.738%`，三个时间步最差也改善`5.519%`。其余速度field保持`3.61`时，这4个field的精细oracle使9个field算术平均改善约`2.71%`；这是选择机制实现前的收益上界，不是可直接报告的通用算法收益。
- [x] 补测全部9个Ocean field、time step 0/15/30的`d^-3.5/-3.6/-3.7`。相对`3.61`，三者跨field平均BD-rate为`-0.330%/-0.068%/+0.242%`，中位数为`-0.054%/+0.012%/+0.049%`。`3.5`略好但收益主要来自temperature、salinity和zTop，velocityX/Z反而略差；`3.6`与`3.61`几乎等价，`3.7`整体略差。因此`3.5`保留为更多数据上的通用候选；当前默认值由旧MSE得到的`3.61`简化为近似等价的`3.6`，历史结果仍保留`3.61`作为基线。结果见`ocean_idw_power_near_3_61_benchmark.csv`、`ocean_idw_power_near_3_61_summary.csv`和`t48_benchmark_ocean_idw_power_near_3_61.py`。
- [x] 补齐全部9个Ocean field、time step 0/15/30的`d^-3.0～d^-3.4`，与已有`3.5～3.7`合成`0.1`间隔扫描。`3.0/3.1/3.2/3.3/3.4`相对`3.61`的跨field平均BD-rate为`-1.028%/-0.939%/-0.889%/-0.743%/-0.570%`，但中位数为`+0.542%/+0.415%/+0.176%/+0.069%/-0.008%`，最差field退化`1.428%/1.095%/0.666%/0.360%/0.280%`。小指数的平均收益由salinity、temperature和zTop拉动，不宜只按算术平均选通用值；`3.5`在平均收益、跨field中位数和最差退化之间更平衡。结果见`ocean_idw_power_3_0_to_3_4_benchmark.csv`、`ocean_idw_power_3_0_to_3_4_summary.csv`和`t49_benchmark_ocean_idw_power_3_0_to_3_4.py`。
- [x] 固定`d^-3.5`，对Ocean全部9个field第0步扫描参考点数`4～10`，每种分别允许cache缺少`0/1/2`个参考点，共21组参数、567次压缩—独立解压；再为代表候选补测time step 15和30。相对`10点/cache-8`，`7点/cache-6`三步平均/中位/最差field BD-rate为`-1.884%/-2.694%/+2.214%`，`8点/cache-6`为`-1.761%/-2.050%/+0.829%`，`9点/cache-7`为`-0.765%/-1.095%/+0.619%`。time step 0预测DAG生成时间分别为`1.612/1.616/1.746 s`，旧`10点/cache-8`为`1.922 s`。联合使用`d^-3.5+8点/cache-6`并直接对比当前`d^-3.61+10点/cache-8`时，9个field全部改善，平均/中位/最差field BD-rate为`-2.088%/-2.138%/-1.040%`；time step 0平均核心压缩/解压时间约缩短`1.0%/3.1%`。这是目前最值得跨数据验证的完整候选配置。用户建议的`4点/cache-3`、`5点/cache-4`虽然让多数速度field和layerThickness改善，但temperature三步平均分别退化`18.080%/10.626%`，不适合作为通用值。进一步补齐4点和6点的cache消融后，4点允许缺`0/1/2`点的三步平均BD-rate为`+1.972%/+2.109%/+3.033%`，6点为`-1.845%/-1.640%/-1.453%`；两者均以不允许缺点的BR最好，允许缺1点可换取约`12%/16%`的预处理加速，允许缺2点不值得。新增结果见`ocean_small_reference_cache_time_benchmark.csv`、`ocean_small_reference_cache_time_summary.csv`和`t52_benchmark_ocean_small_reference_cache_time.py`。
- [x] 在Katrina、LES三个速度field和Tsunami四个field完成`REL=1e-3/9e-4`的四组筛选：随机分轮/全局排序分别结合`d^-3.61+10点/cache-8`和`d^-3.5+8点/cache-6`，全部完成独立解压。以随机10点在`REL=1e-3`的PSNR为目标对两点`log(BR)`局部插值，随机8点在Katrina和Tsunami改善`0.03%～1.70%`，但LES velocity_x/y/z分别退化`3.020%/2.556%/2.345%`，因此不能设为通用默认。全局10点在LES三field改善`16.028%/15.885%/15.759%`，Tsunami四field改善`5.115%/5.874%/2.427%/6.295%`；全局8点在LES只改善约`10.90%～11.54%`，明显不如全局10点，说明顺序与参考点数存在交互。Katrina本次局部插值显示小幅改善，但旧五点BD-rate为`+1.683%`，以后仍以完整曲线为准。结果见`spatial_parameter_screen.csv`、`spatial_parameter_screen_summary.csv`和`t62_screen_spatial_parameters.py`。
- [x] 同一筛选为LES三个field和Tsunami四个field分别训练了二次方loss、4轮、`weight=0.5`的Gauss–Newton矩阵。LES固定第一次采样117580点后训练仅需`0.54～1.14 s`，远低于约`154～237 s`的预测DAG生成；Tsunami使用25%第一次采样，训练约`3.60～4.14 s`。C++的`-adjust-coord-matrix`已从固定三维6参数推广为`dim*(dim+1)/2`参数，二维Tsunami压缩和独立解压闭环通过。
- [x] `t63_benchmark_spatial_order_full_rd.py`完成LES三个field和Tsunami四个field的random/global 10点、UMC五点正式曲线，全部UNSZ结果通过独立解压。global相对random的BD-rate为：LES velocity_x/y/z `-12.218%/-12.292%/-12.395%`，Tsunami momentum_x/y、velocity_x/y `-5.585%/-6.422%/-0.987%/-5.696%`。LES random/global相对UMC分别为`-26.07%～-28.20%/-36.53%～-38.66%`；Tsunami global相对UMC为`+2.808%/-2.519%/-4.596%/-15.019%`。global首次预处理比random慢约`23%～32%`，但LES核心压缩快约`4.7%～5.9%`、解压快约`5.9%～7.8%`。结果见`spatial_order_full_rd.csv`、`spatial_order_full_rd_summary.csv`和`spatial_order_geometry.csv`。
- [x] 新增`-prediction-geometry-stats`，只读统计每轮坐标span、全部参考平均距离和最近参考距离，不改变预测DAG。random后期轮几乎始终覆盖全域；global后期6轮的平均span在LES约为`0.54`、Tsunami约为`0.64～0.68`。但LES后3轮各维span可低至约`0.045～0.142`仍有约12% BD-rate收益，说明不能仅凭空间聚集程度判断是否采用global；需要用训练值的预测误差或估计码长选择。
- [x] LES上的UMC需要约`19.28 GB`峰值内存，8核默认约15GB会OOM。正式基线使用48GB任务，并让UMC一次预处理后依次压缩三个field；这不产生跨field预测，只复用公共网格结构。五个EB的共享预处理约`264.8～270.8 s`。
- [x] 新增实验性block自适应预测器。每块按tree-order确定性抽取256点，以`sum(log2(1+|qb|))`为近似码长，从IDW指数`2.5/3.0/3.5/5.0/7.5`、参考点数`4/6/8/10`及10点局部拟合中选择，共21种配置。采样点本身始终使用公共`d^-3.5/10点`，避免把训练点收益计入被选配置；配置编号以每块1字节写入文件，压缩端按原始采样值选择，解压端读取编号，因此不是免费oracle。所有block共享`max_ref_num=10/cache-8`生成的预测DAG，仅实际预测配置变化。
- [x] 完成Ocean全部9个field、31个time step、REL EB `0.01/0.003/0.001/0.0003/0.0001`的1/2/4/8-block自适应曲线，共180组压缩均通过独立解压PSNR检查。相对现有UMC三点曲线，1/2/4/8块的跨field平均同PSNR BR降低为`15.223%/15.612%/15.843%/17.592%`，中位为`9.517%/9.124%/8.105%/9.112%`。8块在salinity和zTop达到`44.884%/34.331%`，但temperature从1块的`42.622%`降到`39.171%`，多个速度field也略退化。配置选择中局部拟合比例随块数约为`11.1%/16.9%/16.6%/23.6%`；相对已有IDW CSV计时，核心压缩约慢`4.2～5.3`倍、解压约慢`6.6～7.0`倍。结果见`ocean_block_adaptive_benchmark.csv`、`ocean_block_adaptive_summary.csv`和`t53_benchmark_ocean_block_adaptive.py`。
- [x] 使用完全相同的1/2/4/8块划分补跑固定`d^-3.5/10点/cache-8`五点曲线，分离边界与决策影响。固定配置相对UMC的跨field平均BR降低为`12.316%/12.136%/11.972%/11.642%`；2/4/8块相对固定1块的平均边界与顺序代价为`0.282%/0.514%/0.933%`。自适应相对同block固定配置的平均收益为`3.523%/4.532%/4.855%/7.627%`，但中位只有`-0.015%/0.121%/0.615%/0.073%`，说明平均收益集中在少数field。8-block在velocityMeridional/X/Z/Zonal的决策净收益分别为`-0.251%/-0.574%/-0.346%/-0.192%`；temperature的固定边界代价为`1.624%`，自适应相对固定配置仍改善`26.791%`，但低于1-block的`30.065%`。因此temperature退化是边界和决策质量下降共同造成的，其他field也必须分别判断。结果见`ocean_block_fixed_benchmark.csv`、`ocean_block_adaptive_ablation_summary.csv`和`t54_benchmark_ocean_block_adaptive_ablation.py`。
- [x] partition自适应已加入field级置信回退。先汇总全部partition样本，为当前文件选择一个field级配置；局部配置只有在本partition样本上的估计码长至少改善`-partition-adaptive-min-improvement-experimental`指定比例时才被采用，配置编号仍以每partition 1字节写入，不增加新类型元数据。Ocean全部9个field、time step 0、五个REL EB、4 partitions中，0.5%门槛相对原逐partition选择的BD-rate为：layerThickness `-0.042%`、salinity/temperature/velocityMeridional `0`、velocityX `-0.029%`、velocityY `-0.457%`、velocityZ `+0.206%`、velocityZonal `-0.062%`、zTop `-0.064%`；平均/中位改善只有`0.050%/0.029%`。固定门槛不能可靠消除误选，不设为默认；结果见`t71_benchmark_partition_adaptive_confidence.py`、`results/partition_adaptive_confidence_benchmark.csv`和`results/partition_adaptive_confidence_summary.csv`。
- [ ] 如果以后继续partition自适应回退，改用独立验证样本、bootstrap一致性或父partition层次判断；同时比较不含局部拟合的20种纯IDW候选，分离自适应指数/参考点数收益与高成本拟合收益。
- [ ] 局部拟合扩展正在逐项消融，原预测器始终保留。`-partition-fitting-candidates-experimental`现支持`original/weighted/ridge/mixed/residual/extended/all`。距离加权指数为`1/2/3.5`；局部坐标归一化但不加权从未被选择，说明收益主要来自距离权重。Ocean time step 0五点RD中，纯加权在1-block的salinity/temperature降低BR `31.341%/7.318%`，4-block的salinity/temperature/zTop降低`28.295%/6.658%/38.918%`，9-field平均收益为`4.295%/8.208%`。连续混合对每点使用重构参考值计算IDW与加权拟合LOO绝对误差，以另一预测器误差占总误差的比例进行连续加权，不保存selector；相对original，1/4-block平均收益为`6.507%/8.351%`，4-block中位收益为`0.934%`。新增收益包括1-block zTop `25.272%`及4-block velocityX/Z/Zonal `4.160%/0.360%/0.934%`；但相对纯加权，1-block temperature退化`5.790%`、4-block zTop退化`7.511%`，跨field平均为改善`2.165%`和退化`0.184%`，说明当前256点采样近似码长会在两种拟合器间误选。采用连续混合的代表组合核心压缩/解压平均约慢`3～4`倍。IDW残差拟合用参考点留一IDW残差拟合目标残差，但在1/4-block共18个代表组合中一次也未胜出，因此停止完整RD。ridge同样停止。结果见`ocean_weighted_fitting_rd_benchmark.csv`、`ocean_weighted_fitting_rd_summary.csv`、`ocean_mixed_residual_fitting_benchmark.csv`、`ocean_continuous_mix_rd_summary.csv`及`t56～t59`。下一步先研究纯加权与连续混合的置信回退和计算复用，再决定是否测试参考点近似低维时的低维拟合。
- [x] 新增`rbf_fitting`，根据目标及参考点几何关系求解允许负值的局部RBF插值权重，并支持Gaussian、exponential、Matérn-3/2、Matérn-5/2、inverse-multiquadric核、nugget及0/1/2次多项式趋势。Tsunami momentum_x的10点Matérn-3/2、shape `0.03`、nugget `1e-6`、0次趋势相对当前UNSZ、UMC和10点局部二次拟合分别降低`26.836%/30.225%/8.878%` BD-rate。一次趋势几乎无变化，二次趋势明显变差；Matérn-5/2整体比3/2增加`4.878%` BD-rate。完整参考集合的几何权重改为公共预计算后，核心压缩/解压由约`0.583/0.566 s`降到约`0.018/0.012 s`；113885点的一次预计算约`0.561 s`，多time step共用。点集筛选改变参考集合时仍现场求解。结果见`t84～t90`及对应CSV。
- [x] 新增多Huffman树实验，保持最终仍只有一个Zstd阶段。实现`none/round/block/block_round`四种模式，每个点集内先生成多棵自带树头的Huffman流，再把分组元数据和所有流拼接后整体做一次Zstd；解压端按轮和块分别消费量化索引，Ocean和Katrina均通过独立解压PSNR检查。Ocean全部9个field的time step 0、REL `0.001`、2/4块结果表明：block模式18组全部负收益，平均BR增加`1.143%/2.185%`；block×轮平均增加`1.820%/3.648%`；round模式18组中10组正收益，平均BR变化为`+0.215%/-0.059%`，中位降低`0.343%/0.218%`，field间范围约为增加`4.76%`到降低`3.59%`。Katrina首文件2块上round增加`4.04%`、block增加`0.058%`。结论是空间分树不值得继续，按轮分树可作为带默认回退的field级候选；完整RD应等选择规则确定后再跑。结果见`ocean_multi_huffman_benchmark.csv`、`ocean_multi_huffman_summary.csv`和`t55_benchmark_multi_huffman.py`。
- [x] 完成`predict_round_num`、轮间目标点数量比例和逐轮EB比例的第一阶段消融。当前默认采用8轮、前两轮等大、此后目标点数增长比例2以及`round_point_ratio=0.8`；所有方案保持总点覆盖和确定性，最终参数不能在运行时逐field搜索。
- [x] 完成Ocean velocityZonal单time step空间adjust coord基线。仅用`velocityZonal.dat.0`，二次方距离loss、第一次/第二次采样`0.25/0.10`、吸收率`0.5`训练4轮约5.93秒。第1～4轮相对不变换的五点BD-rate依次为`-5.237857%/-7.830259%/-5.052596%/-6.472458%`，第2轮最佳，矩阵为`[[0.60533851,0,0],[0.08032648,0.26534194,0],[0.27221464,0.13329533,1.56998030]]`。结果见`adjust_coord_history/ocean_velocity_zonal_t0_second_early_4.npz`、`adjust_coord_history/ocean_velocity_zonal_t0_adjust_rd.csv`和`t61_benchmark_ocean_velocity_zonal_adjust.py`。
- [x] 为Ocean全部9个field分别训练4近邻早停空间矩阵，并用time step 0、REL `1e-3/9e-4`插值筛选同PSNR BR。`temperature/velocityY/velocityZonal`分别改善`0.896%/3.384%/5.566%`并晋升正式矩阵，其余6个field退化`0.253%～25.594%`而不启用。结果见`t74_screen_ocean_spatial_adjust.py`及`ocean_spatial_adjust_ref4_*.csv`。
- [ ] 跨field预测降为低优先级。已有相关研究表明field间相关性通常不足以超过field内部空间预测，只有少数数据例外且公式复杂；只有在先验相关性统计和简单线性/条件预测基线显示稳定收益后，才考虑修改压缩格式。

## 当前执行顺序

1. 用少量训练文件的预测误差或估计码长在random/global之间离线选择，并在Katrina、LES、Tsunami上验证选择是否正确。
2. 寻找更多数据验证全局排序选择器；`d^-3.5+8点/cache-6`已经确认不是无条件通用参数，默认不替换。
3. 用户定义的有序多点集接口已经闭环，下一阶段开始分区并行。
4. 纯空间、纯时间和联合时空预测的跨数据集消融放到最后；没有足够数据前不继续微调混合selector、四维loss或关键帧。
5. 同一field和矩阵的多EB实验复用预测DAG；初筛允许只运行压缩，正式结果再独立解压，减少当前约`105 s`预测DAG在压缩端和解压端的重复生成。
6. 跨时间串行算法冻结后再实现分区并行；无损编码、复杂局部拟合和细粒度DAG并行暂时不抢占主线。

## 已完成：全局分轮顺序与cache子partition（2026-09-07）

- [x] 使用当前Katrina全部162步五点流程重测先生成全局D-only顺序再分轮。4点IDW高速版的global相对random为`+0.424440%` BD-rate，核心压缩/解压时间倍率为`0.943618/0.939468`；8点Matérn-3/2 RBF高压缩比版为`+1.196001%`和`0.935418/0.920619`。因此它在Katrina是以少量BR换取约`5.6%～7.9%`核心加速的速度候选，不是高压缩比默认。结果见`t112_benchmark_katrina_global_order_current.py`和`results/katrina_global_order_current_rd.csv`。
- [x] 新增`-cache-subpartition-num-experimental`。程序在普通分轮和轮内排序后，将最后3轮按adjust后kd-tree的顶层区域稳定分组，再按新顺序重新生成参考边、逐点EB和码流；区域之间仍能引用全部已恢复点，并非独立partition。该开关自动启用按预测顺序重排数据，使压缩端先把每个文件的数值与点集编号物理重排到DAG顺序，解压端再把输出散射回原始编号；压缩gather和解压scatter分别计入对应核心时间。
- [x] Katrina全部162步五点结果中，`2/4/8/16`个cache子partition相对未重排基线的BD-rate为`-0.110314%/-1.006318%/-1.204893%/-1.147055%`，核心压缩时间倍率为`1.065670/1.080735/1.061593/1.069188`，解压为`1.183758/1.181872/1.178644/1.177312`。`1`个区域只启用物理重排而不改变DAG，BR和PSNR完全不变，但压缩/解压时间倍率为`1.084423/1.196748`；相对于这个数据重排严格对照，真正的空间分组略微追回速度并降低BR。压缩时间包含输入数据到DAG顺序的必要gather，但不包含仅为压缩端误差统计恢复原顺序的scatter；解压时间包含输出数据恢复原顺序的必要scatter。当前瓶颈是上一time step标签仍按原始点号编码，数据重排路径需要额外编号映射、随机访问和最终散射。先优化该路径，再决定是否采纳8区域方案。结果见`t113_benchmark_katrina_cache_subpartition.py`和`results/katrina_cache_subpartition_rd.csv`。

## 当前状态

- [x] 在Katrina和Ocean上，UNSZ的BR/PSNR表现均显著好于UMC；Katrina上的优势尤其明显。UMC的串行压缩和解压更快；预处理在Katrina上UMC更快，在Ocean上UNSZ更快。
- [x] Ocean temperature使用旧四次损失100轮矩阵时，启用和关闭adjust coord的结果基本相同；新的二次损失4轮矩阵在全31步五点曲线上改善`2.722%` BD-rate，确认旧训练过度迭代。
- [x] Katrina的z轴被严重拉伸，原始坐标距离与数值可预测性不成比例；adjust coord能明显改善结果。
- [x] 早期LES和Tsunami结果均未使用adjust coord；最新空间参数筛选已经为每个field独立训练并使用线性矩阵。LES原始坐标是物理空间坐标；Tsunami仍使用原始二维经纬度和通用线性变换，没有采用需要额外论证的球面映射。
- [x] 已将“正常/异常”推广为用户定义的有序点集。Katrina attr0默认分为正常值点集和常量`-99999`点集；Ocean temperature只有一个实际点集。预测DAG只预处理一次，各文件再按自己的标签筛选同点集参考点。
- [x] `.unsz`格式v7/v8和独立解压程序已经形成闭环。一/二点集保持v7，三个及以上点集使用包含n-1个条件标签通道的v8；文件包含点集信息、标签流和各点集数值流，不保存EB、预测参数或参数校验值。
- [x] 最终BR已包含文件头、标签、Huffman和Zstd码流；由坐标确定的预测DAG在解压时重建，不写入每个数据文件。
- [x] 文件I/O现在会检查打开、定位、读写和关闭结果，不再在文件创建失败时把空指针传给`fwrite`。
- [x] `s1_compress`和`s1_decompress`已统一使用显式命名参数，参数顺序不限。时间步通过`-time-step '[a,b]'`或`-time-step '{a1,a2,...}'`选择，误差通过SZ3风格的`-M REL/ABS <eb>`指定。
- [x] 坐标维数可通过`-dim <维数>`显式指定，默认仍为3。Tsunami二维field已完成带线性adjust coord的压缩和独立解压闭环；坐标范围输出、kd-tree、D-only预测顺序、反比例预测、局部仿射拟合和常量下三角坐标变换均已去除三维写死。
- [x] 正确性检查统一相对于用户给定的点集EB，而不要求float舍入后的误差仍严格满足更小的逐轮内部EB。只有`|error|>1.1×用户abs_eb`才报错；压缩和解压均累计全部文件的最大误差比例，在所有文件处理完成后统一判断和报告。
- [x] Katrina attr0的正常点和`-99999`点高度空间聚集：5个代表时间步的最近邻同类比例为95.7%～96.3%，9近邻分类准确率为97.1%～97.6%。Ocean temperature的31个时间步中未发现异常点标记。

## 已完成的关键实验

- [x] 点集格式闭环：Katrina attr0第0个文件在REL EB 0.01下，点集0有123652个正常值并使用反比例预测，点集1有293990个`-99999`并使用常量编码；压缩端和磁盘解压端PSNR一致。Ocean temperature单点集文件不会生成标签流，也已闭环验证。
- [x] 距离指数已统一：Python训练使用欧氏距离的`-3.6`次幂；C++保存平方距离，因此使用`pow(squared_dist,-3.6/2)`。历史实验使用的`3.61`结果保留不变。
- [x] 复查旧`t7_coefficients.py`：它在Katrina attr0时间步80上比较过`1/d^k`和按距离名次变化的`a_i/(d^k+b_i)`。4点与5点拟合分别得到`k=3.607/3.622`，但`a_i,b_i`没有稳定规律，因此旧结论采用统一`d^-3.61`。该实验使用原始值RMSE/MAE、随机最近邻子集和旧坐标矩阵，不是当前闭环BR，不能作为通用参数结论。
- [x] 标签采用硬0/1预测，并按原始点编号保存预测错误位置的差分。连续标签方案、bit-pack、逐字节、游程、预测顺序和按轮编码均未取得更好结果。
- [x] Zstd默认使用与UMC一致的level 3。高等级能略微减小文件，但会影响速度比较口径。
- [x] 标签使用4/5/10/64个近邻时，10和64的存储大小基本相同；10个的错误率略低，标签权重计算量约为64个的1/6.4。
- [x] 联合测试数值预测的实际参考点数`4/5/10`和候选上限`10/15/64`。Katrina attr0第0个文件上，`actual=10,max=10`的正常值流/总BR/PSNR为3601字节/0.265261/52.636202，优于原默认`actual=10,max=64`的3610字节/0.265356/52.609447；预测DAG生成时间由7.48秒降至2.32秒。默认值现已改为`10/10`。
- [x] 测试“选完整轮后再更新可用参考点”：Katrina attr0前3个文件的BR恶化约20.3%，速度没有明显改善，方案已弃用。
- [x] 将候选参考点的未归一化距离权重移到公共预处理中计算，逐文件仅按点集筛选并重新归一化。Ocean temperature第0个文件的核心压缩时间从0.114527秒降至0.012838秒（约8.92倍）；Katrina attr0多点集文件从0.238950秒降至0.034403秒（约6.95倍）。两者BR和PSNR均不变，磁盘解压逐点验证通过。
- [x] 继续优化逐文件压缩：把标签投票/权重和从逐点动态`vector`改为固定栈数组，将预测器类型判断移到点循环外，只在`-label-stats`下生成统计数组，并合并点集划分与值域统计。Katrina attr0前5个文件在REL EB 0.01下的平均核心压缩时间降至约0.021秒，BR和PSNR不变；磁盘解压逐点验证通过。
- [x] Katrina attr0前5个文件上，标签参考点从10减至5可使当时版本的平均核心压缩时间再降低约16%，总BR从0.260483略升至0.261445，PSNR不变。该结果只作为历史消融保留；独立标签参考点参数现已删除，`actual_ref_num`同时控制标签和数值预测使用的参考点数量。
- [x] 同一登录节点上完整测试Katrina attr0的162个时间步、REL EB 0.01：优化后UNSZ核心压缩为3.794880秒、71.315 MB/s，UMC为2.407530秒、112.411 MB/s，串行差距由旧结果的约3.67倍缩小至1.58倍。共享Slurm节点会出现明显CPU争用尖峰，正式速度实验应让两个压缩器在同一shared节点连续运行并重复测试。
- [x] 在同一独占节点上完成Katrina attr0和Ocean temperature、三个REL EB的UNSZ/UMC串行复测。Katrina上UNSZ/UMC的核心压缩吞吐率分别为70.03～72.66/105.70～112.90 MB/s，核心解压吞吐率分别为67.24～69.99/226.55～308.94 MB/s；Ocean上核心压缩吞吐率分别为63.59～70.34/71.92～91.20 MB/s，核心解压吞吐率分别为78.58～107.35/102.59～188.21 MB/s。CSV已刷新为MB/s。
- [x] 已加入`-reorder-data-by-prediction-order-experimental`和`-normalized-weight-experimental`。Ocean temperature全部31个time step在独占节点上各重复5次，原始/重排/预归一化/重排且预归一化四种组合的码流和解压结果均逐字节相同，压缩吞吐率分别为72.857/88.959/98.287/105.644 MB/s，解压吞吐率分别为104.285/136.007/172.134/186.424 MB/s。Katrina全部162个time step上，原始/重排的压缩吞吐率为71.520/70.926 MB/s，解压吞吐率为69.988/66.108 MB/s，说明多点集的标签和原始顺序转换抵消了局部性收益。当前核心计时不包含输入按预测顺序重排、解压结果恢复到原始顺序或压缩后仅供误差统计使用的恢复；输入重排与输出恢复分别计时并输出。
- [x] Ocean-large单点集RBF10热循环已经加入固定参考点快速路径、连续10参考点suffix、静态误差界调度缓存和12-bit Huffman前缀查表。任务`20620599/20620662`均在非独占shared节点上让UNSZ与UMC单线程连续处理同一field三次；REL `1e-3`的总时间比值为`1.0662/1.0247倍`。随后统一五点同节点交替复测得到压缩、解压和两者之和的几何平均倍率为`1.0300/1.3956/1.1346倍`，因此正式五点结果仍比`1.1倍`门槛慢约`3.15%`。速度验收使用两个阶段的绝对时间之和，各阶段相对差异仅用于热点诊断；公共预处理仍不计入该热循环指标。

当前结果远好于UMC，因此不必立刻牺牲效果换速度。如果未来发现速度或内存开销太大，可以继续尝试减小候选点数上限`max_ref_num`和实际参考点数上限`actual_ref_num`；单文件实验表明这样可能显著加快预测DAG生成，同时几乎不损失压缩效果。

## 待办事项（按压缩流程排列）

下面按实际执行顺序排列，而不是按论文优先级排列。

### 0. 数据集、参数和实验入口

- [ ] 寻找更多具有不同网格结构、坐标尺度、异常点分布和物理场特征的数据集，用于验证UNSZ、点集方案和adjust coord结论的普适性。

#### 已知数据集清单（按总空间从小到大）

本地数据的空间是`du -sb`得到的整个目录大小，可能包含格式转换副本、压缩结果或中间文件；未下载数据使用发布页大小或按原始`float32`字段估算。两种口径适合规划存储，但不能直接用于比较有效数据量。

|排名|数据集|状态|总空间|time step数|类型及备注|
|---:|---|---|---:|---:|---|
|1|Tsunami|本地|6.88 MB|1|113885个cell center；目录同时保留NetCDF与转换后数据|
|2|Tpost20|已接入并初测|42.63 MB展开后|20|131072个顶点、615195个四面体；REL 1e-3下与UMC接近|
|3|Ocean|本地|270.87 MB|31|MPAS cell-centered数据，235160个cell、9个field|
|4|Katrina|本地|319.86 MB|162|ADCIRC区域三角形FE，417642个点、826866个三角形|
|5|Chondrule|候选|370.29 MB压缩包|主模拟6，替代模拟另6|是否适合UNSZ还需检查坐标和网格结构|
|—|`small datasets.zip`|本地重复副本|1.37 GB|混合|Katrina、Ocean、LES和Tsunami的旧归档，不是新数据集|
|6|Atrial electrophysiology|候选|1.41 GB压缩包|尚未确认|患者特异有限元网格及跨时间的膜电位VTK结果|
|7|LES|本地|1.44 GB|1|大规模非结构CFD；主要空间来自connectivity|
|8|yfeng/E3SM|h0与h4均已接入并初测|6.54 GB|`h4`为720，`h0`为1|21600个水平cell；h0八个field原坐标UNSZ均优于UMC，AODDUST动态无效点使UNSZ明显优于UMC-INP|
|9|TAPS_compressor|已接入并初测7个field|10.68 GB|1|16002271个cell center；没有可供UMC使用的connectivity，用于DAG预处理扩展性和adjust coord筛选|
|10|ADCIRC Global Storm Tide|Haiyan 1.5 km的zeta/depth已接入|12.3 GB|Katrina每网格16，Haiyan每网格24|网格不完全连通，UMC必须关闭只遍历单连通分量的DPFS重排|
|11|Ocean-large|部分接入|本地约37 GB；论文目标约16.13 GB原始field|官方初始状态1；论文目标12|3693225个cell；官方RRS18to6包与`temperatureSurfaceValue`初始快照已转换并完成五点RD，论文91个field输出仍缺失|
|12|CAM Global RCE|候选|16.2 GB压缩包|不能用单一数量表示|11组海温模拟，每组保留22个月；发布的是处理后的TC轨迹、剖面和降水|
|13|hyuan/pb|本地|32.03 GB|1|1947154个`8³`点谱元，约9.97亿个element-local point|
|14|xmdv_data/TGV|本地|35.25 GB|6|Nek high-order SEM；含大量ASCII和中间格式副本，原始Nek约4.03 GB|
|15|viral-flamespeed|本地|196.26 GB|2|NekRS high-order SEM；3240264个`10³`点谱元|

- [ ] 优先接入`xmdv_data/TGV`作为规模适中的多time step high-order SEM数据；将`viral-flamespeed`留作最终的大规模并行实验。
- [ ] 检查Atrial压缩包中的实际VTK数量，并确认Chondrule是否具有可供UNSZ使用的非结构坐标或网格。
- [x] 将空间训练与时空训练得到的最终坐标变换矩阵保存为可区分的文本文件，并允许压缩和解压从文件读取。
- [ ] 将剩余的数据集、field和算法参数从源码硬编码迁移到配置文件或命令行。
- [ ] 清理旧实验脚本和被Git跟踪的`__pycache__`。
- [ ] 在正式批量测试前做阶段性提交，并保存实验对应的Git提交、命令、节点配置、线程数和运行环境。

### 1. 坐标读取与adjust coord

- [x] Ocean全部field已分别训练自己的4近邻早停空间矩阵。单个中间区间筛选只采纳`temperature/velocityY/velocityZonal`；其余矩阵保留为负结果，不研究field共享矩阵规则。正式大实验只为采纳的矩阵补完整曲线。
- [ ] 如果以后继续提高矩阵模型复杂度，只测试连续的、以坐标为自变量的低阶下三角矩阵。函数参数量、正定/尺度约束和矩阵元数据必须计入方案成本，不能观察单个field结果后任意增加模型复杂度。
- [x] 作废一次定义错误的局部矩阵实验：曾在Ocean zTop上划分2/4个矩阵区域并分别变换坐标，随后却把区域重新合并并建立全局kd-tree，导致跨矩阵接缝的距离没有一致含义。该实验不是“每个压缩partition互不引用”，其`+25.536%/+30.017%`结果不能用于判断独立partition或合理的连续可变矩阵是否有效。错误实现和结果仅保存在`t75_screen_partition_adjust_coord.py`及`ocean_ztop_partition_adjust_screen.csv`，不接入正式路径。
- [ ] LES的`velocity_x/y/z`已经分别得到Gauss–Newton线性矩阵，并写入`adjust_coord_matrices/LES_*_spatial.txt`；下一步建立启用和关闭adjust coord的多EB曲线，确认矩阵自身收益。
- [ ] Tsunami四个field已经分别得到二维Gauss–Newton线性矩阵，并写入`adjust_coord_matrices/Tsunami_*_spatial.txt`；下一步用多EB曲线比较原始二维坐标和线性adjust coord。球面映射只作为需要单独论证的候选方案，必须预先规定适用条件，不能根据压缩结果事后选择。
- [x] 保持二次方loss、Gauss–Newton、4轮、吸收率0.5和单个训练文件不变，将每个采样目标的近邻数从5增加到10。LES训练耗时从`0.43～1.03 s`增至`0.89～1.28 s`，Tsunami从`3.59～4.14 s`增至`4.33～4.90 s`。在当前8轮、`d^-3.6`、10个预测参考点、cache-8和先全局排序再分轮下，LES `velocity_x/y/z`的同PSNR BR变化为`-0.647%/-0.050%/+0.230%`，Tsunami `momentum_x/y、velocity_x/y`为`-2.009%/+5.469%/-13.704%/+10.460%`。因此不把ref10设为统一默认，但保留`*_spatial_ref10_candidate.txt`、`t72_train_spatial_matrices.py`、`t73_compare_spatial_matrix_ref_nums.py`和完整CSV供以后的field级选择。
- [x] 使用相同训练配置将近邻数由5减少到4。最近邻查询和每次残差计算更少，但由于建树、全坐标变换及优化评估次数的波动，总训练时间没有稳定下降；这一参数不会减少正式压缩仍使用的10个预测参考点。LES `velocity_x/y/z`的局部同PSNR BR变化为`-0.810%/-0.131%/+0.223%`，Tsunami `momentum_x/y、velocity_x/y`为`-1.772%/+0.216%/-9.320%/-0.163%`；7个field中5个改善，2个仅退化约0.22%。4近邻已经设为adjust coord训练默认值，ref4矩阵已经晋升为正式矩阵；原ref5矩阵以`*_ref5_baseline.txt`保留，完整筛选结果保存在`ref5_ref4_spatial_matrix_*.csv`。
- [x] 将训练近邻数继续减少至3。LES `velocity_x/y/z`的局部同PSNR BR变化为`-0.149%/-0.210%/+0.296%`，Tsunami `momentum_x/y、velocity_x/y`为`-2.189%/+0.643%/-16.726%/+0.070%`。ref3的极值收益大于ref4，但轻微退化的field由2个增至3个；因此保留`*_spatial_ref3_candidate.txt`和`ref5_ref3_spatial_matrix_*.csv`作为field级候选，完整五点RD优先验证ref4。Tsunami本次ref3训练约`0.54～0.57 s`，LES约`0.86～1.30 s`；查询和残差成本降低，但共享节点负载和优化评估次数会影响总时间，不把单次计时视为稳定加速比。
- [ ] 研究能否用低成本统计量判断field是否需要adjust coord，避免对收益很小的field执行训练；temperature的收益约2.7%，可作为“有小幅收益”的样本。
- [ ] 减少两次采样比例、训练文件数和迭代轮数，记录变换矩阵、近邻集合与最终BR何时稳定。
- [x] 将时空训练的随机超矩形区域采样推广到空间训练，保留原`random_points`并新增两种模式。`random_hyperrectangle_regions`按均匀点分布估算长方体积，在Katrina名义25%的单区域中实际覆盖73.8%～91.7%的点，训练由`9.45 s`增至`56.29 s`；名义比例降至0.001后仍采22723～102701点且同PSNR BR约增加`14.5%`。随后新增`random_fixed_point_regions`：随机选择区域中心，用归一化Chebyshev距离确定轴对齐长方体边界，让4个区域恰好分配原总采样预算；区域允许重叠、无halo，各自建立小树并查询，只合并生成的训练点对后统一拟合公共矩阵。旧离散采样重构前后矩阵SHA完全一致，排除了重构引入的结果变化。固定点数4区域在Katrina/Ocean temperature/LES velocity_x上的`BR,PSNR`分别为`0.280412,57.578262`、`0.386426,51.605179`和`1.756750,67.512842`，对应离散采样为`0.205305,59.421505`、`0.381561,51.603019`和`1.705098,67.638529`，没有发现收益。Katrina小树的建树加查询为`0.576 s`，反而低于离散采样的`0.735 s`；总训练从`9.30 s`增至`32.90 s`是因为Gauss–Newton每轮函数评估数由12～14增加到38～100。LES固定117580点时区域选择需`7.21 s`，总训练`14.39 s`，而离散采样仅`6.78 s`。两种区域模式保留用于复现，但不作为默认，也暂不优化区域选择扫描。
- [x] 继续验证“较大区域内均匀采样”的折中。`random_regions_then_points`让每个区域先包含最终预算4倍的实际点，再均匀抽取25%；LES velocity_x的1/2/4/8区域同PSNR BR变化为`+27.88%/-0.30%/-0.30%/+0.49%`，收益很小且依赖区域数。为避免每个区域扫描全部1274万个坐标，新增25%全局预采样树。直接随机下降到定额block时，SciPy树与C++树切分规则不同；进一步复刻C++的最长边中位切分后也发现LES非均匀分布会产生归一化跨度约0.5的长条block，因此该定义本身不保证紧凑，原SciPy block矩阵仅得`BR=1.507301,PSNR=67.395904`。改为`random_sampled_kd_neighborhoods`后，在约318万全局预采样点上随机选中心并查询4×29395个L∞近邻，单轮采样、建树、查询约`1.53 s`，区域归一化跨度约0.007～0.021、最终预算严格为117580；4轮训练矩阵得到`BR=1.428907,PSNR=67.562379`，同PSNR仍比随机散点矩阵高约`0.20%`。结论是公共预采样树解决了选区速度、定额和紧凑性，但尚未证明局部密度能改善最终矩阵，全部模式只作复现保留。
- [ ] 比较每轮重新采样与固定样本。
- [ ] 减小迭代缓冲，同时检查矩阵是否振荡或接近退化。
- [ ] 检查损失函数、坐标量级和矩阵归一化的关系。
- [x] 比较当前四次方损失`(dv²-distance²)²`、二次方损失`(|dv|-distance)²`和带文件abs EB常数项的对数平方距离损失；保留全部目标作为可切换基线。
  - Katrina时间步0～2使用相同随机种子训练20轮，每5轮保存一次矩阵，并以原四次方100轮矩阵的`PSNR=59.868332 dB,BR=0.276045`为基线做同PSNR搜索：

    | loss | 第5轮BR | 第10轮BR | 第15轮BR | 第20轮BR | 当前最佳轮次 |
    |---|---:|---:|---:|---:|---:|
    | 四次方 | 0.262624 | 0.236681 | 0.242402 | 0.258620 | 10 |
    | 二次方 | 0.215432 | 0.219556 | 0.249196 | 0.272457 | 5 |
    | 对数平方 | 0.219716 | 0.251993 | 0.264016 | 0.276549 | 5 |

  - 三种训练loss都随轮数下降，但真实BR均先改善再退化；四次方同样不能按训练loss或固定100轮选择矩阵。当前最佳是二次方第5轮，比原基线BR低约22.0%；对数平方第5轮低约20.4%，但20轮训练耗时`424.43 s`且经常用满`max_nfev=100`，数值收敛明显差于另外两种。
  - 中间矩阵保存在`adjust_coord_history/*.npz`，逐矩阵结果保存在`adjust_coord_history/katrina_history_benchmark.csv`。后续采样率与吸收率实验得到的二次方第4轮矩阵已经完成全部162步和5个EB的验证；通用停止条件仍需在更多数据集上研究，避免选择轮次本身过拟合。
  - 已为四次方、二次方和对数平方loss实现解析Jacobian，并将逐目标整理点对的Python循环改成NumPy批处理；相同配置和随机种子下矩阵保持不变。二次方原采样配置10轮的“整理点对”耗时降至`0.17 s`，总训练耗时为`17.11 s`。
  - Katrina时间步0～2的二次方采样率与吸收率消融：

    | 配置 | 轮次 | 点对数 | 训练时间 | 同PSNR BR |
    |---|---:|---:|---:|---:|
    | `first=0.25,second=0.25,weight=0.3` | 5 | 391545 | 7.69 s | 0.215432 |
    | `first=0.25,second=0.10,weight=0.3` | 5 | 156620 | 6.66 s | 0.212661 |
    | `first=0.10,second=0.25,weight=0.3` | 5 | 156620 | 18.95 s | 0.231873 |
    | `first=0.10,second=0.10,weight=0.3` | 5 | 62650 | 13.90 s | 0.305474 |
    | `first=0.25,second=0.25,weight=0.5` | 5 | 391545 | 7.32 s | 0.214065 |
    | `first=0.25,second=0.25,weight=0.7` | 5 | 391545 | 32.34 s | 0.245959 |

  - 当前默认配置是第一次采样`0.25`、第二次采样`0.10`、3个训练文件、吸收率`0.5`并在第4轮停止。第一次采样决定可用近邻集合，降低它会改变几何问题并恶化BR；第二次采样只减少拟合目标，风险较小。第二次采样`0.05`、相同吸收率和轮数是训练速度优先候选。
  - 完整消融结果保存在`adjust_coord_history/katrina_efficiency_benchmark.csv`。
- [x] 测试预测残差的广义高斯估计码长目标。外层使用当前矩阵重新选最近参考点，内层在参考点固定时优化矩阵和反距离权重；按文件用abs EB归一化残差，固定广义高斯形状参数`β=0.5`，拟合尺度参数，并用残差宽度为`2×abs_eb`的概率质量估计`-log2(P)`。该目标在Katrina上略优于二次方loss，但只近似零阶符号长度，不能表示Huffman树头、符号顺序和Zstd收益，仍需在更多数据集上验证。
- [x] 在Ocean进一步验证预测码长近似loss。训练使用time step 0、10个IDW参考点、第一次/第二次采样`0.25/0.10`、吸收率`0.5`、4轮和`max_nfev=20`，单field约`7.6～8.6 s`。在time step 0/15/30的局部同PSNR筛选中，temperature相对原坐标改善`1.286%/1.143%/0.540%`，相对二次方ref4为`-0.394%/-0.854%/+0.348%`；velocityZonal相对原坐标改善`7.018%/8.667%/7.469%`，相对二次方ref4继续改善`1.538%/1.668%/1.832%`；zTop从二次方ref4退化约`22.3%～22.9%`变为相对原坐标改善`2.453%/4.519%/2.365%`。三者码长loss矩阵均晋升为正式矩阵；结果见`t74_screen_ocean_spatial_adjust.py`、`t76_validate_ocean_prediction_ggd_adjust.py`和`ocean_spatial_adjust_prediction_ggd_*.csv`。
- [ ] 二次差分目标的点组、方向和归一化方式尚未确定，定义清楚后再实现；之后继续考虑量化索引分布、经验熵以及更直接的BR近似目标。
- [x] Ocean已完成少量同网格跨field矩阵迁移实验；速度field差异很小、temperature专属矩阵有约`5.1%` BD-rate收益。该现象保留为历史结果，但共享规则难以跨数据集复用，不再作为主线。
- [ ] 保存每个矩阵对应的数据集、field、训练参数、随机种子和训练文件编号。

### 2. 公共kd-tree、预测轮和预测顺序

- [x] 优化反向D-only缓存：新增连续`reference_valid`点标记，精确模式的缓存命中/未命中、压缩SHA、BR和PSNR保持不变；新增`-d-only-cache-refill-threshold-experimental 9`，允许缓存剩9个有效参考点时直接按9点计算D、降到8点时才重新查询。Ocean缓存未命中由910504降到485702，平均预处理由`5.282 s`降至`4.104 s`；Katrina由1683703降到901546，预处理由`11.697 s`降至`9.016 s`。三个REL EB、单time step的初步BD-rate分别为Ocean`+0.361%`、Katrina`-0.237%`。
- [x] 将排序候选数与最终预测的`max_ref_num/actual_ref_num`分开，新增`-order-candidate-num-experimental`，最终预测保持`10/10`。候选数5使Ocean/Katrina平均预处理降至`4.298/9.348 s`，初步BD-rate为`+0.267%/-1.136%`；候选数4进一步降至`4.062/8.408 s`，但Ocean退化`+1.553%`，跨数据集稳定性不如5。当前默认仍为精确候选数10；完整结果见`d_only_order_benchmark.csv`。
- [x] 继续比较缓存阈值8～10及其与排序候选数5的组合。保持候选数10时，阈值8将Ocean/Katrina缓存未命中降到352008/646994，平均预处理降至`3.737/7.966 s`，初步BD-rate为`-0.527%/-0.454%`，比阈值9更快且未观察到RD损失。候选数5＋阈值8进一步把预处理降到`2.988/6.760 s`，相对精确10点减少约43%/42%，初步BD-rate为`-0.017%/-0.377%`；候选数5＋阈值9居中。完整结果见`d_only_order_benchmark.csv`。
- [x] Slurm任务`20199747`完成Ocean全部31步、Katrina全部162步的正式五点RD。仅缓存阈值8相对精确10点的BD-rate为`-0.273%/-0.142%`，预测DAG生成预处理由`3.162/6.647 s`降到`2.184/4.676 s`，约缩短`30.9%/29.7%`，已设为默认；`-d-only-cache-refill-threshold-experimental 0`保留精确模式。仅排序候选数5的BD-rate为`+0.619%/-0.814%`，候选数5＋阈值8为`+0.717%/-0.735%`，在Ocean有明确代价，因此两者只保留为实验开关。完整逐档结果见`d_only_full_rd_benchmark.csv`，默认值仍需在新数据上验证。
- [ ] 暂不实施每轮独立的紧凑目标树。后续可让目标树只包含当前轮的`target_ids`，避免在全局树的稀疏有效点中查询。
- [x] 修正kd-tree根节点`valid_num`初始化为0的问题，并将全局第一次反向剔除显式定义为“选择坐标范围中心附近的目标”，不再依赖第一次参考查询因根计数为0而返回空。Ocean/Katrina候选数5＋缓存阈值8的压缩SHA、BR、PSNR及独立解压结果均不变，D-only缓存未命中各减少5次；根节点计数现在始终表示真实有效点数。
- [ ] 将“第一步也使用D”保留为独立顺序实验，与当前显式中心种子比较BR和预处理时间，不再和树状态正确性修复混在一起。
- [x] 新增`-predict-round-num`并完成8/10/12轮与`round_point_ratio=0.7/0.8/0.9`的第一阶段联合消融。采用REL EB `0.02/0.01/0.005/0.0025/0.00125`五点曲线，并以原10轮+0.8为基线：8轮+0.8在Katrina attr0、Ocean temperature、Ocean velocityZonal和Tsunami velocity_y上分别降低BD-rate `0.273615%/0.943852%/0.327420%/1.399576%`，是唯一在4个代表field全部改善的组合；12轮结果不稳定。各field单独最优公比仍不同，Tsunami偏向0.7，Ocean velocityZonal偏向0.9，因此不能逐field运行时调参。8/10/12轮的预测DAG只相差轮边界的数个字节，核心压缩和公共预测DAG生成耗时没有稳定差异。默认已改为8轮+0.8；待更多field、time step和新数据继续验证。
- [x] 新增`-round-size-growth-ratio-experimental`。前两轮保持等大，第三轮开始按指定比例形成几何级数，再归一化到全部点；默认2生成的轮点数和旧整数除数实现完全相同，Katrina压缩文件SHA-256逐字节一致。8轮、EB公比0.8下对比例`1.25/1.5/1.75/2/2.25/2.5/2.75/3/3.5/4`完成4个代表field第0步的五点曲线：相对比例2，Katrina的最好值仍为2；Ocean temperature在4时为`-1.339312%`，Ocean velocityZonal在3.5时为`-3.833390%`，Tsunami velocity_y在1.75时为`-1.883505%`。四field平均以2.5最低，但只改善`0.375045%`，且Katrina/Tsunami分别退化`0.686171%/0.408437%`。各比例的公共预处理、核心压缩和解压没有稳定速度差异，默认继续使用2；串行和4-partition非默认比例均通过独立解压。脚本和完整结果为`t68_benchmark_round_size_growth.py`、`results/round_size_growth_benchmark.csv`和`results/round_size_growth_summary.csv`。
- [x] 在Katrina全部162个time step上完成`10点/cache-8`、`4点/cache-4`和`4点/cache-3`的五点RD及程序内部计时。相对10点基线，4点/cache-4和4点/cache-3的BD-rate分别为`+2.272%/+2.319%`，核心压缩几何平均吞吐率由`43.08 MB/s`升至`54.22/54.26 MB/s`，解压由`52.37 MB/s`升至`71.03/71.26 MB/s`；公共预测DAG由`20,046,652 B`减半至`10,023,424 B`。cache-3将首次预处理由cache-4的`2.520 s`继续降至`2.211 s`，但不改善核心速度，因此4点/cache-4更适合作为忽略公共预处理成本时的快速版候选。完整结果见`t93_benchmark_katrina_reference_num_rd.py`和`results/katrina_reference_num_rd.csv`。
- [x] 将每个block分别分配的`min_coord/max_coord vector`改为树内连续包围盒数组，并在建树时自底向上收紧到真实点范围。Ocean单步非adjust coord预处理由`7.05 s`降至`5.22 s`，Katrina单步由`14.45 s`降至`11.79 s`，压缩文件SHA、BR、PSNR和独立解压结果均不变。当时同时加入的点编号tie-break现已删除；Tsunami height的4点MSD小型对照中，首次DAG生成平均由`0.4169 s`降至`0.3837 s`，但DAG发生变化，REL `1e-3`下的BR由`5.038837`增至`5.129034`。这说明现有数据会遇到分割轴或近邻距离并列，删除该逻辑是速度与压缩效果之间的取舍。
- [x] 用栈内有序小数组替代kd-tree查询中的`priority_queue`。最初容量64时，候选数5＋缓存阈值8下Ocean三次交替测试的排序/最终参考生成/总预处理平均由`2.055/1.051/3.213 s`降至`1.908/0.979/2.980 s`，总预处理缩短`7.2%`；Katrina由`4.308/2.159/6.644 s`降至`4.169/2.028/6.349 s`，缩短`4.4%`。容量现缩到默认路径真实上限11：默认`max_ref_num=10`时，反向参考查询为排除目标自身最多临时保存11项；大于11项自动退回原堆容器。保持tree-order和共享结构时，容量11与64的Ocean总预处理为`2.702/2.720 s`，Katrina为`6.036/5.996 s`，差异在约`0.7%`内且方向不一致，因此容量调整主要减少栈对象大小。两者压缩文件SHA与旧容器一致，独立解压PSNR一致。
- [ ] 如果树查询仍是瓶颈，将递归查询改成显式栈，比较函数调用成本和遍历时间；该方案暂不实施。
- [x] 已按`tree_id`连续复制叶坐标，查询仍通过`tree_id_to_original_id`返回原始编号。相同子树覆盖连续`tree_id`区间，但相邻编号不保证一定空间最近。保持容量11小数组和共享树结构时，关闭/启用tree-order坐标的Ocean三次平均排序/最终参考/总预处理为`1.775/0.946/2.819 s`和`1.645/0.930/2.702 s`；Katrina为`4.045/2.107/6.295 s`和`3.774/2.118/6.036 s`。两者总预处理都缩短约`4.1%`，因此保留。
- [x] `tree`与`reference_tree`现已共享只读拓扑、包围盒、ID映射和tree-order坐标，只分别保存`valid_num`。两棵树的有效状态并非恒定互补，因此不能只保存一份有效计数。共享后压缩文件SHA和独立解压结果不变。
- [x] kd-tree叶节点改为可保存多个点，新增逐点有效标记和`-kd-leaf-size`，查询在叶内连续扫描有效点并继续按`(距离,original_id)`维护精确近邻。Ocean叶大小`1/4/8/16/32/64/128`的单次预处理约为`2.842/2.137/1.910/1.804/1.629/1.736/1.987 s`，Katrina叶大小`1/4/8/16/32/64`约为`6.191/4.883/4.191/3.666/3.476/3.439 s`；对`16/32/64`各重复三次后，Ocean平均为`1.736/1.681/1.719 s`，Katrina为`3.723/3.477/3.491 s`。32在两者上都是最稳妥的共同选择，默认值已改为32；所有叶大小的压缩SHA、BR和PSNR与单点叶一致，独立解压通过。
- [ ] 将每点独立的参考点`vector`改为连续参考边数组。每条边保存`{参考点编号,权重}`，每个目标保存`{起始位置,数量}`；比较内存、预处理、压缩和解压速度，并确保参考点顺序和码流不变。
- [x] 恢复早期“单独文件保存公共预测DAG”的方案，新增`-dag-cache <路径>`：文件不存在时生成并保存轮边界、每个目标编号及其由近到远的参考点，存在时直接读取并根据当前坐标重新计算权重。Ocean文件为`11,287,524 B`，读取时间`0.144 s`、含init的预处理`0.214 s`；Katrina为`20,046,660 B`，读取时间`0.274 s`、不含adjust coord的预处理`0.279 s`。Level 3 Zstd只能压到`6,314,032/9,876,777 B`，仍远大于当前压缩码流。因此该文件只适合作为同一网格反复实验或多field共享的本地缓存；若作为必须传输的压缩元数据，必须计入总BR，目前不可取。生成、读取、prediction-order布局和归一化权重路径均通过压缩SHA及独立解压检查。
- [x] 实验性地按参考点复用程度选择同一轮内的下一个目标。Katrina改善`0.799%`，但Ocean temperature退化`14.570%`，说明增强连续性的代理目标可能破坏预测质量，激进方案已放弃。直接使用训练数据减少相同量化bin之间的切换尚未实现；数值接近不等于Huffman码字或Zstd上下文接近，因此不再使用相邻量化索引差值作为目标，只有预测质量优化基本结束后才考虑条件码长。
- [ ] 获得更多数据集后重新验证最近距离、平均平方距离D、S2和方向集中度；当前暂定反向D-only为主方案，并保留原始顺序和其他评分方案。
- [ ] 预测顺序必须offline生成并适用于所有time step，不能依赖当前文件的真实值。若引入数据值，只能从多个训练文件聚合后学习固定评分函数，或者保存一份所有time step共享的离线顺序。
- [ ] 设计基于多个训练文件的可预测性分数，比较静态分数与随参考集合变化而重新计算分数的收益和成本。

### 3. 逐文件点集划分、值域与误差上限

- [x] 已通过运行期JSON实现任意数量的有序点集。简单分类使用按优先级执行的比较规则；任意复杂分类可由用户代码预先生成逐点`uint8`编号文件。每个点集独立配置常量、反比例、局部拟合或hybrid codec，并显式配置自己的REL或ABS误差上限；单点集也可以使用JSON。

### 4. 点集标签预测与标签流

- [x] 已实现n-1个条件二进制标签通道。通道i跳过已归入较早类型的目标，但把这些点作为值1参考；0表示属于当前类型、1表示不属于，每点只保存预测正确/错误修正bit。每个通道按原始编号过滤后分别执行错误位置差分和Zstd，v8保存各通道逻辑目标数和字节数；最后未分类点属于末尾类型。一/二点集仍使用v7，Katrina默认文件与旧版本SHA一致；三点集Katrina拆分测试的两个通道目标数为417642/294316，IDW、拟合、常量codec及逐点集REL/ABS EB均通过独立解压检查。
- [ ] 在公共预处理中归一化标签预测权重。硬0/1投票仍需分别累加各类权重，因此主要验证其内存访问、浮点舍入和多点集速度影响。
- [ ] 联合调整标签参考点数和距离指数，再比较错误位置差分+Zstd、Golomb/Rice和带上下文的算术编码/rANS。零阶rANS不能利用空间聚集，不能只编码原始0/1流后就下结论。

### 5. 数值预测、逐轮EB与量化

- [ ] 用当前默认cache-8、叶大小32和反向D-only为Ocean全部9个field补齐统一五点RD；旧三点CSV只保留为粗略基线。
- [ ] 在Ocean全部field和Katrina上正式比较固定`round_point_ratio=0.7/0.8/0.9`，用跨field平均BD-rate和留一field检查选择通用值，不允许在运行时针对单个field调参。
- [x] 新增`round_point_ratio_dag`并将最初只看出度的近似改为传递影响力。对于实际同点集IDW预测边`i→j`，以归一化预测权重`α(i→j)`计算`influence[i]=1+Σ_j α(i→j)·influence[j]`；按预测顺序从后向前动态规划，因此某点只直接影响一个目标、但该目标影响大量后代时，影响力也会继续传回来。随后在每轮内按影响力降序分配与`round_point_ratio`完全相同的EB倍率集合，不改变各轮预算，只隔离“给谁更小EB”的影响。压缩端和解压端由预测DAG与点集标签确定性重建调度，不增加码流元数据；当前支持反比例预测和常量点集，独立跨time step预测器与局部拟合暂不支持。第0步、REL EB `0.008/0.010/0.012`的三点二次拟合筛选中，相对原`round_point_ratio=0.8`的BD-rate分别为Katrina attr0 `-0.037208%`、Ocean temperature `-0.522654%`、Ocean velocityZonal `-0.414875%`、Tsunami velocity_y `-2.529300%`；共同PSNR区间分别为`[58.649256,60.999930]`、`[50.527109,52.893570]`、`[47.456961,50.171494]`和`[53.623834,55.864742] dB`。Katrina REL `0.01`时直接边数为1160312、受到后续影响的点数114494、最大传递影响力119178.813、重新分配208816个点，统计、动态规划和排序耗时约`0.033 s`；Ocean调度约`0.048～0.051 s`，Tsunami约`0.018 s`。共享登录节点的配对计时中，Katrina核心压缩/解压由`0.0397/0.0330 s`增至`0.0723/0.0626 s`，Ocean temperature由`0.0273/0.0199 s`增至`0.0737/0.0660 s`；Tsunami压缩由约`0.0121 s`增至`0.0300 s`。收益不足以覆盖当前逐文件成本，因此暂不补LES和正式曲线；若以后恢复，先把单点集调度移到公共预处理并跨time step缓存。4-partition路径和独立解压均已闭环。
- [x] 新增不排序的DAG直接EB映射。`round_point_ratio_dag_direct`在每轮内将`log(influence)`线性映射到该轮原有的对数EB区间，使最小影响力获得最大EB、最大影响力获得最小EB；这不再保持各EB倍率的原始数量分布。`round_point_ratio_dag_variance_direct`按独立高斯近似把传播权重由`α`改为`α²`，其标量动态规划忽略同一源误差沿多条路径汇合时的协方差。8轮、REL EB `0.02/0.01/0.005/0.0025/0.00125`五点曲线中，一次公式相对默认`round_point_ratio=0.8`在Katrina attr0、Ocean temperature、Ocean velocityZonal和Tsunami velocity_y的BD-rate依次为`+0.094070%/-0.244020%/-0.158172%/-0.547067%`，平均`-0.213797%`；二次公式为`+0.269175%/-0.217030%/-0.006786%/+0.052311%`，平均`+0.024418%`。一次公式略稳，二次公式没有总体收益。两者影响力计算约需`0.011～0.039 s`，使共享登录节点上的核心压缩约慢`1.6～2.1`倍；Tsunami二次公式独立解压闭环通过。完整曲线见`py_codes/trash/results/dag_influence_direct_rd.csv`。
- [ ] 实现无需selector bit的混合预测器：使用已经重构的参考点互相做leave-one-out预测，结合局部拟合条件数，在IDW和局部仿射之间确定性选择。压缩端和解压端必须只依赖相同的参考编号、坐标和重构值。
- [ ] 对很小的REL EB增加自动精度升级。`REL<1e-4`可作为初始候选条件，但最终应比较`rel_eb×点集值域`和当前数据尺度上的float ULP；仅把预测运算改成double不能解决最终float输出的表示限制，还需要明确是采用double重构输出，还是在EB小于float可保证范围时切换到精确恢复原始float的编码路径。
- [ ] 为单点集反比例预测预先归一化权重，并使用不再逐点判断点集、codec和常量点集的专用预测循环；在Katrina和Ocean上比较速度，并检查浮点舍入是否改变量化索引、BR或PSNR。多点集数值预测仍需按当前文件标签筛选参考点。
- [ ] 调整反比例预测器的距离指数、参考点数量和零距离处理。
- [ ] 改善局部拟合预测器的数值稳定性，测试正则化、条件数检查和不同参考点数量。
- [ ] 测试按轮动态选择预测器。

### 6. 量化索引的无损编码

- [ ] 在尝试替换编码器前，先为每个field和EB记录量化索引经验熵、零值比例、符号数、尾部分布、预测顺序自相关、Huffman树头、Huffman payload以及Zstd输出长度。若最终流已接近经验熵，无损编码不再作为近期BR主线。
- [ ] 首先拆开比较当前`Huffman`、`Huffman+Zstd level 3`、直接对ZigZag后的`int32_t`量化索引使用Zstd，以及`ZigZag+varint+Zstd`；分别记录数值流BR、编码和解码吞吐率。
- [ ] 用rANS或FSE替换Huffman，分别比较单状态和多状态交错实现；主要与当前`Huffman+Zstd`整体比较，不默认再叠加Zstd。
- [ ] 比较canonical Huffman和当前显式DFS树格式，分别测树头大小、建树、编码和解码速度。
- [ ] 如果继续优化当前Huffman，将无损阶段细分为频率统计/建树、bit packing、Zstd和容器组装，并结合硬件计数器定位`int32_t`实例在Katrina上较慢的原因。
- [ ] 对量化索引尝试Golomb/Rice、ZigZag+varint以及主体/离群索引双流。
- [ ] 将Zstd level做成正式配置并至少比较level 1和3。

### 7. 容器组装、文件输出与边界情况

- [ ] 推导Huffman缓冲区的严格大小上界；当前按元素数分配64倍字节并额外保留4096字节。
- [ ] 检查空量化索引、极大索引范围和极端符号分布。

### 8. 解压与串行闭环冻结

- [ ] 在开始并行前，用最终串行配置重复执行压缩和独立解压，确认预测顺序、压缩文件SHA、逐点误差、BR和PSNR完全可复现。

### 9. 并行实现

- [x] 第一并行原型已经接入OpenMP。使用`-partition-num`和`-thread-num`；kd-tree拓扑、包围盒、tree-order坐标和ID映射通过`shared_ptr`共享，参考树和目标树只各保留一套可变状态。新增`update_point_valid_num_to_partition_root()`，各线程只修改自己的子树。预测DAG排序、参考生成、标签预测、数值预测、量化和重建均按partition动态调度。
- [x] Katrina单文件4个partition测试中，1/4线程的预测DAG、最终压缩文件和解压结果SHA分别完全一致。直接生成预测DAG的预处理由`4.374 s`降到`2.363 s`，核心压缩由`0.0345 s`降到`0.0269 s`，解压由`0.0284 s`降到`0.0230 s`。小数据的数值预测可能被OpenMP调度和串行汇总抵消，不能据此推断大数据扩展性。
- [x] 新增v9独立partition码流。每个partition内的每个条件标签通道分别执行错误位置差分和Zstd；每个点集分别执行Huffman和Zstd。文件头保存`partition_payload_begin[P+1]`的`uint64_t`相对偏移，最终按partition编号确定性拼接。Katrina第0步4分区的1/4线程码流SHA、重构SHA、`BR=0.200938`和`PSNR=58.778694 dB`完全一致；核心压缩为`0.0356/0.0287 s`，解压为`0.0304/0.0252 s`。Ocean temperature第0步4线程闭环通过，`BR=2.296377`，相对旧全局无损流`2.260146`增加`1.603%`，但核心压缩由`0.0321 s`降至`0.0180 s`。这些是登录节点单次小实验，只用于验证实现。
- [ ] 128分区时每块约`7289960/128=56953`个时空点，与原来每个time step分4块的约58790点接近。分区本身也可能通过缩短查询路径和改善cache局部性加快串行预处理，必须同时测速度和RD。
- [ ] 在更多field、EB和更大partition数量下比较全局无损流与v9独立码流，分开统计Huffman树头、payload、Zstd frame和偏移表；如额外树头造成明显BR损失，再实现共享全局Huffman codebook、每partition独立编码与Zstd的折中格式。
- [ ] 如果独立分区边界损失过大，再比较边界halo、跨分区只读参考或DAG逐层并行。细粒度DAG已有足够理论并行度，但调度和确定码流成本更高，因此不作为第一实现。

### 10. 后续压缩方向

- [ ] 尝试空间分块，并比较独立码流、局部Huffman树和局部预测DAG。
- [x] 核对Ocean速度field的读取方式。全31步每个文件均为235160个小端float32且没有NaN/Inf；由`velocityX/Y/Z`和球面坐标重新计算Zonal/Meridional，time step 0/15/30的最大差异约`1e-7`。`conn.dat`含696678条合法edge，edge两端距离中位数约33.9 km，随机点对约8922 km，因此field、坐标和connectivity没有编号错位。
- [x] Ocean速度field相邻time step显示强时间相关。time step 0和1的相关系数为Meridional `0.989718`、X `0.998355`、Y `0.998330`、Z `0.989063`、Zonal `0.998230`；平均绝对变化只占第0步值域的`0.127%～0.162%`。这与较差的空间BR形成强对比，应将跨time step提升为近期主线。
- [x] 实现时间专用跨time-step原型：`-cross-time-predictor-experimental constant_extrapolation`使第0步使用空间预测，后续点在点集ID一致时只用上一重构时间步的同一点，否则回退空间预测。Ocean velocityZonal前2步、`REL=1e-3`下，空间/时间专用的`BR,PSNR`分别为`3.261320,65.134117`和`2.445161,65.078090`，BR降低`25.025%`；独立解压PSNR一致且最大误差比小于1。
- [x] 实现时空联合adjust coord原型`py_codes/trash/t5_cross_time.py`：把前2个time step展平到`(x,y,z,t*s)`，第一次采样保留被选空间点的所有时间副本，训练4×4常量下三角矩阵，并调用C++完成压缩和独立解压。初始`s`为中位空间最近邻距离的`1`倍时得到`BR=2.813046,PSNR=65.233470`；`0.25`倍时得到`2.796275,65.234687`，相对空间基线BR降低`14.259%`。两者有效时间/空间距离比分别为`0.707/0.711`，对初值较稳定；训练约`16.6～19.7 s`。该方案效果不佳，入口已归档。
- [x] 添加时空联合参考统计。两个时间副本的最佳当前结果中，`235110/470320=49.9894%`的目标使用了同位置跨时间参考，但这些边只占全部参考边的`4.9990%`。同点相邻时间差的平均绝对值只有空间最近邻数值差的约`20.5%`，而当前pairwise loss中时间边被约9倍的空间边数量淹没。下一步比较时间/空间loss平衡及预测误差/估计码长loss，然后做五点RD、更多time step、误差传播和关键帧实验。
- [x] 实现并否定第一版固定预算局部采样，原`grouped_ratio`模式保留，新增`local_fixed`和`local_fixed_regions`便于复现。总时空样本约固定为`117580`；8个局部区域在2/4/8步下的空间最近邻中位距离只比完整网格放大`4.9%/8.3%/12.9%`，而同数量全局随机采样会放大`17.2%/65.7%/100.7%`，说明局部密度目标实现正确。但保留选中空间位置的所有时间副本会改变10近邻中的时间/空间边比例，不能作为通用方案。
- [x] 固定预算采样的velocityZonal验证结果：2步、8区域且每轮换区域的`BR,PSNR`为`2.640364,65.197375`，好于旧采样的`2.796275,65.234687`但训练由约19.7秒增至36.5秒。4步固定8区域为`2.208063,65.117477`；固定32区域改善到`2.109330,65.167335`、训练27.4秒，BR与旧采样`2.112715`相当，但PSNR比旧结果`65.447806`低`0.280 dB`。8步固定32区域第1轮即将时间/空间距离比压到`0.0343`，`BR=3.002968`；第4轮为`1.894761,65.259203`，仍差于旧采样的`1.796696,65.567124`，且内层评估数波动到`20/71/55/46`，总训练达135.5秒。因此固定样本数只能固定单次评估成本，还必须限制`max_nfev`才能限制总耗时。
- [x] 实现第二版随机四维区域采样。使用全部可用time step，从真实时空点随机选择区域锚点；区域是四维长方体，各维边长与全范围成相同比例`r=(期望总采样数/(区域数×总点数))^(1/4)`。区域允许重叠，不按实际点密度强行凑数。每个区域从同一个历史矩阵出发独立拟合`C_r`，随后平均`M_r=C_r C_r^T`，从第2轮起以`weight=0.5`和历史metric混合、归一化并Cholesky回公共`C`。旧`grouped_ratio/local_fixed/local_fixed_regions`均保留。
- [x] Slurm任务`20213821`完成Ocean velocityZonal全部31步、`REL=1e-3`的1/2/4/8区域实验，四组均完成压缩和独立解压。空间基线为`BR=3.205020,PSNR=65.603336`，上一时间步同点预测为`1.666862,65.413907`；1/2/4/8区域联合预测依次为`1.633701,65.632898`、`1.596204,65.618626`、`1.560629,65.594190`、`1.559561,65.564669`。相对上一时间步方案的BR变化为`-1.989%/-4.239%/-6.373%/-6.437%`，PSNR均高`0.151～0.219 dB`。4到8区域的BR只再降`0.068%`而PSNR低`0.030 dB`，当前优先4区域。
- [x] 1/2/4/8区域的边长比例为`0.3564/0.2997/0.2520/0.2119`，训练耗时为`14.90/13.10/16.21/21.53 s`。最终矩阵的时间/空间相邻距离比为`0.1941/0.1408/0.0917/0.0932`，跨时间参考边比例为`38.65%/49.41%/64.34%/67.93%`。Ocean点位于球壳而不是均匀填充三维包围盒，各轮实际总采样数波动较大：1区域约`12.1万～33.9万`，8区域约`28.6万～40.3万`；尽管如此训练仍只需约13～22秒。后续比较“保持均匀体积公式”和“按局部密度校正实际预算”，但不能通过删点破坏长方体区域。
- [x] Slurm任务`20213996`补齐velocityZonal全部31步的五点曲线，REL EB为`0.01/0.003/0.001/0.0003/0.0001`，所有点均通过独立解压。逐步空间预测、上一时间步同点预测和4区域联合预测的BR依次为`[0.809599,1.830966,3.205020,4.990108,6.598876]`、`[0.413621,0.903580,1.666862,2.905933,4.291507]`和`[0.329522,0.812511,1.560629,2.783437,4.155315]`。按现有`log(BR)`三次拟合口径，上一时间步相对空间BD-rate为`-45.302%`，4区域联合相对空间为`-50.199%`，联合相对上一时间步为`-8.920%`，最后一项共同PSNR区间为`[46.748,85.061] dB`。逐点结果见`adjust_coord_history/ocean_velocityZonal_cross_time_4region_rd.csv`。
- [x] Slurm任务`20214144`在全部31步、`REL=1e-3`下筛选velocityX、velocityY和temperature的field独立4区域矩阵，训练各需`16.32/17.71/17.79 s`，全部压缩均通过独立解压。velocityX自己矩阵与复用velocityZonal矩阵的`BR,PSNR`为`1.417919,65.684009`和`1.414332,65.702287`，共享矩阵反而略好；velocityY为`1.411733,65.718905`和`1.421670,65.692395`，自己矩阵只低`0.699%` BR；temperature为`0.441465,66.301348`和`0.463549,66.258262`，自己矩阵低`4.764%` BR且高`0.043 dB`。相对上一时间步同点预测，三个field自己矩阵的直接BR分别低`7.352%/8.337%/9.671%`。这些迁移结果只保留为历史观察；由于共享规则难以跨数据集复用，后续仍按field独立训练。结果见`adjust_coord_history/ocean_cross_time_field_screen.csv`。
- [x] Slurm任务`20214409`完成temperature全部31步五点曲线，比较逐步空间、上一时间步同点、temperature专属4区域矩阵和共享velocityZonal矩阵，全部通过独立解压。按现有三次拟合口径，上一时间步相对空间BD-rate为`-72.393%`，专属联合相对空间为`-78.112%`，专属联合相对上一时间步为`-20.743%`，专属矩阵相对共享矩阵为`-5.113%`，最后一项共同PSNR区间为`[47.756,85.562] dB`。专属矩阵的五个`BR`为`[0.041411,0.170901,0.441465,0.991349,1.793926]`，共享矩阵为`[0.044581,0.180587,0.463549,1.032299,1.847437]`；结果见`adjust_coord_history/ocean_temperature_cross_time_4region_rd.csv`。
- [ ] velocityY的0.7%单点专属矩阵收益暂不值得立即补曲线。对尚未覆盖的Ocean field分别训练4区域四维矩阵并做单个中间EB筛选；不再测试共享矩阵分组。
- [x] 扩展前4个Ocean velocityZonal time step的原型实验，`REL=1e-3`下空间/上一步/四维联合的`BR,PSNR`分别为`3.256753,65.184569`、`2.031255,65.110703`和`2.112715,65.447806`。四维联合相对空间BR降低`35.128%`，`74.6596%`的目标使用同位置跨时间参考，这些边占全部参考边的`14.7253%`；学到的相邻时间/空间最近邻中位距离比为`0.2844`，训练约22.1秒。
- [x] 扩展前8个Ocean velocityZonal time step的原型实验，`REL=1e-3`下空间/上一步/四维联合的`BR,PSNR`分别为`3.238786,65.366518`、`1.812268,65.279474`和`1.796696,65.567124`。四维联合相对空间BR降低`44.526%`，比上一步方案BR再低`0.859%`且PSNR高`0.288 dB`；`86.7681%`的目标使用同位置跨时间参考，这些边占全部参考边的`29.0119%`。相邻时间/空间距离比降至`0.1681`，训练约44.2秒；比例采样导致训练耗时近似随time step数线性增长，进一步说明应改为固定总预算的局部采样。
- [x] velocityZonal和temperature的时空联合实验已经同时报告逐步空间、上一时间步同点和联合四维近邻曲线；联合方案相对上一时间步同点的BD-rate分别为`-8.920%/-20.743%`。
- [x] 实现离线一阶内插。首尾时间步使用空间预测，之后按层次顺序递归处理区间中点；点集ID一致时使用两端重构值按物理时间线性插值，否则回退空间预测。Ocean velocityZonal全部31步、五点曲线的BR为`[0.279264,0.654280,1.258877,2.287109,3.552531]`，PSNR为`[47.653100,57.398609,66.398325,76.477892,85.737637]`；相对逐步空间、上一时间步同点和4区域联合预测的BD-rate分别为`-61.170%/-29.065%/-22.455%`。`REL=1e-3`已完成独立解压，PSNR完全一致，最大误差比为`1.000001675`。结果见`adjust_coord_history/ocean_velocityZonal_time_interpolation_rd.csv`。
- [ ] 测试一阶内插中用首文件预测末文件是否优于末文件自身的空间预测；同时测试将长时间序列按固定关键帧间隔分段的一阶内插、只使用过去重构值的因果线性外推`2*v[t-1]-v[t-2]`，以及根据点集标签或重构参考值在时间预测不可靠时无需selector地确定性回退到空间预测。不考虑不规则物理时间间距。
- [ ] 在获得更多多time step数据后统一比较纯空间、纯时间和联合时空预测：先判断每个数据是否总能由两个纯方案之一胜出，再判断联合方案是否存在稳定且不可替代的收益。此前不继续微调混合selector或联合四维训练。
- [x] 用当前4参考点、8轮、EB公比0.8配置筛选Katrina attr0全部162步的四维联合压缩。4区域矩阵训练`72.989 s`，128个独立partition和16线程生成公共DAG需`88.984 s`，核心压缩/解压为`3.097/2.491 s`，即`87.386/108.641 MB/s`，峰值内存`30,196,672 KiB`。REL `1e-3`得到`BR=0.582223,PSNR=72.699202 dB`；在该PSNR处分别对现有五点曲线插值，纯空间UNSZ、上一时刻标签UNSZ和UMC约为`0.357962/0.266721/0.604674 bit/value`，因此联合方案分别变化`+62.65%/+118.29%/-3.71%`。压缩效果和速度均不值得补五点曲线，矩阵、历史和Slurm日志仅作为否定性实验保留。
- [x] 补做固定标签预测器后的Katrina数值方案消融。标签固定为第0步空间预测、后续161步使用上一time step同位置标签；分别取得现有逐time step空间数值流和128个独立partition的统一四维数值流。由于标签修正流与量化索引流独立，混合方案保留联合文件的全部元数据与数值Zstd负载，以跨时间标签Zstd负载替换联合标签负载，并额外保守计入每个time step 8字节的标签流长度。五点估算BR为`0.025569/0.125678/0.535158/1.317791/2.346592`，PSNR为`42.464965/56.079583/72.700781/91.343448/110.795295 dB`；相对上一time step标签＋空间数值的同PSNR BD-rate为`+56.349%`。Slurm任务`20484242`使用32 CPU，5分07秒完成，峰值RSS约25.0 GiB。结果见`katrina_previous_label_joint_value_rd.csv`。
- [x] 比较Katrina标签的高阶因果Lorenzo和时间端点一致预测。数值始终使用逐time step空间预测；三阶、四阶分别使用`Q(3l[t-1]-3l[t-2]+l[t-3])`和`Q(4l[t-1]-6l[t-2]+4l[t-3]-l[t-4])`，历史不足时回退上一时刻；端点方案按层次时间顺序处理，左右端点二分类标签一致时采用端点标签，否则回退空间标签。全部162步、REL `1e-3`下，空间、一阶、三阶、四阶和端点方案的总BR为`0.605492/0.520777/0.530874/0.530846/0.526853`，标签BR为`0.103340/0.018625/0.028722/0.028693/0.024700`，标签错误率为`1.6143%/0.2068%/0.3691%/0.3688%/0.2870%`。全部独立解压通过且PSNR同为`81.640594 dB`。二阶Lorenzo量化后恒等于上一标签，不实现；Katrina保持一阶。脚本和结果为`t100_benchmark_katrina_temporal_label_predictors.py`及`results/katrina_temporal_label_predictors.csv`。
- [ ] 明确联合四维方案的时间访问模型。当前离线统一预测顺序可以引用未来time step；若论文同时讨论流式或随机访问，需要另加只允许过去参考的因果版本或关键帧版本。
- [x] 在adjust后的四维kd-tree上完成`P=1/8/32/64/128`独立分区。Ocean velocityZonal全部31步、`REL=1e-3`的`BR,PSNR`依次为`1.560629,65.594190`、`1.557423,65.597280`、`1.553778,65.589169`、`1.552851,65.595960`、`1.554812,65.581011`。预处理耗时依次为`106.225/99.897/90.065/88.016/85.758 s`，压缩耗时为`1.703/1.574/1.471/1.452/1.433 s`。64分区BR最低且相对全局预处理加快`17.14%`；128分区最快且相对全局预处理加快`19.27%`。这一个field和EB中没有观察到边界RD损失，仍需在更多field和EB上复查。结果见`adjust_coord_history/ocean_velocityZonal_cross_time_partition.csv`。
- [ ] 比较time step独立压缩与跨time step共享公共预测DAG、Huffman频率统计或Zstd字典；共享无损模型只有在树头或跨时间重复模式占用可观BR时才继续。
- [x] 新增`-continuous-reference-array-experimental`，将目标编号、参考起点、参考编号和权重复制到连续数组，同时保留旧`vector`路径用于逐字节对照。Ocean temperature和Katrina attr0各重复10次，数值预测阶段分别加快`5.90%/4.77%`，总压缩加快`5.63%/3.33%`，解压加快`4.83%/5.29%`；两者压缩文件SHA、BR、PSNR和独立解压结果均不变。当前两套结构并存会增加预处理和峰值内存，下一步让连续数组成为唯一内部主存储后再做正式性能实验。结果见`py_codes/trash/results/continuous_reference_array_benchmark.csv`。
- [x] 连续参考数组已成为唯一运行时主存储：目标编号、参考区间、参考编号、原始权重和可选归一化权重均为连续数组，删除每目标独立`vector`及实验开关；IDW、标签、拟合、block自适应、保存/加载和DAG统计统一使用只读数组视图。旧版和新版直接生成的预测DAG逐字节相同，读取同一预测DAG时Ocean/Katrina压缩码流SHA也不变。各交替重复10次，新版Ocean预处理/数值预测/总压缩/解压分别加快`3.84%/1.79%/2.69%/0.31%`，Katrina分别加快`7.21%/5.21%/1.47%/3.65%`；标签阶段的微小波动被总时间覆盖。结果见`py_codes/trash/results/continuous_reference_array_unique_benchmark.csv`。
- [x] 新增`-global-order-before-round-split-experimental`：先对全部点运行一次反向D-only剔除并反转得到全局解码顺序，再按原有每轮点数切分，不再随机分轮。Ocean temperature time step 0五点BD-rate相对原方案改善`13.652%`；Katrina attr0 time step 0恶化`1.683%`，两者独立解压均通过。Ocean每轮三个坐标维度的span均约为全局`99%`以上；Katrina第7/8轮降到`[46.9%,30.9%,3.4%]`和`[39.6%,20.5%,2.1%]`，验证了局部聚集风险。该方案保留但不设默认，随新数据一起判断是否需要空间覆盖约束。结果见`py_codes/trash/results/global_order_before_round_split_rd.csv`。
- [x] 新增利用参考数据搜索公共预测顺序的`-reference-data-order-experimental <time step列表>`。它保留随机分轮、空间邻近的10个路由候选和反向“好点后置”框架，但为每个候选寻找当前仍有效的同点集IDW近邻，在多个参考文件上计算按对应点集值域归一化的绝对预测误差，并优先剔除平均误差最小的候选；最终反转后，好预测的点位于轮内后部。这是一次离线顺序搜索，不是模型训练；结果必须用`-dag-cache`保存，正式压缩其他time step不读取参考文件。
- [x] 第一轮held-out实验使用Katrina time step `40/80/120`作为参考数据，在`0～4`上用当前二次方adjust矩阵和5个REL EB测试。参考数据辅助顺序所得`BR,PSNR`依次为`(0.166098,54.900836)/(0.195547,59.614901)/(0.240543,64.174538)/(0.308678,68.758555)/(0.408373,73.453897)`，相对反向D-only的BD-rate为`-5.434618%`。首次公共预处理由`4.628 s`增至`8.416 s`，逐文件核心压缩基本不变；所有组合均通过独立解压PSNR检查。脚本和结果为`t65_benchmark_training_data_aware_order.py`及`results/training_data_aware_order_benchmark.csv`。
- [x] Ocean temperature held-out实验使用time step `8/16/24`作为参考数据、`0～4`测试，并使用field专属空间adjust矩阵。参考数据辅助顺序所得`BR,PSNR`依次为`(0.139343,47.899505)/(0.289206,51.952774)/(0.563191,56.227663)/(1.001687,60.804660)/(1.647484,65.669105)`；反向D-only的BR为`0.198986/0.390373/0.715693/1.210818/1.909964`，五点BD-rate为`-25.541716%`。首次公共预处理由`2.819 s`增至`4.666 s`，逐文件核心压缩略快；全部组合通过独立解压PSNR检查。脚本和结果为`t66_benchmark_ocean_training_data_aware_order.py`及`results/ocean_temperature_training_data_aware_order_benchmark.csv`。
- [x] 使用默认8轮+0.8完成三组参考/测试位置和全部held-out time step验证。Katrina参考`40/80/120`、`20/80/140`、`10/70/130`并分别测试其余159步，参考数据辅助顺序相对D-only的总BD-rate为`-4.024466%/-3.570563%/-3.660484%`，逐步平均为`-4.512444%/-3.855079%/-3.873397%`；分别改善154/159、159/159和155/159步，最差单步为`+0.879888%/-0.187899%/+0.739721%`。Ocean temperature参考`8/16/24`、`4/15/26`、`2/15/28`并分别测试其余28步，总BD-rate为`-27.613417%/-27.713132%/-27.450648%`，84个“划分×time step”组合全部改善，最差单步仍为`-25.916583%`。同一实验中，全局排序在Katrina总BD-rate为`+0.964301%～+1.041230%`，在Ocean为`-12.236038%～-12.271751%`。Katrina/Ocean参考数据顺序首次预处理约`9.50～9.68/4.94～5.08 s`，D-only首次约`4.65/2.19 s`，加载后均约`0.15～0.25 s`且逐文件压缩没有变慢。Slurm任务`20338893`用4核、峰值约559 MB并在`13:14`内完成；所有文件均独立解压验证。脚本、完整逐步曲线和汇总沿用历史文件名`t67_validate_training_data_aware_order.py`、`results/training_data_aware_order_validation.csv`和`results/training_data_aware_order_validation_summary.csv`。
- [x] 曾为参考数据辅助顺序加入对数量化误差代理，将可选评分定义为对多个参考文件平均`log2(1+|error|/(2×REL_EB×point_set_range))`。Katrina/Ocean temperature使用原参考time step并测试前5个held-out步骤的五点曲线；相对当前平均相对绝对误差评分，代表REL EB `0.02/0.01/0.005/0.00125`在Katrina的BD-rate为`-0.286685%/-0.248477%/-0.002432%/-0.150050%`，Ocean为`+0.219725%/-0.010035%/-0.083267%/+0.166821%`。新评分首次预处理与原评分均约为Katrina `9.3～9.6 s`、Ocean `4.8～5.2 s`，加载结构后的逐文件压缩和解压没有稳定差异。共同参数0.01只有极小正收益，并引入代表EB依赖，决定放弃且从活动C++路径删除。历史脚本和结果为`t69_benchmark_training_order_log_q_score.py`、`results/training_order_log_q_score_benchmark.csv`和`results/training_order_log_q_score_summary.csv`。
- [x] 固定相同held-out集合比较1/2/3个训练文件。Katrina固定排除`40/80/120`，分别训练`80`、`40/120`和`40/80/120`，总BD-rate为`-2.545707%/-3.354168%/-4.024466%`，改善145/159、154/159和154/159步，最差单步为`+2.205309%/+1.104118%/+0.879888%`；首次预处理为`6.51/7.94/9.54 s`。Ocean固定排除`8/16/24`，分别训练`16`、`8/24`和`8/16/24`，总BD-rate为`-26.245928%/-27.266607%/-27.613417%`，三者均改善28/28步；首次预处理为`3.21/4.20/5.02 s`。三个训练文件在两个数据上均得到最低总BR，两个文件分别获得三文件约83.3%/98.7%的收益。Slurm任务`20340160`用4核、峰值约539 MB并在`6:15`内完成，所有文件独立解压通过；结果追加在`t67`的完整CSV和汇总CSV中。
- [x] 比较完整近邻集合上的静态逐点训练分数与当前随参考集合动态重算的分数。Katrina/Ocean temperature前5步五点曲线中，静态分数相对动态分数的BD-rate为`+6.336121%/+42.165027%`，相对D-only为`+0.849468%/+4.714826%`；首次公共预处理分别从`9.222559/4.878707 s`降到`4.223152/2.154114 s`，约快`2.18/2.27`倍，但速度已经与D-only的`4.332269/2.128302 s`基本相同。说明动态参考集合正是训练数据顺序获得额外压缩收益的关键，不能把分数完全冻结。全部20组压缩均通过独立解压PSNR检查；脚本和结果为`t70_benchmark_static_training_score.py`、`results/static_training_score_benchmark.csv`和`results/static_training_score_summary.csv`。
- [x] 在动态参考数据辅助顺序中缓存预测误差：候选点再次出现时，只有上次评分使用的近邻列表发生变化才重新计算。Katrina/Ocean temperature在有历史评分的再次检查中，列表未变比例为`70.270612%/71.551285%`；折算到全部评分请求，缓存命中率为`63.242945%/64.395061%`，分数实际计算次数从`4176050/2351230`降至`1534993/837154`。首次公共预处理由`9.532142/4.878707 s`降至`6.194606/3.281140 s`，加速`1.538776/1.486887`倍。进一步让多个参考文件共用每个目标—近邻组合的距离和IDW权重，预处理继续降至`5.238416/2.843241 s`。最后把未归一化IDW权重与近邻编号一起缓存：缓存减少到9或8个有效点而不触发重查时，编号和权重一起压紧，只重新计算D及各参考文件的加权预测；预处理降至`4.803201/2.386934 s`，比上一版再快`1.090609/1.191169`倍，相对最初实现累计快`1.984539/2.043922`倍。相同版本的D-only为`4.316451/2.014985 s`，参考数据辅助搜索只慢`11.277%/18.459%`。额外权重缓存仅在此实验路径分配，容量为`point_num×(max_ref_num+1)`个double；两数据优化前后的预测DAGSHA均完全一致，并分别通过独立磁盘解压PSNR检查。结果见`results/training_score_cache_benchmark.csv`、`results/reference_data_geometry_weight_benchmark.csv`和`results/reference_data_weight_cache_benchmark.csv`。
- [ ] 扩展参考数据辅助顺序到Ocean其他field和新数据，加入参考数据与测试数据时间分布明显偏移的划分，判断它能否稳定成为field级效果优先方案。
- [x] 在Ocean layerThickness上比较IDW与RBF各自的D-only和参考数据辅助顺序。相对4点IDW+D-only，4点IDW+参考数据顺序降低`22.330%` BD-rate，10点RBF+参考数据RBF评分降低`22.877%`，而10点RBF+D-only退化`8.960%`。RBF只比IDW参考顺序额外改善约`0.55`个百分点，却增加约`8.1%/10.5%`核心压缩/解压时间，首次预处理约为`11.61 s`而IDW为`3.41 s`。该field采用IDW参考数据顺序，说明参考数据辅助DAG与RBF预测器应独立选择。结果见`results/ocean_layerThickness_reference_order.csv`。
- [x] 用当前3文件训练口径重新消融Ocean layerThickness的adjust coord。二次距离loss矩阵使四条IDW/RBF、D-only/参考数据顺序曲线退化`5.338%～7.926%`；预测码长近似loss相对原坐标IDW参考数据顺序仍退化`1.335%`。该field保持原坐标，失败矩阵不进入正式矩阵目录。
- [x] 将参考数据辅助顺序与当前Katrina预测器组合，并严格排除参考time step `40/80/120`后测试其余159步的五点RD。相对各自D-only顺序，4点IDW、5点IDW和8点Matérn-3/2 RBF分别降低`2.525%/2.707%/1.809%` BD-rate；相对当前4点IDW D-only，三者总变化为`-2.525%/-4.157%/-16.068%`。进一步新增`-reference-data-rbf-score-experimental`，让最终使用RBF的点集在搜索顺序时也用同一RBF预测误差评分，并保持异类参考点处退回IDW。它相对RBF D-only降低`4.926%` BD-rate、相对当前4点IDW D-only总计降低`18.733%`；首次公共DAG预处理为`12.705 s`，IDW评分为`6.523 s`，后续结构加载均约`1.6～1.7 s`。全部组合独立解压PSNR一致；脚本和结果为`py_codes/trash/t114_benchmark_katrina_reference_order_predictors.py`与`py_codes/trash/results/katrina_reference_order_predictors_rd.csv`。
- [x] 继续针对Katrina参考数据RBF顺序调优异类参考点回退、参考数和排序候选数。IDW回退指数从`3.6`增加到`12`仍持续改善；显式使用最近同类点相对指数`3.6/12`分别降低`8.830%/2.149%` BD-rate。新回退下10点、shape `0.01`相对8点再降低`2.036%`，4/5/6点均明显退化。排序时每次比较的局部目标候选从10增加到`16/24/32/48/64`，20步筛选分别改善`1.035%/2.415%/2.963%/3.734%/3.658%`；完整159步五点曲线中48候选仍改善`1.761%`。最终10点Matérn-3/2、shape `0.01`、nugget `1e-6`、48候选、cache-10配置相对4点IDW D-only降低`23.298%` BD-rate，相对UMC约降低`60.938%`，核心压缩/解压约为`129.1/194.4 MB/s`，分别约比UMC慢`1.17/1.20`倍。压缩和独立解压PSNR一致。清理后保留最终CSV；空间调优入口现为`t1`内部函数，正式命令行只接受JSON配置名称或路径；完整中间扫描可从提交`4b71e558`恢复。
- [x] 增加参考排序训练文件没有收益：相对3个等间隔参考time step，1个文件退化`3.986%` BD-rate，5个退化`0.496%`，9个仅改善`0.042%`；保持3个。10点RBF排序缓存阈值从10降为`9/8/6`分别退化`1.166%/1.824%/3.663%`，效果优先版保持任一参考点失效即重查。曾尝试从12/16/24/32个空间近邻中优先选择在参考文件上长期与目标同点集的10点，但相对只取最近10点分别退化`7.255%/14.691%/22.761%/26.747%`；说明牺牲局部几何的代价大于避免最近同类点回退的收益，核心路径已删除。

### 11. 实验资源规则

- [ ] 只在最终吞吐率对比时申请独占节点。BR、PSNR、BD-rate和正确性实验使用1核共享任务；反向双树排序的实测峰值RSS约341 MB，低于Anvil单核附带的1.896 GB内存。

## 已完成：批量正确性与性能基线

- [x] 完成磁盘压缩—解压检查。Ocean temperature单点集和Katrina attr0多点集均已闭环；Katrina前3个时间步分别解出123652、123719、123873个正常点，确认每个time step独立保存和解码自己的标签。
- [x] 使用当前点集格式批量复测Katrina attr0的162个时间步和Ocean temperature的31个时间步，在REL EB 0.01、0.001、0.0001下完成与UMC的BR、PSNR、预处理吞吐率和压缩吞吐率比较；全部UNSZ结果均通过磁盘解压和逐点误差检查。
- [x] 增加UNSZ和UMC的核心解压吞吐率并完成独占节点串行性能对比；计时均包含内存中码流解码和预测重建，不包含文件读取、输出写入、误差统计与重排。论文当前不统计峰值内存。
- [x] `s1_decompress`会对每个点按所属点集的`abs_eb`检查误差；`t1_start_c_code.py`现在默认在UNSZ压缩后执行磁盘解压、触发逐点检查，并核对压缩端和解压端的总PSNR。Katrina attr0前3个时间步在REL EB 0.01下检查通过，两端总PSNR均为52.557412。
## 已完成与当前结论：压缩顺序与逐轮误差上限

- [x] 在同一预测轮内联合考虑“候选目标到上一个目标的空间距离”和“候选目标当前的可预测性”来选择下一个目标。空间距离用于增强相邻量化索引的连续性；可预测性用于把容易预测的点推迟到本轮后部。保留当前只按空间距离排序的方案作为基线。
- [x] 已实现第一版确定性几何可预测性顺序`-predictability-order-experimental`：每次先取离上一个目标最近的10个未排序目标，再选择其中离当前有效参考集合最远、因而预计最难预测的点；选中后立即加入参考集合。它只依赖坐标和已经确定的预测顺序，解压端可重建，不使用当前文件真实值。Katrina和Ocean单文件均通过磁盘解压PSNR检查。
- [x] 前5个文件、默认`round_point_ratio=0.8`的初步等PSNR插值结果：几何可预测性顺序使Katrina BR降低约1.19%，Ocean降低约19.65%。预处理时间分别由约2.45秒增至7.39秒、由1.76秒增至4.86秒；当前实现是验证算法收益的原型，尚未优化重复最近邻查询。
- [x] 在独占节点上用Katrina全部162个time step和Ocean全部31个time step完成当前顺序与几何可预测性顺序的5点RD曲线。几何可预测性顺序的BD-rate分别为Katrina `-1.430%`和Ocean `-14.821%`，全部20次压缩均通过磁盘解压PSNR一致性检查。Katrina平均核心压缩吞吐率降低1.39%、解压吞吐率提高5.80%；Ocean分别提高1.61%和1.74%。主要代价仍是共享预处理：登录节点小样本中Katrina约由2.45秒增至7.39秒、Ocean约由1.76秒增至4.86秒，因此下一步应优化排序查询，而不是修改压缩循环。
- [x] 保留与正向“难点优先”不同的反向“好点后置”方案`-reverse-predictability-order-experimental`。从最后一轮向前处理，每轮的有效参考集合包含所有更早轮点和本轮尚未剔除点；每次在离上一个剔除点最近的10个候选中剔除最近参考距离最小的点，最终反转剔除序列。前5文件等PSNR插值中，Katrina和Ocean的BR分别降低约0.16%和10.89%；两者单文件磁盘解压验证通过。
- [x] 完成反向“好点后置”方案的全time-step 5点RD曲线。相对默认顺序，Katrina全部162个time step的BD-rate为`-0.499%`，Ocean全部31个time step为`-7.571%`，全部压缩均通过磁盘解压PSNR一致性检查。它确实有效，但弱于正向“难点优先”的`-1.430%`和`-14.821%`，说明两种机制并不等价。
- [x] 新增反向距离—分布顺序`-reverse-distribution-order-experimental`：每次在离上一个剔除点最近的10个候选中，为每个候选寻找最多10个当前有效参考点，分别按目标—参考平均平方距离`D`以及`t6_stat.py`的归一化分布分数`S2=2D/mean_ref_ref_squared_distance`排名，以两项排名之和剔除预计最好预测的点。Katrina和Ocean前5文件等PSNR BR分别降低约0.69%和18.89%，共享预处理约为16.27秒和18.23秒，约为简单反向方案的2.1倍。
- [x] 完成反向距离—分布顺序的全time-step 5点RD曲线：当前O(k)版本在Katrina的BD-rate为`-0.862%`，优于简单反向方案的`-0.499%`但弱于正向方案的`-1.430%`；Ocean为`-15.732%`，优于简单反向方案的`-7.571%`，也略优于正向方案的`-14.821%`。全部20次压缩均通过磁盘解压PSNR一致性检查。说明参考点空间分布确实包含额外可预测性信息，但不同数据集适合的排序机制不同。
- [x] 将`S2`中的参考点两两平方距离从O(k²)改为质心恒等式`Σ_{i<j}|xi-xj|²=kΣ_i|xi-x0|²-|Σ_i(xi-x0)|²`的O(k)计算；以第一个参考点为原点避免大坐标相减消去。Katrina单文件的压缩文件大小、BR、PSNR和SHA-256均逐字节不变，预处理由约16.16秒降至15.86秒，说明当前主要成本是重复kd-tree查询和临时对象，而不是k=10的两两距离循环。Ocean因大坐标下浮点运算次序改变了少量排序，全量BD-rate由O(k²)版的`-15.758%`轻微变为O(k)版的`-15.732%`，结论不变。
- [x] 修复并重新运行`t6_stat.py`，在Katrina attr0时间步80的10000个随机目标—参考组合上用独立30%测试集预测`log(error)`。仅几何指标模型的Spearman相关为0.332；其中方向合向量`|mean((xi-p)/|xi-p|)|`的置乱重要性最高，超过最近参考点距离，`S2`和三维凸包分数单独较弱。参考值范围和按反距离权重计算的参考值标准差远强于几何指标：仅参考值模型的Spearman相关为0.703，全部指标为0.772。该结论目前只覆盖Katrina的一个时间步和随机参考集合。
- [x] 检查`t6_stat.py`几何距离指标的共线性：最近、最远、平均和RMS距离的Spearman互相关均超过0.95，平均与RMS距离达到0.999813。去掉平均距离后，几何模型预测误差排名的Spearman相关仅由0.332降至0.328，其重要性主要转移给RMS距离和最近距离；进一步只保留最近距离及非距离几何指标时相关为0.312。平均距离几乎没有独立信息，后续可解释评分优先保留最近距离和方向合向量。
- [x] 完成反向距离—分布顺序的前5文件评分消融。最近有效参考点距离就是已有的`-reverse-predictability-order-experimental`，在Katrina和Ocean分别节省0.160%和10.889% BR；平均平方距离`D only`分别节省2.071%和21.995%；`S2 only`分别节省`-0.118%`和12.980%；`rank(D)+rank(S2)`分别节省0.686%和18.853%。D在两个数据集上都是新最佳反向评分，S2单独较弱，并且加入D后仍会干扰D的排序。
- [x] 完成D-only反向顺序的全time-step五点RD曲线。Katrina全部162个time step的BD-rate为`-1.731%`，优于最近距离的`-0.499%`、D＋S2的`-0.862%`和正向方案的`-1.430%`；Ocean全部31个time step为`-18.734%`，优于最近距离的`-7.571%`、D＋S2的`-15.732%`和正向方案的`-14.821%`。全部20组压缩—解压均通过PSNR一致性检查。说明随机参考集合上的单指标Spearman相关性不能替代完整顺序实验，当前应以平均平方距离D作为默认候选评分。
- [x] 开始公平优化正向“难点优先”搜索，保留原最近距离方案并新增`-forward-mean-distance-only-order-experimental`：每次在距离上一个目标最近的10个候选中，优先选择到最多10个已有参考点的平均平方距离D最大的点。前5文件同PSNR实验中，Katrina的BR节省由原正向1.191%提高到1.501%，Ocean由19.648%提高到19.734%，证明正向评分仍有优化空间；但仍低于反向D-only的2.071%和21.995%，当前不做完整RD曲线。
- [x] 完成正向评分消融，全部采用“分数越大越难预测、越早处理”的同一框架。Katrina前5文件同PSNR BR节省分别为：D 1.501%、方向集中度`-1.781%`、S2 `-1.342%`、`rank(D)+rank(方向集中度)` `-0.207%`、`rank(D)+rank(S2)`0.534%；Ocean分别为19.734%、2.634%、4.767%、12.336%和13.889%。D-only在两个数据集上都是最好正向评分，但仍低于反向D-only的2.071%和21.995%；方向集中度和S2单独较弱，与D等权组合也会干扰D，因此不运行新正向方案的全time-step RD曲线，保留所有实验开关供更多数据集复查。
- [x] 优化D-only串行预处理并保持码流逐字节相同。新增专用评分路径，跳过无用的最近距离、方向集中度、S2和排名排序，复用查询队列，并缓存候选点的D及其参考点；由于反向参考集合只删除点，只要缓存参考点仍有效，D就不需要重新查询。Katrina和Ocean前5文件的缓存命中率分别为60.97%和61.27%，共享预处理由16.383/12.312秒降至8.978/6.754秒，分别加速1.82倍和1.82倍。两者优化前后的组合压缩文件SHA-256、BR和PSNR完全相同，磁盘解压PSNR检查通过。
- [x] 新增反向方向集中度`-reverse-direction-only-order-experimental`、最近距离＋方向集中度`-reverse-direction-order-experimental`以及方向集中度＋`S2`的`-reverse-direction-distribution-order-experimental`，均只使用坐标并可由解压端重建。前5文件等PSNR实验中，方向集中度单独在Katrina和Ocean分别节省0.226%和15.233% BR，最近距离＋方向集中度分别节省`-0.116%`和18.562%，方向集中度＋`S2`分别节省0.470%和14.327%。方向集中度＋`S2`在Katrina优于单独方向集中度、在Ocean反而更差，且两者均未超过相应数据集上的最好方案，因此不做全time-step测试。`t6_stat.py`中的误差相关性不能直接替代最终BR实验。
- [x] 实验性加入三种几何EB序列，均由`-eb-schedule-experimental <round|point|round_point> <起始倍率>`启用：`round`逐轮增长且最后一轮使用完整用户EB，`point`沿全部预测点增长且最后一点使用完整EB，`round_point`在每轮内部从本轮倍率逐点增长到下一轮倍率且最后一轮使用完整EB。所有倍率不大于1，压缩和解压采用相同外部参数且不写入文件头；默认路径保持不变。
- [x] Katrina attr0和Ocean temperature前5个文件、REL EB 0.01的初步结果表明，全局逐点序列会让数量巨大的后期轮长期使用偏小EB，BR明显恶化；温和逐轮调度有收益。`round_point 0.9`相对默认值使Katrina的BR从0.260487降至0.259932、PSNR从52.109增至53.020，使Ocean的BR从0.345780降至0.344889、PSNR从47.217增至47.573。单文件磁盘解压和逐点误差检查通过。
- [x] 新增真正以相邻两轮EB倍率为参数的`round_ratio`和`round_point_ratio`，保留上述以第一轮倍率为参数的原方案。用默认方案在REL EB 0.01、0.008、0.006、0.004、0.002下建立BR–PSNR曲线，并插值估算相同PSNR的默认BR。Katrina前5个文件上`round_ratio 0.7`和`round_point_ratio 0.7`分别节省约9.69%和9.56% BR；Ocean上`round_ratio 0.9`和`round_point_ratio 0.9`分别节省约14.70%和14.93% BR。倍率过小会恶化结果：0.5在Katrina分别增加约3.97%和0.53% BR，在Ocean分别增加约42.31%和33.74% BR。`round_point_ratio 0.7`完成单文件磁盘解压和逐点误差检查。
- [x] 当前统一采用`round_point_ratio=0.8`，不根据运行时数据调节；压缩和解压的默认值均已修改，原始固定EB方案可用`-eb-schedule-experimental none 1`保留为实验基线。
- [x] 在独占节点上用Katrina全部162个time step和Ocean全部31个time step完成`round_point_ratio=0.8`与`none`的5点BR–PSNR曲线，REL EB为0.02、0.01、0.005、0.0025、0.00125。三次多项式积分得到的BD-rate分别为Katrina `-7.120%`和Ocean `-5.757%`，重叠PSNR区间分别为[61.312,77.862]和[46.597,63.562]；全部20次压缩均通过磁盘解压PSNR一致性检查。0.8使平均核心压缩吞吐率分别降低约2.13%和3.00%，平均核心解压吞吐率分别降低约0.36%和2.66%，当前认为RD收益明显高于串行速度代价。

## 已完成与当前结论：串行性能优化

- [x] 实验性地按照预测顺序重排内部坐标、数据、标签和参考点编号，使预测循环连续写入目标数据并提高参考点读取的缓存局部性。Ocean单点集显著加速，Katrina多点集略微减速；原始编号路径和实验开关均保留，BR、PSNR和逐点结果不变。
- [x] 将量化索引和Huffman符号固定为`int32_t`，计算后先检查范围再转换；Huffman树头的有符号offset改为按原始位模式读写，文件格式升级为v7。正权重反比例预测位于参考值凸包内，REL EB不小于约`2.33e-10`时理论上可由有符号32位量化索引覆盖；ABS模式和局部拟合预测器若越界会明确报错。Katrina attr0和Ocean temperature各一个文件在REL EB 0.0001下完成磁盘压缩—解压验证，量化索引范围分别为`[-2484,2507]`和`[-2226,2076]`。同一独占节点上与临时`int64_t`版本各重复5次：Katrina总压缩慢0.65%、解压快0.35%，Ocean总压缩快0.07%、解压快0.59%，端到端均可视为不变；Katrina/Ocean的无损编码阶段分别慢11.5%和2.8%，说明当前Huffman实现的32位模板实例没有速度优势。BR仅由0.279005降至0.278998、由0.346908降至0.346899。保留`int32_t`主要是为了量化索引内存减半和明确取值范围，不把它作为速度优化宣传。

## 已完成与当前结论：并行可行性

- [x] 新增`-parallelism-stats`并统计当前D-only预测参考图的真实DAG，而不是把预测轮直接当作并行层。Katrina有417642个点、4176365条边和930个依赖层，work/span并行度为449.08，只有15.55%的目标直接依赖上一个目标；Ocean有235160个点、2351545条边和553层，并行度为425.24，直接依赖上一个目标的比例为13.64%。按每点等成本和逐层barrier估算，8线程理想加速分别为7.910倍和7.899倍，利用率为98.87%和98.74%。说明空间连续顺序没有把整个图变成长链，现有全局预测DAG理论上具有充足的细粒度并行性。

## 已完成与当前结论：点集标签与参考结构

- [x] 验证“上一time step预测标签＋当前空间DAG预测数值”的逐目标融合。该组合没有复用标签和数值的IDW权重，但会合并目标遍历、量化索引收集和部分临时数组操作。Katrina全部162步、REL `1e-3`、4参考点/cache-4下三次交替重复：分阶段、融合原始权重、融合预归一化数值权重的核心压缩/解压吞吐率几何平均分别为`119.298/187.105`、`170.885/221.110`和`164.629/225.018 MB/s`。三种路径的BR均为`0.520777`、PSNR均为`81.640594 dB`、组合压缩文件SHA-256均为`ee73c15cf1c0975a0e6fbaa5da5677c85426786f7c1158c5073dae719df52c3a`，独立解压结果也相同。融合原始权重相比分阶段路径加快`43.242%/18.174%`；预归一化权重相对融合原始权重使压缩慢约`3.66%`、解压快约`1.77%`，因此默认使用原始权重。融合差值还包含专用循环相对通用分阶段路径减少分支、除法、统计和临时数组的收益，不能把全部加速归因于同步本身。结果见`py_codes/trash/results/katrina_cross_time_label_fused_time.csv`。

- [x] 使用`-prediction-order-label-experimental`比较按预测顺序和原始点编号顺序编码标签预测错误位置。Katrina全部162个time step上，标签字节数由1569979增至1580348，总BR由0.279004624增至0.280230673；压缩吞吐率由71.520降至67.834 MB/s，解压吞吐率基本不变。预测顺序标签在BR和压缩速度上都更差，因此继续默认按原始编号编码。
- [x] 新增`-fused-two-point-set-prediction-experimental`，在串行、非partition的两点集IDW路径中复用同一参考数组。压缩端一次遍历同时累加标签权重和真实点集的数值加权和；解压端先恢复目标标签，常量点集直接重构，只有非常量点集再次读取参考值。Katrina全部162步、REL `1e-3`、4参考点/cache-4下交替重复3次，核心压缩从`55.469`提高到`60.954 MB/s`，加快`9.889%`；解压从`70.068`提高到`80.071 MB/s`，加快`14.275%`。六次运行的BR均为`0.605492`、PSNR均为`81.640594 dB`，组合压缩文件SHA-256均为`6750206091b82e21a36e882219510c23507a90918abd446cbfd792717ac0acff`。当前格式不变，开关只影响执行路径；后续需推广到独立partition。三个以上点集仍受逐标签通道解码顺序限制，需单独设计按目标交错恢复方案。结果见`py_codes/trash/results/katrina_fused_prediction_time.csv`。
- [x] 将已有`-normalized-weight-experimental`扩展到融合两点集路径。标签使用预归一化权重仍得到相同二分类；正常数值目标的前`actual_ref_num`个参考点全部同类时直接点积，混入其他点集时退回未归一化权重筛选与现场归一化。Katrina全部162步、REL `1e-3`、4参考点/cache-4中，20928463次正常数值预测有19010520次命中，比例为`90.835720%`。三次交替重复的核心压缩从`61.248`提高到`62.890 MB/s`，加快`2.681%`；解压从`79.213`提高到`80.492 MB/s`，加快`1.615%`。六次运行的BR、PSNR和组合SHA均相同。结果见`py_codes/trash/results/katrina_fused_normalized_weight_time.csv`。
- [x] 删除融合压缩路径的逐点临时量化索引数组及其后的全点二次扫描。融合循环现在按预测轮遍历，在生成量化索引后直接写入对应点集和Huffman分组，同时更新逐点集和逐轮统计。Katrina全部162步、REL `1e-3`、4参考点/cache-4下各重复3次：原始权重路径由`61.248`提高到`64.152 MB/s`，加快`4.741%`；预归一化权重路径由`62.890`提高到`65.510 MB/s`，加快`4.166%`。新旧结果的BR均为`0.605492`、PSNR均为`81.640594 dB`，组合压缩文件SHA-256均为`6750206091b82e21a36e882219510c23507a90918abd446cbfd792717ac0acff`。旧两趟结果见`katrina_fused_normalized_weight_two_pass_time.csv`，当前一趟结果见`katrina_fused_normalized_weight_time.csv`。
- [x] 细分原`point_set`阶段后发现Katrina一个规则、两个点集的分类本身每文件只需约`0.0012 s`，主要成本其实是默认EB调度每文件重新生成417642个倍率，约需`0.00875 s`，162步累计约`1.42 s`。两个点集现使用规则类型移到循环外的专用分类循环；与当前field数据无关的`none/round/point/round_point/round_ratio/round_point_ratio`倍率在第一次生成后由全部time step复用，解压端同样复用。三种DAG影响力模式依赖当前文件的点集标签，仍逐文件计算。Katrina全部162步、REL `1e-3`、4参考点/cache-4、融合且预归一化路径重复3次后，核心压缩/解压几何平均达到`107.011/138.288 MB/s`；相对只使用专用分类、尚未缓存倍率时的`67.889/80.661 MB/s`提高`57.626%/71.443%`。BR仍为`0.605492`、PSNR仍为`81.640594 dB`且组合SHA不变。结果见`py_codes/trash/results/katrina_fused_cached_eb_time.csv`。
- [x] 将当前文件、上一time step和时间插值历史的运行期点集编号从`uint32_t`改为`uint8_t`。配置格式原本已经限制最多256个点集，因此没有减少接口能力；Katrina每个time step的当前标签数组由约1.67 MB降为约0.42 MB。全部162步、REL `1e-3`、4参考点/cache-4、融合且预归一化路径重复3次后，核心压缩/解压由`107.011/138.288`提高到`111.233/144.599 MB/s`，即提高`3.945%/4.564%`。BR仍为`0.605492`、PSNR仍为`81.640594 dB`，组合压缩文件SHA-256保持不变。结果见`py_codes/trash/results/katrina_uint8_point_set_id_time.csv`。每字节保存两个点集编号的4-bit方案虽然能再减半标签内存，但会增加解包操作和并行写同一字节的数据竞争，暂不实现。
- [x] 为融合、预归一化的两点集路径实现固定4参考点专用循环。标签修正符号和相关临时数组由`uint32_t`缩为`uint8_t`；默认原始编号标签流直接按`pred_id`读取；解压按轮遍历，不再为每个文件生成417642个`uint32_t` round编号；固定4点分支一次读取四组参考编号、标签和归一化权重，直接计算两个标签权重和，全同类时执行固定4项数值点积，只有不匹配时才退回通用筛选；默认数据无关EB序列还缓存倍率倒数，使量化用乘法代替逐点除法。Katrina完整162步、REL `1e-3`、4参考点/cache-4重复3次，核心压缩/解压达到`136.984/198.991 MB/s`，相对`uint8_t`点集编号版本的`111.233/144.599 MB/s`累计提高`23.151%/37.616%`。BR仍为`0.605492`、PSNR仍为`81.640594 dB`，组合SHA-256仍为`6750206091b82e21a36e882219510c23507a90918abd446cbfd792717ac0acff`。尝试让四组编号和权重跨越标签修正分支一直保留到数值重构，使解压下降到`183.964 MB/s`，推测是寄存器压力增大，因此已经回退。保留结果见`katrina_uint8_label_direct_time.csv`、`katrina_cached_inverse_eb_time.csv`、`katrina_ref4_fixed_fast_path_time.csv`和`katrina_ref4_decompress_reuse_time.csv`。
- [x] 继续排查逐文件大数组复制和无效写入。Huffman编码过去为每个符号分配并清零64字节输出缓冲；根据当前树头和最长码字推导，将安全上界缩为`value_num*(2*sizeof(T)+1)+64`，`int32_t`即约9字节/符号。密集符号路径直接在最终`tree.vecfreq`中计数，删除同样大小的局部`freq`数组。Zstd解压新增指针和长度入口，直接读取总容器中的标签和Huffman子流。压缩端将每文件重构数组的全零填充改成保留容量直接覆盖；DAG保证任何参考点都已在当前文件中重构。逐文件PSNR计算并入主程序本来就要执行的误差界限和总PSNR扫描，核心压缩不再第二次遍历全部点。Katrina完整162步、REL `1e-3`、4参考点/cache-4三次测试中，Huffman缓冲修改使无损阶段中位时间由`0.227262`降至`0.185624 s`；合并重复验证后该阶段外的`0.0585 s`全点扫描降为0。相同机器状态下，融合预归一化压缩由`136.491`提高到`144.985 MB/s`，约`6.223%`，解压为`199.203 MB/s`且没有稳定变化；BR、PSNR和组合SHA不变。解压端曾同样取消重构数组清零，但吞吐率由`199.411`降至`196.779 MB/s`，推测顺序清零对随后散乱参考访问具有缓存预热作用，因此恢复。结果见`katrina_before_copy_elimination_rerun.csv`、`katrina_copy_elimination_decompress_clear.csv`和`katrina_duplicate_validation_removed.csv`。
- [x] 融合两点集热循环顺序维护当前参考区间起点，每个目标只读取一次下一个`pred_ref_begin`，随后由同一对边界构造参考编号、未归一化权重和预归一化权重视图。原实现三次调用访问函数，每次都重复读取同一对边界。Katrina完整162步、REL `1e-3`、4参考点/cache-4、融合且预归一化路径两组各重复3次，压缩吞吐率分别为`148.376/147.057 MB/s`，相对修改前稳定结果`144.985 MB/s`提高约`1.4%~2.3%`；解压分别为`214.225/213.041 MB/s`，相对`199.203 MB/s`提高约`6.9%~7.5%`。两组BR均为`0.605492`、PSNR均为`81.640594 dB`、组合SHA-256均为`6750206091b82e21a36e882219510c23507a90918abd446cbfd792717ac0acff`。结果见`katrina_single_ref_boundary_load.csv`和`katrina_single_ref_boundary_load_repeat.csv`。
- [x] 已尝试跨time step复用融合标签修正工作数组，省去后续每文件约0.42 MB数组的重新分配和清零。Katrina三次测试中，压缩吞吐率由上一组`147.057`降到`144.413 MB/s`，原始权重路径也由`126.201`降到`124.257 MB/s`；节省的写入量太小，交换并延长vector生命周期没有带来收益，因此回退。结果保留在`katrina_reused_label_scratch.csv`。
- [x] 已尝试把融合路径的预归一化权重由`double`缩为`float`。Katrina BR仅由`0.605492`变为`0.605494`，PSNR由`81.640594`变为`81.640612 dB`；压缩最快一组达到`153.190 MB/s`，但解压降至约`206~211 MB/s`。而且保留double/float运行时分支会使未启用float的默认double解压也从约`213`降至约`205 MB/s`。该方案只有约2%~4%的压缩收益，却损害解压和默认路径，因此整体回退。结果见`katrina_float_normalized_weight.csv`和`katrina_float_normalized_weight_grouped_branch.csv`。
- [x] 已尝试把每点DAG参考区间边界从`uint64_t`缩为`uint32_t`，Katrina边界数组可由约3.34 MB降到1.67 MB。两组各3次测试中，归一化路径相对同组原始权重路径的压缩优势只有`15.499%/14.800%`，低于64位版本同组的`16.525%`，解压也没有稳定收益，因此恢复64位。结果见`katrina_uint32_ref_begin_isolated.csv`和`katrina_uint32_ref_begin_isolated_repeat.csv`。
- [x] 删除融合4点预测函数中的逐点内部一致性检查：目标和参考点集编号只能为0/1已经由两点集配置、分类和标签解码保证，归一化权重数量已经由`prepare_pred_refs_structure()`验证，因此不再在约2093万次数值预测中重复判断。外部输入和预测DAG验证仍然保留。Katrina完整162步、REL `1e-3`两组各3次测试中，压缩吞吐率为`160.047/157.175 MB/s`，相对上一版`148.376/147.057 MB/s`提高约`6.9%~8.8%`；解压为`214.693/215.579 MB/s`，基本不变。BR、PSNR和组合SHA不变。结果见`katrina_removed_redundant_hot_checks.csv`和`katrina_removed_redundant_hot_checks_repeat.csv`。
- [x] 继续把融合路径中可推导的计数移出逐点循环。标签通道目标数等于处理完的`pred_ref_id`，总标签错误数等于唯一标签通道的错误数，数值预测数等于全部非常量点集的点数；解压处理数也等于循环结束后的`pred_ref_id`。二进制位置差分解码器只会生成0/1，因此融合解压不再逐点检查符号上限。Katrina完整162步、REL `1e-3`三次测试达到`160.235/217.818 MB/s`压缩/解压；相对紧邻上一组`158.399/217.751 MB/s`约提高`1.159%/0.031%`。BR仍为`0.605492`、PSNR仍为`81.640594 dB`且组合SHA不变。结果见`katrina_removed_more_redundant_hot_checks.csv`和`katrina_post_loop_fused_counters.csv`。
- [x] 删除融合量化热循环中仅用于诊断的逐点`qb_num/qb_min/qb_max`更新。每个点集的量化索引数量在循环结束后由各Huffman分组vector长度汇总；单Huffman模式直接取其预处理阶段本来就会计算的最小值和最大值，多Huffman实验模式才额外顺序扫描分组。Katrina完整162步、REL `1e-3`两组各3次测试的压缩吞吐率为`164.317/161.286 MB/s`；第二组含一次登录节点同步降速，但归一化相对原始权重路径的优势仍由上一项`30.680%`提高到`31.136%`，第一组为`31.362%`，说明收益虽小但方向稳定。BR、PSNR、组合SHA和打印的量化范围均不变。结果见`katrina_reuse_huffman_qb_range.csv`和`katrina_reuse_huffman_qb_range_repeat.csv`。
- [x] `constant_extrapolation`零阶外插过去在压缩和解压主循环中把完整`current_decompressed_data`及`point_set_id_vec`复制到历史数组。现在延后到当前文件输出与误差验证结束后交换`current/previous`两个vector的所有权，下一步重用换回的旧容量，删除每个time step的数据和标签全量复制。Katrina全部162步、REL `1e-3`的压缩—独立解压通过：当前4参考点通用路径中，纯空间为`BR=0.605492,PSNR=81.640594 dB`及`87.833/121.229 MB/s`压缩/解压；只用上一时刻同位置预测标签为`BR=0.520777`、PSNR不变及`121.430/197.971 MB/s`；直接用零阶外插数值和一阶内插在Katrina上均明显增加BR，不继续作为该数据的候选。
- [ ] 继续分别处理其他大型数组操作。时间插值目前把每个已恢复端点完整复制进`reconstructed_time_*`，可转移当前buffer所有权，但全部端点仍必须占用内存；prediction-order在压缩输入和解压输出各执行一次全点散射，只有让文件I/O、误差验证和跨时间历史直接理解内部编号才能减少；多点集第1个以后条件标签流是紧凑变长流，为按预测顺序随机读取而建立的内部`uint8_t`数组不能直接删除。这些均不影响当前纯空间、原始编号布局的Katrina核心路径。

## 已完成与当前结论：无损编码调研

- [x] 阅读Jinyang Liu等人在IEEE BigData 2021发表的《Improving Lossy Compression for SZ by Exploring the Best-Fit Lossless Compression Techniques》。论文保留SZ的自定义Huffman，再在其输出字节上比较briefLZ、compress、deflate、fari、gipfeli、lzfse、zling和Zstd；5个数据集和多个EB上的结论是Zstd相对Zlib具有近似压缩比和更高速度，整体压缩速度最多提升约40%。这支持当前`Huffman+Zstd`作为合理基线，但没有回答是否应该替换Huffman。

## 逐点预测器oracle

- [x] Ocean velocityZonal time step 0、REL `0.001`、4个block上的28候选逐点oracle：基线为`BR=3.639088,PSNR=66.153392 dB`；oracle沿自身重构链递归选择真实误差最小的预测器后，量化流的估算完整BR为`1.964110`，降低`46.027%`，PSNR升至`68.948101 dB`。但28类selector的理论最低码长为`3.852738 bit/point`，最好的Huffman+Zstd仍需114043字节，即`3.880 bit/point`；计入selector后估算BR为`5.844055`，反而增加`60.591%`。这说明候选族存在很大的上限收益，主要瓶颈是无法廉价传递逐点决策。下一步测试解码端可重现的逐点LOO选择，以及将候选缩减为少数预测器后的binary/few-class oracle。诊断开关为`-pointwise-oracle-statistics-experimental`，脚本为`t60_benchmark_pointwise_oracle.py`。
