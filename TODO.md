# UNSZ简要状态与TODO

本文件只保留关键结论和尚未实施的主线；完整参数、历史消融、数据集清单及弃用原因见`TODO_DETAIL.md`。

## 当前已经确认的结论

- Katrina：UNSZ明显优于UMC。其z轴严重拉伸，原始欧氏距离与可预测性不匹配，因此adjust coord收益显著；当前二次方距离loss矩阵相对旧四次方100轮矩阵改善`12.182%` BD-rate。
- Ocean：4-block自适应预测器在全部31个time step、五点RD曲线上相对UMC平均降低BR `15.843%`，中位降低`8.105%`。收益主要集中在salinity、temperature和zTop，速度field通常只有约`5%～10%`。
- LES：目前只测过`REL=0.01`且未启用adjust coord，UNSZ同EB下BR比UMC低约`19%～21%`、PSNR高约8 dB；还缺正式同PSNR曲线。
- Tsunami：二维压缩和独立解压已经闭环，但缺少可靠的UMC多EB对比和二维adjust coord。
- 当前空间基线采用反向D-only顺序、10个候选及参考点、cache重查阈值8、32点kd-tree叶节点、`round_point_ratio=0.8`、`int32_t`量化索引以及Huffman+Zstd level 3。
- `round_point_ratio=0.8`相对固定EB在Katrina/Ocean temperature上改善`7.120%/5.757%` BD-rate。32点kd-tree叶相对单点叶使Ocean/Katrina预处理缩短约`41%/44%`，不改变预测结果。

### Ocean全部31步的现有五点RD结果

下表采用4个完全独立空间block和原21候选block自适应预测器；正数表示同PSNR下UNSZ的BR低于UMC。

|field|UNSZ相对UMC的BR降低|
|---|---:|
|layerThickness|8.11%|
|salinity|36.94%|
|temperature|42.11%|
|velocityMeridional|7.82%|
|velocityX|5.60%|
|velocityY|5.99%|
|velocityZ|9.77%|
|velocityZonal|5.04%|
|zTop|21.21%|

局部加权拟合和连续混合进一步利好平滑标量field，但当前逐点计算约使核心压缩和解压慢`3～4`倍。IDW残差拟合与ridge没有收益，已经停止继续测试。

### Ocean速度field的重要新结论

- velocityZonal文件格式、float32字节序、点编号和坐标对应关系均正确。由`velocityX/Y/Z`和球面坐标重新计算Zonal/Meridional，在time step 0、15、30上的最大误差约为`1e-7`；`conn.dat`的edge两端距离中位数约33.9 km，而随机点对约8922 km，因此field、坐标和connectivity没有错位。
- 速度field空间上较难预测，但时间上很连续。time step 0和1的相关系数为：Meridional `0.98972`、X `0.99836`、Y `0.99833`、Z `0.98906`、Zonal `0.99823`；相邻步平均绝对变化只有各自第0步值域的约`0.127%～0.162%`。
- velocityZonal单time step空间adjust coord已经建立基线：仅用`velocityZonal.dat.0`训练二次方距离loss，第2轮矩阵在五点RD曲线上相对不变换降低`7.830%` BD-rate，第1/3/4轮分别降低`5.238%/5.053%/6.472%`。因此以后测试时空联合adjust coord时，必须与这个已优化空间基线比较，不能把单纯空间变换的收益算作跨时间收益。
- velocityZonal的28候选逐点oracle忽略selector时将BR从`3.6391`降至`1.9641`，但28类selector需要约`3.880 bit/point`，计入后BR反而达到`5.8441`。现有候选存在能力上限，真正困难是构造解压端可重现且廉价的逐点决策。

## 建议接下来直接做

1. 先为velocityZonal比较两种跨time-step原型：A为第0步使用空间压缩、后续点只由上一重构时间步的同一点预测；B为把time step作为第4维、联合所有时空点做adjust coord和空间近邻预测，使参考选择自动在空间与时间方向之间权衡。两者都与使用第2轮单步矩阵的逐time step空间基线比较BR、PSNR、误差传播和吞吐率，再补velocityX/Y。
2. 如果继续空间预测器主线，将28候选缩减成少数代表候选，先测试“当前block预测器与距离加权拟合”的binary oracle并计入1-bit selector；然后实现解压端可重复的逐点LOO选择，验证能否在不存selector时保留oracle收益。
3. 在Katrina和LES上验证`d^-3.5+8点/cache-6`。它在Ocean 9个field、3个time step上相对当前`d^-3.61+10点/cache-8`全部改善，平均/中位/最差field的BD-rate为`-2.09%/-2.14%/-1.04%`，并略微加速压缩和解压；跨数据验证通过后再修改默认值。

## 剩余工作：按压缩流程排列

### 0. 数据与实验基线

- 接入更多不同网格、坐标尺度、异常分布和物理场的数据，优先选择规模适中的多time step数据；先处理本地`xmdv_data/TGV`，大规模`viral-flamespeed`留作并行实验。
- 为LES补齐多EB的UNSZ/UMC同PSNR曲线，为Tsunami补齐正确的UMC二维曲线。
- 用最终候选配置为Ocean全部field重新生成统一五点RD、各阶段流长度和吞吐率，避免混用不同历史版本。

### 1. 坐标与adjust coord

- 为Ocean其余7个field训练早停线性矩阵；velocityZonal已经完成field独立基线，`velocityX/Y/Z`先共享矩阵，其余标量field独立初筛。
- 为LES速度field训练共享矩阵；实现二维adjust coord并测试Tsunami。
- 之后再比较坐标相关低阶矩阵、分块常矩阵、平方loss、对数平方loss及预测码长近似loss；必须计入参数和分块元数据。

### 2. kd-tree、参考结构和预测顺序

- 当前反向D-only已经足够稳定；只在预处理仍是瓶颈时继续减少重复查询、改显式查询栈或共享双树拓扑。
- 若继续优化量化索引连续性，保持参考DAG不变，仅重排同轮拓扑可交换目标，统计最终Zstd流和BR；不再采用已经使Ocean temperature退化`14.57%`的激进参考复用优先方案。
- 固定相同坐标、相同距离和浮点tie时的编号tie-break，保证公共预测结构跨机器可复现。

### 3. 空间预测器和block决策

- 为block采样选择加入父块/全field配置回退和置信门槛，避免速度field出现负决策收益。
- 测试少候选binary/few-class oracle、确定性逐点LOO及其计算复用；不能读取当前目标值，也不能把selector当作免费信息。
- 只有上述方案仍不足时，再考虑局部低维拟合、IDW残差的其他形式或更复杂的局部模型。

### 4. EB、量化和码流

- 联合消融预测轮数、每轮点数增长比例和逐轮EB比例；获得更多数据后再冻结通用参数，不进行逐field运行时调参。
- 比较按预测DAG影响力分配EB与当前仅按轮号分配EB。
- 按轮多Huffman只有很小且不稳定的收益，空间block分树已经确认变差；暂不继续完整RD。

### 5. 点集和标签

- 将当前两点集推广为用户定义的有序多点集分类函数；n个点集使用n-1个条件标签通道，每个点集可独立选择codec和EB。
- 每个time step独立生成标签；优先类型的点在后续标签预测中视为“不属于当前类型”，但仍可作为已知标签参与预测。

### 6. 跨time step

- 时间专用基线：第0步按空间压缩，后续点只使用上一重构时间步的同一点预测；它最直接利用时间连续性，也最容易分析误差传播。
- 时空联合方案：将每个点表示为`(x,y,z,t*s)`，把所有time step合成一个4维点集并做adjust coord，再按统一近邻结构压缩。学习到较小的有效时间尺度时，同位置跨时间点会自然成为近邻；否则仍可优先使用同时间步的空间近邻。这是自动选择空间/时间方向，不保证一定优于时间专用预测。
- 已完成Ocean velocityZonal前2步、`REL=1e-3`原型：独立空间预测为`BR=3.261320,PSNR=65.134117`；上一重构时间步同点预测为`2.445161,65.078090`，BR降低`25.025%`；四维联合预测为`2.796275,65.234687`，BR降低`14.259%`。这只是单个EB的初步结果，不是同PSNR BD-rate结论。
- 四维联合训练的第一次采样已改为对空间点采样后保留其所有时间副本。初始时间间距为空间最近邻中位数的`1`倍或`0.25`倍时，最终有效时间/空间距比均约为`0.71`，说明对初值较稳定。约50%的时空目标使用了同位置跨时间参考，但跨时间边只占全部参考边约5%。
- 当前四维pairwise loss中，每个目标通常只有1条时间边和约9条空间边，因此时间连续性仍被低估。下一步先平衡时间/空间loss权重，或改用预测误差/估计码长loss，然后再做五点RD曲线。
- 后续比较时间线性外推、空间与时间确定性混合、时空联合参考以及关键帧方案。
- 分别评估误差传播、随机访问、关键帧间隔、跨时间点集标签变化和解压并行性。
- 之后再测试共享Huffman树或Zstd字典；跨field预测继续保持低优先级。

### 7. 串行冻结与并行

- 串行算法冻结前重复压缩和独立解压，检查逐点误差、BR、PSNR、预测结构和文件SHA的确定性。
- 然后实现DAG逐层并行预测、量化和重建，用固定输出槽或prefix sum保持码流顺序；Huffman和Zstd先保持串行。
- 如果细粒度DAG调度开销过大，再比较固定kd-tree粗分区或微分区动态领取。

## 暂缓事项

- 当前无损编码需求不明确，暂不继续rANS/FSE、Golomb/Rice、varint或canonical Huffman实验。
- 公共预测结构文件只作为同网格、多field和多time step实验的本地缓存，不能作为免费压缩元数据。
- 跨field预测通常弱于field内空间或时间预测，除非新数据统计显示稳定强相关，否则不进入近期主线。
