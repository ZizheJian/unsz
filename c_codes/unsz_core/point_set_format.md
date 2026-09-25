# UNSZ点集格式

不使用`-point-set-config`时全部数据属于唯一点集，codec和EB分别来自命令行`-predictor`和`-M`。使用`-point-set-config <JSON路径>`时，配置文件定义一个或两个点集，并忽略命令行中可能存在的`-predictor`和`-M`。

`labels`数组按优先级保存点集。除最后一个label外，每项都包含`method`、`eb_mode`、`eb_value`和`condition`；第一条成立的条件决定点集编号，全部不成立时属于最后一个label。最后一个label省略`condition`，`constant`还省略`eb_mode`和`eb_value`。比较符支持`<`、`<=`、`>`、`>=`、`==`和`!=`。只有一个label时也不写`condition`。点集分类只读取当前field的数值，不接受额外标签文件。

同一文件中的每个点集独立统计点数和值域，只在同点集的已重建点之间执行数值预测。每个非常量点集必须独立选择反比例或RBF预测，并显式配置自己的REL或ABS EB。常量点集不配置EB，也不保存量化索引。RBF的核、shape、nugget和多项式趋势次数是公共几何参数，压缩和解压时必须相同；完整候选参考集合的RBF权重会在预处理时计算并跨time step复用，点集筛选改变参考集合时现场重新求解。

## 二分类标签通道

两个点集使用一条二分类标签通道。程序预测每个点属于点集1还是点集2，并只保存0=预测正确、1=预测错误的修正bit。两个点集是对等类别，存储方式不依赖哪个点集占多数。串行v7格式将修正bit按原始点编号排列，只保存值1位置的差分，再使用Zstd level 3压缩；并行v9格式在每个partition内按预测顺序执行相同的位置差分和Zstd。每个time step独立生成标签，因此同一空间点的类型可以随time step变化。单点集数据没有标签通道。

## 容器版本

一或两个实际点集使用v7格式，因此现有Ocean和Katrina码流不增加文件头开销。

v7依次保存：

1. magic、版本、点数、数据类型字节数和点集数量；
2. 唯一标签通道经位置差分和Zstd后的流长度与字节；
3. 每个点集的codec、点数、最小值、最大值、量化索引数量，以及Huffman+Zstd数值流。

启用空间partition时使用v9格式。公共文件头保存每个点集的codec、点数和值域，随后保存partition数量和`partition_payload_begin[P+1]`相对偏移表。每个partition payload独立保存：

1. 每个条件标签通道的目标数量，以及错误位置差分后的独立Zstd frame；
2. 每个点集的量化索引数量，以及独立Huffman编码后再独立Zstd得到的frame。

partition之间不共享预测参考、Huffman树或Zstd frame，因此压缩和解压都可以并行执行。偏移使用`uint64_t`，payload按照partition编号顺序拼接，所以不同线程数不会改变文件内容；解压时仍需提供和压缩时相同的partition划分参数。

数值codec写入容器，但相对或绝对误差上限仍不写入。使用JSON时压缩器和解压器必须使用相同配置；不使用JSON的单点集模式则必须在两端通过`-M REL <eb>`或`-M ABS <eb>`提供相同参数。REL模式始终根据对应点集自己的最大值和最小值换算ABS EB。

预测DAG仍由外部坐标重建。预测轮数、候选/实际参考点数量、距离指数、RBF参数、坐标变换、误差参数和实验开关都不保存在文件中，压缩和解压时必须由实验脚本保证一致。
