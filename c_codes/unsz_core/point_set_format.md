# UNSZ点集格式

不使用`-point-set-config`时全部数据属于唯一点集，codec和EB分别来自命令行`-predictor`和`-M`。使用`-point-set-config <JSON路径>`时，配置文件定义一个或多个点集，并忽略命令行中可能存在的`-predictor`和`-M`。

`rule_type=ordered`时，`labels`数组按优先级保存点集。除最后一个label外，每项都包含`method`、`eb_mode`、`eb_value`和`condition`；第一条成立的条件决定点集编号，全部不成立时属于最后一个label。最后一个label省略`condition`，`constant`还省略`eb_mode`和`eb_value`。比较符支持`<`、`<=`、`>`、`>=`、`==`和`!=`。只有一个label时也不写`condition`。

`rule_type=label_file`允许用户用任意外部代码为每个time step生成逐点`uint8`点集编号。标签文件信息写在根对象的`label_file`中，各label不写`condition`。路径必须包含`{time_step}`，相对路径以JSON所在目录为准；每个文件必须恰好包含`point_num`个编号。该方式目前不能与训练数据感知的公共预测顺序一起使用。

同一文件中的每个点集独立统计点数和值域，只在同点集的已重建点之间执行数值预测。每个非常量点集必须独立选择反比例、局部拟合或`hybrid_loo`压缩，并显式配置自己的REL或ABS EB。常量点集不配置EB，也不保存量化索引。

## 条件标签通道

`n`个点集使用`n-1`个递进二分类标签通道：

1. 通道`i`只处理尚未归入点集`0...i-1`的点。
2. 通道值0表示属于点集`i`，值1表示不属于点集`i`。
3. 已归入较早点集的点不再产生当前通道的修正bit，但在当前通道中作为值1的参考点参与后续预测。
4. 每次预测只保存0=预测正确、1=预测错误的修正bit；最后仍未分类的点自动属于点集`n-1`。

串行v7/v8格式将每个通道的修正bit按原始点编号过滤并排列，只保存值1位置的差分，再单独使用Zstd level 3压缩。并行v9格式在每个partition内按预测顺序执行相同的位置差分和Zstd。每个time step独立生成标签，因此同一空间点的类型可以随时间变化。单点集数据没有标签通道。

## 容器版本

一或两个实际点集继续使用v7格式，因此现有Ocean和Katrina码流不增加文件头开销，并且旧v7文件仍可由新版解压器读取。出现三个及以上实际点集时使用v8格式。

v7依次保存：

1. magic、版本、点数、数据类型字节数和点集数量；
2. 唯一标签通道经位置差分和Zstd后的流长度与字节；
3. 每个点集的codec、点数、最小值、最大值、量化索引数量，以及Huffman+Zstd数值流。

v8将第2项替换为标签通道数量，并为每个通道保存逻辑目标数量、Zstd字节数和Zstd字节。其余点集数值流布局与v7相同。

启用空间partition时使用v9格式。公共文件头保存每个点集的codec、点数和值域，随后保存partition数量和`partition_payload_begin[P+1]`相对偏移表。每个partition payload独立保存：

1. 每个条件标签通道的目标数量，以及错误位置差分后的独立Zstd frame；
2. 每个点集的量化索引数量，以及独立Huffman编码后再独立Zstd得到的frame。

partition之间不共享预测参考、Huffman树或Zstd frame，因此压缩和解压都可以并行执行。偏移使用`uint64_t`，payload按照partition编号顺序拼接，所以不同线程数不会改变文件内容；解压时仍需提供和压缩时相同的partition划分参数。

实验参数`-multi-huffman-experimental`支持`none`、`round`、`partition`和`partition_round`。非`none`模式下，每个点集按预测轮、partition或二者组合划分量化索引，分别建立带树头的Huffman流；分组数量、各组量化索引数量、各Huffman流长度和全部Huffman字节拼接后只做一次Zstd。该模式不写入通用文件头，解压时必须指定与压缩时相同的参数。

数值codec写入容器，但相对或绝对误差上限仍不写入。使用JSON时压缩器和解压器必须使用相同配置；不使用JSON的单点集模式则必须在两端通过`-M REL <eb>`或`-M ABS <eb>`提供相同参数。REL模式始终根据对应点集自己的最大值和最小值换算ABS EB。

预测结构仍由外部坐标重建。预测轮数、候选/实际参考点数量、距离指数、坐标变换、误差参数和实验开关都不保存在文件中，压缩和解压时必须由实验脚本保证一致。
