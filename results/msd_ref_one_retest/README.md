# 1点MSD复测：Katrina与Tsunami

2026-09-15，commit `0c2818bbaa200ae8cfd580cc6b489348f1428a7d`，单进程且`OMP_NUM_THREADS=1`。压缩器/解压器SHA256分别为`ad7a619861b32f6706ed6a0a4238fe788513aaa1031079d3fe2be075f68f0413`/`8003198dad2fbb5f5f9645956f11c849ad7535849122023f8c75d597664c187d`。所有DAG从空缓存重新生成；每个压缩码流均独立解压，压缩端与解压端PSNR相同。

两种MSD配置只改变排序评分参考点数及相同数值的评分缓存补查阈值：1或10。最终DAG和数值RBF仍使用最多10个参考点；空间参数固定为global D-only/MSD、16排序候选、Matérn-3/2、shape 0.03、nugget 1e-6、连续EB底数0.85和shift 0.5、不使用输入重排。Tsunami无坐标变换和时间预测；Katrina使用`adjust_coord_matrices/Katrina_attr0_spatial.txt`、`point_set_configs/Katrina_attr0_rbf_rel_1e-3.json`，标签零阶外插，数值零阶外插＋空间修正。**点集配置内的`method`必须是`rbf_fitting`；命令行`-predictor`不能覆盖其中的IDW方法。**

下表用UMC在REL `1e-3`的PSNR作代表点，在各曲线的相邻两个点间按PSNR对BR直接线性内插。`BR变化`是1点相对10点，负数代表1点更好；`CR/UMC`是同PSNR的UMC BR除以1点MSD的BR。UMC曲线来自既有`results/current/fields`，**没有在本轮重跑UMC**。核心时间为REL `1e-3`压缩＋解压，不含预处理。

| 数据与field | 代表PSNR/dB | BR：1点 | BR：10点 | BR变化 | CR/UMC：1点 | 核心时间1点/10点，ms |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tsunami bathy | 64.905 | 4.2408 | 4.2348 | 0.14% | 1.053× | 9.16/7.72 |
| Tsunami height | 65.464 | 4.5408 | 4.6018 | -1.33% | 0.988× | 9.23/7.81 |
| Tsunami momentum_x | 66.492 | 2.2476 | 2.2388 | 0.39% | 1.394× | 8.65/7.31 |
| Tsunami momentum_y | 66.477 | 2.3882 | 2.3408 | 2.03% | 1.316× | 8.68/7.53 |
| Tsunami velocity_x | 67.028 | 0.7005 | 0.7331 | -4.45% | 1.522× | 8.22/6.78 |
| Tsunami velocity_y | 67.158 | 0.6491 | 0.6965 | -6.80% | 1.449× | 8.16/6.54 |
| Katrina attr0 | 81.466 | 0.3364 | 0.3559 | -5.47% | 2.534× | 3524/3658* |

`*`Katrina核心时间是2026-09-15同节点四组交替测速的平均值，见下节。Tsunami六个field中，1点使height及两个velocity的同PSNR BR降低，bathy和两个momentum略升；六个field的核心时间反而都增加约15%～25%。首次生成Tsunami DAG的预处理时间为1点`1.141 s`、10点`2.322 s`；Katrina为1点`5.241 s`，上一轮10点`13.606 s`。后两者不是同轮交替速度比较，且这些登录节点单次计时不能作为正式吞吐率结论。

Katrina全部162步在REL `1e-3`与`0.001396`两点的BR/PSNR分别为`0.337513/81.537077 dB`与`0.295423/78.808002 dB`，和本日旧1点MSD实验逐位一致；两点独立解压PSNR与压缩端完全一致。Katrina10点数据来自`results/katrina_msd_reference_sweep/README.md`中的上一轮同SHA二进制实验。原始测量值见`tsunami_curves.csv`与`katrina_curves.csv`；本轮输出暂存于`/tmp/unsz_msd_retest_zE0Av0`，新生成DAG在测试结束后移入数据目录专用trash，不留活跃缓存。

Tsunami输入SHA256：coords `94a59ab3449db90b31fa28923099e19e5ef187e3ff2b15b9bb9019f462b1a163`；bathy `fba1d364fe3ce0f515b4dd8b0b4e4c0548b4a330bf01c0b59fe4c4f097d6f2a2`；height `d88135acaec164d28b479bf129171cf9e534ae53028a3d92b047dfa14c696125`；momentum_x `ca624dd911a76fb7b1032377431f11e8bab3a32cc774baefe7b907152e1098d1`；momentum_y `368c7d9a0df015ee05ededc70e3c9a9427cbe1c7fc1f3dda939cc76ccadc71d8`；velocity_x `8e849789f05fd880ff739eae395d8655721ce321b57e9fd4d930d6e5a3d6ea7e`；velocity_y `653a2c0ebc59c70edee371baf5cc4b3ece8ee058eaa552a7c9d10da3873bc0a0`。Katrina公共坐标、矩阵和点集配置的SHA256见`results/katrina_msd_reference_sweep/README.md`。

## Katrina核心时间交替复测

同一登录节点、`thread_num=1`、REL `1e-3`、全部162步，按`1→10、10→1、1→10、10→1`运行四组；每次加载相应DAG，RBF权重预计算与DAG加载都在核心计时之外。压缩和解压分别读取程序内部时间，先完整压缩，再独立解压。全部八次BR、PSNR固定为1点`0.337513/81.537077 dB`和10点`0.355880/81.466042 dB`，解压PSNR均与压缩端一致。原始计时见`katrina_core_paired.csv`。

| 组别 | 1点核心总时间/s | 10点核心总时间/s | 1点相对10点 |
| ---: | ---: | ---: | ---: |
| 1 | 3.578414 | 3.617980 | -1.09% |
| 2 | 3.516690 | 3.628936 | -3.09% |
| 3 | 3.483608 | 3.671255 | -5.11% |
| 4 | 3.517161 | 3.712656 | -5.27% |
| 平均 | 3.523968 | 3.657707 | -3.66% |

两侧使用的参考边数均为`4176365`。1点DAG中`62.89%`的目标使用预测顺序中前一个点作为参考点、最长依赖层数`552`；10点分别为`34.75%`和`279`。因此，Tsunami的“相邻依赖更多→核心更慢”并非普遍规律：在Katrina的双点集时间流程中，1点虽有更多相邻依赖，本次核心计时仍略快。具体收益只有约4%，仍应视为单节点实验结果，不把跨轮次旧数据的约10%差异当成稳定收益。
