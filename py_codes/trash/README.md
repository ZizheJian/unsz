# 历史Python实验

这里保存已经完成、当前不再作为项目入口，但以后可能复用其分析或实验逻辑的Python程序。

- 根目录只保留当前使用的`t0_start_single_c_code.py`、`t1_start_c_code.py`、`t2_adjust_coord.py`和统一可视化程序`t3_visualize.py`。效果不佳的联合四维入口已移到本目录的`t5_cross_time.py`。
- `screen_more_fields.py`、`run_spatial_field_benchmark.py`和`benchmark_field_reference_order.py`的功能已并入根目录`t1_start_c_code.py`，旧文件仅用于查看历史接口；`slurm/screen_more_fields.slurm`是对应的历史批处理入口。
- `t5`～`t83`中的主要分析与消融程序移到这里；相互导入关系保持原文件名。其中`t64_benchmark_katrina_coord_scale.py`比较Katrina原坐标adjust、等范围scale以及二者组合的五点RD，`t65_benchmark_training_data_aware_order.py`和`t66_benchmark_ocean_training_data_aware_order.py`保存第一轮小规模参考数据辅助顺序实验，`t67_validate_training_data_aware_order.py`使用三组参考位置、全部held-out time step和1/2/3个参考文件正式验证Katrina与Ocean temperature，`t68_benchmark_round_size_growth.py`比较8轮下不同目标点数增长比例，`t69_benchmark_training_order_log_q_score.py`保存平均相对绝对误差与对数量化误差代理的历史比较，`t70_benchmark_static_training_score.py`比较动态分数与完整近邻集合上的静态逐点分数，`t71_benchmark_partition_adaptive_confidence.py`比较逐partition配置与0.5% field级置信回退，`t72_train_spatial_matrices.py`使用可指定的近邻数重新训练Ocean、LES和Tsunami纯空间矩阵，`t73_compare_spatial_matrix_ref_nums.py`在其他压缩参数不变时比较最近5点和指定近邻数训练矩阵的同PSNR BR，`t74_screen_ocean_spatial_adjust.py`筛选Ocean全部field的4近邻空间矩阵。`t75_screen_partition_adjust_coord.py`错误地在局部矩阵变换后继续进行全局预测，只作为无效实验保留；`t76_validate_ocean_prediction_ggd_adjust.py`在未参与训练的time step 15和30验证Ocean预测码长近似loss矩阵；`t77_benchmark_katrina_cross_time_label_rd.py`测试Katrina上一time step同位置标签预测的五点RD并生成正式曲线；`t78`～`t81`依次完成Tsunami momentum_x预测器初筛、五点RD、预测码长loss矩阵验证及其与全局顺序的组合实验，`t82_benchmark_tsunami_quadratic_fitting.py`比较局部二次拟合使用6/8/10个参考点的五点RD，`t83_ablate_tsunami_quadratic_matrix.py`固定10点二次拟合后消融默认二次方loss与预测码长loss矩阵；历史脚本名保留原称呼，`t69/t70`对应的活动C++评分开关均已删除，需要切换到相应历史提交才能重跑。
- `t84`～`t90`筛选Tsunami momentum_x的RBF核、shape、nugget、多项式趋势和参考点数，并补齐Matérn-3/2与Matérn-5/2五点RD。
- `adjust_coord_matrices/Tsunami_momentum_x_*`、`figures/tsunami_momentum_x_*`和`results/tsunami_momentum_x_superseded/`保存该field已经放弃的坐标矩阵、矩阵训练记录、旧图、旧shape `0.03`筛选结果以及误用旧矩阵生成的current结果。当前正式配置使用原始坐标、RBF10、shape `0.1`、global排序和MSD，不应从这些归档文件恢复配置。
- `t91_benchmark_additional_datasets.py`筛选Tpost20、E3SM-h4、ADCIRC Haiyan和TAPS；对应原始数据转换现由各数据集独立的`py_codes/prepare_<dataset>.py`完成，Slurm入口为`slurm/run_additional_datasets.slurm`。
- `t92_benchmark_katrina_reference_num.py`在Katrina全部162步、REL EB `1e-3`下初筛10点/cache-8、4点/cache-4和4点/cache-3，并启用详细标签统计；`t93_benchmark_katrina_reference_num_rd.py`关闭高开销标签诊断，完成三组配置的正式五点RD和程序内部计时。
- `t94_benchmark_katrina_fused_prediction.py`在Katrina全部162步、REL EB `1e-3`、4点/cache-4下交替重复测试分离与融合的两点集IDW预测，使用程序内部计时并核对组合压缩文件SHA；原始结果位于`results/katrina_fused_prediction_time.csv`。
- `t95_benchmark_katrina_fused_normalized_weight.py`在相同数据和配置下交替比较融合路径使用原始权重与“全同类时使用预归一化权重”的快速路径，统计命中率、内部吞吐率和组合SHA；当前一趟量化索引收集结果位于`results/katrina_fused_normalized_weight_time.csv`，修改前的两趟结果保存在`results/katrina_fused_normalized_weight_two_pass_time.csv`。
- 同一个`t95`入口也用于验证两点集专用分类和跨time step复用EB倍率。专用分类、尚未缓存倍率的结果保存在`results/katrina_fused_normalized_weight_fast_classification_time.csv`，加入倍率缓存后的结果保存在`results/katrina_fused_cached_eb_time.csv`。
- `t95`还用于比较运行期点集编号从`uint32_t`缩为`uint8_t`后的吞吐率，结果保存在`results/katrina_uint8_point_set_id_time.csv`。
- `t95`随后验证紧凑标签符号、直接标签流、按轮解压、EB倍率倒数和固定4参考点专用循环；最终结果位于`results/katrina_ref4_fixed_fast_path_time.csv`，被回退的跨分支局部变量复用结果位于`results/katrina_ref4_decompress_reuse_time.csv`。
- `t95`还用于排查逐文件大型内存操作。修改前同机复测、缩小Huffman缓冲并保留解压清零、删除重复PSNR扫描的结果分别位于`results/katrina_before_copy_elimination_rerun.csv`、`results/katrina_copy_elimination_decompress_clear.csv`和`results/katrina_duplicate_validation_removed.csv`。
- `t95`随后验证融合热循环只读取一次DAG参考区间边界，结果保存在`results/katrina_single_ref_boundary_load.csv`和`results/katrina_single_ref_boundary_load_repeat.csv`。
- `results/katrina_reused_label_scratch.csv`记录复用融合标签工作数组但速度略降的回退实验。
- `results/katrina_float_normalized_weight.csv`和`results/katrina_float_normalized_weight_grouped_branch.csv`记录将预归一化权重缩为`float`后压缩略快、解压和默认double路径变慢的回退实验。
- `results/katrina_uint32_ref_begin_isolated.csv`和`results/katrina_uint32_ref_begin_isolated_repeat.csv`记录将DAG参考区间边界缩为`uint32_t`但没有速度收益的回退实验。
- `results/katrina_removed_redundant_hot_checks.csv`和`results/katrina_removed_redundant_hot_checks_repeat.csv`记录把已由外层保证的内部一致性检查移出融合4点热循环后的速度收益。
- `results/katrina_removed_more_redundant_hot_checks.csv`和`results/katrina_post_loop_fused_counters.csv`记录继续删除重复符号检查并把可推导计数移出逐点循环后的结果。
- `results/katrina_reuse_huffman_qb_range.csv`和`results/katrina_reuse_huffman_qb_range_repeat.csv`记录量化索引数量改为循环后汇总、范围复用Huffman预处理结果后的测速。
- `t96_benchmark_katrina_current_spatial.py`使用当前4参考点融合快速路径重跑Katrina五点空间RD和核心吞吐率，并与数据目录中的UMC NBP-IDW-INP曲线对齐。
- `slurm/run_katrina_current_joint_screen.slurm`在Katrina全部162步、REL EB `1e-3`下训练四维矩阵，并用128个独立partition和16线程筛选当前时空联合预测的BR及核心吞吐率。
- `results/katrina_current_joint_screen.csv`记录上述任务`20476218`的结果；联合方案同PSNR下明显差于当前纯空间和上一时刻标签方案，因此没有继续补五点曲线。
- `t97_benchmark_katrina_current_cross_time_label.py`使用当前4参考点融合路径重跑Katrina上一time step标签预测的五点RD；`t98_benchmark_katrina_cross_time_label_fused.py`在REL `1e-3`下交替重复比较分阶段、融合原始权重和融合预归一化数值权重三条路径，结果分别位于`results/katrina_current_cross_time_label_rd.csv`和`results/katrina_cross_time_label_fused_time.csv`。
- `results`保存这些实验产生的历史CSV；`results/adjust_coord_history`保存已被正式矩阵取代的采样原型和区域数消融矩阵。关键结论已经整理到根目录的`TODO.md`和`TODO_DETAIL.md`。
- `slurm`保存已完成但可能复用的Slurm实验脚本，`results/slurm_logs`保存对应的原始标准输出；日志由`.gitignore`忽略，只用于本地复查。
- 归档程序中的部分路径按它们原先位于项目根目录时编写。重新启用某个实验时，应先把它移回根目录，或者修正其中的`project_root`和输出路径。

没有保留以下内容：

- 仅通过修改另一个脚本的全局变量来启动实验的薄包装器；
- 已被最终基线取代的adjust_coord中间矩阵。

这些被清理的文件仍可从清理前的本地Git提交`978753a5`恢复。
