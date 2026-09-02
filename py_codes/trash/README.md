# 历史Python实验

这里保存已经完成、当前不再作为项目入口，但以后可能复用其分析或实验逻辑的Python程序。

- 根目录只保留当前使用的`t0_start_single_c_code.py`、`t1_start_c_code.py`、`t2_adjust_coord.py`、统一可视化程序`t3_visualize.py`和跨时间入口`t5_cross_time.py`。
- `t5`～`t68`中的主要分析与消融程序移到这里；相互导入关系保持原文件名。其中`t64_benchmark_katrina_coord_scale.py`比较Katrina原坐标adjust、等范围scale以及二者组合的五点RD，`t65_benchmark_training_data_aware_order.py`和`t66_benchmark_ocean_training_data_aware_order.py`保存第一轮小规模训练顺序实验，`t67_validate_training_data_aware_order.py`使用三组训练位置、全部held-out time step和1/2/3个训练文件正式验证Katrina与Ocean temperature，`t68_benchmark_round_size_growth.py`比较8轮下不同目标点数增长比例。
- `results`保存这些实验产生的历史CSV；`results/adjust_coord_history`保存已被正式矩阵取代的采样原型和区域数消融矩阵。关键结论已经整理到根目录的`TODO.md`和`TODO_DETAIL.md`。
- `slurm`保存已完成但可能复用的Slurm实验脚本，`results/slurm_logs`保存对应的原始标准输出；日志由`.gitignore`忽略，只用于本地复查。
- 归档程序中的部分路径按它们原先位于项目根目录时编写。重新启用某个实验时，应先把它移回根目录，或者修正其中的`project_root`和输出路径。

没有保留以下内容：

- 仅通过修改另一个脚本的全局变量来启动实验的薄包装器；
- 已被最终基线取代的adjust_coord中间矩阵。

这些被清理的文件仍可从清理前的本地Git提交`978753a5`恢复。
