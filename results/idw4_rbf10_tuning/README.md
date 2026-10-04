# IDW4与RBF10调参

第一阶段固定各数据当前正式流程、global排序、1点MSD、16个排序候选和当前EB schedule，在REL `1e-3`筛选IDW4指数及RBF10核函数与shape。每个数据的IDW4候选共享一份4点DAG，RBF10候选共享一份10点DAG；缓存只保存顺序和参考编号，加载后根据当前参数重新生成权重。

IDW4扫描指数`1/2/3.6/5/8`。RBF10固定nugget `1e-6`，扫描Gaussian `0.1/0.25/0.5`、exponential `0.05/0.1/0.25`、Matérn-3/2 `0.01/0.03/0.1`、Matérn-5/2 `0.03/0.1`和inverse-multiquadric `0.25`。

Katrina第一阶段使用原始坐标，因为当前正式矩阵专门按Matérn-3/2 RBF10、shape `0.03`训练。筛选后分别为IDW4和RBF10候选训练坐标矩阵，再以完整五点曲线决策。其余四个数据当前均使用原始坐标。

运行：

```bash
bash results/idw4_rbf10_tuning/submit_screen.sh
```
