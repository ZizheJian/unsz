# 历史Adjust coord矩阵

此目录仅保留旧Python Adjust coord实验生成的下三角矩阵，供历史结果复现。正式压缩、解压和可视化路径均不再读取这里的文件。

当前C++预处理直接训练对角坐标缩放，并把实际使用的结果写到数据目录中的`<field>.unsz.coord_transform.txt`；格式和判据见项目根目录README的“生成坐标变换矩阵”一节。
