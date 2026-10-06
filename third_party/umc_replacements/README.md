# UMC replacement files

这些文件基于官方UMC提交`52a86589de096cadfd8ab83719dc4c3e697e5886`：

- 上游仓库：`https://github.com/xuanwu02/umc.git`
- 替换文件：`include/compress.hpp`、`include/decompress.hpp`
- 修改范围：增加核心压缩/解压和数据重排计时，将误差统计移出核心计时，并将坐标与connectivity读取排除在预处理计时之外；预测、量化、无损编码和码流格式不变。

`third_party/build_umc.sh`会联网更新官方remote，确认子模块处于上述commit且没有未知源码修改，然后把这两份完整文件复制到子模块并构建。替换文件已经安装时，脚本只验证内容后重新构建。
