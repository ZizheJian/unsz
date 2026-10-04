# RBF kernel粗筛

固定RBF10、shape=0.03、nugget=1e-6，在Katrina、Tsunami、LES、E3SM和Haiyan的正式流程及完整五点RD曲线上比较gaussian、exponential、matern32、matern52和inverse_multiquadric。

这一轮只用于kernel粗筛。同一数值shape在不同kernel中不代表完全相同的有效尺度，因此差距较小的kernel不能据此直接淘汰；下一轮会为候选kernel测试额外shape。

每个field相对当前matern32计算共同PSNR区间上的分段线性BD-rate。多field数据集在field间取几何平均，最终五个数据集等权。

Katrina诊断确认五个kernel均有417632个目标成功生成RBF权重，仅最开始的10个目标使用最近点fallback，因此kernel间的BD-rate差异并非fallback数量造成。单次权重生成时间分别为gaussian 0.242873 s、exponential 0.263141 s、matern32 0.289553 s、matern52 0.299472 s和inverse_multiquadric 0.209117 s。

五数据等权BD-rate相对matern32分别为gaussian -5.849154%、exponential -5.562051%、matern52 -4.316823%、inverse_multiquadric -5.986111%。Exponential在五个数据集汇总中均未退化，IMQ取得最低总BD-rate但测速波动较大。Haiyan诊断也确认五个kernel均有2187822个目标成功生成RBF权重，仅最开始10个目标fallback；其速度差异不是求解失败数量造成。
