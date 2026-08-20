// 已归档的点集标签预测方案（2026-08-19）
//
// 每个点先用已经无损恢复的硬标签0/1进行空间分类，保存：
//     error_bit=predicted_id!=actual_id
// 解压时使用：
//     actual_id=predicted_id^error_bit
// 当前方案没有被删除，仍可通过point_set_predictor="inverse_proportion"启用。
// 归档此说明是为了与连续标签状态实验区分。

/*
uint32_t predicted_id=predict_point_set_id(pred_id,ref_ids,2);
uint32_t actual_id=point_set_id_vec[pred_id];
uint32_t error_bit=predicted_id!=actual_id;

// 解压
uint32_t actual_id=predicted_id^error_bit;
point_set_id_vec[pred_id]=actual_id;//后续点直接使用修正后的硬标签
*/
