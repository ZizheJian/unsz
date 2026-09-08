import t103_screen_katrina_rbf as benchmark

benchmark.idw_result_path=benchmark.project_root/"py_codes/trash/results/katrina_current_idw_parameter_rd.csv"
benchmark.result_path=benchmark.project_root/"py_codes/trash/results/katrina_rbf_ref8_optimized_rd.csv"
benchmark.time_step_num=162
benchmark.rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
benchmark.shape_vec=(0.01,)
benchmark.ref_num=8
benchmark.cache_threshold=8
benchmark.extra_arguments=["-rbf-filtered-fallback-idw-experimental","-fused-two-point-set-prediction-experimental","-linear-previous-label-experimental"]
benchmark.verify_decompression=True

benchmark.main()
