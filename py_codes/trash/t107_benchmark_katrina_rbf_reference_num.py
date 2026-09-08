import t103_screen_katrina_rbf as benchmark

benchmark.idw_result_path=benchmark.project_root/"py_codes/trash/results/katrina_current_idw_parameter_rd.csv"
benchmark.time_step_num=162
benchmark.rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
benchmark.extra_arguments=["-rbf-filtered-fallback-idw-experimental"]
for ref_num,shape in ((4,0.003),(5,0.003),(6,0.01),(8,0.01)):
    benchmark.ref_num=ref_num
    benchmark.cache_threshold=ref_num
    benchmark.shape_vec=(shape,)
    benchmark.result_path=benchmark.project_root/f"py_codes/trash/results/katrina_rbf_ref{ref_num}_rd.csv"
    benchmark.main()
