import t101_screen_katrina_idw_parameters as benchmark

benchmark.result_path=benchmark.project_root/"py_codes/trash/results/katrina_fast_idw_rd.csv"
benchmark.summary_path=benchmark.project_root/"py_codes/trash/results/katrina_fast_idw_rd_summary.csv"
benchmark.time_step_num=162
benchmark.rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
benchmark.config_vec=(
    {"name":"ref4_power3.6","ref_num":4,"cache_threshold":4,"power":3.6},
    {"name":"ref4_power2_cache3","ref_num":4,"cache_threshold":3,"power":2},
)

benchmark.main()
