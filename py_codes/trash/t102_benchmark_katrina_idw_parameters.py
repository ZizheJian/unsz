import t101_screen_katrina_idw_parameters as benchmark

benchmark.result_path=benchmark.project_root/"py_codes/trash/results/katrina_current_idw_parameter_rd.csv"
benchmark.summary_path=benchmark.project_root/"py_codes/trash/results/katrina_current_idw_parameter_rd_summary.csv"
benchmark.time_step_num=162
benchmark.rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
benchmark.config_vec=(
    {"name":"ref4_power3.6","ref_num":4,"cache_threshold":4,"power":3.6},
    {"name":"ref4_power3.6_cache3","ref_num":4,"cache_threshold":3,"power":3.6},
    {"name":"ref4_power2","ref_num":4,"cache_threshold":4,"power":2},
    {"name":"ref4_power2.5","ref_num":4,"cache_threshold":4,"power":2.5},
    {"name":"ref5_power2","ref_num":5,"cache_threshold":5,"power":2},
    {"name":"ref5_power2.5","ref_num":5,"cache_threshold":5,"power":2.5},
    {"name":"ref5_power3","ref_num":5,"cache_threshold":5,"power":3},
)

benchmark.main()
