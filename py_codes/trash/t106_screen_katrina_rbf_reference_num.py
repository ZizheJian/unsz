import t103_screen_katrina_rbf as benchmark

benchmark.shape_vec=(0.003,0.01,0.03,0.1)
benchmark.extra_arguments=["-rbf-filtered-fallback-idw-experimental"]
for ref_num in (4,5,6,8):
    benchmark.ref_num=ref_num
    benchmark.cache_threshold=ref_num
    benchmark.result_path=benchmark.project_root/f"py_codes/trash/results/katrina_rbf_ref{ref_num}_screen.csv"
    benchmark.main()
