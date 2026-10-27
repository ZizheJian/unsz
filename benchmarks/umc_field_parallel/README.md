# UMC field-level parallel run

UMC NBP-IDW-INP processes all slices of one field sequentially because its invalid-point state and neighbor relations are updated every 10 slices. Independent fields run in parallel without changing UMC.

All formal runs allocate 128 CPUs on one node. The 32-worker and 64-worker checks assign fields to all workers. The 128-worker run creates 128 worker slots. Ocean-large contains 95 fields,so 95 workers receive one field each and the remaining 33 receive no task. The largest static worker loads are 1,920,972,and 972 slices,respectively. Fields are assigned solely by their slice counts,and all assignments attain their respective lower bounds.

Run the complete `REL=1e-3` memory and performance check:

```bash
mkdir -p results/ocean_large_umc_32_field_workers
sbatch benchmarks/umc_field_parallel/run_umc_32_field_workers.slurm
mkdir -p results/ocean_large_umc_64_field_workers
sbatch benchmarks/umc_field_parallel/run_umc_64_field_workers.slurm
mkdir -p results/ocean_large_umc_128_cpus
sbatch benchmarks/umc_field_parallel/run_umc_128_cpus.slurm
```

Every field is checked against the completed serial reference. Per-field peak RSS,aggregate cgroup memory,and the compression and decompression critical paths are recorded separately.
