# SZ3-1D 128-CPU allocation selection

SZ3-1D processes each original-order Ocean-large slice independently. Long-lived slice workers divide all 36,264 slices by round-robin assignment. Inside each worker,SZ3 divides one slice into contiguous point-index ranges. Every configuration therefore uses slice-level and point-level parallelism without launching one process per slice.

Build the existing SZ3-1D executable:

```bash
cmake -S benchmarks/tensor_1d -B build/tensor_1d -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build/tensor_1d --parallel 2 --target benchmark_sz3
```

Run four 128-CPU allocations as separate jobs:

```bash
sbatch --job-name=sz3_1d_128workers_1thread benchmarks/sz3_1d_parallel/run_sz3_1d_128cpu_four_allocations.slurm 128x1
sbatch --job-name=sz3_1d_64workers_2threads benchmarks/sz3_1d_parallel/run_sz3_1d_128cpu_four_allocations.slurm 64x2
sbatch --job-name=sz3_1d_32workers_4threads benchmarks/sz3_1d_parallel/run_sz3_1d_128cpu_four_allocations.slurm 32x4
sbatch --job-name=sz3_1d_16workers_8threads benchmarks/sz3_1d_parallel/run_sz3_1d_128cpu_four_allocations.slurm 16x8
```

Each run checks every reconstructed value against its slice-relative error bound and requires its compressed size to match the existing complete-dataset result made with the same number of threads per slice. Reported compression and decompression times are the respective maximum accumulated times among workers. Slurm wall time and peak RSS are recorded separately.
