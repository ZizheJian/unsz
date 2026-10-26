# PFPL parallel experiments

PFPL processes every original-order Ocean-large slice independently. Each long-lived slice worker receives a static round-robin list of complete slices and invokes the unchanged PFPL OpenMP implementation. The worker count and the threads used inside each slice multiply to the active CPU count. Input reading,slice-range calculation,and error validation remain outside the reported compression and decompression times.

The existing single-thread five-error-bound results remain the rate-distortion results. Parallel runs verify that every field has exactly the same input and compressed byte counts as those results.

Build the existing PFPL executable:

```bash
cmake -S benchmarks/tensor_1d -B build/tensor_1d -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build/tensor_1d --parallel 2 --target benchmark_pfpl
```

Screen three 128-CPU allocations at `REL=1e-3` as three separate jobs:

```bash
mkdir -p results/ocean_large_pfpl_128cpu_selection
first=$(sbatch --parsable --job-name=pfpl_128workers_1thread benchmarks/pfpl_parallel/run_pfpl_128cpu_128x1_64x2_or_32x4.slurm 128x1)
second=$(sbatch --parsable --job-name=pfpl_64workers_2threads --dependency=afterany:$first benchmarks/pfpl_parallel/run_pfpl_128cpu_128x1_64x2_or_32x4.slurm 64x2)
sbatch --job-name=pfpl_32workers_4threads --dependency=afterany:$second benchmarks/pfpl_parallel/run_pfpl_128cpu_128x1_64x2_or_32x4.slurm 32x4
```

Each job processes all 95 fields and 36,264 slices. Every worker is pinned to a disjoint CPU group and remains alive for its complete slice list,so the experiment does not launch one process per slice. Compression and decompression time are the maximum accumulated times among workers. The Slurm wall time separately captures input and validation overhead.
