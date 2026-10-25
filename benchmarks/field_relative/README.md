# Ocean-large field-relative experiment

This experiment compares single-thread UNSZ,SZ3,and full QoZ 2.0 level 4. Each field uses one absolute error bound computed as `REL*(field_max-field_min)`,where `-99999` is excluded from the range. The complete stored tensor,including exceptional values,is passed to each compressor and checked after decompression.

Fields with one height slice use `[time,point]`. Fields with 80 or 81 height slices use `[time,height,point]`. The point index is always the fastest dimension. QoZ level 4 divides its tuning sample block repeatedly by the shortest dimension,so representing `[time,1,point]` as a three-dimensional tensor reaches a zero sample-block size and triggers an integer division by zero in upstream `SZInterp.hpp`. Removing the singleton axis preserves the tensor and follows the normal two-dimensional API.

SZ3 is pinned by `third_party/SZ3`. QoZ 2.0 is pinned by `third_party/QoZ`,and the wrapper explicitly sets `QoZ=4`,`TUNING_TARGET_RD`,and ABS mode. All three programs use one software thread. File input,field-range calculation,output,and validation are excluded from codec time.

Build the tensor baselines:

```bash
cmake -S benchmarks/field_relative -B build/field_relative -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build/field_relative --parallel 2
```

Run the maximum-field pilot at REL `1e-3`:

```bash
mkdir -p results/ocean_large_field_relative_pilot
sbatch --cpus-per-task=8 --time=00:30:00 benchmarks/field_relative/run_pilot.slurm unsz
sbatch --cpus-per-task=64 --time=00:30:00 benchmarks/field_relative/run_pilot.slurm sz3
sbatch --cpus-per-task=64 --time=01:00:00 benchmarks/field_relative/run_pilot.slurm qoz
```

The CPU allocations provide memory; each program remains single-threaded. The pilot uses `timeMonthly_avg_vertVelocityTop`,whose shape is `[12,81,3693225]`. Each result directory contains the exact binary,manifest,per-field JSON,logs,and summary. Temporary compressed streams are removed only after independent decompression and error validation pass.
