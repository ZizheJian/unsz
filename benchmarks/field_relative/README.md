# Ocean-large per-field error-bound experiments

This experiment measures whether tensor compressors narrow their rate-distortion gap from UNSZ when they receive a per-field error bound and can process the complete tensor. It compares UNSZ,SZ3-ND,and QoZ 2.0 level 4. Each field uses one absolute error bound computed as `REL*(field_max-field_min)`,where `-99999` is excluded from the range. The complete stored tensor,including exceptional values,is passed to each compressor and checked after decompression.

Fields with one height slice use `[time,point]`. Fields with 80 or 81 height slices use `[time,height,point]`. The point index is always the fastest dimension. QoZ level 4 divides its tuning sample block repeatedly by the shortest dimension,so representing `[time,1,point]` as a three-dimensional tensor reaches a zero sample-block size and triggers an integer division by zero in upstream `SZInterp.hpp`. Removing the singleton axis preserves the tensor and follows the normal two-dimensional API.

SZ3 and QoZ 2.0 are pinned by `third_party/SZ3` and `third_party/QoZ`. The QoZ wrapper explicitly sets `QoZ=4`,`TUNING_TARGET_RD`,and ABS mode. Reading independent slices,the field-range scan,error validation,and independent partition-buffer allocation and release use the assigned CPU group. These phases are excluded from reported compression and decompression time,and their separate timings are recorded in each field result.

Build the tensor baselines:

```bash
cmake -S benchmarks/field_relative -B build/field_relative -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build/field_relative --parallel 2
```

Run five error bounds on `timeMonthly_avg_vertVelocityTop`,the largest vertical field:

```bash
mkdir -p results/ocean_large_field_relative_pilot
sbatch --cpus-per-task=8 --time=01:30:00 benchmarks/field_relative/run_one_field_five_ebs.slurm unsz
sbatch --cpus-per-task=64 --time=01:00:00 benchmarks/field_relative/run_one_field_five_ebs.slurm qoz
```

The CPU allocations provide memory; each program remains single-threaded. The field shape is `[12,81,3693225]`. Each result directory contains the exact binary,manifest,per-field JSON,logs,and summary. Temporary compressed streams are removed only after independent decompression and error validation pass. After both jobs finish,compare the curves over their common PSNR range:

```bash
./unsz_env/bin/python benchmarks/field_relative/summarize_one_field_five_ebs.py
```

The summary compares two matched-PSNR gaps on the same field: QoZ-ND versus UNSZ under per-field bounds,and SZ3-1D versus UNSZ under per-slice bounds. Positive values mean the tensor compressor needs more bits than UNSZ at the same reconstruction quality. The change between these gaps measures how much the tensor compressor benefits from the more favorable per-field protocol. The script also reports the existing all-95-field per-slice result as a cross-check.

SZ3 and QoZ divide each field along the point index and run one single-threaded compressor instance per partition. Compare two SZ3 128-CPU allocations over all 95 fields at REL `1e-3`:

```bash
first=$(sbatch --parsable --job-name=sz3_4fields_32parts --time=00:20:00 benchmarks/field_relative/run_sz3_128cpu_4x32_or_2x64.slurm 4x32)
sbatch --job-name=sz3_2fields_64parts --time=00:35:00 --dependency=afterany:$first benchmarks/field_relative/run_sz3_128cpu_4x32_or_2x64.slurm 2x64
```

The configurations are 4 fields with 32 point partitions each and 2 fields with 64 partitions each. Every field in a configuration uses the same partition count. Compression time includes point partitioning,and decompression time includes merging the reconstructed partitions. Each job supports field-level restart.

After selecting one allocation,run all five error bounds with:

```bash
sbatch benchmarks/field_relative/run_all_95_fields.slurm qoz <field-workers> <partitions-per-field>
sbatch benchmarks/field_relative/run_all_95_fields.slurm sz3 <field-workers> <partitions-per-field>
```

For each field,the five error bounds run in one process. The process reads the original field and computes its range once,then creates fresh point partitions from the unchanged original values for each error bound. It releases the current error bound's partition,compressed,and reconstructed buffers before starting the next one,so reuse does not increase the peak-memory model.
