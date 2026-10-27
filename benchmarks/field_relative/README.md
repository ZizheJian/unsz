# Ocean-large per-field error-bound experiments

This experiment measures how close tensor compressors come to UNSZ when they receive the more favorable per-field error bound and can process the complete tensor. It compares UNSZ, SZ3-ND, and QoZ 2.0 level 4. Each field uses one absolute bound, `REL*(field_max-field_min)`, where `-99999` is excluded from the range. The complete stored tensor, including exceptional values, is passed to each compressor and checked after decompression.

Fields with one height slice use `[time,point]`. Fields with 80 or 81 height slices use `[time,height,point]`. The point index is always the fastest dimension. QoZ level 4 divides its tuning sample block repeatedly by the shortest dimension, so representing `[time,1,point]` as a three-dimensional tensor reaches a zero sample-block size and triggers an integer division by zero in upstream `SZInterp.hpp`. Removing the singleton axis preserves the tensor and follows the normal two-dimensional API.

SZ3 and QoZ 2.0 are pinned by `third_party/SZ3` and `third_party/QoZ`. The QoZ wrapper explicitly sets `QoZ=4`, `TUNING_TARGET_RD`, and ABS mode. Each field is split along the point index into equal partitions, and every partition is compressed by one single-threaded codec instance. Compression time includes splitting the field into partitions, and decompression time includes merging the reconstructed partitions. Reading slices, the field-range scan, error validation, and partition-buffer allocation and release are excluded and recorded separately in each field result.

Build the tensor baselines:

```bash
cmake -S benchmarks/field_relative -B build/field_relative -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
cmake --build build/field_relative --parallel 2
```

Run all five EBs on 128 CPUs with 4 field workers and 32 partitions per field:

```bash
sbatch --job-name=sz3_field_rd benchmarks/field_relative/run_all_95_fields.slurm sz3 4 32
sbatch --job-name=qoz_field_rd benchmarks/field_relative/run_all_95_fields.slurm qoz 4 32
sbatch --job-name=unsz_field_rd benchmarks/two_level/run_five_ebs.slurm field 16
```

The five EBs of one field run in one process, which reads the field once and creates fresh partitions for every EB. Completed fields are skipped on restart. If the codec aborts on a field, the failure is recorded in `summary.json` and the worker continues with its other fields. Fields are assigned by `optimal_assignment` in `benchmarks/common.py`. UNSZ uses the same two-level runner as the per-slice protocol with a `FIELD_REL` point-set configuration, so fields still share the spatial DAG. Results go to `results/ocean_large_per_field_rd_<cpus>/<codec>/<workers>_fields_<partitions>_partitions/rel_<eb>/`, each with `fields.csv` and `summary.json` in the common format. PSNR normalizes the squared error of every valid value, which excludes `-99999`, by the valid range of the complete field. The results in `results/ocean_large_field_relative_five_ebs/` predate this statistic.
