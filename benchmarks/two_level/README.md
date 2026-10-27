# Ocean-large two-level parallel execution

These experiments run one UNSZ process on 128 CPU cores at REL `1e-3`. They compare 8 field workers with 16 point partitions, 16 field workers with 8 point partitions, 32 field workers with 4 point partitions, and 64 field workers with 2 point partitions. Fields that keep the original coordinates share one lazily constructed spatial DAG and its RBF weights. Fields that enable coordinate transformation build private spatial structures.

Every assignment covers all 95 fields and 36,264 slices. The deterministic size-only assignments for 8, 16, 32, and 64 workers have maximum loads of 4,812, 2,880, 1,920, and 972 slices. All four maxima are optimal when complete fields are indivisible. The 16-worker assignment is stored in `schedule.json`.

Validate the metadata, schedule, binaries, and commands without starting the experiment:

```bash
./unsz_env/bin/python benchmarks/two_level/run.py --prepare-only
./unsz_env/bin/python benchmarks/two_level/run.py --field-worker-num 8 --prepare-only
./unsz_env/bin/python benchmarks/two_level/run.py --field-worker-num 32 --prepare-only
./unsz_env/bin/python benchmarks/two_level/run.py --field-worker-num 64 --prepare-only
```

Install the current build and submit the formal run:

```bash
cmake --install build
sbatch benchmarks/two_level/run.slurm
sbatch benchmarks/two_level/run_8x16.slurm
sbatch benchmarks/two_level/run_32x4.slurm
sbatch benchmarks/two_level/run_64x2.slurm
```

The job writes its manifest, compression and decompression logs, resource report, and final `summary.json` below `results/ocean_large_two_level/`. The reported end-to-end time is the sum of the maximum compression-side and decompression-side worker times. These worker times include preprocessing and codec work while excluding field-data input, compressed output, transform-metadata input/output, and validation, consistently with the single-thread results. Temporary compressed files are deleted only after all 95 fields pass compression-side and decompression-side error checks. Pass `--keep-output` to retain them for diagnosis.
