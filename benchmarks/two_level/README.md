# Ocean-large two-level parallel execution

This experiment runs one UNSZ process on 128 CPU cores at REL `1e-3`. Sixteen field workers execute the fixed complete-field assignment in `schedule.json`. Each worker uses eight OpenMP threads and eight spatial partitions. Fields that keep the original coordinates share one lazily constructed spatial DAG and its RBF weights. Fields that enable coordinate transformation build private spatial structures.

The fixed assignment covers all 95 fields and 36,264 slices. Its maximum worker load is 2,880 slices. This is optimal when complete fields are indivisible because 37 fields contain either 960 or 972 slices, so at least five of 16 workers must receive three such fields.

Validate the metadata, schedule, binaries, and commands without starting the experiment:

```bash
./unsz_env/bin/python benchmarks/two_level/run.py --prepare-only
```

Install the current build and submit the formal run:

```bash
cmake --install build
sbatch benchmarks/two_level/run.slurm
```

The job writes its manifest, compression and decompression logs, resource report, and final `summary.json` below `results/ocean_large_two_level/`. The reported end-to-end time is the sum of the maximum compression-side and decompression-side worker times. These worker times include preprocessing and codec work while excluding field-data input, compressed output, transform-metadata input/output, and validation, consistently with the single-thread results. Temporary compressed files are deleted only after all 95 fields pass compression-side and decompression-side error checks. Pass `--keep-output` to retain them for diagnosis.
