# UNSZ field-worker and partition parallelism on Ocean-large

`run.py` runs one UNSZ compression process and one decompression process over the complete Ocean-large dataset. The CPUs are split into field workers, and each worker uses the remaining factor as OpenMP threads and spatial partitions. Fields that keep the original coordinates share one lazily constructed spatial DAG and its RBF weights.

Fields are assigned by `optimal_assignment` in `benchmarks/common.py`. It first minimizes the largest per-worker slice count exactly and then minimizes the largest per-worker field count, so small fields are not stacked on a few workers. Ocean-large fields have only three slice counts, which makes the exact search a small dynamic program.

`--protocol slice` uses the per-slice REL point-set configuration. `--protocol field` uses the matching `FIELD_REL` configuration, which converts REL with the valid range of the complete field. Both protocols run the same code path.

Validate the assignment and commands without running:

```bash
./unsz_env/bin/python benchmarks/two_level/run.py --protocol slice --field-worker-num 16 --prepare-only
```

Run all five EBs on 128 CPUs with 16 workers and 8 partitions, or choose another protocol, worker count, or CPU count:

```bash
sbatch benchmarks/two_level/run_five_ebs.slurm
sbatch benchmarks/two_level/run_five_ebs.slurm field 16
sbatch --cpus-per-task=64 benchmarks/two_level/run_five_ebs.slurm slice 8
```

Results go to `results/ocean_large_per_<protocol>_rd_<cpus>/unsz/<workers>_fields_<partitions>_partitions/rel_<eb>/`. Each directory holds `manifest.json`, both logs, `fields.csv`, and `summary.json`. UNSZ prints one `UNSZ FIELD RESULT` line per field and side, and the runner builds `fields.csv` from these lines. Field PSNR uses the valid range of the complete field, and aggregated PSNR averages the per-field NMSE weighted by value count, as in the plots. The compression side and the decompression side each report the largest cumulative worker time, including preprocessing, plus the dependency-plan time shared by all workers. Their sum is the total codec time. File input, compressed output, transform-metadata I/O, and validation are excluded.
