# SZ3-1D and PFPL slice-parallel runs on Ocean-large

`run.py --codec sz3|pfpl` compresses every Ocean-large slice independently as a one-dimensional float32 array. Slices are distributed round-robin over slice workers, and each slice uses the codec's own OpenMP threads. The absolute bound of each slice is REL times its valid range, excluding `-99999`. The benchmark binaries come from `benchmarks/tensor_1d`.

Run all five EBs on 128 CPUs, or choose another allocation:

```bash
sbatch benchmarks/slice_parallel/run.slurm sz3 32 4
sbatch benchmarks/slice_parallel/run.slurm pfpl 64 2
sbatch --cpus-per-task=64 benchmarks/slice_parallel/run.slurm pfpl 32 2
```

Results go to `results/ocean_large_per_slice_rd_<cpus>/<sz3_1d|pfpl>/<workers>_slice_workers_<threads>_threads_per_slice/rel_<eb>/`. Completed EBs are skipped. `--summarize-only` rebuilds `fields.csv` and `summary.json` from the existing per-worker slice CSVs. Field PSNR uses the valid range of the complete field, matching UNSZ, UMC, and the plots. Per-field stream sizes are checked against the single-thread results for PFPL and against results with the same internal thread count for SZ3, whose OpenMP blocking changes the stream. The compression and decompression sides each report the largest cumulative worker time. Input reads, range scans, and validation are excluded.
