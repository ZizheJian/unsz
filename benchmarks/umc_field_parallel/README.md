# UMC field-level parallel runs on Ocean-large

UMC NBP-IDW-INP processes all slices of one field sequentially, and its public MPI path cannot process the complete Ocean-large input. Each field worker therefore runs the official serial UMC on its complete fields, and independent fields run concurrently. Each worker uses one CPU.

Fields are assigned by `optimal_assignment` in `benchmarks/common.py`, which first minimizes the largest per-worker slice count and then the largest per-worker field count. With 128 workers each of the 95 fields gets its own worker.

```bash
./unsz_env/bin/python benchmarks/umc_field_parallel/run_field_workers.py --field-workers 128 --rel-eb 1e-3 --prepare-only
sbatch benchmarks/umc_field_parallel/run.slurm 128
```

Results go to `results/ocean_large_per_slice_rd_<workers>/umc/<workers>_field_workers/rel_<eb>/`. Every field must reproduce the BR and PSNR of the serial reference in `results/ocean_large_tensor_full_umc/serial_rel_<eb>`, and completed fields are skipped on restart. UMC writes its streams next to the input slices, and the runner deletes them after each field so the data directory does not run out of its file-count quota. The on-disk validation also accumulates the squared errors of valid values normalized by each slice's valid range, which gives the per-slice PSNR. Per-field peak RSS and the aggregate cgroup memory peak are recorded.
