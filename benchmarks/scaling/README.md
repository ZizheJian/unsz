# Ocean-large parallel scaling

This experiment runs UNSZ,SZ3,and PFPL on the complete 95-field Ocean-large dataset at REL `1e-3`. Each Slurm job runs one codec at one software thread count. The submission script chains the jobs with `afterany` dependencies so that a failed point does not prevent later codecs or thread counts from running.

Inspect commands without submitting:

```bash
./unsz_env/bin/python benchmarks/scaling/submit.py --dry-run
```

Submit the missing thread counts. The formal single-thread Ocean-large results are reused as the `p=1` reference:

```bash
./unsz_env/bin/python benchmarks/scaling/submit.py
```

Explicitly rerun `p=1` only when a fresh reference is required:

```bash
./unsz_env/bin/python benchmarks/scaling/submit.py --threads 1
```

Prepare and validate one configuration without running a codec:

```bash
./unsz_env/bin/python benchmarks/scaling/run.py --codec unsz --threads 4 --prepare-only
```

Results are written below `results/ocean_large_scaling/`. Each completed codec point contains a `summary.json` that is written only after all 95 fields,36264 slices,498.9 GiB of input,and pointwise error checks pass. Per-codec resource files record peak RSS with `/usr/bin/time` inside the Slurm step.

Aggregate the reused single-thread results together with every completed scaling point and regenerate the summary table and figure:

```bash
./unsz_env/bin/python benchmarks/scaling/summarize.py
```

This writes `results/ocean_large_scaling/summary.csv` and the three-panel scaling figure below `results/ocean_large_scaling/figures/`. Missing thread counts are omitted and appear automatically after their `summary.json` becomes available.

## Completed results

The completed sweep contains all 18 combinations of three codecs and six thread counts. Every point processes 95 fields,36,264 slices,and 498.9 GiB at REL `1e-3`.

| Codec | Best compression speedup | Best decompression speedup | Bit-rate range |
|---|---:|---:|---:|
| UNSZ | 6.24x at 8 threads | 9.05x at 16 threads | 0.544--0.599 bits/value |
| SZ3 | 15.52x at 32 threads | 22.21x at 32 threads | 5.480--5.554 bits/value |
| PFPL | 10.84x at 16 threads | 18.10x at 16 threads | 14.940 bits/value |

The authoritative machine-readable aggregation is `results/ocean_large_scaling/summary.csv`. The Slurm logs,per-slice CSV files,input lists,and copied executables are reproducible intermediate artifacts and are excluded from version control.

The experiment does not request memory explicitly. Runs with two or four software threads reserve eight Slurm CPUs only because the UNSZ low-partition peak exceeds the memory attached to fewer CPUs. The software thread count remains two or four respectively. The reused single-thread run also reserved eight Slurm CPUs while executing one software thread.
