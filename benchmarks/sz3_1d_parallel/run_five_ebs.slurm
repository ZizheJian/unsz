#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:20:00
#SBATCH --job-name=sz3_slice_rd
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_per_slice_rd_128/sz3_1d/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
if [[ ${SLURM_CPUS_PER_TASK:-0} -ne 128 ]];then echo 'SZ3-1D per-slice RD requires 128 allocated CPUs'>&2;exit 2;fi
result_root=results/ocean_large_per_slice_rd_128/sz3_1d
mkdir -p "$result_root/resources"
for rel_eb in 1e-1 1e-2 1e-3 1e-4 1e-5;do
    output_dir="$result_root/32_slice_workers_4_threads_per_slice/rel_$rel_eb"
    if [[ -f "$output_dir/summary.json" ]]&&jq -e --argjson rel_eb "$rel_eb" '.active_cpus==128 and .slice_workers==32 and .threads_per_slice==4 and .field_num==95 and .slice_num==36264 and .valid==true' "$output_dir/summary.json">/dev/null;then echo "SKIP completed REL=$rel_eb";continue;fi
    resource_path="$result_root/resources/rel_${rel_eb}_job_${SLURM_JOB_ID}.txt"
    srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task=128 --cpu-bind=none /usr/bin/time -v -o "$resource_path" ./unsz_env/bin/python -u benchmarks/sz3_1d_parallel/run_all_slices.py --rel-eb "$rel_eb" --slice-workers 32 --threads-per-slice 4 --output-root "$result_root"
done
