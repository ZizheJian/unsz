#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:30:00
#SBATCH --job-name=unsz_slice_rd
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_per_slice_rd_128/unsz/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
if [[ ${SLURM_CPUS_PER_TASK:-0} -ne 128 ]];then echo 'UNSZ per-slice RD requires 128 allocated CPUs'>&2;exit 2;fi
export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=2
export OMP_PROC_BIND=spread,close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
result_root=results/ocean_large_per_slice_rd_128/unsz/16_fields_8_partitions
mkdir -p "$result_root"
for rel_eb in 1e-1 1e-2 1e-3 1e-4 1e-5;do
    output_dir="$result_root/rel_$rel_eb"
    if [[ -f "$output_dir/summary.json" ]]&&jq -e --argjson rel_eb "$rel_eb" '.rel_eb==$rel_eb and .cpu_num==128 and .field_worker_num==16 and .partition_num==8 and .field_num==95 and .slice_num==36264 and .max_error_over_eb<=1.01' "$output_dir/summary.json">/dev/null;then echo "SKIP completed REL=$rel_eb";continue;fi
    mkdir -p "$output_dir"
    resource_path="$output_dir/resources_job_${SLURM_JOB_ID}.txt"
    srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task=128 --cpu-bind=none /usr/bin/time -v -o "$resource_path" ./unsz_env/bin/python -u benchmarks/two_level/run.py --rel-eb "$rel_eb" --field-worker-num 16 --output-root "$output_dir" --launcher benchmarks/two_level/run_five_ebs.slurm
done
