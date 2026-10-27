#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:30:00
#SBATCH --job-name=umc_32_field_workers
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_umc_32_field_workers/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
if [[ ${SLURM_CPUS_PER_TASK:-0} -ne 128 ]];then echo 'This experiment requires 128 allocated CPUs'>&2;exit 2;fi
result_root=results/ocean_large_umc_32_field_workers
mkdir -p "$result_root/resources"
srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task=128 --cpu-bind=none /usr/bin/time -v -o "$result_root/resources/job_${SLURM_JOB_ID}.txt" ./unsz_env/bin/python -u benchmarks/umc_field_parallel/run_field_workers.py --rel-eb 1e-3 --field-workers 32 --output-root "$result_root"
