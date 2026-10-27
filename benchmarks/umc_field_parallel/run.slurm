#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=08:00:00
#SBATCH --job-name=umc_field_parallel
#SBATCH --output=/home/x-zjian1/unsz/results/%x_%j.log

# 用法：sbatch [--cpus-per-task=N] run.slurm <field-workers> [rel-eb...]
# 每个worker占一个CPU；不指定EB时运行五个正式EB，已完成的field自动跳过
set -euo pipefail
cd /home/x-zjian1/unsz
if [[ $# -lt 1 ]];then echo 'Usage: sbatch run.slurm <field-workers> [rel-eb...]'>&2;exit 2;fi
field_workers=$1;shift
rel_ebs=("$@");if [[ ${#rel_ebs[@]} -eq 0 ]];then rel_ebs=(1e-1 1e-2 1e-3 1e-4 1e-5);fi
eb_arguments=();for rel_eb in "${rel_ebs[@]}";do eb_arguments+=(--rel-eb "$rel_eb");done
srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=none /usr/bin/time -v -o "results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_resources.txt" ./unsz_env/bin/python -u benchmarks/umc_field_parallel/run_field_workers.py --field-workers "$field_workers" "${eb_arguments[@]}"
