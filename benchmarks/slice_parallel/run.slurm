#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:40:00
#SBATCH --job-name=slice_parallel
#SBATCH --output=/home/x-zjian1/unsz/results/%x_%j.log

# 用法：sbatch [--cpus-per-task=N] run.slurm <sz3|pfpl> <slice-workers> <threads-per-slice> [rel-eb...]
# 不指定EB时运行五个正式EB；slice-workers×threads-per-slice必须等于申请的CPU数
set -euo pipefail
cd /home/x-zjian1/unsz
if [[ $# -lt 3 ]];then echo 'Usage: sbatch run.slurm <sz3|pfpl> <slice-workers> <threads-per-slice> [rel-eb...]'>&2;exit 2;fi
codec=$1;slice_workers=$2;threads_per_slice=$3;shift 3
rel_ebs=("$@");if [[ ${#rel_ebs[@]} -eq 0 ]];then rel_ebs=(1e-1 1e-2 1e-3 1e-4 1e-5);fi
eb_arguments=();for rel_eb in "${rel_ebs[@]}";do eb_arguments+=(--rel-eb "$rel_eb");done
srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=none /usr/bin/time -v -o "results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_resources.txt" ./unsz_env/bin/python -u benchmarks/slice_parallel/run.py --codec "$codec" --slice-workers "$slice_workers" --threads-per-slice "$threads_per_slice" "${eb_arguments[@]}"
