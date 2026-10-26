#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --job-name=field_rel_screen
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_field_relative_pilot/%x_%j.log

set -euo pipefail
if [[ $# -ne 1 ]]||[[ "$1" != unsz && "$1" != qoz ]];then
    echo "Usage: sbatch run_one_field_five_ebs.slurm <unsz|qoz>" >&2
    exit 2
fi
cd /home/x-zjian1/unsz
codec=$1
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
result_root=results/ocean_large_field_relative_pilot
mkdir -p "$result_root/resources"
for rel_eb in 1e-1 1e-2 1e-3 1e-4 1e-5;do
    resource_path="$result_root/resources/${codec}_rel_${rel_eb}_job_${SLURM_JOB_ID}.txt"
    srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores /usr/bin/time -v -o "$resource_path" ./unsz_env/bin/python -u benchmarks/field_relative/run_one_field.py --codec "$codec" --rel-eb "$rel_eb" --field timeMonthly_avg_vertVelocityTop --output-root "$result_root"
done
