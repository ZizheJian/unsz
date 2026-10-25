#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --job-name=field_rel_pilot
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_field_relative_pilot/%x_%j.log

set -euo pipefail
if [[ $# -ne 1 ]]||[[ "$1" != unsz && "$1" != sz3 && "$1" != qoz ]];then
    echo "Usage: sbatch run_pilot.slurm <unsz|sz3|qoz>" >&2
    exit 2
fi
cd /home/x-zjian1/unsz
codec=$1
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p results/ocean_large_field_relative_pilot
srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores /usr/bin/time -v -o "results/ocean_large_field_relative_pilot/${codec}_resources.txt" ./unsz_env/bin/python -u benchmarks/field_relative/run.py --codec "$codec" --rel-eb 1e-3 --field timeMonthly_avg_vertVelocityTop --output-root results/ocean_large_field_relative_pilot
