#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:45:00
#SBATCH --job-name=ocean_two_level
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_two_level/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
if [[ ${SLURM_CPUS_PER_TASK:-0} -ne 128 ]]; then echo "Two-level experiment requires 128 CPUs" >&2; exit 1; fi
export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=2
export OMP_PROC_BIND=spread,close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p results/ocean_large_two_level
srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task=128 --cpu-bind=cores /usr/bin/time -v -o results/ocean_large_two_level/resources.txt ./unsz_env/bin/python -u benchmarks/two_level/run.py
