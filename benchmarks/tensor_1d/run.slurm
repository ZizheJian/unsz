#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --job-name=ocean_1d_baseline
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_tensor_1d/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun --exact --ntasks=1 --cpus-per-task=1 --cpu-bind=cores /usr/bin/time -v ./unsz_env/bin/python -u benchmarks/tensor_1d/run.py --codec "$1"
