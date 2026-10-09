#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --job-name=ocean_tensor_full_1e-3
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_tensor_full/run_rel_1e-3_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun --exact --ntasks=1 --cpus-per-task=8 --cpu-bind=cores ./unsz_env/bin/python results/ocean_large_tensor_full/run_rel_1e-3.py --require-memory-validation
