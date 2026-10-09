#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:35:00
#SBATCH --job-name=ocean_remaining_sparse_mem8
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_tensor_full/test_remaining_sparse_mem8_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
./unsz_env/bin/python results/ocean_large_tensor_full/run_rel_1e-3.py --prepare-only
srun --exact --ntasks=1 --cpus-per-task=8 --cpu-bind=cores ./unsz_env/bin/python tests/profile_ocean_field.py results/ocean_large_tensor_full/sparse_window/config_rel_1e-3.json --field timeMonthly_avg_layerThickness --output-dir results/ocean_large_tensor_full/layerThickness_sparse_mem8
srun --exact --ntasks=1 --cpus-per-task=8 --cpu-bind=cores ./unsz_env/bin/python tests/profile_ocean_field.py results/ocean_large_tensor_full/sparse_window/config_rel_1e-3.json --field timeMonthly_avg_vertVelocityTop --output-dir results/ocean_large_tensor_full/vertVelocityTop_sparse_mem8
