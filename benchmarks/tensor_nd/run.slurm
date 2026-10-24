#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --job-name=ocean_sz3_3d
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_tensor_nd/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
result_dir=results/ocean_large_tensor_nd/sz3/rel_1e-3
data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/tensor_data
field=timeMonthly_avg_density
mkdir -p "$result_dir"
srun --exact --ntasks=1 --cpus-per-task=64 --cpu-bind=cores /usr/bin/time -v build/tensor_nd/benchmark_sz3 "$data_dir" "$field" 12 80 1e-3 global-relative "$result_dir/${field}_global_relative.json"
srun --exact --ntasks=1 --cpus-per-task=64 --cpu-bind=cores /usr/bin/time -v build/tensor_nd/benchmark_sz3 "$data_dir" "$field" 12 80 1e-3 strict-slice-relative "$result_dir/${field}_strict_slice_relative.json"
