#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-3%1
#SBATCH --job-name=ocean_unsz_4eb
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_tensor_full/remaining_%A_%a.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
experiment_ebs=(1e-1 1e-2 1e-4 1e-5)
experiment_index=${SLURM_ARRAY_TASK_ID:?Expected a Slurm array task}
if ((experiment_index<0||experiment_index>=${#experiment_ebs[@]})); then exit 1; fi
srun --exact --ntasks=1 --cpus-per-task=8 --cpu-bind=cores /usr/bin/time -v ./unsz_env/bin/python -u results/ocean_large_tensor_full/run_rel_1e-3.py --rel-eb "${experiment_ebs[experiment_index]}" --require-memory-validation
