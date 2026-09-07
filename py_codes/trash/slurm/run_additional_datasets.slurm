#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --job-name=unsz_new_data
#SBATCH --output=/home/x-zjian1/unsz/py_codes/trash/results/slurm_logs/additional_datasets_%j.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
read -r -a rel_eb_vec <<< "${REL_EB_LIST:-${REL_EB:-1e-3}}"
./unsz_env/bin/python py_codes/trash/t91_benchmark_additional_datasets.py -dataset "$DATASET" -rel-eb "${rel_eb_vec[@]}"
