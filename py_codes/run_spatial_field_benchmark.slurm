#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=unsz_spatial_rd
#SBATCH --output=/home/x-zjian1/unsz/results/logs/spatial_rd_%A_%a.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

IFS=':' read -r -a field_array <<< "$FIELD_LIST"
field="${field_array[$SLURM_ARRAY_TASK_ID]}"
./unsz_env/bin/python py_codes/run_spatial_field_benchmark.py -dataset "$DATASET" -field "$field"
