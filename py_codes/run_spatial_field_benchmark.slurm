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
config_path="${SLURM_TMPDIR:-/tmp}/unsz_field_config.json"
./unsz_env/bin/python -c 'import sys,t1_start_c_code as t;t.write_selected_dataset_config(sys.argv[1],sys.argv[2],sys.argv[3])' "$DATASET" "$field" "$config_path"
./unsz_env/bin/python t1_start_c_code.py "$config_path"
