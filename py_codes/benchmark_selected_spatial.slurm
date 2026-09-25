#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=unsz_selected
#SBATCH --output=/home/x-zjian1/unsz/results/logs/selected_%j.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

config_path="${SLURM_TMPDIR:-/tmp}/unsz_field_config.json"
./unsz_env/bin/python -c 'import sys,t1_start_c_code as t;t.write_selected_dataset_config(sys.argv[1],sys.argv[2],sys.argv[3])' "$DATASET" "$VARIABLE" "$config_path"
./unsz_env/bin/python t1_start_c_code.py "$config_path"
