#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=unsz_training_num
#SBATCH --output=/home/x-zjian1/unsz/py_codes/trash/results/slurm_logs/training_file_num_%j.out

set -e
cd /home/x-zjian1/unsz
unsz_env/bin/python py_codes/trash/t67_validate_training_data_aware_order.py --training-file-num
