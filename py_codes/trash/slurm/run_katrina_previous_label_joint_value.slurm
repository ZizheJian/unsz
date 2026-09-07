#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --job-name=unsz_katrina_mixed_time
#SBATCH --output=/home/x-zjian1/unsz/py_codes/trash/results/slurm_logs/katrina_previous_label_joint_value_%j.log

set -euo pipefail
export OMP_NUM_THREADS=32
cd /home/x-zjian1/unsz
./unsz_env/bin/python py_codes/trash/t99_benchmark_katrina_previous_label_joint_value.py
