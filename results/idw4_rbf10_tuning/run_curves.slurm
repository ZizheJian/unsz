#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/home/x-zjian1/unsz/results/idw4_rbf10_tuning/curve_%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
./unsz_env/bin/python results/idw4_rbf10_tuning/benchmark_curves.py "$DATASET"
