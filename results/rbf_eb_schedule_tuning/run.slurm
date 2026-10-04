#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/home/x-zjian1/unsz/results/rbf_eb_schedule_tuning/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
: "${DATASET:?必须通过sbatch --export=DATASET=...指定数据集}"
./unsz_env/bin/python results/rbf_eb_schedule_tuning/benchmark.py "$DATASET" "${PARAMETER:-shift}"
