#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --job-name=idw4_rbf10
#SBATCH --output=/home/x-zjian1/unsz/results/idw4_rbf10_tuning/screen_%x_%j.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
: "${DATASET:?必须通过sbatch --export=DATASET=...指定数据集}"
./unsz_env/bin/python results/idw4_rbf10_tuning/benchmark.py "$DATASET"
