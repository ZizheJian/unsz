#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --job-name=rbf_ref_tuning
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --output=/home/x-zjian1/unsz/results/rbf_reference_num_tuning/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
: "${DATASET:?必须通过sbatch --export=ALL,DATASET=...指定数据集}"
./unsz_env/bin/python results/rbf_reference_num_tuning/benchmark.py "$DATASET" ${REFERENCE_PAIRS:-}
