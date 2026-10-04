#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --job-name=e3sm_label_flow
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --output=/home/x-zjian1/unsz/results/e3sm_label_flow_tuning/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
./unsz_env/bin/python results/e3sm_label_flow_tuning/benchmark.py
