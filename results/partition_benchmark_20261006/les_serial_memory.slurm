#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --job-name=unsz_les_serial_mem
#SBATCH --output=/home/x-zjian1/unsz/results/partition_benchmark_20261006/les_serial_%j.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
./unsz_env/bin/python t1_start_c_code.py results/partition_benchmark_20261006/config.json --skip-build --task les_serial_memory
