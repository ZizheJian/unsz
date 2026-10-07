#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=unsz_taps_p4
#SBATCH --output=/home/x-zjian1/unsz/results/partition_benchmark_20261006/taps_%j.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
./unsz_env/bin/python t1_start_c_code.py results/partition_benchmark_20261006/config.json --skip-build --task taps_partition_4
