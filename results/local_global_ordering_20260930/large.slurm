#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
#SBATCH --job-name=unsz_local_global
#SBATCH --output=/home/x-zjian1/unsz/results/local_global_ordering_20260930/large_%j.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
./unsz_env/bin/python results/local_global_ordering_20260930/benchmark.py adcirc_haiyan_depth ocean_large_temperature_surface_value les_velocity_x taps_temperature --trials 1 --output results/local_global_ordering_20260930/large.csv
