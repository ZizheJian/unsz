#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=final_e3sm
#SBATCH --output=/home/x-zjian1/unsz/results/current/formal_runs/e3sm_%j.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
./unsz_env/bin/python t1_start_c_code.py e3sm-h4-aoddust-fused --skip-build
