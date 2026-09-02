#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --job-name=unsz_les_order
#SBATCH --output=/home/x-zjian1/unsz/py_codes/trash/results/spatial_order_full_rd_les_%j.log

cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
./unsz_env/bin/python py_codes/trash/t63_benchmark_spatial_order_full_rd.py -dataset LES -method unsz
