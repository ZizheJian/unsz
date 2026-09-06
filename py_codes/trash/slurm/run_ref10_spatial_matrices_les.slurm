#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --job-name=unsz_les_ref10_matrix
#SBATCH --output=/home/x-zjian1/unsz/py_codes/trash/results/slurm_logs/ref10_spatial_matrices_les_%j.log

cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
./unsz_env/bin/python py_codes/trash/t72_train_spatial_matrices.py -dataset LES -actual-ref-num 10
