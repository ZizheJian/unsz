#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-7%1
#SBATCH --time=01:00:00
#SBATCH --job-name=unsz_configs
#SBATCH --output=/home/x-zjian1/unsz/results/current/logs/%A_%a.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
config_vec=(adcirc-haiyan-1.5km e3sm-h4 katrina les ocean-large ocean taps tpost20)
./unsz_env/bin/python t1_start_c_code.py "${config_vec[$SLURM_ARRAY_TASK_ID]}"
