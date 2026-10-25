#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=unsz_spatial_rd
#SBATCH --output=/home/x-zjian1/unsz/results/current/logs/spatial_rd_%A_%a.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
: "${CONFIG:?必须设置源JSON配置名称或路径}"

./unsz_env/bin/python t1_start_c_code.py "$CONFIG"
