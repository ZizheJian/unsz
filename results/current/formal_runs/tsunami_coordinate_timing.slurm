#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --job-name=tsunami_coord_time
#SBATCH --output=/home/x-zjian1/unsz/results/current/formal_runs/tsunami_coordinate_timing_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=4

for repeat in 1 2
do
    ./unsz_env/bin/python t1_start_c_code.py tsunami-coordinate-transform-timing --skip-build
done
