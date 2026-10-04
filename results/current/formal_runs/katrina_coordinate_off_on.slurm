#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=katrina_coord
#SBATCH --output=/home/x-zjian1/unsz/results/current/formal_runs/katrina_coordinate_off_on_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=4
./unsz_env/bin/python t1_start_c_code.py katrina-coordinate-transform-off-on --skip-build
