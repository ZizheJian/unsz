#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:10:00
#SBATCH --array=0-13%4
#SBATCH --job-name=unsz-coordinate-group-loss
#SBATCH --output=results/coordinate_transform_group_loss/run_%A_%a.log

set -euo pipefail
cd /home/x-zjian1/unsz
python3 results/coordinate_transform_group_loss/benchmark.py
