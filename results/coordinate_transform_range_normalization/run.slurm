#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=unsz-coordinate-range
#SBATCH --output=results/coordinate_transform_range_normalization/run_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
cmake --build build --parallel 4
cmake --install build
python3 results/coordinate_transform_range_normalization/benchmark.py
