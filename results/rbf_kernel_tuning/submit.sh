#!/bin/bash
set -euo pipefail
cd /home/x-zjian1/unsz
script=results/rbf_kernel_tuning/run.slurm
sbatch --job-name=kernel_tsunami --mem=4G --time=00:45:00 --export=ALL,DATASET=tsunami "$script"
sbatch --job-name=kernel_les --mem=32G --time=01:30:00 --export=ALL,DATASET=les "$script"
sbatch --job-name=kernel_e3sm --mem=8G --time=00:45:00 --export=ALL,DATASET=e3sm "$script"
sbatch --job-name=kernel_haiyan --mem=16G --time=00:45:00 --export=ALL,DATASET=haiyan "$script"
sbatch --job-name=kernel_katrina --mem=8G --time=00:45:00 --export=ALL,DATASET=katrina "$script"
