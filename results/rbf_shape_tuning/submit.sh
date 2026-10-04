#!/bin/bash
set -euo pipefail
cd /home/x-zjian1/unsz
script=results/rbf_shape_tuning/run.slurm
sbatch --job-name=shape_tsunami --mem=4G --time=00:30:00 --export=ALL,DATASET=tsunami "$script"
sbatch --job-name=shape_les --mem=32G --time=01:15:00 --export=ALL,DATASET=les "$script"
sbatch --job-name=shape_e3sm --mem=8G --time=00:30:00 --export=ALL,DATASET=e3sm "$script"
sbatch --job-name=shape_haiyan --mem=16G --time=00:30:00 --export=ALL,DATASET=haiyan "$script"
sbatch --job-name=shape_katrina --mem=8G --time=00:30:00 --export=ALL,DATASET=katrina "$script"
