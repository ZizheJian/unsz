#!/bin/bash
set -euo pipefail
cd /home/x-zjian1/unsz
script=results/rbf_eb_base_tuning/run.slurm
sbatch --job-name=ebbase_tsunami --mem=4G --time=00:20:00 --export=ALL,DATASET=tsunami "$script"
sbatch --job-name=ebbase_les --mem=32G --time=00:45:00 --dependency=afterok:21030202 --export=ALL,DATASET=les "$script"
sbatch --job-name=ebbase_e3sm --mem=8G --time=00:20:00 --export=ALL,DATASET=e3sm "$script"
sbatch --job-name=ebbase_haiyan --mem=16G --time=00:20:00 --export=ALL,DATASET=haiyan "$script"
sbatch --job-name=ebbase_katrina --mem=8G --time=00:20:00 --export=ALL,DATASET=katrina "$script"
