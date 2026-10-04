#!/bin/bash
set -euo pipefail
cd /home/x-zjian1/unsz
script=results/local_global_final/run.slurm
sbatch --job-name=scope_tsunami --mem=4G --time=00:30:00 --export=ALL,DATASET=tsunami "$script"
sbatch --job-name=scope_les --mem=32G --time=01:00:00 --export=ALL,DATASET=les "$script"
sbatch --job-name=scope_e3sm --mem=8G --time=00:30:00 --export=ALL,DATASET=e3sm "$script"
sbatch --job-name=scope_haiyan --mem=16G --time=00:30:00 --export=ALL,DATASET=haiyan "$script"
sbatch --job-name=scope_katrina --mem=8G --time=00:30:00 --export=ALL,DATASET=katrina "$script"
