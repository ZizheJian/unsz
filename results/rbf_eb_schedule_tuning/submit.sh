#!/bin/bash
set -euo pipefail
cd /home/x-zjian1/unsz
script=results/rbf_eb_schedule_tuning/run.slurm
sbatch --job-name=ebshift_tsunami --mem=4G --time=00:25:00 --export=ALL,DATASET=tsunami "$script"
sbatch --job-name=ebshift_les --mem=32G --time=01:00:00 --export=ALL,DATASET=les "$script"
sbatch --job-name=ebshift_e3sm --mem=8G --time=00:25:00 --export=ALL,DATASET=e3sm "$script"
sbatch --job-name=ebshift_haiyan --mem=16G --time=00:25:00 --export=ALL,DATASET=haiyan "$script"
sbatch --job-name=ebshift_katrina --mem=8G --time=00:25:00 --export=ALL,DATASET=katrina "$script"
