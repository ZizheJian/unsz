#!/bin/bash
set -euo pipefail
cd /home/x-zjian1/unsz
script=results/rbf_eb_base_tuning/run.slurm
sbatch --job-name=eb075_tsunami --mem=4G --time=00:15:00 --export=ALL,DATASET=tsunami "$script"
sbatch --job-name=eb075_les --mem=32G --time=00:30:00 --export=ALL,DATASET=les "$script"
sbatch --job-name=eb075_e3sm --mem=8G --time=00:15:00 --export=ALL,DATASET=e3sm "$script"
sbatch --job-name=eb075_haiyan --mem=16G --time=00:15:00 --export=ALL,DATASET=haiyan "$script"
sbatch --job-name=eb075_katrina --mem=8G --time=00:15:00 --export=ALL,DATASET=katrina "$script"
