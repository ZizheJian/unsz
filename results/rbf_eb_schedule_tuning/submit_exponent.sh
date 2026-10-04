#!/bin/bash
set -euo pipefail
cd /home/x-zjian1/unsz
script=results/rbf_eb_schedule_tuning/run.slurm
sbatch --job-name=ebexp_tsunami --mem=4G --time=00:25:00 --export=ALL,DATASET=tsunami,PARAMETER=exponent "$script"
sbatch --job-name=ebexp_les --mem=32G --time=01:00:00 --export=ALL,DATASET=les,PARAMETER=exponent "$script"
sbatch --job-name=ebexp_e3sm --mem=8G --time=00:25:00 --export=ALL,DATASET=e3sm,PARAMETER=exponent "$script"
sbatch --job-name=ebexp_haiyan --mem=16G --time=00:25:00 --export=ALL,DATASET=haiyan,PARAMETER=exponent "$script"
sbatch --job-name=ebexp_katrina --mem=8G --time=00:25:00 --export=ALL,DATASET=katrina,PARAMETER=exponent "$script"
