#!/bin/bash
set -e
cd /home/x-zjian1/unsz
slurm=results/idw4_rbf10_tuning/run_screen.slurm
sbatch --job-name=tune_tsunami --mem=4G --time=00:20:00 --export=ALL,DATASET=tsunami "$slurm"
sbatch --job-name=tune_les --mem=32G --time=01:00:00 --export=ALL,DATASET=les "$slurm"
sbatch --job-name=tune_e3sm --mem=8G --time=00:20:00 --export=ALL,DATASET=e3sm "$slurm"
sbatch --job-name=tune_haiyan --mem=16G --time=00:30:00 --export=ALL,DATASET=haiyan "$slurm"
sbatch --job-name=tune_katrina --mem=8G --time=00:30:00 --export=ALL,DATASET=katrina "$slurm"
