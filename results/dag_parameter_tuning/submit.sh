#!/bin/bash
set -euo pipefail
cd /home/x-zjian1/unsz
script=results/dag_parameter_tuning/run.slurm
sbatch --job-name=dag_tsunami --mem=4G --time=00:45:00 --export=ALL,DATASET=tsunami "$script"
sbatch --job-name=dag_les --mem=32G --time=02:00:00 --export=ALL,DATASET=les "$script"
sbatch --job-name=dag_e3sm --mem=8G --time=00:45:00 --export=ALL,DATASET=e3sm "$script"
sbatch --job-name=dag_haiyan --mem=16G --time=00:45:00 --export=ALL,DATASET=haiyan "$script"
sbatch --job-name=dag_katrina --mem=8G --time=00:45:00 --export=ALL,DATASET=katrina "$script"
