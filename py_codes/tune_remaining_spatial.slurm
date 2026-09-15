#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-10%11
#SBATCH --job-name=unsz_config_tune
#SBATCH --output=/home/x-zjian1/unsz/results/logs/config_tune_%A_%a.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

dataset_vec=(LES LES LES ADCIRC-Haiyan-1.5km TAPS TAPS TAPS TAPS TAPS TAPS TAPS)
variable_vec=(velocity_x velocity_y velocity_z depth TEMPERATURE PRESSURE TKE VELOCITY_X RMS_T CO2 C11H22)
slug_vec=(les_velocity_x les_velocity_y les_velocity_z haiyan_depth taps_TEMPERATURE taps_PRESSURE taps_TKE taps_VELOCITY_X taps_RMS_T taps_CO2 taps_C11H22)
dataset=${dataset_vec[$SLURM_ARRAY_TASK_ID]}
variable=${variable_vec[$SLURM_ARRAY_TASK_ID]}
slug=${slug_vec[$SLURM_ARRAY_TASK_ID]}
data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/$dataset
matrix_path=adjust_coord_matrices/${dataset}_${variable}_spatial.txt
arguments=(./unsz_env/bin/python py_codes/tune_field_spatial.py -data-dir "$data_dir" -variable "$variable" -time-step-num 1 -dim 3 -test-time-step 0 -candidate-set full -result "results/${slug}_spatial_tune.csv")
if [[ -f "$matrix_path" ]]; then
  arguments+=(-matrix-file "$matrix_path")
fi
"${arguments[@]}"
