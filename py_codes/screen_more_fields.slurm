#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=unsz_more_fields
#SBATCH --output=/home/x-zjian1/unsz/results/logs/more_fields_%A_%a.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

field_config_vec=(ADCIRC-Haiyan-1.5km-depth TAPS-PRESSURE TAPS-TKE TAPS-VELOCITY_X TAPS-RMS_T TAPS-CO2 TAPS-C11H22)
folder_vec=(ADCIRC-Haiyan-1.5km TAPS TAPS TAPS TAPS TAPS TAPS)
variable_vec=(depth PRESSURE TKE VELOCITY_X RMS_T CO2 C11H22)
field_config=${field_config_vec[$SLURM_ARRAY_TASK_ID]}
folder=${folder_vec[$SLURM_ARRAY_TASK_ID]}
variable=${variable_vec[$SLURM_ARRAY_TASK_ID]}
data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/$folder
matrix_path=adjust_coord_matrices/${folder}_${variable}_spatial.txt

./unsz_env/bin/python t2_adjust_coord.py -data-dir "$data_dir" -variable "$variable" -time-step-num 1 -dim 3 -matrix-output-path "$matrix_path"
./unsz_env/bin/python py_codes/screen_more_fields.py -field "$field_config" -rel-eb 1e-1 1e-2 1e-3 1e-4 1e-5
./unsz_env/bin/python py_codes/screen_more_fields.py -field "$field_config" -use-adjust-matrix -rel-eb 1e-1 1e-2 1e-3 1e-4 1e-5
