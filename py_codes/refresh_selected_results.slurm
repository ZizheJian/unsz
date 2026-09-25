#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=unsz_refresh
#SBATCH --output=/home/x-zjian1/unsz/results/current/logs/%A_%a.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
dataset_field_vec=(Katrina:attr0 Ocean:layerThickness Ocean:salinity Ocean:temperature Ocean:velocityMeridional Ocean:velocityX Ocean:velocityY Ocean:velocityZ Ocean:velocityZonal Ocean:zTop Ocean-large:temperatureSurfaceValue LES:velocity_x LES:velocity_y LES:velocity_z Tsunami:bathy Tsunami:height Tsunami:momentum_x Tsunami:momentum_y Tsunami:velocity_x Tsunami:velocity_y Tpost20:scalar E3SM-h4:AODDUST E3SM-h4:Mass_dst_srf ADCIRC-Haiyan-1.5km:depth ADCIRC-Haiyan-1.5km:zeta TAPS:TEMPERATURE TAPS:PRESSURE TAPS:TKE TAPS:VELOCITY_X TAPS:RMS_T TAPS:CO2 TAPS:C11H22)
IFS=: read -r dataset field <<< "${dataset_field_vec[$SLURM_ARRAY_TASK_ID]}"
config_path="${SLURM_TMPDIR:-/tmp}/unsz_field_config.json"
./unsz_env/bin/python -c 'import sys,t1_start_c_code as t;t.write_selected_dataset_config(sys.argv[1],sys.argv[2],sys.argv[3])' "$dataset" "$field" "$config_path"
./unsz_env/bin/python t1_start_c_code.py "$config_path"
