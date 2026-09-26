#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=unsz_rbf_speed
#SBATCH --output=/home/x-zjian1/unsz/results/logs/ocean_large_rbf_speed_%j.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

source_data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/data
speed_data_dir=$(mktemp -d /tmp/unsz_ocean_large_speed.XXXXXX)
trap 'rm -rf "$speed_data_dir"' EXIT
ln -s "$source_data_dir/coords.dat" "$speed_data_dir/coords.dat"
ln -s "$source_data_dir/conn.dat" "$speed_data_dir/conn.dat"
for time_step in 0 1 2
do
    ln -s "$source_data_dir/temperatureSurfaceValue.dat.0" "$speed_data_dir/temperatureSurfaceValue.dat.$time_step"
done
export UNSZ_SPEED_DATA_DIR="$speed_data_dir"
common=(./unsz_env/bin/python -c 'import sys;from py_codes.reference_order import main;main(sys.argv[1:])' -data-dir "$speed_data_dir" -variable temperatureSurfaceValue -time-step-num 3 -dim 3 -configuration rbf_split_rounds_then_sort_mean_reference_distance -rbf-ref-num 10 -rbf-shape 0.03 -rel-eb 0.001 -decompression-stage-time)
"${common[@]}" -reorder-data-by-prediction-order -dag-cache-result results/prediction_structures/ocean_large_restart_temperatureSurfaceValue_rbf10_s003.dag -result results/ocean_large_rbf10_speed_reordered_fast.csv
./unsz_env/bin/python -c 'import os;from t1_start_c_code import find_task,run_umc;f=find_task("ocean-large","Ocean-large","temperatureSurfaceValue");f={**f,"data_dir":os.environ["UNSZ_SPEED_DATA_DIR"],"steps":3};run_umc(f,1e-3)'
