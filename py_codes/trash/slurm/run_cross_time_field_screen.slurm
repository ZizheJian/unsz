#!/bin/bash
#SBATCH --job-name=unsz-cross-time-fields
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=cross-time-field-screen-%j.out

set -euo pipefail
export OMP_NUM_THREADS=1
cd /home/x-zjian1/unsz
task_tmp_dir="/tmp/unsz_cross_time_fields_${SLURM_JOB_ID}"
mkdir -p "$task_tmp_dir"
zonal_history="/home/x-zjian1/unsz/adjust_coord_history/ocean_velocityZonal_joint_t31_random_hyperrectangle_regions_r4.npz"

for field in velocityX velocityY temperature
do
    ./unsz_env/bin/python t5_cross_time.py -variable "$field" -time-step-num 0 -rel-eb 0.001 -cross-time-first-sample-mode random_hyperrectangle_regions -cross-time-first-sample-num 117580 -cross-time-first-sample-region-num 4 -iteration-num 4 -second-sample-ratio 0.1 -new-matrix-weight 0.5 -max-nfev 100 -work-dir "$task_tmp_dir/train_${field}" -joint-only -train-only
done

for field in velocityX velocityY temperature
do
    field_history="/home/x-zjian1/unsz/adjust_coord_history/ocean_${field}_joint_t31_random_hyperrectangle_regions_r4.npz"
    ./unsz_env/bin/python t5_cross_time.py -variable "$field" -time-step-num 0 -rel-eb 0.001 -cross-time-first-sample-mode random_hyperrectangle_regions -cross-time-first-sample-num 117580 -cross-time-first-sample-region-num 4 -work-dir "$task_tmp_dir/own_${field}" -joint-matrix-history-input "$field_history" -spatial-matrix 1 0 1 0 0 1
    ./unsz_env/bin/python t5_cross_time.py -variable "$field" -time-step-num 0 -rel-eb 0.001 -cross-time-first-sample-mode random_hyperrectangle_regions -cross-time-first-sample-num 117580 -cross-time-first-sample-region-num 4 -work-dir "$task_tmp_dir/zonal_${field}" -joint-matrix-history-input "$zonal_history" -joint-only
done
