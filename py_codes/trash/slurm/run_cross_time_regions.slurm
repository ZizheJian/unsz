#!/bin/bash
#SBATCH --job-name=unsz-cross-time-regions
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=cross-time-regions-%j.out

set -euo pipefail
export OMP_NUM_THREADS=1
cd /home/x-zjian1/unsz
task_tmp_dir="/tmp/unsz_cross_time_regions_${SLURM_JOB_ID}"
mkdir -p "$task_tmp_dir"

for region_num in 1 2 4 8
do
    ./unsz_env/bin/python py_codes/trash/t5_cross_time.py -variable velocityZonal -time-step-num 0 -cross-time-first-sample-mode random_hyperrectangle_regions -cross-time-first-sample-num 117580 -cross-time-first-sample-region-num "$region_num" -iteration-num 4 -second-sample-ratio 0.1 -new-matrix-weight 0.5 -max-nfev 100 -work-dir "$task_tmp_dir/train_r${region_num}" -joint-only -train-only
done

for region_num in 1 2 4 8
do
    extra_args=("-joint-only")
    if [ "$region_num" -eq 1 ]
    then
        extra_args=()
    fi
    history_path="/home/x-zjian1/unsz/adjust_coord_history/ocean_velocityZonal_joint_t31_random_hyperrectangle_regions_r${region_num}.npz"
    ./unsz_env/bin/python py_codes/trash/t5_cross_time.py -variable velocityZonal -time-step-num 0 -cross-time-first-sample-mode random_hyperrectangle_regions -cross-time-first-sample-num 117580 -cross-time-first-sample-region-num "$region_num" -work-dir "$task_tmp_dir/codec_r${region_num}" -joint-matrix-history-input "$history_path" "${extra_args[@]}"
done
