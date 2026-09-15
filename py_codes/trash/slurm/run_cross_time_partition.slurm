#!/bin/bash
#SBATCH --job-name=unsz-cross-time-partition
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=cross-time-partition-%j.out

set -euo pipefail
export OMP_NUM_THREADS=1
cd /home/x-zjian1/unsz
task_tmp_dir="/tmp/unsz_cross_time_partition_${SLURM_JOB_ID}"
history_path="/home/x-zjian1/unsz/adjust_coord_history/ocean_velocityZonal_joint_t31_random_hyperrectangle_regions_r4.npz"
mkdir -p "$task_tmp_dir"

for block_num in 1 8 32 64 128
do
    ./unsz_env/bin/python py_codes/trash/t5_cross_time.py -variable velocityZonal -time-step-num 0 -rel-eb 0.001 -joint-only -compress-only -joint-block-num "$block_num" -joint-matrix-history-input "$history_path" -work-dir "$task_tmp_dir/block_${block_num}"
done
