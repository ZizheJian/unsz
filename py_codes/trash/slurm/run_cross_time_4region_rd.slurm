#!/bin/bash
#SBATCH --job-name=unsz-cross-time-rd
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=cross-time-4region-rd-%j.out

set -euo pipefail
export OMP_NUM_THREADS=1
cd /home/x-zjian1/unsz
task_tmp_dir="/tmp/unsz_cross_time_rd_${SLURM_JOB_ID}"
mkdir -p "$task_tmp_dir"
history_path="/home/x-zjian1/unsz/adjust_coord_history/ocean_velocityZonal_joint_t31_random_hyperrectangle_regions_r4.npz"

for rel_eb in 0.01 0.003 0.0003 0.0001
do
    eb_name="${rel_eb//./_}"
    ./unsz_env/bin/python py_codes/trash/t5_cross_time.py -variable velocityZonal -time-step-num 0 -rel-eb "$rel_eb" -cross-time-first-sample-mode random_hyperrectangle_regions -cross-time-first-sample-num 117580 -cross-time-first-sample-region-num 4 -work-dir "$task_tmp_dir/eb_${eb_name}" -joint-matrix-history-input "$history_path"
done
