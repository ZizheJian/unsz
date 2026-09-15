#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --job-name=unsz_katrina_joint
#SBATCH --output=/home/x-zjian1/unsz/py_codes/trash/results/slurm_logs/katrina_current_joint_%j.log

set -euo pipefail
export OMP_NUM_THREADS=16
cd /home/x-zjian1/unsz
task_tmp_dir="/tmp/unsz_katrina_joint_${SLURM_JOB_ID}"
mkdir -p "$task_tmp_dir"

./unsz_env/bin/python py_codes/trash/t5_cross_time.py -data-dir /anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina -variable attr0 -spatial-dim 3 -time-step-num 0 -rel-eb 0.001 -point-set-config point_set_configs/Katrina_attr0_rel_1e-3.json -cross-time-first-sample-mode random_hyperrectangle_regions -cross-time-first-sample-num 117580 -cross-time-first-sample-region-num 4 -second-sample-num 11758 -iteration-num 4 -new-matrix-weight 0.5 -max-nfev 100 -work-dir "$task_tmp_dir" -matrix-history-path adjust_coord_history/katrina_attr0_joint_t162_random_hyperrectangle_regions_r4.npz -joint-matrix-output-path adjust_coord_matrices/Katrina_attr0_spatiotemporal_t162_random_hyperrectangle_regions_r4.txt -joint-only -joint-block-num 128 -thread-num 16
