#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --array=0-4
#SBATCH --time=00:30:00
#SBATCH --job-name=unsz_final_temporal
#SBATCH --output=/home/x-zjian1/unsz/results/current/formal_runs/unsz_final_%A_%a.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
config_vec=(katrina e3sm-h4 adcirc-haiyan-1.5km ocean tpost20)
echo "git_commit=$(git rev-parse HEAD)"
echo "hostname=$(hostname)"
if [ "$SLURM_ARRAY_TASK_ID" -eq 2 ];then
    exec ./unsz_env/bin/python t1_start_c_code.py adcirc-haiyan-1.5km --skip-build --compressor unsz --task adcirc_haiyan_zeta
fi
exec ./unsz_env/bin/python t1_start_c_code.py "${config_vec[$SLURM_ARRAY_TASK_ID]}" --skip-build --compressor unsz
