#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=taps_auto_coord
#SBATCH --output=/home/x-zjian1/unsz/results/remaining_adjust_coord_screen/taps_auto_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
./unsz_env/bin/python -c 'import t1_start_c_code as t;task=t.load_json_tasks(t.resolve_json_path("remaining-single-step-auto-screen"))[0];t.run_task(task)'
