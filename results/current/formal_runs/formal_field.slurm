#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=unsz_formal
#SBATCH --output=/home/x-zjian1/unsz/results/current/formal_runs/%x_%j.log

set -euo pipefail
: "${CONFIG:?必须通过sbatch --export传入CONFIG}"
: "${FIELD:?必须通过sbatch --export传入FIELD}"
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "git_commit=$(git rev-parse HEAD)"
echo "node=$(hostname),config=$CONFIG,field=$FIELD"
./unsz_env/bin/python t1_start_c_code.py "$CONFIG" --field "$FIELD" --skip-build
