#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:30:00
#SBATCH --job-name=unsz_field_rd
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_field_relative_five_ebs/unsz/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
if [[ ${SLURM_CPUS_PER_TASK:-0} -ne 128 ]];then echo 'UNSZ per-field RD requires 128 allocated CPUs'>&2;exit 2;fi
export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=2
export OMP_PROC_BIND=spread,close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
result_root=results/ocean_large_field_relative_five_ebs
mkdir -p "$result_root/unsz"
resource_path="$result_root/unsz/resources_job_${SLURM_JOB_ID}.txt"
srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task=128 --cpu-bind=none /usr/bin/time -v -o "$resource_path" ./unsz_env/bin/python -u benchmarks/field_relative/run_all_unsz_95_fields.py --rel-eb 1e-1 --rel-eb 1e-2 --rel-eb 1e-3 --rel-eb 1e-4 --rel-eb 1e-5 --field-workers 16 --partitions-per-field 8 --output-root "$result_root"
