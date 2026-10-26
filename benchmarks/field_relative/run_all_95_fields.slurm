#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:30:00
#SBATCH --job-name=all_95_fields
#SBATCH --output=/home/x-zjian1/unsz/results/%x_%j.log

set -euo pipefail
case "${1:-}" in sz3|qoz);;*)echo 'Usage: sbatch run_all_95_fields.slurm <sz3|qoz> <field-workers> <partitions-per-field>'>&2;exit 2;;esac
if [[ $# -ne 3 ]];then echo 'Usage: sbatch run_all_95_fields.slurm <sz3|qoz> <field-workers> <partitions-per-field>'>&2;exit 2;fi
if [[ ${SLURM_CPUS_PER_TASK:-0} -ne 128 ]];then echo 'Field-relative full experiment requires 128 allocated CPUs'>&2;exit 2;fi
cd /home/x-zjian1/unsz
codec=$1
field_workers=$2
partitions_per_field=$3
if ((field_workers<1||partitions_per_field<1||field_workers*partitions_per_field>128));then echo 'Invalid field/partition allocation'>&2;exit 2;fi
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
result_root=results/ocean_large_field_relative_five_ebs
mkdir -p "$result_root/resources"
resource_path="$result_root/resources/${codec}_job_${SLURM_JOB_ID}.txt"
srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task=128 --cpu-bind=none /usr/bin/time -v -o "$resource_path" ./unsz_env/bin/python -u benchmarks/field_relative/run_all_95_fields.py --codec "$codec" --rel-eb 1e-1 --rel-eb 1e-2 --rel-eb 1e-3 --rel-eb 1e-4 --rel-eb 1e-5 --field-workers "$field_workers" --partitions-per-field "$partitions_per_field"
