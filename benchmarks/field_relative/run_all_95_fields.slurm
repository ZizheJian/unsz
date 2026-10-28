#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:30:00
#SBATCH --job-name=all_95_fields
#SBATCH --output=/home/x-zjian1/unsz/results/%x_%j.log

# 用法：sbatch [--cpus-per-task=N] run_all_95_fields.slurm <sz3|qoz> <field-workers> <partitions-per-field> [rel-eb...]
# 不指定EB时运行五个正式EB；field-workers×partitions-per-field个CPU被使用，申请的CPU可以更多，多出的CPU只提供内存
set -euo pipefail
usage='Usage: sbatch run_all_95_fields.slurm <sz3|qoz> <field-workers> <partitions-per-field> [rel-eb...]'
case "${1:-}" in sz3|qoz);;*)echo "$usage">&2;exit 2;;esac
if [[ $# -lt 3 ]];then echo "$usage">&2;exit 2;fi
cd /home/x-zjian1/unsz
codec=$1;field_workers=$2;partitions_per_field=$3;shift 3
rel_ebs=("$@");if [[ ${#rel_ebs[@]} -eq 0 ]];then rel_ebs=(1e-1 1e-2 1e-3 1e-4 1e-5);fi
eb_arguments=();for rel_eb in "${rel_ebs[@]}";do eb_arguments+=(--rel-eb "$rel_eb");done
cpu_num=$((field_workers*partitions_per_field))
if ((field_workers<1||partitions_per_field<1||cpu_num>SLURM_CPUS_PER_TASK));then echo 'field-workers×partitions-per-field must not exceed the allocated CPUs'>&2;exit 2;fi
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
result_root=results/ocean_large_per_field_rd_${cpu_num}
mkdir -p "$result_root/resources"
resource_path="$result_root/resources/${codec}_job_${SLURM_JOB_ID}.txt"
srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=none /usr/bin/time -v -o "$resource_path" ./unsz_env/bin/python -u benchmarks/field_relative/run_all_95_fields.py --codec "$codec" "${eb_arguments[@]}" --field-workers "$field_workers" --partitions-per-field "$partitions_per_field"
