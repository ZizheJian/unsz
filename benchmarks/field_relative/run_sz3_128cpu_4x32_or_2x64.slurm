#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:35:00
#SBATCH --job-name=sz3_128cpu
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_field_relative_sz3_128cpu_4x32_or_2x64/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
if [[ $# -ne 1 ]];then echo 'Usage: sbatch run_sz3_128cpu_4x32_or_2x64.slurm <4x32|2x64>'>&2;exit 2;fi
case "$1" in
    4x32)field_workers=4;partitions_per_field=32;;
    2x64)field_workers=2;partitions_per_field=64;;
    *)echo 'Usage: sbatch run_sz3_128cpu_4x32_or_2x64.slurm <4x32|2x64>'>&2;exit 2;;
esac
if [[ ${SLURM_CPUS_PER_TASK:-0} -ne 128 ]];then echo 'This experiment requires 128 allocated CPUs'>&2;exit 2;fi
binary=build/field_relative/compress_one_field_with_sz3_partitions
if [[ ! -x "$binary" ]];then echo "Missing executable: $binary" >&2;exit 1;fi
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
result_root=results/ocean_large_field_relative_sz3_128cpu_4x32_or_2x64
mkdir -p "$result_root/resources"
resource_path="$result_root/resources/${field_workers}fields_${partitions_per_field}parts_job_${SLURM_JOB_ID}.txt"
srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task=128 --cpu-bind=none /usr/bin/time -v -o "$resource_path" ./unsz_env/bin/python -u benchmarks/field_relative/run_all_95_fields.py --codec sz3 --rel-eb 1e-3 --field-workers "$field_workers" --partitions-per-field "$partitions_per_field" --output-root "$result_root"
