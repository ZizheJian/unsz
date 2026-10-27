#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:08:00
#SBATCH --job-name=sz3_1d_128cpu
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_sz3_1d_128cpu_selection/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
if [[ $# -ne 1 ]];then echo 'Usage: sbatch run_sz3_1d_128cpu_four_allocations.slurm <128x1|64x2|32x4|16x8>'>&2;exit 2;fi
case "$1" in
    128x1)slice_workers=128;threads_per_slice=1;;
    64x2)slice_workers=64;threads_per_slice=2;;
    32x4)slice_workers=32;threads_per_slice=4;;
    16x8)slice_workers=16;threads_per_slice=8;;
    *)echo 'Usage: sbatch run_sz3_1d_128cpu_four_allocations.slurm <128x1|64x2|32x4|16x8>'>&2;exit 2;;
esac
if [[ ${SLURM_CPUS_PER_TASK:-0} -ne 128 ]];then echo 'This experiment requires 128 allocated CPUs'>&2;exit 2;fi
binary=build/tensor_1d/benchmark_sz3
if [[ ! -x "$binary" ]];then echo "Missing executable: $binary" >&2;exit 1;fi
result_root=results/ocean_large_sz3_1d_128cpu_selection
mkdir -p "$result_root/resources"
resource_path="$result_root/resources/${slice_workers}workers_${threads_per_slice}threads_job_${SLURM_JOB_ID}.txt"
srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task=128 --cpu-bind=none /usr/bin/time -v -o "$resource_path" ./unsz_env/bin/python -u benchmarks/sz3_1d_parallel/run_all_slices.py --rel-eb 1e-3 --slice-workers "$slice_workers" --threads-per-slice "$threads_per_slice" --output-root "$result_root"
