#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=wholenode
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=124
#SBATCH --time=00:10:00
#SBATCH --job-name=unsz_les_p124
#SBATCH --output=/home/x-zjian1/unsz/results/partition_benchmark_20261006/les_p124_current_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
source_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/LES
work_dir=$(mktemp -d /tmp/unsz_les_p124_XXXXXX)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/data"
ln -s "$source_dir/coords.dat" "$work_dir/data/coords.dat"
ln -s "$source_dir/velocity_x.dat.0" "$work_dir/data/velocity_x.dat.0"
cp build/s1_compress build/s1_decompress "$work_dir/"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_DYNAMIC=false
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

run_case()
{
    local partition_num=$1
    local thread_num=$2
    local repeat_id=$3
    export OMP_NUM_THREADS=$thread_num
    echo "===== partition=$partition_num threads=$thread_num repeat=$repeat_id compression ====="
    /usr/bin/time -v "$work_dir/s1_compress" -M REL 0.001 -data-dir "$work_dir/data" -field velocity_x -time-step '{0}' -coordinate-dim 3 -partition-num "$partition_num" -thread-num "$thread_num" -compression-stage-time
    echo "===== partition=$partition_num threads=$thread_num repeat=$repeat_id decompression ====="
    /usr/bin/time -v "$work_dir/s1_decompress" -M REL 0.001 -data-dir "$work_dir/data" -field velocity_x -time-step '{0}' -coordinate-dim 3 -partition-num "$partition_num" -thread-num "$thread_num" -decompression-stage-time
}

for repeat_id in 1 2 3
do
    run_case 128 124 "$repeat_id"
done
