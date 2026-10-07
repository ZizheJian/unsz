#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --job-name=unsz_katrina_gate
#SBATCH --output=/home/x-zjian1/unsz/results/partition_benchmark_20261006/katrina_release_gate_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
source_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina
work_dir=$(mktemp -d /tmp/unsz_katrina_gate_XXXXXX)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/data"
ln -s "$source_dir/coords.dat" "$work_dir/data/coords.dat"
for time_step in {0..161}
do
    ln -s "$source_dir/attr0.dat.$time_step" "$work_dir/data/attr0.dat.$time_step"
done
cp build/s1_compress build/s1_decompress "$work_dir/"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_DYNAMIC=false
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

for partition_num in 1 4
do
    export OMP_NUM_THREADS=$partition_num
    echo "===== partition=$partition_num threads=$partition_num compression ====="
    /usr/bin/time -v "$work_dir/s1_compress" -point-set-config /home/x-zjian1/unsz/point_set_configs/Katrina_attr0_rbf_rel_1e-3.json -data-dir "$work_dir/data" -field attr0 -time-step '[0,161]' -coordinate-dim 3 -partition-num "$partition_num" -thread-num "$partition_num" -compression-stage-time
    echo "===== partition=$partition_num threads=$partition_num decompression ====="
    /usr/bin/time -v "$work_dir/s1_decompress" -point-set-config /home/x-zjian1/unsz/point_set_configs/Katrina_attr0_rbf_rel_1e-3.json -data-dir "$work_dir/data" -field attr0 -time-step '[0,161]' -coordinate-dim 3 -partition-num "$partition_num" -thread-num "$partition_num" -decompression-stage-time
done
