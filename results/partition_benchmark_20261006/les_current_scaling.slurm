#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:15:00
#SBATCH --job-name=unsz_les_scaling
#SBATCH --output=/home/x-zjian1/unsz/results/partition_benchmark_20261006/les_current_scaling_%j.log

set -e
cd /home/x-zjian1/unsz
source_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/LES
work_dir=$(mktemp -d /tmp/unsz_les_scaling_XXXXXX)
mkdir -p "$work_dir/data"
ln -s "$source_dir/coords.dat" "$work_dir/data/coords.dat"
ln -s "$source_dir/velocity_x.dat.0" "$work_dir/data/velocity_x.dat.0"
cp install/bin/s1_compress install/bin/s1_decompress "$work_dir/"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

for partition_num in 1 2 4 8 16
do
    export OMP_NUM_THREADS=$partition_num
    echo "===== partition=$partition_num compression ====="
    "$work_dir/s1_compress" -M REL 0.001 -data-dir "$work_dir/data" -field velocity_x -time-step '{0}' -coordinate-dim 3 -partition-num "$partition_num" -thread-num "$partition_num" -compression-stage-time
    echo "===== partition=$partition_num decompression ====="
    "$work_dir/s1_decompress" -M REL 0.001 -data-dir "$work_dir/data" -field velocity_x -time-step '{0}' -coordinate-dim 3 -partition-num "$partition_num" -thread-num "$partition_num" -decompression-stage-time
done
