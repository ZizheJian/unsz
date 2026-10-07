#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --job-name=unsz_les_core_stages
#SBATCH --output=/home/x-zjian1/unsz/results/partition_benchmark_20261006/les_core_stages_%j.log

set -e
cd /home/x-zjian1/unsz
data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/LES
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo '===== serial compression ====='
export OMP_NUM_THREADS=1
./install/bin/s1_compress -M REL 0.001 -data-dir "$data_dir" -field velocity_x -time-step '{0}' -coordinate-dim 3 -compression-stage-time
echo '===== serial decompression ====='
./install/bin/s1_decompress -M REL 0.001 -data-dir "$data_dir" -field velocity_x -time-step '{0}' -coordinate-dim 3 -decompression-stage-time

echo '===== partition-4 compression ====='
export OMP_NUM_THREADS=4
./install/bin/s1_compress -M REL 0.001 -data-dir "$data_dir" -field velocity_x -time-step '{0}' -coordinate-dim 3 -partition-num 4 -thread-num 4 -compression-stage-time
echo '===== partition-4 decompression ====='
./install/bin/s1_decompress -M REL 0.001 -data-dir "$data_dir" -field velocity_x -time-step '{0}' -coordinate-dim 3 -partition-num 4 -thread-num 4 -decompression-stage-time

rm -f "$data_dir/velocity_x.dat.0.unsz" "$data_dir/velocity_x.dat.0.unsz.bin" "$data_dir/velocity_x.unsz.coord_transform.txt"
