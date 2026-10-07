#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:08:00
#SBATCH --job-name=unsz_ocean_large_pilot
#SBATCH --output=/home/x-zjian1/unsz/results/partition_benchmark_20261006/ocean_large_pilot_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
source_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/data
work_dir=$(mktemp -d /tmp/unsz_ocean_large_pilot_XXXXXX)
trap 'rm -rf "$work_dir"' EXIT
mkdir -p "$work_dir/data"
ln -s "$source_dir/coords.dat" "$work_dir/data/coords.dat"
ln -s "$source_dir/refBottomDepth.dat" "$work_dir/data/refBottomDepth.dat"
ln -s "$source_dir/temperatureSurfaceValue.dat.0" "$work_dir/data/temperatureSurfaceValue.dat.0"
for time_step in {0..79}
do
    ln -s "$source_dir/temperature.dat.$time_step" "$work_dir/data/temperature.dat.$time_step"
done
cp build/s1_compress build/s1_decompress "$work_dir/"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=4
export OMP_DYNAMIC=false
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

echo "===== surface compression ====="
/usr/bin/time -v "$work_dir/s1_compress" -M REL 0.001 -data-dir "$work_dir/data" -field temperatureSurfaceValue -time-step '{0}' -coordinate-dim 3 -partition-num 4 -thread-num 4 -compression-stage-time
echo "===== surface decompression ====="
/usr/bin/time -v "$work_dir/s1_decompress" -M REL 0.001 -data-dir "$work_dir/data" -field temperatureSurfaceValue -time-step '{0}' -coordinate-dim 3 -partition-num 4 -thread-num 4 -decompression-stage-time

echo "===== vertical compression ====="
/usr/bin/time -v "$work_dir/s1_compress" -point-set-config /home/x-zjian1/unsz/point_set_configs/Ocean_large_temperature_rbf_rel_1e-3.json -data-dir "$work_dir/data" -field temperature -time-step '[0,79]' -time-step-coordinate-file "$work_dir/data/refBottomDepth.dat" -coordinate-dim 3 -partition-num 4 -thread-num 4 -compression-stage-time
echo "===== vertical decompression ====="
/usr/bin/time -v "$work_dir/s1_decompress" -point-set-config /home/x-zjian1/unsz/point_set_configs/Ocean_large_temperature_rbf_rel_1e-3.json -data-dir "$work_dir/data" -field temperature -time-step '[0,79]' -time-step-coordinate-file "$work_dir/data/refBottomDepth.dat" -coordinate-dim 3 -partition-num 4 -thread-num 4 -decompression-stage-time
