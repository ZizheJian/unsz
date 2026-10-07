#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=wholenode
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --job-name=umc_ocean_128
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_umc_128/%x_%j.log

set -euo pipefail
project=/home/x-zjian1/unsz
export UMC_OCEAN_ROOT=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/umc_parallel
partition_binary="$project/third_party/umc/build/parallel_src/test_partition"
compress_binary="$project/third_party/umc/build/parallel_src/test_par_compress"
decompress_binary="$project/third_party/umc/build/parallel_src/test_par_decompress"
mkdir -p "$UMC_OCEAN_ROOT/mesh/partition" "$UMC_OCEAN_ROOT/compressed/idw"

if [[ ! -f "$UMC_OCEAN_ROOT/mesh/partition/128.part" ]];then
    echo "阶段：生成128个METIS partition"
    /usr/bin/time -v "$partition_binary" 128
fi

echo "阶段：UMC NBP压缩，time step 0，91 fields，REL=1e-3"
srun --exact --ntasks=128 --cpus-per-task=1 --cpu-bind=cores "$compress_binary" 0 1 0 0 4 1e-3

echo "阶段：UMC NBP解压，time step 0，91 fields，REL=1e-3"
srun --exact --ntasks=128 --cpus-per-task=1 --cpu-bind=cores "$decompress_binary" 0 1 0 0 4 1e-3
