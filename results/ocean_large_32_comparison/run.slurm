#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=ocean32_compare
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_32_comparison/%x_%j.log

set -euo pipefail
project=/home/x-zjian1/unsz
result_dir="$project/results/ocean_large_32_comparison"
data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data
export UMC_OCEAN_ROOT=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/umc_parallel
partition_binary="$project/third_party/umc/build/parallel_src/test_partition"
umc_compress_binary="$project/third_party/umc/build/parallel_src/test_par_compress"
umc_decompress_binary="$project/third_party/umc/build/parallel_src/test_par_decompress"
unsz_compress_binary="$project/build/s1_compress"
unsz_decompress_binary="$project/build/s1_decompress"
mkdir -p "$result_dir/logs" "$UMC_OCEAN_ROOT/mesh/partition" "$UMC_OCEAN_ROOT/compressed/idw"

if [[ ! -f "$UMC_OCEAN_ROOT/mesh/partition/32.part" ]];then
    /usr/bin/time -v "$partition_binary" 32 >"$result_dir/logs/umc_partition.log" 2>&1
fi

export OMP_NUM_THREADS=1
for time_step in $(seq 0 11);do
    srun --exact --ntasks=32 --cpus-per-task=1 --cpu-bind=cores "$umc_compress_binary" 0 1 "$time_step" 0 4 1e-3 >"$result_dir/logs/umc_compress_$time_step.log"
done
for time_step in $(seq 0 11);do
    srun --exact --ntasks=32 --cpus-per-task=1 --cpu-bind=cores "$umc_decompress_binary" 0 1 "$time_step" 0 4 1e-3 >"$result_dir/logs/umc_decompress_$time_step.log"
done

export OMP_NUM_THREADS=32
export OMP_DYNAMIC=false
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
invalid_field=timeMonthly_avg_bruntVaisalaFreqML
regular_field_expression="{$(grep -v "^${invalid_field}$" "$UMC_OCEAN_ROOT/varnames_91.txt" | paste -sd,)}"
point_set_config="$project/point_set_configs/Ocean_large_monthly_rbf_rel_1e-3.json"
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$unsz_compress_binary" -M REL 0.001 -data-dir "$data_dir" -field-list "$regular_field_expression" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 -compression-stage-time >"$result_dir/logs/unsz_regular_compress.log"
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$unsz_decompress_binary" -M REL 0.001 -data-dir "$data_dir" -field-list "$regular_field_expression" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 >"$result_dir/logs/unsz_regular_decompress.log"
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$unsz_compress_binary" -point-set-config "$point_set_config" -data-dir "$data_dir" -field "$invalid_field" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 -compression-stage-time >"$result_dir/logs/unsz_invalid_compress.log"
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$unsz_decompress_binary" -point-set-config "$point_set_config" -data-dir "$data_dir" -field "$invalid_field" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 >"$result_dir/logs/unsz_invalid_decompress.log"

python3 "$result_dir/summarize.py" | tee "$result_dir/summary.txt"
