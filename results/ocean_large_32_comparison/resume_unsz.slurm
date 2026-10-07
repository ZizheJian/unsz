#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=ocean32_resume
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_32_comparison/%x_%j.log

set -euo pipefail
project=/home/x-zjian1/unsz
result_dir="$project/results/ocean_large_32_comparison"
data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data
umc_root=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/umc_parallel
unsz_compress_binary="$project/build/s1_compress"
unsz_decompress_binary="$project/build/s1_decompress"
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=false
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
invalid_field=timeMonthly_avg_bruntVaisalaFreqML
regular_field_expression="{$(grep -v "^${invalid_field}$" "$umc_root/varnames_91.txt" | paste -sd,)}"
point_set_config="$project/point_set_configs/Ocean_large_monthly_rbf_rel_1e-3.json"

if [[ -f "$result_dir/logs/unsz_regular_decompress.log" ]];then
    cp "$result_dir/logs/unsz_regular_decompress.log" "$result_dir/logs/unsz_regular_decompress.failed.log"
fi
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$unsz_decompress_binary" -M REL 0.001 -data-dir "$data_dir" -field-list "$regular_field_expression" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 >"$result_dir/logs/unsz_regular_decompress.log"
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$unsz_compress_binary" -point-set-config "$point_set_config" -data-dir "$data_dir" -field "$invalid_field" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 -compression-stage-time >"$result_dir/logs/unsz_invalid_compress.log"
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$unsz_decompress_binary" -point-set-config "$point_set_config" -data-dir "$data_dir" -field "$invalid_field" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 >"$result_dir/logs/unsz_invalid_decompress.log"
python3 "$result_dir/summarize.py" >"$result_dir/summary.txt"
cat "$result_dir/summary.txt"
