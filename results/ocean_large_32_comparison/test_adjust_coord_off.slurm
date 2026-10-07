#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --job-name=ocean90_coord_off
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_32_comparison/%x_%j.log

set -euo pipefail
project=/home/x-zjian1/unsz
result_dir="$project/results/ocean_large_32_comparison"
source_data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data
off_data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data_adjust_coord_off
umc_root=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/umc_parallel
invalid_field=timeMonthly_avg_bruntVaisalaFreqML
mkdir -p "$off_data_dir"
ln -sfn "$source_data_dir/coords.dat" "$off_data_dir/coords.dat"
ln -sfn "$source_data_dir/conn.dat" "$off_data_dir/conn.dat"
while IFS= read -r field;do
    if [[ "$field" == "$invalid_field" ]];then
        continue
    fi
    for time_step in $(seq 0 11);do
        ln -sfn "$source_data_dir/$field.dat.$time_step" "$off_data_dir/$field.dat.$time_step"
    done
done <"$umc_root/varnames_91.txt"

export OMP_NUM_THREADS=32
export OMP_DYNAMIC=false
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
regular_field_expression="{$(grep -v "^${invalid_field}$" "$umc_root/varnames_91.txt" | paste -sd,)}"
srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$project/build/s1_compress" -M REL 0.001 -data-dir "$off_data_dir" -field-list "$regular_field_expression" -time-step '[0,11]' -coordinate-dim 3 -coordinate-transform off -partition-num 32 -thread-num 32 -compression-stage-time >"$result_dir/logs/unsz_regular_adjust_coord_off_compress.log"
python3 "$result_dir/summarize_adjust_coord_comparison.py" >"$result_dir/adjust_coord_comparison.txt"
cat "$result_dir/adjust_coord_comparison.txt"
