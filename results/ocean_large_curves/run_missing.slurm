#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --job-name=ocean_curve
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_curves/%x_%j.log

set -euo pipefail
project=/home/x-zjian1/unsz
result_dir="$project/results/ocean_large_curves"
comparison_dir="$project/results/ocean_large_32_comparison"
source_data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data
curve_data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data_curve
export UMC_OCEAN_ROOT=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/umc_parallel
invalid_field=timeMonthly_avg_bruntVaisalaFreqML
regular_field_expression="{$(grep -v "^${invalid_field}$" "$UMC_OCEAN_ROOT/varnames_91.txt" | paste -sd,)}"
mkdir -p "$result_dir/logs" "$curve_data_dir"
ln -sfn "$source_data_dir/coords.dat" "$curve_data_dir/coords.dat"
ln -sfn "$source_data_dir/conn.dat" "$curve_data_dir/conn.dat"
while IFS= read -r field;do
    for time_step in $(seq 0 11);do
        ln -sfn "$source_data_dir/$field.dat.$time_step" "$curve_data_dir/$field.dat.$time_step"
    done
done <"$UMC_OCEAN_ROOT/varnames_91.txt"

"$project/unsz_env/bin/python" "$result_dir/collect_metrics.py" --rel-eb 0.001 --log-dir "$comparison_dir/logs" --unsz-data-dir "$source_data_dir" --output "$result_dir/metrics_1e-3.json"

export OMP_NUM_THREADS=1
partition_binary="$project/third_party/umc/build/parallel_src/test_partition"
umc_compress_binary="$project/third_party/umc/build/parallel_src/test_par_compress"
umc_decompress_binary="$project/third_party/umc/build/parallel_src/test_par_decompress"
if [[ ! -f "$UMC_OCEAN_ROOT/mesh/partition/32.part" ]];then
    "$partition_binary" 32 >"$result_dir/logs/umc_partition.log" 2>&1
fi

for exponent in 1 2 4 5;do
    rel_eb="1e-$exponent"
    label="1e-$exponent"
    log_dir="$result_dir/logs/$label"
    mkdir -p "$log_dir"
    for time_step in $(seq 0 11);do
        srun --exact --ntasks=32 --cpus-per-task=1 --cpu-bind=cores "$umc_compress_binary" 0 1 "$time_step" 0 4 "$rel_eb" >"$log_dir/umc_compress_$time_step.log"
    done
    for time_step in $(seq 0 11);do
        srun --exact --ntasks=32 --cpus-per-task=1 --cpu-bind=cores "$umc_decompress_binary" 0 1 "$time_step" 0 4 "$rel_eb" >"$log_dir/umc_decompress_$time_step.log"
    done

    export OMP_NUM_THREADS=32
    export OMP_DYNAMIC=false
    export OMP_PROC_BIND=spread
    export OMP_PLACES=cores
    srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$project/build/s1_compress" -M REL "$rel_eb" -data-dir "$curve_data_dir" -field-list "$regular_field_expression" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 -compression-stage-time >"$log_dir/unsz_regular_compress.log"
    srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$project/build/s1_decompress" -M REL "$rel_eb" -data-dir "$curve_data_dir" -field-list "$regular_field_expression" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 >"$log_dir/unsz_regular_decompress.log"
    point_set_config="$project/point_set_configs/Ocean_large_monthly_rbf_rel_$label.json"
    srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$project/build/s1_compress" -point-set-config "$point_set_config" -data-dir "$curve_data_dir" -field "$invalid_field" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 -compression-stage-time >"$log_dir/unsz_invalid_compress.log"
    srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$project/build/s1_decompress" -point-set-config "$point_set_config" -data-dir "$curve_data_dir" -field "$invalid_field" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 >"$log_dir/unsz_invalid_decompress.log"
    "$project/unsz_env/bin/python" "$result_dir/collect_metrics.py" --rel-eb "$rel_eb" --log-dir "$log_dir" --unsz-data-dir "$curve_data_dir" --output "$result_dir/metrics_$label.json"
    find "$curve_data_dir" -maxdepth 1 -type f \( -name '*.unsz' -o -name '*.unsz.bin' \) -delete
    export OMP_NUM_THREADS=1
done

export MPLCONFIGDIR="$result_dir/matplotlib"
mkdir -p "$MPLCONFIGDIR"
"$project/unsz_env/bin/python" "$result_dir/plot.py"
