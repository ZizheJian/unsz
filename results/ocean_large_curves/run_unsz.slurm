#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:45:00
#SBATCH --job-name=ocean_unsz
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_curves/%x_%j.log

set -euo pipefail
project=/home/x-zjian1/unsz
result_dir="$project/results/ocean_large_curves"
source_data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data
curve_data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data_curve
umc_ocean_root=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/umc_parallel
field_expression="{$(paste -sd, "$umc_ocean_root/varnames_91.txt")}"
mkdir -p "$result_dir/logs" "$curve_data_dir"
ln -sfn "$source_data_dir/coords.dat" "$curve_data_dir/coords.dat"
ln -sfn "$source_data_dir/conn.dat" "$curve_data_dir/conn.dat"
while IFS= read -r field;do
    for time_step in $(seq 0 11);do
        ln -sfn "$source_data_dir/$field.dat.$time_step" "$curve_data_dir/$field.dat.$time_step"
    done
done <"$umc_ocean_root/varnames_91.txt"

export OMP_NUM_THREADS=32
export OMP_DYNAMIC=false
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
for exponent in ${EXPONENTS:-1 2 3 4 5};do
    rel_eb="1e-$exponent"
    label="1e-$exponent"
    log_dir="$result_dir/logs/$label"
    mkdir -p "$log_dir"
    metric_path="$result_dir/metrics_$label.json"
    if [[ "${FORCE:-false}" != true ]]&&[[ -f "$metric_path" ]]&&grep -q '"collector_version": 4' "$metric_path";then
        continue
    fi
    output_num=$(find "$curve_data_dir" -maxdepth 1 -type f -name '*.unsz' | wc -l)
    compression_metric_num=$(grep -c '^normalized squared error sum=' "$log_dir/unsz_compress.log" 2>/dev/null||true)
    decompression_metric_num=$(grep -c '^normalized squared error sum=' "$log_dir/unsz_decompress.log" 2>/dev/null||true)
    compression_timing_num=$(grep -c '^total compression preprocessing time excluding' "$log_dir/unsz_compress.log" 2>/dev/null||true)
    decompression_timing_num=$(grep -c '^total decompression preprocessing time excluding' "$log_dir/unsz_decompress.log" 2>/dev/null||true)
    if [[ "$output_num" -ne 1092 || "$compression_metric_num" -ne 91 || "$decompression_metric_num" -ne 91 || "$compression_timing_num" -ne 1 || "$decompression_timing_num" -ne 1 ]];then
        point_set_config="$project/point_set_configs/Ocean_large_monthly_rbf_rel_$label.json"
        srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$project/install/bin/s1_compress" -point-set-config "$point_set_config" -data-dir "$curve_data_dir" -field-list "$field_expression" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 -compression-stage-time >"$log_dir/unsz_compress.log"
        srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$project/install/bin/s1_decompress" -point-set-config "$point_set_config" -data-dir "$curve_data_dir" -field-list "$field_expression" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 -skip-decompressed-output >"$log_dir/unsz_decompress.log"
    fi
    "$project/unsz_env/bin/python" "$result_dir/collect_metrics.py" --rel-eb "$rel_eb" --log-dir "$log_dir" --unsz-data-dir "$curve_data_dir" --reuse-umc-metrics "$metric_path" --output "$metric_path"
    find "$curve_data_dir" -maxdepth 1 -type f \( -name '*.unsz' -o -name '*.unsz.bin' \) -delete
done

export MPLCONFIGDIR="${TMPDIR:-/tmp}/unsz-matplotlib-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
"$project/unsz_env/bin/python" "$result_dir/plot.py"
