#!/bin/bash
#SBATCH -A cis240669
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:06:00
#SBATCH -J unsz_katrina_layout

set -euo pipefail
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
project=/home/x-zjian1/unsz
result_dir="$project/results/katrina_reference_layout_benchmark"
source_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina
test_data_dir="/tmp/unsz_katrina_layout_${SLURM_JOB_ID}"
mkdir -p "$test_data_dir"
ln -s "$source_dir/coords.dat" "$test_data_dir/coords.dat"
for time_step in $(seq 0 161);do
    ln -s "$source_dir/attr0.dat.$time_step" "$test_data_dir/attr0.dat.$time_step"
done

common=(-point-set-config "$project/point_set_configs/Katrina_attr0_rbf_rel_1e-3.json" -predictor rbf -data-dir "$test_data_dir" -field attr0 -time-step '[0,161]' -coordinate-dim 3 -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -adjust-coord-matrix-file "$project/adjust_coord_matrices/Katrina_attr0_spatial.txt" -label-value-execution fused -value-flow constant_extrapolation_spatial_residual -label-flow constant_extrapolation -dag-cache "$result_dir/katrina_global.dag")

run_variant()
{
    local repeat=$1
    local variant=$2
    local force_reference_offsets=0
    if [[ "$variant" == baseline ]];then
        force_reference_offsets=1
    fi
    env UNSZ_BENCHMARK_FORCE_REFERENCE_OFFSETS="$force_reference_offsets" "$result_dir/runtime_s1_compress" "${common[@]}" -compression-stage-time >"$result_dir/${variant}_${repeat}_compress.log"
    env UNSZ_BENCHMARK_FORCE_REFERENCE_OFFSETS="$force_reference_offsets" "$result_dir/runtime_s1_decompress" "${common[@]}" -decompression-stage-time >"$result_dir/${variant}_${repeat}_decompress.log"
}

run_variant warmup baseline
run_variant warmup optimized
for repeat in $(seq 1 15);do
    if ((repeat%2==1));then
        run_variant "$repeat" baseline
        run_variant "$repeat" optimized
    else
        run_variant "$repeat" optimized
        run_variant "$repeat" baseline
    fi
done

rm -f "$result_dir/baseline_warmup_compress.log" "$result_dir/baseline_warmup_decompress.log" "$result_dir/optimized_warmup_compress.log" "$result_dir/optimized_warmup_decompress.log"
rm -f "$test_data_dir"/*
rmdir "$test_data_dir"
