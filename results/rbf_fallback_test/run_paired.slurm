#!/bin/bash
#SBATCH -A cis240669
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:04:00
#SBATCH -J unsz_rbf_fast

set -euo pipefail
project=/home/x-zjian1/unsz
result_dir="$project/results/rbf_fallback_test"
source_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina
test_data_dir="/tmp/unsz_rbf_fallback_${SLURM_JOB_ID}"
mkdir -p "$test_data_dir"
ln -s "$source_dir/coords.dat" "$test_data_dir/coords.dat"
for time_step in $(seq 0 161);do
    ln -s "$source_dir/attr0.dat.$time_step" "$test_data_dir/attr0.dat.$time_step"
done

common=(-point-set-config "$project/point_set_configs/Katrina_attr0_rbf_rel_1e-3.json" -predictor rbf -data-dir "$test_data_dir" -field attr0 -time-step '[0,161]' -coordinate-dim 3 -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -adjust-coord-matrix-file "$project/adjust_coord_matrices/Katrina_attr0_spatial.txt" -label-value-execution fused -value-flow constant_extrapolation_spatial_residual -label-flow constant_extrapolation -dag-cache "$result_dir/katrina_global.dag")

run_variant()
{
    local repeat=$1
    local name=$2
    local compress_binary=$3
    local decompress_binary=$4
    "$compress_binary" "${common[@]}" -compression-stage-time >"$result_dir/preserved_${name}_${repeat}_compress.log"
    "$decompress_binary" "${common[@]}" >"$result_dir/preserved_${name}_${repeat}_decompress.log"
}

for repeat in $(seq 1 8);do
    if ((repeat%2==1));then
        run_variant "$repeat" baseline "$project/install/bin/s1_compress" "$project/install/bin/s1_decompress"
        run_variant "$repeat" optimized "$project/build/s1_compress" "$project/build/s1_decompress"
    else
        run_variant "$repeat" optimized "$project/build/s1_compress" "$project/build/s1_decompress"
        run_variant "$repeat" baseline "$project/install/bin/s1_compress" "$project/install/bin/s1_decompress"
    fi
done

rm -f "$test_data_dir"/*
rmdir "$test_data_dir"
