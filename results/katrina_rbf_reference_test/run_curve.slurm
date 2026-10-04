#!/bin/bash
#SBATCH -A cis240669
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:06:00
#SBATCH -J unsz_katrina_ref_curve

set -euo pipefail
project=/home/x-zjian1/unsz
result_dir="$project/results/katrina_rbf_reference_test/full_curve"
source_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina
test_data_dir="/tmp/unsz_katrina_reference_curve_${SLURM_JOB_ID}"
mkdir -p "$test_data_dir" "$result_dir"
ln -s "$source_dir/coords.dat" "$test_data_dir/coords.dat"
for time_step in $(seq 0 161);do
    ln -s "$source_dir/attr0.dat.$time_step" "$test_data_dir/attr0.dat.$time_step"
done

common=(-predictor rbf -data-dir "$test_data_dir" -field attr0 -time-step '[0,161]' -coordinate-dim 3 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -adjust-coord-matrix-file "$project/adjust_coord_matrices/Katrina_attr0_spatial.txt" -label-value-execution fused -value-flow constant_extrapolation_spatial_residual -label-flow constant_extrapolation)

run_variant()
{
    local eb=$1
    local repeat=$2
    local ref_num=$3
    local eb_dir="$result_dir/$eb"
    mkdir -p "$eb_dir"
    local output_prefix="$eb_dir/rbf${ref_num}_${repeat}"
    local ref_arguments=(-prediction-min-ref-num "$ref_num" -prediction-max-ref-num "$ref_num" -dag-cache "$result_dir/katrina_rbf${ref_num}.dag")
    "$project/build/s1_compress" "${common[@]}" -point-set-config "$project/point_set_configs/Katrina_attr0_rbf_rel_${eb}.json" "${ref_arguments[@]}" -compression-stage-time >"${output_prefix}_compress.log"
    "$project/build/s1_decompress" "${common[@]}" -point-set-config "$project/point_set_configs/Katrina_attr0_rbf_rel_${eb}.json" "${ref_arguments[@]}" -decompression-stage-time >"${output_prefix}_decompress.log"
}

run_variant 1e-3 warmup 4
run_variant 1e-3 warmup 10
for eb in 1e-1 1e-2 1e-3 1e-4 1e-5;do
    for repeat in $(seq 1 4);do
        if ((repeat%2==1));then
            run_variant "$eb" "$repeat" 4
            run_variant "$eb" "$repeat" 10
        else
            run_variant "$eb" "$repeat" 10
            run_variant "$eb" "$repeat" 4
        fi
    done
done

rm -f "$result_dir/1e-3/rbf4_warmup_compress.log" "$result_dir/1e-3/rbf4_warmup_decompress.log" "$result_dir/1e-3/rbf10_warmup_compress.log" "$result_dir/1e-3/rbf10_warmup_decompress.log"
rm -f "$test_data_dir"/*
rmdir "$test_data_dir"
