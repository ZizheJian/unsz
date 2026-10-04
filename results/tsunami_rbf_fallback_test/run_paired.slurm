#!/bin/bash
#SBATCH -A cis240669
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:04:00
#SBATCH -J unsz_tsunami_fast

set -euo pipefail
project=/home/x-zjian1/unsz
result_dir="$project/results/tsunami_rbf_fallback_test"
source_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Tsunami
test_data_dir="/tmp/unsz_tsunami_rbf_fallback_${SLURM_JOB_ID}"
field_vec=(bathy height momentum_x momentum_y velocity_x velocity_y)
mkdir -p "$test_data_dir"
ln -s "$source_dir/coords.dat" "$test_data_dir/coords.dat"
for field in "${field_vec[@]}";do
    ln -s "$source_dir/$field.dat.0" "$test_data_dir/$field.dat.0"
done

common=(-point-set-config "$result_dir/single_point_set_rel_1e-3.json" -predictor rbf -data-dir "$test_data_dir" -time-step '[0,0]' -coordinate-dim 2 -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -dag-cache "$result_dir/tsunami_global.dag")

run_variant()
{
    local repeat=$1
    local name=$2
    local compress_binary=$3
    local decompress_binary=$4
    for field in "${field_vec[@]}";do
        "$compress_binary" "${common[@]}" -field "$field" >"$result_dir/${name}_${repeat}_${field}_compress.log"
        "$decompress_binary" "${common[@]}" -field "$field" >"$result_dir/${name}_${repeat}_${field}_decompress.log"
    done
}

for repeat in $(seq 1 12);do
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
