#!/bin/bash
#SBATCH -A cis240669
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 5
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -J unsz_multihead

set -euo pipefail
project=/home/x-zjian1/unsz
test_dir="$project/results/multihead_test"
data_dir="$test_dir/data"
source_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina
mkdir -p "$data_dir"
ln -sf "$source_dir/coords.dat" "$data_dir/coords.dat"
for step in $(seq 0 161);do
    ln -sf "$source_dir/attr0.dat.$step" "$data_dir/attr0.dat.$step"
done

common=(-point-set-config "$project/point_set_configs/Katrina_attr0_rbf_rel_1e-3.json" -predictor rbf -data-dir "$data_dir" -field attr0 -time-step '[0,161]' -coordinate-dim 3 -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -adjust-coord-matrix-file "$project/adjust_coord_matrices/Katrina_attr0_spatial.txt" -label-value-execution fused -value-flow constant_extrapolation_spatial_residual -label-flow constant_extrapolation -dag-cache "$test_dir/katrina_global.dag")

run_variant()
{
    local repeat=$1
    local name=$2
    shift 2
    "$project/install/bin/s1_compress" "${common[@]}" "$@" >"$test_dir/${name}_${repeat}_compress.log"
    "$project/install/bin/s1_decompress" "${common[@]}" "$@" >"$test_dir/${name}_${repeat}_decompress.log"
}

for repeat in 1 2 3;do
    if ((repeat%2==1));then
        variants=(baseline preserve_interleaved preserve_separate execution_interleaved execution_separate)
    else
        variants=(execution_separate execution_interleaved preserve_separate preserve_interleaved baseline)
    fi
    for variant in "${variants[@]}";do
        case "$variant" in
            baseline)
                run_variant "$repeat" "$variant"
                ;;
            preserve_interleaved)
                run_variant "$repeat" "$variant" -multihead-execution-head-num-experimental 2 -multihead-eb-order-experimental preserve
                ;;
            preserve_separate)
                run_variant "$repeat" "$variant" -multihead-execution-head-num-experimental 2 -multihead-eb-order-experimental preserve -multihead-separate-qb-streams-experimental
                ;;
            execution_interleaved)
                run_variant "$repeat" "$variant" -multihead-execution-head-num-experimental 2 -multihead-eb-order-experimental execution
                ;;
            execution_separate)
                run_variant "$repeat" "$variant" -multihead-execution-head-num-experimental 2 -multihead-eb-order-experimental execution -multihead-separate-qb-streams-experimental
                ;;
        esac
    done
done
