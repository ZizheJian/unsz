#!/bin/bash
#SBATCH --job-name=unsz-katrina-flows
#SBATCH --account=cis240669
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=/home/x-zjian1/unsz/results/katrina_anchor_stride_experimental/full_benchmark_%j.log

set -euo pipefail

project=/home/x-zjian1/unsz
source_data=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina
dag=$project/results/current/prediction_structures/katrina_attr0.dag
config=$project/point_set_configs/Katrina_attr0_rbf_rel_1e-3.json
matrix=$project/adjust_coord_matrices/Katrina_attr0_spatial.txt
work_root=${SLURM_TMPDIR:-/tmp}/unsz_katrina_selected_flows
common=(-variable attr0 -time-step '[0,161]' -dim 3 -point-set-config "$config" -adjust-coord-matrix-file "$matrix" -dag-cache "$dag" -predict-round-num 8 -max-ref-num 10 -actual-ref-num 10 -inverse-distance-power 3.6 -kd-leaf-size 32 -d-only-cache-refill-threshold-experimental 10 -order-candidate-num-experimental 48 -eb-schedule-experimental round_point_ratio 0.8 -rbf-kernel matern32 -rbf-shape 0.01 -rbf-nugget 1e-6 -rbf-filtered-fallback-nearest-experimental -fused-two-point-set-prediction-experimental)

run_flow()
{
    name=$1
    shift
    data_dir=$work_root/$name
    mkdir -p "$data_dir"
    ln -sf "$source_data/coords.dat" "$data_dir/coords.dat"
    for time_step in $(seq 0 161)
    do
        ln -sf "$source_data/attr0.dat.$time_step" "$data_dir/attr0.dat.$time_step"
    done
    echo "FLOW_BEGIN $name"
    "$project/install/bin/s1_compress" -data-dir "$data_dir" "${common[@]}" "$@" | tee "$project/results/katrina_anchor_stride_experimental/${name}_compress.log"
    "$project/install/bin/s1_decompress" -data-dir "$data_dir" "${common[@]}" "$@" | tee "$project/results/katrina_anchor_stride_experimental/${name}_decompress.log"
    echo "FLOW_END $name"
}

run_flow constant -cross-time-label-predictor-experimental previous -cross-time-predictor-experimental constant_extrapolation_spatial_residual
run_flow interpolation -cross-time-label-predictor-experimental endpoint_agreement -cross-time-predictor-experimental linear_interpolation_spatial_residual
run_flow hybrid_stride8 -time-anchor-stride-experimental 8
