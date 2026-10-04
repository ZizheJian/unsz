#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH --job-name=unsz_preprocessing_timing
#SBATCH --output=/home/x-zjian1/unsz/results/coordinate_transform/preprocessing_timing_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz

install/bin/s1_compress -M REL 0.001 -predictor rbf -data-dir /anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Tsunami -field momentum_x -time-step '{0}' -coordinate-dim 2 -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -dag-cache /tmp/unsz_preprocessing_timing_tsunami_${SLURM_JOB_ID}.dag -coordinate-transform auto > results/coordinate_transform/preprocessing_timing_tsunami.log

install/bin/s1_compress -M REL 0.001 -predictor rbf -data-dir /anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/LES -field velocity_x -time-step '{0}' -coordinate-dim 3 -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -dag-cache /tmp/unsz_preprocessing_timing_les_${SLURM_JOB_ID}.dag -coordinate-transform auto > results/coordinate_transform/preprocessing_timing_les.log

install/bin/s1_compress -point-set-config point_set_configs/E3SM_h4_AODDUST_rbf_rel_1e-3.json -predictor rbf -data-dir /anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/E3SM-h4 -field AODDUST -time-step '[0,719]' -coordinate-dim 3 -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -dag-cache /tmp/unsz_preprocessing_timing_e3sm_${SLURM_JOB_ID}.dag -coordinate-transform auto -label-value-execution fused -value-flow anchor_interpolation_stride_8 > results/coordinate_transform/preprocessing_timing_e3sm.log

install/bin/s1_compress -point-set-config point_set_configs/ADCIRC_Haiyan_zeta_rbf_rel_1e-3.json -predictor rbf -data-dir /anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/ADCIRC-Haiyan-1.5km -field zeta -time-step '[0,23]' -coordinate-dim 3 -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -dag-cache /tmp/unsz_preprocessing_timing_haiyan_${SLURM_JOB_ID}.dag -coordinate-transform auto -label-value-execution fused -value-flow anchor_interpolation_stride_8_spatial_residual -label-flow anchor_interpolation_stride_8 > results/coordinate_transform/preprocessing_timing_haiyan.log

install/bin/s1_compress -point-set-config point_set_configs/Katrina_attr0_rbf_rel_1e-3.json -predictor rbf -data-dir /anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina -field attr0 -time-step '[0,161]' -coordinate-dim 3 -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -dag-cache /tmp/unsz_preprocessing_timing_katrina_${SLURM_JOB_ID}.dag -coordinate-transform auto -label-value-execution fused -value-flow constant_extrapolation_spatial_residual -label-flow constant_extrapolation > results/coordinate_transform/preprocessing_timing_katrina.log
