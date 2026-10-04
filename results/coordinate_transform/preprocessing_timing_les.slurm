#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:05:00
#SBATCH --job-name=unsz_les_preprocessing
#SBATCH --output=/home/x-zjian1/unsz/results/coordinate_transform/preprocessing_timing_les_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
install/bin/s1_compress -M REL 0.001 -predictor rbf -data-dir /anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/LES -field velocity_x -time-step '{0}' -coordinate-dim 3 -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -dag-cache /tmp/unsz_preprocessing_timing_les_${SLURM_JOB_ID}.dag -coordinate-transform auto > results/coordinate_transform/preprocessing_timing_les.log
