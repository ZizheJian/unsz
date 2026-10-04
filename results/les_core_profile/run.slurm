#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --job-name=les_core_profile
#SBATCH --output=/home/x-zjian1/unsz/results/les_core_profile/run_%j.log

set -e
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

for repetition in 1 2 3 4 5
do
    echo "repetition=$repetition compression"
    install/bin/s1_compress -M REL 0.001 -predictor rbf -data-dir /anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/LES -field velocity_x -time-step '{0}' -coordinate-dim 3 -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -dag-cache /home/x-zjian1/unsz/results/les_spatial/rbf10_global.dag -compression-stage-time
    echo "repetition=$repetition decompression"
    install/bin/s1_decompress -M REL 0.001 -predictor rbf -data-dir /anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/LES -field velocity_x -time-step '{0}' -coordinate-dim 3 -prediction-min-ref-num 10 -prediction-max-ref-num 10 -ordering-scope global -ordering-score-max-ref-num 1 -ordering-score-min-ref-num 1 -ordering-candidate-num 16 -rbf-kernel matern32 -rbf-shape 0.03 -rbf-nugget 1e-06 -eb-schedule inverse_rank_log2 0.85 -eb-schedule-log2-shift 0.5 -eb-schedule-max-log2-exponent 7 -dag-cache /home/x-zjian1/unsz/results/les_spatial/rbf10_global.dag -decompression-stage-time
done
