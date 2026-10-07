#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --job-name=ocean_metrics
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_curves/%x_%j.log

set -euo pipefail
project=/home/x-zjian1/unsz
result_dir="$project/results/ocean_large_curves"
curve_data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data_curve
"$project/unsz_env/bin/python" "$result_dir/collect_metrics.py" --rel-eb 1e-5 --log-dir "$result_dir/logs/1e-5" --unsz-data-dir "$curve_data_dir" --output "$result_dir/metrics_1e-5.json"
export MPLCONFIGDIR="$result_dir/matplotlib"
mkdir -p "$MPLCONFIGDIR"
"$project/unsz_env/bin/python" "$result_dir/plot.py"
find "$curve_data_dir" -maxdepth 1 -type f \( -name '*.unsz' -o -name '*.unsz.bin' \) -delete
