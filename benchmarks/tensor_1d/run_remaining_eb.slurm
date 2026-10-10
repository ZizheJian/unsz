#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --array=0-3%1
#SBATCH --job-name=ocean_1d_remaining
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_tensor_1d/%x_%A_%a.log

set -euo pipefail
cd /home/x-zjian1/unsz
case "${1:-}" in sz3|pfpl|zfp) ;; *) echo 'Expected sz3,pfpl,or zfp' >&2; exit 1;; esac
baseline_ebs=(1e-1 1e-2 1e-4 1e-5)
baseline_index=${SLURM_ARRAY_TASK_ID:?Expected a Slurm array task}
if ((baseline_index<0||baseline_index>=${#baseline_ebs[@]})); then exit 1; fi
./unsz_env/bin/python -c 'import hashlib,json,sys;from pathlib import Path;codec=sys.argv[1];expected=json.loads((Path("results/ocean_large_tensor_1d")/codec/"rel_0.001_t1/manifest.json").read_text())["binary_sha256"];actual=hashlib.sha256((Path("build/tensor_1d")/f"benchmark_{codec}").read_bytes()).hexdigest();assert actual==expected,"Binary differs from completed REL 1e-3 baseline";print(f"Verified {codec} binary SHA256={actual}",flush=True)' "$1"
echo "BEGIN codec=$1,REL=${baseline_ebs[baseline_index]},threads=1"
bash benchmarks/tensor_1d/run.slurm "$1" "${baseline_ebs[baseline_index]}"
