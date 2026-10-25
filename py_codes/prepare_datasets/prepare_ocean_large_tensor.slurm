#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --job-name=prepare_ocean_tensor
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_tensor_prepare_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
arguments=()
if [[ -n "${FIELDS:-}" ]];then
    IFS=':' read -ra fields<<<"$FIELDS"
    for field in "${fields[@]}";do
        arguments+=("-field" "$field")
    done
fi
./unsz_env/bin/python py_codes/prepare_datasets/prepare_ocean_large_tensor.py "${arguments[@]}"
