#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=ocean_adjust_rd
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_adjust_coord_screen/%x_%j.log

set -euo pipefail
project=/home/x-zjian1/unsz
result_dir="$project/results/ocean_large_adjust_coord_screen"
data_dir=/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/monthly_data
mkdir -p "$result_dir/logs"
export OMP_NUM_THREADS=32
export OMP_DYNAMIC=false
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
mapfile -t fields <"$result_dir/fields.txt"
for rel_eb in 0.0005 0.002;do
    eb_name=${rel_eb//./p}
    for field in "${fields[@]}";do
        for mode in on off;do
            srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$project/build/s1_compress" -M REL "$rel_eb" -data-dir "$data_dir" -field "$field" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 -coordinate-transform "$mode" -compression-stage-time >"$result_dir/logs/${field}_${mode}_${eb_name}_compress.log"
            srun --exact --ntasks=1 --cpus-per-task=32 --cpu-bind=cores "$project/build/s1_decompress" -M REL "$rel_eb" -data-dir "$data_dir" -field "$field" -time-step '[0,11]' -coordinate-dim 3 -partition-num 32 -thread-num 32 -coordinate-transform "$mode" >"$result_dir/logs/${field}_${mode}_${eb_name}_decompress.log"
        done
    done
done
python3 "$result_dir/summarize_local_rd.py" | tee "$result_dir/local_rd_summary.txt"
