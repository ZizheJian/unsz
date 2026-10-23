#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --job-name=ocean_scaling
#SBATCH --output=/home/x-zjian1/unsz/results/ocean_large_scaling/%x_%j.log

set -euo pipefail
cd /home/x-zjian1/unsz
scaling_codec=${1:?Expected codec}
scaling_threads=${2:?Expected thread count}
case "$scaling_codec" in unsz|sz3|pfpl) ;; *) echo "Unsupported codec: $scaling_codec" >&2; exit 1;; esac
case "$scaling_threads" in 1|2|4|8|16|32) ;; *) echo "Unsupported thread count: $scaling_threads" >&2; exit 1;; esac
required_cpus=$((scaling_threads<8?8:scaling_threads))
if [[ ${SLURM_CPUS_PER_TASK:-0} -ne $required_cpus ]]; then echo "p=$scaling_threads requires --cpus-per-task=$required_cpus" >&2; exit 1; fi
export OMP_NUM_THREADS=$scaling_threads
export OMP_DYNAMIC=false
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p results/ocean_large_scaling
echo "BEGIN scaling codec=$scaling_codec,threads=$scaling_threads,slurm_cpus=$SLURM_CPUS_PER_TASK"
srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores /usr/bin/time -v -o "results/ocean_large_scaling/${scaling_codec}_p${scaling_threads}_resources.txt" ./unsz_env/bin/python -u benchmarks/scaling/run.py --codec "$scaling_codec" --threads "$scaling_threads"
echo "END scaling codec=$scaling_codec,threads=$scaling_threads"
