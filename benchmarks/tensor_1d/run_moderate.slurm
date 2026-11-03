#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --job-name=moderate_1d
#SBATCH --output=/home/x-zjian1/unsz/results/%x_%j.log

# 与run_configurations.slurm中UNSZ和UMC的单线程计时使用同样的资源；SZ3-1D和PFPL在同一节点上依次运行五个REL EB
set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "git_revision=$(git rev-parse HEAD) hostname=$(hostname) slurm_job_id=$SLURM_JOB_ID"
sha256sum build/tensor_1d/benchmark_sz3 build/tensor_1d/benchmark_pfpl
for codec in sz3 pfpl;do
    echo "BEGIN codec=$codec $(date -Is)"
    ./unsz_env/bin/python -u benchmarks/tensor_1d/run_moderate.py --codec "$codec"
    echo "END codec=$codec $(date -Is)"
done
