#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --job-name=unsz_ordering_timing
#SBATCH --output=/home/x-zjian1/unsz/results/ablation/logs/%x_%j.log

# 用法：sbatch benchmarks/ablation/ordering_timing.slurm
# 三种预测顺序交替串行运行，单线程，申请的CPU只用于获得足够内存并减少其他作业的干扰
set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
./unsz_env/bin/python benchmarks/ablation/ordering_timing.py
