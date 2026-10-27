#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --job-name=unsz_configs
#SBATCH --output=/home/x-zjian1/unsz/results/current/logs/%x_%j.log

# 用法：sbatch [--time=...] run_configurations.slurm <config...>
# 依次把configs中的数据集JSON交给t1；UNSZ与UMC都单线程运行，同一数据集的两个压缩器在同一节点上依次运行
# 使用已经安装并测试过的二进制，不在作业中重新构建，避免并发作业同时写构建目录
set -euo pipefail
cd /home/x-zjian1/unsz
if [[ $# -lt 1 ]];then echo 'Usage: sbatch run_configurations.slurm <config...>'>&2;exit 2;fi
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
echo "git_revision=$(git rev-parse HEAD) hostname=$(hostname) slurm_job_id=$SLURM_JOB_ID"
sha256sum install/bin/s1_compress install/bin/s1_decompress third_party/umc/build/test/test_compress third_party/umc/build/test/test_decompress
for config in "$@";do
    echo "BEGIN config=$config $(date -Is)"
    ./unsz_env/bin/python -u t1_start_c_code.py "$config" --skip-build
    echo "END config=$config $(date -Is)"
done
