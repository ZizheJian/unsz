#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=02:30:00
#SBATCH --job-name=unsz_slice_rd
#SBATCH --output=/home/x-zjian1/unsz/results/%x_%j.log

# 用法：sbatch [--cpus-per-task=N] run_five_ebs.slurm [slice|field] [field-workers] [rel-eb...]
# 默认在128核上用16个field worker、每个worker 8个partition运行per-slice协议的五个EB；已完成的EB自动跳过
# 环境变量UNSZ_CPU_NUM指定实际使用的核数（默认为申请的CPU数），UNSZ_DIR_SUFFIX给结果目录加后缀
set -euo pipefail
cd /home/x-zjian1/unsz
protocol=${1:-slice};field_workers=${2:-16};shift $(($#<2?$#:2))
rel_ebs=("$@");if [[ ${#rel_ebs[@]} -eq 0 ]];then rel_ebs=(1e-1 1e-2 1e-3 1e-4 1e-5);fi
cpu_num=${UNSZ_CPU_NUM:-$SLURM_CPUS_PER_TASK};partitions=$((cpu_num/field_workers));dir_suffix=${UNSZ_DIR_SUFFIX:-}
if ((cpu_num>SLURM_CPUS_PER_TASK));then echo 'UNSZ_CPU_NUM must not exceed the allocated CPUs'>&2;exit 2;fi
export OMP_DYNAMIC=false
export OMP_MAX_ACTIVE_LEVELS=2
export OMP_PROC_BIND=spread,close
export OMP_PLACES=cores
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
for rel_eb in "${rel_ebs[@]}";do
    output_dir="results/ocean_large_per_${protocol}_rd_${cpu_num}${dir_suffix}/unsz/${field_workers}_fields_${partitions}_partitions/rel_$rel_eb"
    if [[ -f "$output_dir/summary.json" ]]&&jq -e --arg protocol "$protocol" --argjson rel_eb "$rel_eb" --argjson cpu_num "$cpu_num" --argjson field_workers "$field_workers" '.protocol==$protocol and .rel_eb==$rel_eb and .cpu_num==$cpu_num and .field_worker_num==$field_workers and .field_num==95 and .valid' "$output_dir/summary.json">/dev/null;then echo "SKIP completed REL=$rel_eb";continue;fi
    mkdir -p "$output_dir"
    srun --exact --kill-on-bad-exit=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=none /usr/bin/time -v -o "$output_dir/resources_job_${SLURM_JOB_ID}.txt" ./unsz_env/bin/python -u benchmarks/two_level/run.py --protocol "$protocol" --rel-eb "$rel_eb" --field-worker-num "$field_workers" --cpu-num "$cpu_num" --output-root "$output_dir" --launcher benchmarks/two_level/run_five_ebs.slurm
done
