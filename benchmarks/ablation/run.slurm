#!/bin/bash
#SBATCH --account=cis240669
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=03:00:00
#SBATCH --job-name=unsz_ablation
#SBATCH --output=/home/x-zjian1/unsz/results/ablation/logs/%x_%j.log

# 用法：sbatch benchmarks/ablation/run.slurm <变体>...
# 变体配置由make_configs.py生成。每个变体的各task并行运行，只比较压缩比，因此并行不影响结果，但CSV中的耗时不能使用
# 使用已安装的二进制，不重新构建；环境变量UNSZ_INSTALL_DIR指定安装目录（默认install），ABLATION_PARALLEL_TASKS设置并行task数，默认24
set -euo pipefail
cd /home/x-zjian1/unsz
export OMP_NUM_THREADS=1
parallel_tasks=${ABLATION_PARALLEL_TASKS:-24}
for variant in "$@"; do
    config=results/ablation/configs/$variant.json
    log_dir=results/ablation/$variant/logs
    mkdir -p "$log_dir"
    ./unsz_env/bin/python - "$variant" <<'EOF'
import hashlib,json,os,socket,subprocess,sys,time
from pathlib import Path
variant=sys.argv[1]
install_dir=Path(os.environ.get('UNSZ_INSTALL_DIR','install'))
binaries={name:hashlib.sha256((install_dir/'bin'/name).read_bytes()).hexdigest() for name in ('s1_compress','s1_decompress')}
manifest={'variant':variant,'install_dir':str(install_dir),'git_revision':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),'binary_sha256':binaries,'node':socket.gethostname(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'start_time':time.strftime('%Y-%m-%d %H:%M:%S')}
Path(f'results/ablation/{variant}/manifest.json').write_text(json.dumps(manifest,indent=1)+'\n')
EOF
    ./unsz_env/bin/python -c "import json,sys;print('\n'.join(task['name'] for task in json.load(open(sys.argv[1]))))" "$config" | xargs -P "$parallel_tasks" -I{} sh -c "./unsz_env/bin/python t1_start_c_code.py $config --skip-build --task {} > $log_dir/{}.log 2>&1 && echo 'done {}' || echo 'FAILED {}'"
    echo "variant $variant finished"
done
