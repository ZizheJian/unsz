import argparse,hashlib,json,os,re,subprocess,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project_root))
import t1_start_c_code as benchmark
from py_codes.parse_unsz_output import parse_unsz_output

parser=argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--field',required=True)
parser.add_argument('--output-dir',required=True)
parser.add_argument('--dry-run',action='store_true')
arguments=parser.parse_args()
output_dir=Path(arguments.output_dir).resolve()
output_dir.mkdir(parents=True,exist_ok=True)
tasks=benchmark.select_tasks(benchmark.load_json_tasks(arguments.config),None,[arguments.field],['unsz'])
if len(tasks)!=1:
    raise RuntimeError('内存测试应匹配一个task')
task=tasks[0]
config=task|task.get('unsz',{})|{'field':arguments.field}
config=benchmark.expand_unsz_pipelines(config)[0]
label_paths=config.get('label_config_path_list',[])
if len(label_paths)!=1:
    raise RuntimeError('内存测试应匹配一个误差界配置')
config['label_config_path']=label_paths[0]
commands=benchmark.build_unsz_commands(config,Path(config['data_dir']))
if arguments.dry_run:
    print(json.dumps([[str(value) for value in command] for command in commands],indent=2))
    sys.exit(0)
logs=[]
memory={}
for stage,command in zip(('compression','decompression'),commands):
    command=[str(value) for value in command]
    command[0]=str(output_dir/Path(command[0]).name)
    log_path=output_dir/f'{stage}.log'
    memory_path=output_dir/f'{stage}.memory.txt'
    print(f'BEGIN {stage}: {command[0]}',flush=True)
    with log_path.open('w') as log_file:
        result=subprocess.run(['/usr/bin/time','-v','-o',str(memory_path),*command],stdout=log_file,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1'})
    print(memory_path.read_text(),flush=True)
    result.check_returncode()
    for line in memory_path.read_text().splitlines():
        if 'Maximum resident set size (kbytes):' in line:
            memory[stage]=int(line.rsplit(':',1)[1])
    logs.append(log_path.read_text())
metrics=parse_unsz_output(*logs)
maximum_error=re.search(r'maximum error over eb= ([^\s,]+)',logs[1])
if maximum_error is None or not 0<=float(maximum_error[1])<=1.01:
    raise RuntimeError('内存测试缺少有效的逐点误差验证结果')
summary={'field':arguments.field,'max_rss_kib':memory,'max_error_over_eb':float(maximum_error[1]),'metrics':metrics,'binary_sha256':{Path(command[0]).name:hashlib.sha256((output_dir/Path(command[0]).name).read_bytes()).hexdigest() for command in commands}}
(output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
benchmark.cleanup_test_outputs(config,'unsz')
print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
