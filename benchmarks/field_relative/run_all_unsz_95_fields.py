import argparse,json,math,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import project_root,rel_eb_tags

# per-field协议的UNSZ与per-slice协议使用同一个两级并行入口，只把点集配置从REL换成FIELD_REL，因此同样跨field共享空间DAG
def main():
    parser=argparse.ArgumentParser(description='Run UNSZ on all Ocean-large fields under per-field error bounds')
    parser.add_argument('--rel-eb',type=float,action='append',required=True,choices=tuple(rel_eb_tags))
    parser.add_argument('--field-workers',type=int,default=16)
    parser.add_argument('--partitions-per-field',type=int,default=8)
    parser.add_argument('--output-root',type=Path,default=project_root/'results/ocean_large_field_relative_five_ebs')
    arguments=parser.parse_args()
    if len(set(arguments.rel_eb))!=len(arguments.rel_eb):parser.error('rel-eb不能重复')
    cpu_num=arguments.field_workers*arguments.partitions_per_field
    configuration_dir=arguments.output_root/'unsz'/f'{arguments.field_workers}_fields_{arguments.partitions_per_field}_partitions'
    for rel_eb in arguments.rel_eb:
        output_dir=configuration_dir/f'rel_{rel_eb:g}'
        summary_path=output_dir/'summary.json'
        if summary_path.is_file():
            summary=json.loads(summary_path.read_text())
            if summary.get('protocol')=='field'and summary.get('rel_eb')==rel_eb and summary.get('field_num')==95 and summary.get('cpu_num')==cpu_num and summary.get('field_worker_num')==arguments.field_workers and summary.get('valid'):
                print(f'SKIP completed REL={rel_eb:g}',flush=True)
                continue
        subprocess.run([sys.executable,'-u',str(project_root/'benchmarks/two_level/run.py'),'--protocol','field','--rel-eb',str(rel_eb),'--field-worker-num',str(arguments.field_workers),'--cpu-num',str(cpu_num),'--output-root',str(output_dir),'--launcher',str(Path(__file__).with_suffix('.slurm'))],cwd=project_root,check=True)

if __name__=='__main__':main()
