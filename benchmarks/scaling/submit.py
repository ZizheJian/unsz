import argparse,shlex,subprocess
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
codecs=('unsz','sz3','pfpl')
resources={'unsz':{1:(8,'12:00:00'),2:(8,'10:00:00'),4:(8,'06:00:00'),8:(8,'06:00:00'),16:(16,'06:00:00'),32:(32,'06:00:00')},'sz3':{1:(8,'03:00:00'),2:(8,'03:00:00'),4:(8,'02:00:00'),8:(8,'02:00:00'),16:(16,'02:00:00'),32:(32,'02:00:00')},'pfpl':{1:(8,'02:00:00'),2:(8,'02:00:00'),4:(8,'02:00:00'),8:(8,'02:00:00'),16:(16,'02:00:00'),32:(32,'02:00:00')}}
default_threads=(2,4,8,16,32)
parser=argparse.ArgumentParser(description='Submit complete Ocean-large scaling jobs')
parser.add_argument('--codec',action='append',choices=codecs)
parser.add_argument('--threads',type=int,action='append',choices=tuple(resources['unsz']))
parser.add_argument('--dry-run',action='store_true')
arguments=parser.parse_args()
(project_root/'results/ocean_large_scaling').mkdir(parents=True,exist_ok=True)
previous_job_id=None
for codec in arguments.codec or codecs:
    for threads in arguments.threads or default_threads:
        suffix=f'p{threads}' if codec=='unsz' else f't{threads}'
        summary_path=project_root/f'results/ocean_large_scaling/{codec}/rel_0.001_{suffix}/summary.json'
        if summary_path.is_file():
            print(f'skip complete codec={codec},threads={threads}',flush=True)
            continue
        cpus,time_limit=resources[codec][threads]
        command=['sbatch','--parsable',f'--cpus-per-task={cpus}',f'--time={time_limit}',f'--job-name=ocean_{codec}_p{threads}']
        if previous_job_id:
            command.append(f'--dependency=afterany:{previous_job_id}')
        command.extend((str(project_root/'benchmarks/scaling/run.slurm'),codec,str(threads)))
        print(shlex.join(command),flush=True)
        if arguments.dry_run:
            previous_job_id='PREVIOUS_JOB'
        else:
            previous_job_id=subprocess.check_output(command,cwd=project_root,text=True).strip()
            print(f'submitted codec={codec},threads={threads},job={previous_job_id}',flush=True)
