import argparse,csv,hashlib,json,math,os,socket,subprocess,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root/'py_codes'))
from plot_current_benchmarks import default_configs,excluded_fields

# 在中等规模数据集上单线程运行SZ3-1D和PFPL：每个slice按原始点序作为一维数组，ABS EB为REL乘以该slice排除-99999后的range
# 码率按全部值计算，PSNR只统计有效值并按各slice的range归一化，与results/current中UNSZ和UMC的统计方式相同
codecs={'sz3':'sz3_1d','pfpl':'pfpl'}

def run_field(binary,task,field,rel_eb,work_dir):
    paths=[Path(task['data_dir'])/f'{field}.dat.{slice_id}' for slice_id in range(task['slice_num'])]
    if not all(path.is_file() for path in paths):raise FileNotFoundError(f'输入slice不完整：{task["name"]}:{field}')
    input_list=work_dir/f'{task["name"]}_{field}.inputs.txt'
    input_list.write_text(''.join(str(path)+'\n' for path in paths))
    slices_path=work_dir/f'{task["name"]}_{field}.slices.csv'
    with (work_dir/f'{task["name"]}_{field}.log').open('w') as log:
        returncode=subprocess.run([str(binary),str(input_list),str(rel_eb),'1',str(slices_path)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OMP_DYNAMIC':'false'}).returncode
    with slices_path.open(newline='') as source:rows=list(csv.DictReader(source))
    if returncode or len(rows)!=len(paths):raise RuntimeError(f'{task["name"]}:{field}失败，returncode={returncode}，slices={len(rows)}')
    normalized_squared_error=0.0
    for row in rows:
        slice_range=float(row['valid_max'])-float(row['valid_min']) if int(row['valid_point_num']) else 0.0
        if slice_range>0:normalized_squared_error+=float(row['valid_squared_error'])/slice_range**2
        elif float(row['valid_squared_error'])!=0:raise RuntimeError(f'有效值为常量的slice未被精确恢复：{field}')
    value_num=sum(int(row['point_num']) for row in rows)
    valid_value_num=sum(int(row['valid_point_num']) for row in rows)
    compressed_bytes=sum(int(row['compressed_bytes']) for row in rows)
    nmse=normalized_squared_error/valid_value_num
    return {'task':task['name'],'field':field,'rel_eb':rel_eb,'slice_num':len(rows),'value_num':value_num,'valid_value_num':valid_value_num,'compressed_bytes':compressed_bytes,'br':8*compressed_bytes/value_num,'psnr':math.inf if nmse==0 else -10*math.log10(nmse),'compression_time':sum(float(row['compression_time']) for row in rows),'decompression_time':sum(float(row['decompression_time']) for row in rows),'max_error_over_eb':max(float(row['max_error_over_eb']) for row in rows),'valid':all(row['valid']=='1' for row in rows)}

def main():
    parser=argparse.ArgumentParser(description='在中等规模数据集上单线程运行SZ3-1D或PFPL')
    parser.add_argument('--codec',required=True,choices=tuple(codecs))
    parser.add_argument('--rel-eb',type=float,action='append')
    parser.add_argument('--output-root',type=Path,default=project_root/'results/moderate_tensor_1d')
    arguments=parser.parse_args()
    rel_ebs=arguments.rel_eb or [1e-1,1e-2,1e-3,1e-4,1e-5]
    binary=project_root/'build/tensor_1d'/f'benchmark_{arguments.codec}'
    fields=[(config,task,field) for config in default_configs for task in json.loads((project_root/'configs'/(config+'.json')).read_text()) for field in task['field_list'] if (config,field) not in excluded_fields]
    for rel_eb in rel_ebs:
        output_dir=arguments.output_root/codecs[arguments.codec]/f'rel_{rel_eb:g}'
        work_dir=output_dir/'slices'
        work_dir.mkdir(parents=True,exist_ok=True)
        # 记录主机和作业号；论文中的计时来自run_moderate.slurm在计算节点上的运行
        (output_dir/'manifest.json').write_text(json.dumps({'codec':codecs[arguments.codec],'rel_eb':rel_eb,'threads':1,'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip(),'hostname':socket.gethostname(),'slurm_job_id':os.environ.get('SLURM_JOB_ID')},indent=2)+'\n')
        rows=[]
        for config,task,field in fields:
            rows.append(run_field(binary,task,field,rel_eb,work_dir))
            print(json.dumps(rows[-1],ensure_ascii=False),flush=True)
        with (output_dir/'fields.csv').open('w',newline='') as destination:
            writer=csv.DictWriter(destination,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        if not all(row['valid'] for row in rows):raise RuntimeError(f'REL={rel_eb:g}的误差检查失败')

if __name__=='__main__':main()
