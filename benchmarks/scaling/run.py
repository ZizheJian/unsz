import argparse,csv,hashlib,json,math,os,subprocess,sys
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
thread_counts=(1,2,4,8,16,32)
rel_eb=1e-3
default_data_dir=Path('/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/tensor_data')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_metadata(data_dir):
    metadata_path=data_dir/'tensor_metadata.json'
    metadata=json.loads(metadata_path.read_text())
    fields=list(metadata['fields'])
    slice_num=sum(math.prod(metadata['fields'][field]['shape']) for field in fields)
    shape_counts={shape:sum(tuple(config['shape'])==shape for config in metadata['fields'].values()) for shape in ((12,1),(12,80),(12,81))}
    if len(fields)!=95 or slice_num!=36264 or shape_counts!={(12,1):58,(12,80):33,(12,81):4}:
        raise RuntimeError(f'完整Ocean-large metadata不匹配：fields={len(fields)},slices={slice_num},shapes={shape_counts}')
    metadata_sha256=sha256(metadata_path)
    return metadata_path,metadata,fields,slice_num,metadata_sha256

def validate_slurm(threads,prepare_only):
    required_cpus=max(8,threads)
    if prepare_only:
        return None
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('正式scaling必须在Slurm作业中运行')
    allocated_cpus=int(os.environ.get('SLURM_CPUS_PER_TASK','0'))
    if allocated_cpus!=required_cpus:
        raise RuntimeError(f'p={threads}必须申请{required_cpus}个Slurm CPU，实际为{allocated_cpus}')
    if len(os.sched_getaffinity(0))<allocated_cpus:
        raise RuntimeError('进程CPU affinity小于Slurm CPU分配')
    return allocated_cpus

def validate_point_set():
    path=project_root/'point_set_configs/Ocean_large_monthly_rbf_rel_1e-3.json'
    expected={'version':3,'labels':[{'method':'predict','eb_mode':'REL','eb_value':rel_eb,'condition':{'operator':'!=','value':-99999}},{'method':'constant'}]}
    if json.loads(path.read_text())!=expected:
        raise RuntimeError('Ocean-large REL 1e-3点集配置与正式规则不一致')
    return path

def binary_identity(codec):
    if codec=='unsz':
        binaries=[project_root/'install/bin/s1_compress',project_root/'install/bin/s1_decompress']
        actual={path.name:sha256(path) for path in binaries}
        return binaries,actual
    binary=project_root/f'build/tensor_1d/benchmark_{codec}'
    actual=sha256(binary)
    return [binary],{binary.name:actual}

def write_manifest(output_dir,codec,threads,allocated_cpus,metadata_path,metadata_sha256,point_set_path,slice_num,input_bytes,binary_sha256):
    sources=(Path(__file__).resolve(),Path(__file__).with_name('run.slurm'),project_root/'t1_start_c_code.py',project_root/'py_codes/parse_unsz_output.py',project_root/'benchmarks/tensor_1d/run.py')
    manifest={'codec':codec,'rel_eb':rel_eb,'thread_num':threads,'partition_num':threads if codec=='unsz' else None,'slurm_cpus':allocated_cpus,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_nodelist':os.environ.get('SLURM_NODELIST'),'hostname':os.uname().nodename,'field_num':95,'slice_num':slice_num,'input_bytes':input_bytes,'metadata_path':str(metadata_path),'metadata_sha256':metadata_sha256,'point_set_config_sha256':sha256(point_set_path),'binary_sha256':binary_sha256,'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip(),'source_sha256':{str(path.relative_to(project_root)):sha256(path) for path in sources}}
    path=output_dir/'scaling_manifest.json'
    if path.exists():
        previous=json.loads(path.read_text())
        for key in ('codec','rel_eb','thread_num','partition_num','field_num','slice_num','input_bytes','metadata_sha256','point_set_config_sha256','binary_sha256','source_sha256'):
            if previous.get(key)!=manifest.get(key):
                raise RuntimeError(f'已有scaling结果的{key}不同，禁止混合')
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return path

def run_unsz(arguments,data_dir,fields,point_set_path,output_dir):
    run_id=os.environ.get('SLURM_JOB_ID',f'local_{os.getpid()}')
    stream_dir=data_dir.parent/'scaling_outputs'/f'p{arguments.threads}_job{run_id}'
    stream_dir.mkdir(parents=True,exist_ok=True)
    config=[{'data_dir':str(data_dir),'output_dir':str(stream_dir),'csv_path':str(output_dir/'{field}.csv'),'compressor_list':['unsz'],'coordinate_dim':3,'name':f'ocean_large_scaling_p{arguments.threads}','field_list':fields,'tensor_metadata':'tensor_metadata.json','cleanup_after_each_test':True,'unsz':{'partition_num':arguments.threads,'thread_num':arguments.threads,'coordinate_transform':'auto','point_set_config_path_list':[str(point_set_path.relative_to(project_root))]}}]
    config_path=output_dir/'config.json'
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    if not arguments.prepare_only:
        subprocess.run([sys.executable,str(project_root/'t1_start_c_code.py'),str(config_path),'--skip-build'],cwd=project_root,check=True)
    return config_path

def run_baseline(arguments,data_dir):
    command=[sys.executable,str(project_root/'benchmarks/tensor_1d/run.py'),'--codec',arguments.codec,'--data-dir',str(data_dir),'--rel-eb',str(rel_eb),'--threads',str(arguments.threads),'--output-root',arguments.output_root]
    if not arguments.prepare_only:
        subprocess.run(command,cwd=project_root,check=True)
    return command

def summarize_results(output_dir,codec,threads,fields,slice_num,expected_input_bytes):
    if codec=='unsz':
        rows=[]
        for field in fields:
            path=output_dir/f'{field}.csv'
            if not path.is_file():
                raise RuntimeError(f'UNSZ scaling缺少field结果：{field}')
            with path.open(newline='',encoding='utf-8') as source:
                field_rows=list(csv.DictReader(source))
            if len(field_rows)!=1:
                raise RuntimeError(f'UNSZ scaling的field结果数量不是1：{field}={len(field_rows)}')
            rows.extend(field_rows)
        completed_slice_num=sum(int(row['time_step_num']) for row in rows)
        valid=all(float(row['max_error_over_eb'])<=1.01 for row in rows)
    else:
        path=output_dir/'summary.csv'
        if not path.is_file():
            raise RuntimeError(f'{codec} scaling缺少summary.csv')
        with path.open(newline='',encoding='utf-8') as source:
            rows=list(csv.DictReader(source))
        completed_slice_num=sum(int(row['slice_num']) for row in rows)
        valid=all(row['valid'].lower()=='true' and float(row['max_error_over_eb'])<=1.01 for row in rows)
    input_bytes=sum(int(row['input_bytes']) for row in rows)
    if len(rows)!=len(fields) or completed_slice_num!=slice_num or input_bytes!=expected_input_bytes or not valid:
        raise RuntimeError(f'{codec} scaling结果不完整或误差超界：fields={len(rows)},slices={completed_slice_num},input_bytes={input_bytes},valid={valid}')
    summary={'codec':codec,'rel_eb':rel_eb,'thread_num':threads,'partition_num':threads if codec=='unsz' else None,'field_num':len(rows),'slice_num':completed_slice_num,'input_bytes':input_bytes,'br':sum(float(row['br'])*int(row['input_bytes']) for row in rows)/input_bytes,'compression_preprocessing_time':sum(float(row['compression_preprocessing_time']) for row in rows),'compression_time':sum(float(row['compression_time']) for row in rows),'decompression_preprocessing_time':sum(float(row['decompression_preprocessing_time']) for row in rows),'decompression_time':sum(float(row['decompression_time']) for row in rows),'max_error_over_eb':max(float(row['max_error_over_eb']) for row in rows)}
    summary['compression_total_time']=summary['compression_preprocessing_time']+summary['compression_time']
    summary['decompression_total_time']=summary['decompression_preprocessing_time']+summary['decompression_time']
    (output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary

def main():
    parser=argparse.ArgumentParser(description='Run one complete Ocean-large scaling point')
    parser.add_argument('--codec',required=True,choices=('unsz','sz3','pfpl'))
    parser.add_argument('--threads',required=True,type=int,choices=thread_counts)
    parser.add_argument('--data-dir',type=Path,default=default_data_dir)
    parser.add_argument('--output-root',default='results/ocean_large_scaling')
    parser.add_argument('--prepare-only',action='store_true')
    arguments=parser.parse_args()
    data_dir=arguments.data_dir.resolve()
    metadata_path,_,fields,slice_num,metadata_sha256=validate_metadata(data_dir)
    coordinate_bytes=(data_dir/'coords.dat').stat().st_size
    if coordinate_bytes%12:
        raise RuntimeError('coords.dat不是三维float32坐标数组')
    expected_input_bytes=slice_num*(coordinate_bytes//12)*4
    point_set_path=validate_point_set()
    allocated_cpus=validate_slurm(arguments.threads,arguments.prepare_only)
    _,binary_sha256=binary_identity(arguments.codec)
    if arguments.codec=='unsz':
        output_dir=(project_root/arguments.output_root/'unsz'/f'rel_0.001_p{arguments.threads}').resolve()
    else:
        output_dir=(project_root/arguments.output_root/arguments.codec/f'rel_0.001_t{arguments.threads}').resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    manifest_path=write_manifest(output_dir,arguments.codec,arguments.threads,allocated_cpus,metadata_path,metadata_sha256,point_set_path,slice_num,expected_input_bytes,binary_sha256)
    os.environ.update({'OMP_NUM_THREADS':str(arguments.threads),'OMP_DYNAMIC':'false','OMP_PROC_BIND':'close','OMP_PLACES':'cores','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    artifact=run_unsz(arguments,data_dir,fields,point_set_path,output_dir) if arguments.codec=='unsz' else run_baseline(arguments,data_dir)
    summary=None if arguments.prepare_only else summarize_results(output_dir,arguments.codec,arguments.threads,fields,slice_num,expected_input_bytes)
    print(json.dumps({'codec':arguments.codec,'threads':arguments.threads,'slurm_cpus':allocated_cpus,'field_num':len(fields),'slice_num':slice_num,'manifest':str(manifest_path),'configuration':str(artifact),'summary':summary},ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
