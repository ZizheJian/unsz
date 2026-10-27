import argparse,hashlib,heapq,json,math,os,re,shlex,shutil,subprocess
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
benchmark_dir=Path(__file__).resolve().parent
default_data_dir=Path('/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/tensor_data')
rel_eb_tags={1e-1:'1e-1',1e-2:'1e-2',1e-3:'1e-3',1e-4:'1e-4',1e-5:'1e-5'}
configurations={8:{'thread_num':16,'maximum_load':4812},16:{'thread_num':8,'maximum_load':2880},32:{'thread_num':4,'maximum_load':1920},64:{'thread_num':2,'maximum_load':972}}

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def assign_by_slice_count(metadata_fields,worker_num):
    workers=[{'fields':[],'slice_num':0} for _ in range(worker_num)]
    heap=[(0,worker_id) for worker_id in range(worker_num)]
    heapq.heapify(heap)
    for field in sorted(metadata_fields,key=lambda name:(-math.prod(metadata_fields[name]['shape']),name)):
        load,worker_id=heapq.heappop(heap)
        field_load=math.prod(metadata_fields[field]['shape'])
        workers[worker_id]['fields'].append(field)
        workers[worker_id]['slice_num']+=field_load
        heapq.heappush(heap,(load+field_load,worker_id))
    return workers

def load_and_validate(data_dir,field_worker_num,rel_eb):
    metadata_path=data_dir/'tensor_metadata.json'
    metadata=json.loads(metadata_path.read_text())
    metadata_fields=metadata['fields']
    slice_num=sum(math.prod(config['shape']) for config in metadata_fields.values())
    shape_counts={shape:sum(tuple(config['shape'])==shape for config in metadata_fields.values()) for shape in ((12,1),(12,80),(12,81))}
    if len(metadata_fields)!=95 or slice_num!=36264 or shape_counts!={(12,1):58,(12,80):33,(12,81):4}:
        raise RuntimeError(f'完整Ocean-large metadata不匹配：fields={len(metadata_fields)},slices={slice_num},shapes={shape_counts}')
    thread_num=configurations[field_worker_num]['thread_num']
    schedule_path=benchmark_dir/'schedule.json' if field_worker_num==16 else None
    if schedule_path is not None:
        schedule=json.loads(schedule_path.read_text())
        workers=schedule['workers']
        if schedule['worker_num']!=16 or schedule['thread_num']!=8 or schedule['partition_num']!=8 or len(workers)!=16:
            raise RuntimeError('16×8固定分配文件的配置不匹配')
    else:
        workers=assign_by_slice_count(metadata_fields,field_worker_num)
    fields=[]
    worker_ids=[]
    actual_loads=[]
    for worker_id,worker in enumerate(workers):
        worker_fields=worker['fields']
        actual_load=sum(math.prod(metadata_fields[field]['shape']) for field in worker_fields)
        if actual_load!=worker['slice_num']:
            raise RuntimeError(f'worker {worker_id}的slice_num不匹配：{worker["slice_num"]}!={actual_load}')
        fields.extend(worker_fields)
        worker_ids.extend([worker_id]*len(worker_fields))
        actual_loads.append(actual_load)
    if len(fields)!=len(set(fields)) or set(fields)!=set(metadata_fields):
        raise RuntimeError('固定schedule没有恰好覆盖全部95个fields')
    expected_maximum_load=configurations[field_worker_num]['maximum_load']
    if max(actual_loads)!=expected_maximum_load:
        raise RuntimeError(f'{field_worker_num}×{thread_num}固定分配的最大slice load不是{expected_maximum_load}：{actual_loads}')
    point_set_path=project_root/f'point_set_configs/Ocean_large_monthly_rbf_rel_{rel_eb_tags[rel_eb]}.json'
    expected={'version':3,'labels':[{'method':'predict','eb_mode':'REL','eb_value':rel_eb,'condition':{'operator':'!=','value':-99999}},{'method':'constant'}]}
    if json.loads(point_set_path.read_text())!=expected:
        raise RuntimeError(f'Ocean-large REL {rel_eb:g}点集配置与正式规则不一致')
    coordinate_bytes=(data_dir/'coords.dat').stat().st_size
    if coordinate_bytes%12:
        raise RuntimeError('coords.dat不是三维float32坐标数组')
    point_num=coordinate_bytes//12
    return metadata_path,schedule_path,point_set_path,fields,worker_ids,slice_num,point_num,actual_loads,workers

def validate_slurm(prepare_only):
    if prepare_only:
        return None
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('正式two-level实验必须在Slurm作业中运行')
    allocated_cpus=int(os.environ.get('SLURM_CPUS_PER_TASK','0'))
    if allocated_cpus!=128:
        raise RuntimeError(f'two-level实验必须申请128个CPU，实际为{allocated_cpus}')
    if len(os.sched_getaffinity(0))<128:
        raise RuntimeError('进程CPU affinity没有覆盖128个CPU')
    return allocated_cpus

def build_command(binary,data_dir,stream_dir,metadata_path,point_set_path,fields,worker_ids,field_worker_num,thread_num,decompress):
    command=[str(binary),'-point-set-config',str(point_set_path),'-data-dir',str(data_dir),'-output-dir',str(stream_dir),'-field-list','{'+','.join(fields)+'}','-field-worker-num',str(field_worker_num),'-field-worker-map','{'+','.join(str(worker_id) for worker_id in worker_ids)+'}','-tensor-metadata',str(metadata_path),'-coordinate-dim','3','-coordinate-transform','auto','-partition-num',str(thread_num),'-thread-num',str(thread_num)]
    if decompress:
        command.append('-skip-decompressed-output')
    return command

def run_logged(command,log_path):
    with log_path.open('w',encoding='utf-8') as log:
        subprocess.run(command,cwd=project_root,stdout=log,stderr=subprocess.STDOUT,check=True)

def read_one_float(text,pattern,name):
    matches=re.findall(pattern,text,re.MULTILINE)
    if len(matches)!=1:
        raise RuntimeError(f'{name}应当出现一次，实际为{len(matches)}次')
    return float(matches[0])

def validate_log(text,fields,side):
    completed_fields=re.findall(r'END UNSZ FIELD ([^\r\n]+)',text)
    if len(completed_fields)!=len(fields) or set(completed_fields)!=set(fields):
        raise RuntimeError(f'{side}没有完整处理95个fields：{len(completed_fields)}')
    error_ratios=[float(value) for value in re.findall(r'全部文件最大误差/用户误差上限= ([^\s]+)',text)]
    if len(error_ratios)!=len(fields) or max(error_ratios)>1.01:
        raise RuntimeError(f'{side}误差验证失败：count={len(error_ratios)},max={max(error_ratios,default=math.inf)}')
    return max(error_ratios)

def transform_statistics(stream_dir,fields):
    enabled=0
    for field in fields:
        path=stream_dir/f'{field}.unsz.coord_transform.txt'
        match=re.search(r'^enabled ([01])$',path.read_text(),re.MULTILINE)
        if match is None:
            raise RuntimeError(f'坐标变换文件格式错误：{path}')
        enabled+=int(match.group(1))
    return enabled,len(fields)-enabled

def write_manifest(output_dir,allocated_cpus,metadata_path,schedule_path,point_set_path,binaries,slice_num,point_num,field_worker_num,thread_num,loads,workers,commands,rel_eb,launcher_path):
    slurm_path=launcher_path or benchmark_dir/('run.slurm' if field_worker_num==16 else f'run_{field_worker_num}x{thread_num}.slurm')
    sources=[Path(__file__).resolve(),slurm_path,project_root/'c_codes/compress_cli.cpp',project_root/'c_codes/s1_decompress.cpp',project_root/'c_codes/unsz_core/spatial_prediction_state.tpp',project_root/'c_codes/unsz_core/unsz.hpp']
    if schedule_path is not None:
        sources.append(schedule_path)
    manifest={'rel_eb':rel_eb,'field_worker_num':field_worker_num,'thread_num':thread_num,'partition_num':thread_num,'slurm_cpus':allocated_cpus,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_nodelist':os.environ.get('SLURM_NODELIST'),'hostname':os.uname().nodename,'field_num':95,'slice_num':slice_num,'point_num':point_num,'assignment_basis':'descending slice count,assigned to the least-loaded worker','worker_slice_loads':loads,'worker_fields':[worker['fields'] for worker in workers],'metadata_path':str(metadata_path),'metadata_sha256':sha256(metadata_path),'point_set_config_sha256':sha256(point_set_path),'binary_sha256':{path.name:sha256(path) for path in binaries},'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip(),'source_sha256':{str(path.relative_to(project_root)):sha256(path) for path in sources},'commands':{name:shlex.join(command) for name,command in commands.items()}}
    if schedule_path is not None:
        manifest['schedule_sha256']=sha256(schedule_path)
    path=output_dir/'manifest.json'
    path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return path

def main():
    parser=argparse.ArgumentParser(description='Run an Ocean-large UNSZ 128-CPU allocation experiment')
    parser.add_argument('--data-dir',type=Path,default=default_data_dir)
    parser.add_argument('--rel-eb',type=float,choices=tuple(rel_eb_tags),default=1e-3)
    parser.add_argument('--field-worker-num',type=int,choices=tuple(configurations),default=16)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--launcher',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--keep-output',action='store_true')
    arguments=parser.parse_args()
    data_dir=arguments.data_dir.resolve()
    rel_eb=arguments.rel_eb
    field_worker_num=arguments.field_worker_num
    thread_num=configurations[field_worker_num]['thread_num']
    metadata_path,schedule_path,point_set_path,fields,worker_ids,slice_num,point_num,loads,workers=load_and_validate(data_dir,field_worker_num,rel_eb)
    allocated_cpus=validate_slurm(arguments.prepare_only)
    binaries=[project_root/'install/bin/s1_compress',project_root/'install/bin/s1_decompress']
    if any(not path.is_file() for path in binaries):
        raise RuntimeError('缺少已安装的UNSZ二进制文件')
    run_id=os.environ.get('SLURM_JOB_ID',f'local_{os.getpid()}')
    stream_root=data_dir.parent/'two_level_outputs'
    stream_dir=stream_root/f'job_{run_id}'
    default_output_root=Path('results/ocean_large_two_level') if field_worker_num==16 else Path(f'results/ocean_large_unsz_128cpu/{field_worker_num}_field_workers_{thread_num}_partitions')
    output_dir=(project_root/(arguments.output_root or default_output_root)).resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    compression_log=output_dir/'compression.log'
    decompression_log=output_dir/'decompression.log'
    compression_command=build_command(binaries[0],data_dir,stream_dir,metadata_path,point_set_path,fields,worker_ids,field_worker_num,thread_num,False)
    decompression_command=build_command(binaries[1],data_dir,stream_dir,metadata_path,point_set_path,fields,worker_ids,field_worker_num,thread_num,True)
    launcher_path=arguments.launcher.resolve()if arguments.launcher else None
    manifest_path=write_manifest(output_dir,allocated_cpus,metadata_path,schedule_path,point_set_path,binaries,slice_num,point_num,field_worker_num,thread_num,loads,workers,{'compression':compression_command,'decompression':decompression_command},rel_eb,launcher_path)
    if arguments.prepare_only:
        print(json.dumps({'manifest':str(manifest_path),'worker_slice_loads':loads,'compression_command':shlex.join(compression_command),'decompression_command':shlex.join(decompression_command)},ensure_ascii=False),flush=True)
        return
    if stream_dir.exists():
        raise RuntimeError(f'临时输出目录已经存在：{stream_dir}')
    stream_dir.mkdir(parents=True)
    print(f'BEGIN two-level compression,output={stream_dir}',flush=True)
    run_logged(compression_command,compression_log)
    transformed_field_num,shared_field_num=transform_statistics(stream_dir,fields)
    compressed_bytes=sum(path.stat().st_size for path in stream_dir.glob('*.unsz'))
    print('BEGIN two-level decompression',flush=True)
    run_logged(decompression_command,decompression_log)
    compression_text=compression_log.read_text()
    decompression_text=decompression_log.read_text()
    compression_max_error=validate_log(compression_text,fields,'压缩端')
    decompression_max_error=validate_log(decompression_text,fields,'解压端')
    compression_time=read_one_float(compression_text,r'^two-level reported compression-side time= ([^ ]+) s$','two-level压缩端时间')
    decompression_time=read_one_float(decompression_text,r'^two-level reported decompression-side time= ([^ ]+) s$','two-level解压端时间')
    compression_wall_time=read_one_float(compression_text,r'^two-level compression wall time= ([^ ]+) s$','two-level压缩端wall time')
    decompression_wall_time=read_one_float(decompression_text,r'^two-level decompression wall time= ([^ ]+) s$','two-level解压端wall time')
    input_bytes=point_num*slice_num*4
    summary={'rel_eb':rel_eb,'field_worker_num':field_worker_num,'thread_num':thread_num,'partition_num':thread_num,'cpu_num':128,'field_num':len(fields),'slice_num':slice_num,'input_bytes':input_bytes,'compressed_bytes':compressed_bytes,'br':compressed_bytes*8/(point_num*slice_num),'transformed_field_num':transformed_field_num,'shared_dag_field_num':shared_field_num,'compression_time':compression_time,'decompression_time':decompression_time,'end_to_end_time':compression_time+decompression_time,'compression_wall_time':compression_wall_time,'decompression_wall_time':decompression_wall_time,'max_error_over_eb':max(compression_max_error,decompression_max_error),'shared_dag_compression_reuse_count':compression_text.count('Reusing shared spatial prediction DAG and weights'),'shared_dag_decompression_reuse_count':decompression_text.count('Reusing shared spatial prediction DAG and weights')}
    single_thread_path=project_root/f'results/ocean_large_scaling/unsz/rel_{rel_eb:g}_p1/summary.json'
    if single_thread_path.is_file():
        single_thread=json.loads(single_thread_path.read_text())
        single_thread_time=single_thread['compression_total_time']+single_thread['decompression_total_time']
        summary['single_thread_end_to_end_time']=single_thread_time
        summary['speedup_over_single_thread']=single_thread_time/summary['end_to_end_time']
    (output_dir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    if not arguments.keep_output:
        if stream_dir.parent!=stream_root:
            raise RuntimeError(f'拒绝清理意外目录：{stream_dir}')
        shutil.rmtree(stream_dir)
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
