import argparse,concurrent.futures,csv,hashlib,json,math,os,re,subprocess,threading,time
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
default_data_dir=Path('/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/tensor_data')

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def read_metadata(data_dir,selected_fields):
    metadata_path=data_dir/'tensor_metadata.json'
    metadata=json.loads(metadata_path.read_text())
    all_fields=list(metadata.get('fields',{}))
    shape_counts={}
    for field in all_fields:
        shape=metadata['fields'][field].get('shape')
        if not isinstance(shape,list)or len(shape)!=2 or any(not isinstance(value,int)or value<=0 for value in shape):raise RuntimeError(f'无效的tensor shape：{field}')
        shape_counts[tuple(shape)]=shape_counts.get(tuple(shape),0)+1
    if len(all_fields)!=95 or shape_counts!={(12,1):58,(12,80):33,(12,81):4}:raise RuntimeError(f'完整Ocean-large metadata不匹配：fields={len(all_fields)},shapes={shape_counts}')
    fields=selected_fields or all_fields
    if any(field not in metadata['fields']for field in fields):raise RuntimeError('指定field不在tensor metadata中')
    return metadata_path,metadata,fields

def make_assignments(fields,worker_num,metadata):
    field_loads={field:math.prod(metadata['fields'][field]['shape'])for field in fields}
    assignments=[[]for _ in range(worker_num)]
    loads=[0]*worker_num
    for field in sorted(fields,key=lambda name:(-field_loads[name],name)):
        worker_id=min(range(worker_num),key=lambda index:(loads[index],index))
        assignments[worker_id].append(field)
        loads[worker_id]+=field_loads[field]
    average_lower_bound=(sum(field_loads.values())+worker_num-1)//worker_num
    threshold_lower_bound=max(((sum(load>=threshold for load in field_loads.values())+worker_num-1)//worker_num)*threshold for threshold in set(field_loads.values()))
    lower_bound=max(average_lower_bound,threshold_lower_bound)
    if max(loads)!=lower_bound:raise RuntimeError(f'静态分配没有达到可证明的最小最大负载：当前={max(loads)},下界={lower_bound}')
    return assignments,loads,lower_bound

def cleanup(data_dir,field,slice_num):
    for slice_id in range(slice_num):
        source=data_dir/f'{field}.dat.{slice_id}'
        Path(str(source)+'.umc').unlink(missing_ok=True)
        Path(str(source)+'.umc.out').unlink(missing_ok=True)

def read_one_row(path):
    with Path(path).open(newline='')as source:rows=list(csv.DictReader(source))
    if len(rows)!=1:raise RuntimeError(f'结果行数不是1：{path}')
    return rows[0]

def same_float(left,right):
    if left is None:left=math.inf
    if right is None:right=math.inf
    left=float(left);right=float(right)
    return left==right if math.isinf(left)or math.isinf(right)else math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12)

def memory_usage_path():
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        hierarchy,controllers,relative=line.split(':',2)
        if 'memory'in controllers.split(','):
            path=Path('/sys/fs/cgroup/memory')/relative.lstrip('/')/'memory.usage_in_bytes'
            if path.is_file():return path
    return None

def monitor_memory(stop_event,result):
    path=memory_usage_path()
    result['path']=str(path)if path else None
    result['peak_bytes']=0
    if path is None:return
    while True:
        try:result['peak_bytes']=max(result['peak_bytes'],int(path.read_text()))
        except OSError:pass
        if stop_event.wait(0.25):break

def load_completed(path,field,rel_eb,reference):
    if not path.is_file():return None
    result=json.loads(path.read_text())
    if result.get('field')!=field or result.get('rel_eb')!=rel_eb or not result.get('valid'):return None
    if not same_float(result['br'],reference['br'])or not same_float(result['psnr'],reference['psnr']):return None
    return result

def run_field(worker_id,field,cpu,data_dir,metadata,rel_eb,output_dir,config_dir,log_dir,resource_dir,reference):
    result_path=output_dir/f'{field}.json'
    completed=load_completed(result_path,field,rel_eb,reference)
    if completed is not None:
        print(f'SKIP worker={worker_id},field={field}',flush=True)
        return completed
    slice_num=math.prod(metadata['fields'][field]['shape'])
    csv_path=output_dir/f'{field}.csv'
    config_path=config_dir/f'worker_{worker_id:02d}_{field}.json'
    log_path=log_dir/f'{field}.log'
    resource_path=resource_dir/f'{field}.txt'
    cleanup(data_dir,field,slice_num)
    config=[{'name':'ocean_large_umc_field_parallel','data_dir':str(data_dir),'csv_path':str(csv_path),'compressor_list':['umc'],'field_list':[field],'time_step_num':slice_num,'coordinate_dim':3,'rel_eb_list':[rel_eb],'cleanup_after_each_test':True,'umc':{'from_edge':1,'reorder':1,'method':'NBP-IDW-INP','neighbor_num':4}}]
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    environment={**os.environ,'OMP_NUM_THREADS':'1','OMP_DYNAMIC':'false','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    command=['/usr/bin/time','-v','-o',str(resource_path),'taskset','--cpu-list',str(cpu),str(project_root/'unsz_env/bin/python'),'-u',str(project_root/'t1_start_c_code.py'),str(config_path),'--skip-build']
    print(f'BEGIN worker={worker_id},field={field},slices={slice_num},cpu={cpu}',flush=True)
    start=time.perf_counter()
    try:
        with log_path.open('w')as log:returncode=subprocess.run(command,cwd=project_root,stdout=log,stderr=subprocess.STDOUT,env=environment).returncode
        process_wall_time=time.perf_counter()-start
        if returncode:raise RuntimeError(f'UMC field失败：{field},returncode={returncode},见{log_path}')
        row=read_one_row(csv_path)
        if row['compressor']!='umc' or float(row['rel_eb'])!=rel_eb or int(row['time_step_num'])!=slice_num:raise RuntimeError(f'UMC结果参数不匹配：{field}')
        if int(row['input_bytes'])!=int(reference['input_bytes'])or not same_float(row['br'],reference['br'])or not same_float(row['psnr'],reference['psnr']):raise RuntimeError(f'UMC并行结果与串行参考结果不一致：{field}')
        resource_text=resource_path.read_text()
        match=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',resource_text)
        if match is None:raise RuntimeError(f'UMC缺少峰值内存记录：{field}')
        result={'field':field,'worker_id':worker_id,'slice_num':slice_num,'input_bytes':int(row['input_bytes']),'rel_eb':rel_eb,'br':float(row['br']),'psnr':None if math.isinf(float(row['psnr']))else float(row['psnr']),'compression_preprocessing_time':float(row['compression_preprocessing_time']),'compression_time':float(row['compression_time']),'decompression_preprocessing_time':float(row['decompression_preprocessing_time']),'decompression_time':float(row['decompression_time']),'field_process_wall_time':process_wall_time,'max_rss_kib':int(match.group(1)),'matches_serial_reference':True,'valid':True}
        write_json(result_path,result)
        print(f'END worker={worker_id},field={field},wall_time={process_wall_time:.3f},max_rss_kib={result["max_rss_kib"]}',flush=True)
        return result
    finally:
        cleanup(data_dir,field,slice_num)
        config_path.unlink(missing_ok=True)

def run_worker(worker_id,fields,cpu,data_dir,metadata,rel_eb,output_dir,config_dir,log_dir,resource_dir,references):
    return worker_id,[run_field(worker_id,field,cpu,data_dir,metadata,rel_eb,output_dir,config_dir,log_dir,resource_dir,references[field])for field in fields]

def main():
    parser=argparse.ArgumentParser(description='Run UMC on independent fields with static field-level parallelism')
    parser.add_argument('--data-dir',type=Path,default=default_data_dir)
    parser.add_argument('--rel-eb',type=float,default=1e-3)
    parser.add_argument('--field-workers',type=int,default=32)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--field',action='append')
    parser.add_argument('--prepare-only',action='store_true')
    arguments=parser.parse_args()
    if arguments.rel_eb!=1e-3:parser.error('当前内存验证只运行REL=1e-3')
    if arguments.field_workers<=0 or arguments.field_workers&(arguments.field_workers-1):parser.error('field-workers必须是2的幂')
    arguments.data_dir=arguments.data_dir.resolve()
    metadata_path,metadata,fields=read_metadata(arguments.data_dir,arguments.field)
    formal_complete_dataset=arguments.field is None
    if formal_complete_dataset and arguments.field_workers not in(32,64,128):parser.error('完整正式实验只允许32、64或128个requested field workers')
    worker_num=arguments.field_workers
    assignments,loads,lower_bound=make_assignments(fields,worker_num,metadata)
    active_worker_num=sum(bool(worker_fields)for worker_fields in assignments)
    expected_maximum_load={32:1920,64:972,128:972}.get(worker_num)
    if formal_complete_dataset and(max(loads)!=expected_maximum_load or lower_bound!=expected_maximum_load):raise RuntimeError(f'完整实验静态分配应达到{expected_maximum_load} slices，实际loads={loads},下界={lower_bound}')
    umc_root=project_root/'third_party/umc'
    for name in('compress.hpp','decompress.hpp'):
        if(umc_root/'include'/name).read_bytes()!=(project_root/'third_party/umc_replacements/include'/name).read_bytes():raise RuntimeError(f'UMC源码不符合项目记录的替换文件：{name}')
    binaries={name:umc_root/'build/test'/name for name in('test_compress','test_decompress')}
    binary_sha256={name:sha256(path)for name,path in binaries.items()}
    serial_dir=project_root/'results/ocean_large_tensor_full_umc/serial_rel_1e-3'
    serial_manifest=json.loads((serial_dir/'manifest.json').read_text())
    if binary_sha256!=serial_manifest['binary_sha256']:raise RuntimeError('UMC binary与串行参考结果不同')
    references={field:read_one_row(serial_dir/f'{field}.csv')for field in fields}
    output_root=(arguments.output_root or project_root/f'results/ocean_large_umc_{worker_num}_field_workers').resolve()
    configuration_dir=(output_root/f'{worker_num}_workers_rel_0.001').resolve()
    output_dir=configuration_dir/'fields'
    config_dir=configuration_dir/'configs'
    log_dir=configuration_dir/'logs'
    resource_dir=configuration_dir/'resources'
    for path in(output_dir,config_dir,log_dir,resource_dir):path.mkdir(parents=True,exist_ok=True)
    slurm_name='run_umc_128_cpus.slurm' if worker_num==128 else f'run_umc_{worker_num}_field_workers.slurm'
    source_paths=[Path(__file__).resolve(),project_root/'benchmarks/umc_field_parallel'/slurm_name,project_root/'t1_start_c_code.py',project_root/'py_codes/parse_umc_output.py',umc_root/'include/compress.hpp',umc_root/'include/decompress.hpp']
    manifest={'codec':'umc-nbp-idw-inp','field_workers':worker_num,'field_workers_with_tasks':active_worker_num,'program_threads_per_worker':1,'allocated_cpus':128 if formal_complete_dataset else None,'field_num':len(fields),'fields':fields,'assignments':assignments,'assignment_slice_loads':loads,'assignment_max_slice_load_lower_bound':lower_bound,'assignment_is_proven_optimal':max(loads)==lower_bound,'rel_eb':arguments.rel_eb,'batch_size':10,'metadata_sha256':sha256(metadata_path),'binary_sha256':binary_sha256,'source_sha256':{str(path.relative_to(project_root)):sha256(path)for path in source_paths},'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip(),'umc_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=umc_root,text=True).strip(),'serial_reference_directory':str(serial_dir),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_nodelist':os.environ.get('SLURM_NODELIST')}
    manifest_path=configuration_dir/'manifest.json'
    if manifest_path.is_file():
        previous=json.loads(manifest_path.read_text())
        for key in('codec','field_workers','field_workers_with_tasks','program_threads_per_worker','allocated_cpus','field_num','fields','assignments','assignment_slice_loads','assignment_max_slice_load_lower_bound','assignment_is_proven_optimal','rel_eb','batch_size','metadata_sha256','binary_sha256','source_sha256','git_revision','umc_revision','serial_reference_directory'):
            if previous.get(key)!=manifest.get(key):raise RuntimeError(f'已有manifest的{key}不同')
    write_json(manifest_path,manifest)
    print(json.dumps({'manifest':str(manifest_path),'worker_num':worker_num,'loads':loads,'proven_lower_bound':lower_bound,'prepare_only':arguments.prepare_only},ensure_ascii=False),flush=True)
    if arguments.prepare_only:return
    if formal_complete_dataset:
        if 'SLURM_JOB_ID'not in os.environ or int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=128:raise RuntimeError('完整UMC实验必须申请128 CPUs')
    available_cpus=sorted(os.sched_getaffinity(0))
    if len(available_cpus)<worker_num:raise RuntimeError(f'可用CPU不足：需要{worker_num},实际{len(available_cpus)}')
    cpu_stride=max(1,len(available_cpus)//worker_num)
    worker_cpus=[available_cpus[worker_id*cpu_stride]for worker_id in range(worker_num)]
    stop_event=threading.Event();memory_result={}
    monitor=threading.Thread(target=monitor_memory,args=(stop_event,memory_result),daemon=True)
    monitor.start()
    start=time.perf_counter();worker_results={}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num)as executor:
            futures=[executor.submit(run_worker,worker_id,worker_fields,worker_cpus[worker_id],arguments.data_dir,metadata,arguments.rel_eb,output_dir,config_dir,log_dir,resource_dir,references)for worker_id,worker_fields in enumerate(assignments)]
            for future in concurrent.futures.as_completed(futures):
                worker_id,results=future.result();worker_results[worker_id]=results
    finally:
        stop_event.set();monitor.join()
    execution_wall_time=time.perf_counter()-start
    compression_worker_times=[sum(result['compression_preprocessing_time']+result['compression_time']for result in worker_results[worker_id])for worker_id in range(worker_num)]
    decompression_worker_times=[sum(result['decompression_preprocessing_time']+result['decompression_time']for result in worker_results[worker_id])for worker_id in range(worker_num)]
    execution_worker_times=[sum(result['field_process_wall_time']for result in worker_results[worker_id])for worker_id in range(worker_num)]
    max_rss_upper_bound_kib=sum(max((result['max_rss_kib']for result in worker_results[worker_id]),default=0)for worker_id in range(worker_num))
    summary={'codec':'umc-nbp-idw-inp','field_workers':worker_num,'field_workers_with_tasks':active_worker_num,'program_threads_per_worker':1,'allocated_cpus':128 if formal_complete_dataset else len(available_cpus),'active_cpus':active_worker_num,'field_num':len(fields),'slice_num':sum(loads),'rel_eb':arguments.rel_eb,'reported_compression_time':max(compression_worker_times),'reported_decompression_time':max(decompression_worker_times),'reported_end_to_end_time':max(compression_worker_times)+max(decompression_worker_times),'compression_worker_times':compression_worker_times,'decompression_worker_times':decompression_worker_times,'execution_worker_times':execution_worker_times,'execution_wall_time':execution_wall_time,'worker_slice_loads':loads,'assignment_max_slice_load_lower_bound':lower_bound,'assignment_is_proven_optimal':max(loads)==lower_bound,'max_rss_upper_bound_kib':max_rss_upper_bound_kib,'cgroup_memory_path':memory_result.get('path'),'cgroup_memory_peak_bytes':memory_result.get('peak_bytes'),'matches_serial_reference':all(result['matches_serial_reference']for results in worker_results.values()for result in results),'valid':True}
    write_json(configuration_dir/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':main()
