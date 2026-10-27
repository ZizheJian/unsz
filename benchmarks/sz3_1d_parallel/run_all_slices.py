import argparse,csv,hashlib,json,math,os,shutil,subprocess,time
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
default_data_dir=Path('/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/tensor_data')

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def json_safe(value):
    if isinstance(value,float)and not math.isfinite(value):return None
    if isinstance(value,dict):return {key:json_safe(item)for key,item in value.items()}
    if isinstance(value,list):return [json_safe(item)for item in value]
    return value

def write_json(path,value):
    Path(path).write_text(json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def read_metadata(data_dir,selected_fields,max_slices):
    metadata_path=data_dir/'tensor_metadata.json'
    metadata=json.loads(metadata_path.read_text())
    all_fields=list(metadata.get('fields',{}))
    if selected_fields:
        if any(field not in metadata['fields']for field in selected_fields):raise RuntimeError('指定field不在tensor metadata中')
        fields=selected_fields
    else:
        fields=all_fields
        shape_counts={}
        for field in fields:
            shape=metadata['fields'][field].get('shape')
            if not isinstance(shape,list)or len(shape)!=2 or any(not isinstance(value,int)or value<=0 for value in shape):raise RuntimeError(f'无效的tensor shape：{field}')
            shape_counts[tuple(shape)]=shape_counts.get(tuple(shape),0)+1
        if len(fields)!=95 or shape_counts!={(12,1):58,(12,80):33,(12,81):4}:raise RuntimeError(f'完整Ocean-large metadata不匹配：fields={len(fields)},shapes={shape_counts}')
    slices=[]
    for field in fields:
        slice_num=math.prod(metadata['fields'][field]['shape'])
        if max_slices is not None:slice_num=min(slice_num,max_slices)
        if slice_num<=0:raise RuntimeError(f'field没有slice：{field}')
        first=data_dir/f'{field}.dat.0'
        last=data_dir/f'{field}.dat.{slice_num-1}'
        if not first.is_file()or not last.is_file():raise FileNotFoundError(f'输入切片不完整：{field}')
        input_bytes=first.stat().st_size
        if input_bytes<=0 or input_bytes%4 or last.stat().st_size!=input_bytes:raise RuntimeError(f'输入切片大小无效：{field}')
        for slice_id in range(slice_num):slices.append({'field':field,'slice_id':slice_id,'path':data_dir/f'{field}.dat.{slice_id}','input_bytes':input_bytes})
    return metadata_path,fields,slices

def load_reference(rel_eb,threads_per_slice,fields,binary_sha256):
    candidates=[project_root/'results/ocean_large_scaling/sz3'/f'rel_{rel_eb:g}_t{threads_per_slice}',project_root/'results/ocean_large_tensor_1d/sz3'/f'rel_{rel_eb:g}_t{threads_per_slice}']
    for reference_dir in candidates:
        summary_path=reference_dir/'summary.csv'
        manifest_path=reference_dir/'manifest.json'
        if not summary_path.is_file()or not manifest_path.is_file():continue
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('binary_sha256')!=binary_sha256:continue
        with summary_path.open(newline='')as source:rows={row['field']:row for row in csv.DictReader(source)}
        if any(field not in rows for field in fields):continue
        return {'directory':str(reference_dir),'fields':{field:{'compressed_bytes':int(rows[field]['compressed_bytes']),'input_bytes':int(rows[field]['input_bytes'])}for field in fields}}
    return None

def run_workers(assignments,threads_per_slice,binary,rel_eb,worker_dir,available_cpus):
    environment={**os.environ,'OMP_NUM_THREADS':str(threads_per_slice),'OMP_THREAD_LIMIT':str(threads_per_slice),'OMP_DYNAMIC':'false','OMP_PROC_BIND':'close','OMP_PLACES':'cores','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    processes=[]
    start=time.perf_counter()
    for worker_id,assigned in enumerate(assignments):
        input_list=worker_dir/f'worker_{worker_id:03d}.inputs.txt'
        slices_csv=worker_dir/f'worker_{worker_id:03d}.slices.csv'
        log_path=worker_dir/f'worker_{worker_id:03d}.log'
        input_list.write_text(''.join(str(item['path'])+'\n'for item in assigned))
        cpus=available_cpus[worker_id*threads_per_slice:(worker_id+1)*threads_per_slice]
        cpu_set=','.join(str(cpu)for cpu in cpus)
        log=log_path.open('w')
        command=['taskset','--cpu-list',cpu_set,str(binary),str(input_list),str(rel_eb),str(threads_per_slice),str(slices_csv)]
        process=subprocess.Popen(command,cwd=project_root,stdout=log,stderr=subprocess.STDOUT,env=environment)
        processes.append((worker_id,process,log,log_path,slices_csv))
    failures=[]
    for worker_id,process,log,log_path,_ in processes:
        returncode=process.wait()
        log.close()
        if returncode:failures.append((worker_id,returncode,str(log_path)))
    execution_wall_time=time.perf_counter()-start
    if failures:raise RuntimeError(f'SZ3-1D worker失败：{failures}')
    return [item[4]for item in processes],execution_wall_time

def summarize(assignments,csv_paths,fields,reference,threads_per_slice,execution_wall_time):
    field_rows={field:[]for field in fields}
    compression_worker_times=[]
    decompression_worker_times=[]
    worker_slice_counts=[]
    for worker_id,(assigned,csv_path)in enumerate(zip(assignments,csv_paths)):
        if not csv_path.is_file():raise FileNotFoundError(f'worker结果不存在：{csv_path}')
        with csv_path.open(newline='')as source:rows=list(csv.DictReader(source))
        if len(rows)!=len(assigned):raise RuntimeError(f'worker {worker_id}结果数量错误：{len(rows)}!={len(assigned)}')
        compression_worker_times.append(sum(float(row['compression_time'])for row in rows))
        decompression_worker_times.append(sum(float(row['decompression_time'])for row in rows))
        worker_slice_counts.append(len(rows))
        for local_slice_id,(item,row)in enumerate(zip(assigned,rows)):
            if int(row['slice_id'])!=local_slice_id:raise RuntimeError(f'worker {worker_id}的slice顺序错误')
            if int(row['point_num'])*4!=item['input_bytes']:raise RuntimeError(f'输入字节数不匹配：{item["path"]}')
            field_rows[item['field']].append(row)
    field_summaries=[]
    for field in fields:
        rows=field_rows[field]
        if not rows:raise RuntimeError(f'field没有结果：{field}')
        value_num=sum(int(row['point_num'])for row in rows)
        valid_rows=[row for row in rows if int(row['valid_point_num'])]
        data_range=max((float(row['valid_max'])for row in valid_rows),default=0)-min((float(row['valid_min'])for row in valid_rows),default=0)
        squared_error=sum(float(row['squared_error'])for row in rows)
        compressed_bytes=sum(int(row['compressed_bytes'])for row in rows)
        expected=reference['fields'][field]if reference else None
        summary={'field':field,'slice_num':len(rows),'value_num':value_num,'input_bytes':value_num*4,'compressed_bytes':compressed_bytes,'br':8*compressed_bytes/value_num,'psnr':math.inf if squared_error==0 else 10*math.log10(data_range*data_range/(squared_error/value_num))if data_range>0 else-math.inf,'compression_time_sum':sum(float(row['compression_time'])for row in rows),'decompression_time_sum':sum(float(row['decompression_time'])for row in rows),'max_error_over_eb':max(float(row['max_error_over_eb'])for row in rows),'zero_eb_slice_num':sum(int(row['zero_eb'])for row in rows),'valid':all(row['valid']=='1'for row in rows),'reference_compressed_bytes':expected['compressed_bytes']if expected else None,'reference_input_bytes':expected['input_bytes']if expected else None,'matches_reference_size':compressed_bytes==expected['compressed_bytes']and value_num*4==expected['input_bytes']if expected else None}
        field_summaries.append(summary)
    if reference and not all(summary['matches_reference_size']for summary in field_summaries):raise RuntimeError('并行SZ3-1D码流大小与相同内部线程数的参考结果不一致')
    value_num=sum(summary['value_num']for summary in field_summaries)
    compressed_bytes=sum(summary['compressed_bytes']for summary in field_summaries)
    summary={'codec':'sz3-1d','slice_workers':len(assignments),'threads_per_slice':threads_per_slice,'active_cpus':len(assignments)*threads_per_slice,'field_num':len(fields),'slice_num':sum(summary['slice_num']for summary in field_summaries),'value_num':value_num,'input_bytes':value_num*4,'compressed_bytes':compressed_bytes,'br':8*compressed_bytes/value_num,'reported_compression_time':max(compression_worker_times),'reported_decompression_time':max(decompression_worker_times),'reported_end_to_end_time':max(compression_worker_times)+max(decompression_worker_times),'compression_worker_times':compression_worker_times,'decompression_worker_times':decompression_worker_times,'worker_slice_counts':worker_slice_counts,'execution_wall_time':execution_wall_time,'compression_preprocessing_time':0,'decompression_preprocessing_time':0,'max_error_over_eb':max(field['max_error_over_eb']for field in field_summaries),'matches_reference_size':all(field['matches_reference_size']for field in field_summaries)if reference else None,'valid':all(field['valid']for field in field_summaries)}
    return field_summaries,summary

def main():
    parser=argparse.ArgumentParser(description='Run SZ3-1D with parallelism across complete slices and within each slice')
    parser.add_argument('--data-dir',type=Path,default=default_data_dir)
    parser.add_argument('--rel-eb',type=float,default=1e-3)
    parser.add_argument('--slice-workers',type=int,required=True)
    parser.add_argument('--threads-per-slice',type=int,required=True)
    parser.add_argument('--output-root',type=Path,default=project_root/'results/ocean_large_sz3_1d_128cpu_selection')
    parser.add_argument('--field',action='append')
    parser.add_argument('--max-slices',type=int)
    arguments=parser.parse_args()
    if not math.isfinite(arguments.rel_eb)or arguments.rel_eb<=0:parser.error('rel-eb必须是有限正数')
    if arguments.slice_workers<=0 or arguments.slice_workers&(arguments.slice_workers-1):parser.error('slice-workers必须是2的幂')
    if arguments.threads_per_slice<=0 or arguments.threads_per_slice&(arguments.threads_per_slice-1):parser.error('threads-per-slice必须是2的幂')
    if arguments.max_slices is not None and arguments.max_slices<=0:parser.error('max-slices必须为正数')
    arguments.data_dir=arguments.data_dir.resolve()
    metadata_path,fields,slices=read_metadata(arguments.data_dir,arguments.field,arguments.max_slices)
    formal_complete_dataset=not arguments.field and arguments.max_slices is None
    if formal_complete_dataset and len(slices)!=36264:raise RuntimeError(f'完整Ocean-large应有36264个slices，实际{len(slices)}')
    required_cpus=arguments.slice_workers*arguments.threads_per_slice
    if 'SLURM_JOB_ID'in os.environ and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=required_cpus:raise RuntimeError(f'Slurm CPU数量必须等于{required_cpus}')
    available_cpus=sorted(os.sched_getaffinity(0))
    if len(available_cpus)<required_cpus:raise RuntimeError(f'需要{required_cpus}个CPU，当前affinity只有{len(available_cpus)}个')
    binary=(project_root/'build/tensor_1d/benchmark_sz3').resolve()
    if not binary.is_file():raise FileNotFoundError(binary)
    binary_sha256=sha256(binary)
    reference=load_reference(arguments.rel_eb,arguments.threads_per_slice,fields,binary_sha256)if arguments.max_slices is None else None
    configuration=f'{arguments.slice_workers}_slice_workers_{arguments.threads_per_slice}_threads_per_slice'
    output_dir=(arguments.output_root/configuration/f'rel_{arguments.rel_eb:g}').resolve()
    worker_dir=output_dir/'workers'
    worker_dir.mkdir(parents=True,exist_ok=True)
    snapshot=output_dir/binary.name
    if not snapshot.is_file():shutil.copy2(binary,snapshot)
    if sha256(snapshot)!=binary_sha256:raise RuntimeError('保存的SZ3-1D binary SHA不一致')
    assignments=[slices[worker_id::arguments.slice_workers]for worker_id in range(arguments.slice_workers)]
    source_paths=[Path(__file__).resolve(),project_root/'benchmarks/sz3_1d_parallel/run_sz3_1d_128cpu_four_allocations.slurm']
    five_eb_launcher=project_root/'benchmarks/sz3_1d_parallel/run_five_ebs.slurm'
    if five_eb_launcher.is_file():source_paths.append(five_eb_launcher)
    manifest={'codec':'sz3-1d','configuration':configuration,'slice_workers':arguments.slice_workers,'threads_per_slice':arguments.threads_per_slice,'active_cpus':required_cpus,'assignment_rule':'global slice order round-robin; independent of measured runtime','worker_slice_counts':[len(assigned)for assigned in assignments],'field_num':len(fields),'slice_num':len(slices),'input_bytes':sum(item['input_bytes']for item in slices),'rel_eb':arguments.rel_eb,'error_bound_rule':'REL*(slice valid max-slice valid min)','binary_sha256':binary_sha256,'metadata_sha256':sha256(metadata_path),'source_sha256':{str(path.relative_to(project_root)):sha256(path)for path in source_paths},'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip(),'sz3_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root/'third_party/SZ3',text=True).strip(),'reference_directory':reference['directory']if reference else None,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_nodelist':os.environ.get('SLURM_NODELIST')}
    manifest_path=output_dir/'manifest.json'
    if manifest_path.is_file():
        previous=json.loads(manifest_path.read_text())
        for key in('codec','configuration','slice_workers','threads_per_slice','active_cpus','assignment_rule','worker_slice_counts','field_num','slice_num','input_bytes','rel_eb','error_bound_rule','binary_sha256','metadata_sha256','source_sha256','git_revision','sz3_revision','reference_directory'):
            if previous.get(key)!=manifest.get(key):raise RuntimeError(f'已有manifest的{key}不同')
    write_json(manifest_path,manifest)
    csv_paths,execution_wall_time=run_workers(assignments,arguments.threads_per_slice,snapshot,arguments.rel_eb,worker_dir,available_cpus)
    field_summaries,summary=summarize(assignments,csv_paths,fields,reference,arguments.threads_per_slice,execution_wall_time)
    with (output_dir/'fields.csv').open('w',newline='')as destination:
        writer=csv.DictWriter(destination,fieldnames=list(field_summaries[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(json_safe(field_summaries))
    write_json(output_dir/'summary.json',summary)
    print(json.dumps(json_safe(summary),ensure_ascii=False,allow_nan=False),flush=True)
    if not summary['valid']:raise RuntimeError('SZ3-1D误差检查失败')

if __name__=='__main__':main()
