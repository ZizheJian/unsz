import argparse,concurrent.futures,csv,json,math,os,re,subprocess,sys,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import check_cpus,default_data_dir,field_slice_num,git_revision,optimal_assignment,project_root,quality_columns,quality_summary,read_metadata,rel_eb_tags,sha256,shape_name,slurm_environment,statistics_version,worker_times,write_fields_csv,write_json

# UMC没有可处理完整Ocean-large的并行入口，因此每个worker用官方串行程序依次处理分给它的完整field
umc_root=project_root/'third_party/umc'

def cleanup(data_dir,field,slice_num):
    # UMC把码流写在原始数据旁边；每个field结束后立即删除，避免占满数据目录的文件数量配额
    for slice_id in range(slice_num):
        source=data_dir/f'{field}.dat.{slice_id}'
        Path(str(source)+'.umc').unlink(missing_ok=True)
        Path(str(source)+'.umc.out').unlink(missing_ok=True)

def read_one_row(path):
    with Path(path).open(newline='')as source:rows=list(csv.DictReader(source))
    if len(rows)!=1:raise RuntimeError(f'结果行数不是1：{path}')
    return rows[0]

def same_float(left,right):
    left,right=float(left if left not in(None,'')else math.inf),float(right if right not in(None,'')else math.inf)
    return left==right if math.isinf(left)or math.isinf(right)else math.isclose(left,right,rel_tol=1e-12,abs_tol=1e-12)

def memory_usage_path():
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        _,controllers,relative=line.split(':',2)
        if 'memory'in controllers.split(','):
            path=Path('/sys/fs/cgroup/memory')/relative.lstrip('/')/'memory.usage_in_bytes'
            if path.is_file():return path
    return None

def monitor_memory(stop_event,result):
    path=memory_usage_path()
    result['path']=str(path)if path else None
    result['peak_bytes']=0
    while path is not None:
        try:result['peak_bytes']=max(result['peak_bytes'],int(path.read_text()))
        except OSError:pass
        if stop_event.wait(0.25):break

def run_field(worker_id,field,cpu,data_dir,metadata,rel_eb,output_dir,reference):
    result_path=output_dir/'fields'/f'{field}.json'
    if result_path.is_file():
        result=json.loads(result_path.read_text())
        if 'normalized_squared_error'in result and result.get('rel_eb')==rel_eb and result.get('valid')and same_float(result['br'],reference['br'])and same_float(result['psnr'],reference['psnr']):
            print(f'SKIP worker={worker_id},field={field}',flush=True)
            return result
    slice_num=field_slice_num(metadata,field)
    csv_path=output_dir/'fields'/f'{field}.csv'
    config_path=output_dir/'configs'/f'{field}.json'
    log_path=output_dir/'logs'/f'{field}.log'
    resource_path=output_dir/'resources'/f'{field}.txt'
    cleanup(data_dir,field,slice_num)
    write_json(config_path,[{'name':'ocean_large_umc_field_parallel','data_dir':str(data_dir),'csv_path':str(csv_path),'compressor_list':['umc'],'field_list':[field],'time_step_num':slice_num,'coordinate_dim':3,'rel_eb_list':[rel_eb],'cleanup_after_each_test':True,'umc':{'from_edge':1,'reorder':1,'method':'NBP-IDW-INP','neighbor_num':4}}])
    environment={**os.environ,'OMP_NUM_THREADS':'1','OMP_DYNAMIC':'false','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    command=['/usr/bin/time','-v','-o',str(resource_path),'taskset','--cpu-list',str(cpu),str(project_root/'unsz_env/bin/python'),'-u',str(project_root/'t1_start_c_code.py'),str(config_path),'--skip-build']
    print(f'BEGIN worker={worker_id},field={field},slices={slice_num},cpu={cpu}',flush=True)
    start=time.perf_counter()
    try:
        with log_path.open('w')as log:returncode=subprocess.run(command,cwd=project_root,stdout=log,stderr=subprocess.STDOUT,env=environment).returncode
        process_wall_time=time.perf_counter()-start
        if returncode:raise RuntimeError(f'UMC field失败：{field},returncode={returncode},见{log_path}')
        row=read_one_row(csv_path)
        if row['compressor']!='umc'or float(row['rel_eb'])!=rel_eb or int(row['time_step_num'])!=slice_num:raise RuntimeError(f'UMC结果参数不匹配：{field}')
        if int(row['input_bytes'])!=int(reference['input_bytes'])or not same_float(row['br'],reference['br'])or not same_float(row['psnr'],reference['psnr']):raise RuntimeError(f'UMC并行结果与串行参考结果不一致：{field}')
        match=re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',resource_path.read_text())
        if match is None:raise RuntimeError(f'UMC缺少峰值内存记录：{field}')
        value_num=int(row['input_bytes'])//4
        if int(row['value_num'])!=value_num:raise RuntimeError(f'UMC逐slice误差统计的值数量不一致：{field}')
        result={'field':field,'shape':shape_name(metadata,field),'slice_num':slice_num,'value_num':value_num,'compressed_bytes':round(float(row['br'])*value_num/8),'br':float(row['br']),**quality_columns(float(row['slice_normalized_valid_squared_error']),int(row['valid_value_num']),float(row['slice_normalized_squared_error']),value_num),'field_range_psnr':float(row['psnr']),'compression_preprocessing_time':float(row['compression_preprocessing_time']),'compression_time':float(row['compression_time']),'decompression_preprocessing_time':float(row['decompression_preprocessing_time']),'decompression_time':float(row['decompression_time']),'valid':True,'worker_id':worker_id,'rel_eb':rel_eb,'field_process_wall_time':process_wall_time,'max_rss_kib':int(match.group(1))}
        write_json(result_path,result)
        print(f'END worker={worker_id},field={field},wall_time={process_wall_time:.3f},max_rss_kib={result["max_rss_kib"]}',flush=True)
        return result
    finally:
        cleanup(data_dir,field,slice_num)
        config_path.unlink(missing_ok=True)

def main():
    parser=argparse.ArgumentParser(description='Run UMC on independent Ocean-large fields with static field-level parallelism')
    parser.add_argument('--data-dir',type=Path,default=default_data_dir)
    parser.add_argument('--rel-eb',type=float,action='append',required=True,choices=tuple(rel_eb_tags))
    parser.add_argument('--field-workers',type=int,required=True)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--field',action='append')
    parser.add_argument('--prepare-only',action='store_true')
    arguments=parser.parse_args()
    worker_num=arguments.field_workers
    if worker_num<=0 or worker_num&(worker_num-1):parser.error('field-workers必须是2的幂')
    data_dir=arguments.data_dir.resolve()
    metadata_path,metadata,fields=read_metadata(data_dir,arguments.field)
    formal_complete_dataset=not arguments.field
    assignments,loads,optimum,lower_bound=optimal_assignment({field:field_slice_num(metadata,field)for field in fields},worker_num)
    for name in('compress.hpp','decompress.hpp'):
        if(umc_root/'include'/name).read_bytes()!=(project_root/'third_party/umc_replacements/include'/name).read_bytes():raise RuntimeError(f'UMC源码不符合项目记录的替换文件：{name}')
    binary_sha256={name:sha256(umc_root/'build/test'/name)for name in('test_compress','test_decompress')}
    output_root=arguments.output_root or project_root/f'results/ocean_large_per_slice_rd_{worker_num}/umc'
    available_cpus=None
    for rel_eb in arguments.rel_eb:
        serial_dir=project_root/f'results/ocean_large_tensor_full_umc/serial_rel_{rel_eb_tags[rel_eb]}'
        if binary_sha256!=json.loads((serial_dir/'manifest.json').read_text())['binary_sha256']:raise RuntimeError(f'UMC binary与串行参考结果不同：{serial_dir}')
        references={field:read_one_row(serial_dir/f'{field}.csv')for field in fields}
        output_dir=(output_root/f'{worker_num}_field_workers'/f'rel_{rel_eb:g}').resolve()
        for name in('fields','configs','logs','resources'):(output_dir/name).mkdir(parents=True,exist_ok=True)
        source_paths=[Path(__file__).resolve(),project_root/'benchmarks/common.py',project_root/'t1_start_c_code.py',project_root/'py_codes/parse_umc_output.py',umc_root/'include/compress.hpp',umc_root/'include/decompress.hpp']
        write_json(output_dir/'manifest.json',{'codec':'umc-nbp-idw-inp','protocol':'slice','field_workers':worker_num,'program_threads_per_worker':1,'cpu_num':worker_num,'field_num':len(fields),'assignment_rule':'exact minimum of the maximum worker slice load, then of the maximum worker field count','assignments':assignments,'worker_slice_loads':loads,'worker_slice_load_lower_bound':lower_bound,'rel_eb':rel_eb,'batch_size':10,'metadata_sha256':sha256(metadata_path),'binary_sha256':binary_sha256,'source_sha256':{str(path.relative_to(project_root)):sha256(path)for path in source_paths},'git_revision':git_revision(),'umc_revision':git_revision(umc_root),'serial_reference_directory':str(serial_dir.relative_to(project_root)),**slurm_environment()})
        print(json.dumps({'output_dir':str(output_dir),'loads':loads,'optimum':optimum,'max_fields_per_worker':max(len(worker_fields)for worker_fields in assignments)},ensure_ascii=False),flush=True)
        if arguments.prepare_only:continue
        if available_cpus is None:available_cpus=check_cpus(worker_num,formal_complete_dataset)
        stop_event=threading.Event();memory_result={}
        monitor=threading.Thread(target=monitor_memory,args=(stop_event,memory_result),daemon=True)
        monitor.start()
        start=time.perf_counter();worker_results={}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num)as executor:
                futures=[executor.submit(lambda worker_id,worker_fields:(worker_id,[run_field(worker_id,field,available_cpus[worker_id],data_dir,metadata,rel_eb,output_dir,references[field])for field in worker_fields]),worker_id,worker_fields)for worker_id,worker_fields in enumerate(assignments)]
                for future in concurrent.futures.as_completed(futures):
                    worker_id,results=future.result();worker_results[worker_id]=results
        finally:
            stop_event.set();monitor.join()
        rows=[result for worker_id in range(worker_num)for result in worker_results[worker_id]]
        write_fields_csv(output_dir/'fields.csv',rows)
        times=worker_times([sum(result['compression_preprocessing_time']+result['compression_time']for result in worker_results[worker_id])for worker_id in range(worker_num)],[sum(result['decompression_preprocessing_time']+result['decompression_time']for result in worker_results[worker_id])for worker_id in range(worker_num)])
        summary={'codec':'umc-nbp-idw-inp','protocol':'slice','statistics_version':statistics_version,'rel_eb':rel_eb,'field_workers':worker_num,'cpu_num':worker_num,'field_num':len(fields),'slice_num':sum(loads),'compressed_bytes':sum(row['compressed_bytes']for row in rows),**quality_summary(rows),**times,'execution_wall_time':time.perf_counter()-start,'worker_slice_loads':loads,'max_rss_upper_bound_kib':sum(max((result['max_rss_kib']for result in worker_results[worker_id]),default=0)for worker_id in range(worker_num)),'cgroup_memory_peak_bytes':memory_result.get('peak_bytes'),'matches_serial_reference':True,'valid':True}
        write_json(output_dir/'summary.json',summary)
        print(json.dumps({key:value for key,value in summary.items()if not isinstance(value,(list,dict))},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
