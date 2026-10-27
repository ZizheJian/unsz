import argparse,concurrent.futures,csv,json,math,os,shutil,subprocess,time
from pathlib import Path
from run_all_95_fields import make_assignments
from run_one_field import aggregate,default_data_dir,json_safe,parse_unsz_field,read_metadata,sha256,write_json

project_root=Path(__file__).resolve().parents[2]

def load_completed(path,rel_eb,thread_num):
    if not path.is_file():return None
    result=json.loads(path.read_text())
    if result.get('compressor')=='unsz'and result.get('rel_eb')==rel_eb and result.get('thread_num')==thread_num and result.get('valid'):return result
    return None

def run_worker(worker_id,fields,rel_ebs,thread_num,cpus,data_dir,metadata,output_dirs,stream_root,binaries):
    cpu_set=','.join(str(cpu)for cpu in cpus)
    results={rel_eb:[]for rel_eb in rel_ebs}
    for field in fields:
        shape=metadata['fields'][field]['shape']
        for rel_eb in rel_ebs:
            output_dir=output_dirs[rel_eb]
            result_path=output_dir/f'{field}.json'
            result=load_completed(result_path,rel_eb,thread_num)
            if result is not None:
                results[rel_eb].append(result)
                print(f'SKIP worker={worker_id},field={field},rel_eb={rel_eb:g}',flush=True)
                continue
            arguments=argparse.Namespace(data_dir=data_dir,rel_eb=rel_eb,threads=thread_num,compression_binary=binaries['s1_compress'],decompression_binary=binaries['s1_decompress'],stream_root=stream_root/f'rel_{rel_eb:g}',cpu_set=cpu_set)
            print(f'BEGIN worker={worker_id},field={field},shape={shape},partitions={thread_num},rel_eb={rel_eb:g},cpus={cpu_set}',flush=True)
            start=time.perf_counter()
            result=parse_unsz_field(arguments,field,shape,output_dir)
            result['worker_id']=worker_id
            result['field_process_wall_time']=time.perf_counter()-start
            write_json(result_path,result)
            results[rel_eb].append(result)
            print(f'END worker={worker_id},field={field},rel_eb={rel_eb:g},br={result["br"]:.6g},compression_time={result["compression_time"]:.6g},decompression_time={result["decompression_time"]:.6g},max_error_over_eb={result["max_error_over_eb"]:.9g}',flush=True)
    return worker_id,results

def main():
    parser=argparse.ArgumentParser(description='Run UNSZ on all Ocean-large fields with field-relative error bounds and 128 CPUs')
    parser.add_argument('--data-dir',type=Path,default=default_data_dir)
    parser.add_argument('--rel-eb',type=float,action='append',required=True)
    parser.add_argument('--field-workers',type=int,default=16)
    parser.add_argument('--partitions-per-field',type=int,default=8)
    parser.add_argument('--output-root',type=Path,default=project_root/'results/ocean_large_field_relative_five_ebs')
    parser.add_argument('--stream-root',type=Path)
    parser.add_argument('--field',action='append')
    arguments=parser.parse_args()
    rel_ebs=arguments.rel_eb
    if any(not math.isfinite(rel_eb)or rel_eb<=0 for rel_eb in rel_ebs):parser.error('rel-eb必须是有限正数')
    if len(set(rel_ebs))!=len(rel_ebs):parser.error('rel-eb不能重复')
    if arguments.field_workers<=0 or arguments.field_workers&(arguments.field_workers-1):parser.error('field-workers必须是2的幂')
    if arguments.partitions_per_field<=0 or arguments.partitions_per_field&(arguments.partitions_per_field-1):parser.error('partitions-per-field必须是2的幂')
    arguments.data_dir=arguments.data_dir.resolve()
    metadata_path,metadata=read_metadata(arguments.data_dir)
    fields=arguments.field or list(metadata['fields'])
    if any(field not in metadata['fields']for field in fields):parser.error('指定field不在tensor metadata中')
    formal_complete_dataset=not arguments.field
    if formal_complete_dataset and len(fields)!=95:raise RuntimeError(f'预期95个fields，实际{len(fields)}个')
    point_num=(arguments.data_dir/f'{fields[0]}.dat.0').stat().st_size//4
    assignments,loads,max_load_lower_bound=make_assignments(fields,arguments.field_workers,metadata,point_num)
    required_cpus=arguments.field_workers*arguments.partitions_per_field
    if formal_complete_dataset and required_cpus!=128:raise RuntimeError(f'正式实验必须使用128个CPU，当前配置为{required_cpus}')
    if 'SLURM_JOB_ID'in os.environ and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=required_cpus:raise RuntimeError(f'Slurm CPU数量必须等于{required_cpus}')
    available_cpus=sorted(os.sched_getaffinity(0))
    if len(available_cpus)<required_cpus:raise RuntimeError(f'需要{required_cpus}个CPU，当前affinity只有{len(available_cpus)}个')
    configuration=f'{arguments.field_workers}_fields_{arguments.partitions_per_field}_partitions'
    configuration_dir=(arguments.output_root/'unsz'/configuration).resolve()
    configuration_dir.mkdir(parents=True,exist_ok=True)
    binary_dir=configuration_dir/'binaries'
    binary_dir.mkdir(exist_ok=True)
    source_binaries={'s1_compress':project_root/'install/bin/s1_compress','s1_decompress':project_root/'install/bin/s1_decompress'}
    if any(not path.is_file()for path in source_binaries.values()):raise RuntimeError('缺少已安装的UNSZ二进制文件')
    binary_sha256={name:sha256(path)for name,path in source_binaries.items()}
    binaries={}
    for name,source in source_binaries.items():
        snapshot=binary_dir/name
        if not snapshot.is_file():shutil.copy2(source,snapshot)
        if sha256(snapshot)!=binary_sha256[name]:raise RuntimeError(f'保存的二进制SHA不一致：{snapshot}')
        binaries[name]=snapshot
    output_dirs={rel_eb:configuration_dir/f'rel_{rel_eb:g}'for rel_eb in rel_ebs}
    source_paths=[Path(__file__).resolve(),Path(__file__).with_name('run_one_field.py'),Path(__file__).with_name('run_all_95_fields.py'),Path(__file__).with_name('run_all_unsz_95_fields.slurm')]
    git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip()
    for rel_eb,output_dir in output_dirs.items():
        output_dir.mkdir(parents=True,exist_ok=True)
        manifest={'codec':'unsz','configuration':configuration,'field_workers':arguments.field_workers,'partitions_per_field':arguments.partitions_per_field,'active_cpus':required_cpus,'field_num':len(fields),'fields':fields,'assignments':assignments,'assignment_value_loads':loads,'assignment_max_value_load_lower_bound':max_load_lower_bound,'rel_eb':rel_eb,'rel_ebs_in_process':rel_ebs,'codec_error_bound_mode':'ABS','abs_eb_rule':'rel_eb*(field_max-field_min),excluding -99999','metadata_sha256':sha256(metadata_path),'binary_sha256':binary_sha256,'git_revision':git_revision,'source_sha256':{str(path.relative_to(project_root)):sha256(path)for path in source_paths}}
        manifest_path=output_dir/'manifest.json'
        if manifest_path.is_file()and json.loads(manifest_path.read_text())!=manifest:raise RuntimeError(f'已有manifest不同：{manifest_path}')
        write_json(manifest_path,manifest)
    environment={'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_nodelist':os.environ.get('SLURM_NODELIST'),'slurm_cpus':int(os.environ['SLURM_CPUS_PER_TASK'])if os.environ.get('SLURM_CPUS_PER_TASK')else None,'hostname':os.uname().nodename}
    write_json(configuration_dir/f'environment_{os.environ.get("SLURM_JOB_ID",str(os.getpid()))}.json',environment)
    run_id=os.environ.get('SLURM_JOB_ID',f'local_{os.getpid()}')
    stream_parent=(arguments.stream_root or arguments.data_dir.parent/'field_relative_unsz_outputs').resolve()
    stream_root=stream_parent/f'job_{run_id}'
    if stream_root.exists()and any(stream_root.iterdir()):raise RuntimeError(f'临时输出目录非空：{stream_root}')
    stream_root.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    worker_results={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.field_workers)as executor:
        futures=[]
        for worker_id,worker_fields in enumerate(assignments):
            cpus=available_cpus[worker_id*arguments.partitions_per_field:(worker_id+1)*arguments.partitions_per_field]
            futures.append(executor.submit(run_worker,worker_id,worker_fields,rel_ebs,arguments.partitions_per_field,cpus,arguments.data_dir,metadata,output_dirs,stream_root,binaries))
        for future in concurrent.futures.as_completed(futures):
            worker_id,results=future.result()
            worker_results[worker_id]=results
    execution_wall_time=time.perf_counter()-start
    for rel_eb,output_dir in output_dirs.items():
        results=[]
        for field in fields:
            result_path=output_dir/f'{field}.json'
            result=load_completed(result_path,rel_eb,arguments.partitions_per_field)
            if result is None:raise RuntimeError(f'缺少有效field结果：rel_eb={rel_eb:g},field={field}')
            results.append(result)
        with (output_dir/'fields.csv').open('w',newline='')as destination:
            fieldnames=sorted({key for result in results for key in result if key!='command'})
            writer=csv.DictWriter(destination,fieldnames=fieldnames,extrasaction='ignore',lineterminator='\n')
            writer.writeheader()
            writer.writerows(json_safe(results))
        per_worker=[worker_results[worker_id][rel_eb]for worker_id in range(arguments.field_workers)]
        compression_worker_times=[sum(result['compression_preprocessing_time']+result['compression_time']for result in worker)for worker in per_worker]
        decompression_worker_times=[sum(result['decompression_preprocessing_time']+result['decompression_time']for result in worker)for worker in per_worker]
        summary=aggregate(results,'unsz',rel_eb)
        summary.update({'configuration':configuration,'field_workers':arguments.field_workers,'partitions_per_field':arguments.partitions_per_field,'active_cpus':required_cpus,'compression_worker_times':compression_worker_times,'decompression_worker_times':decompression_worker_times,'reported_compression_time':max(compression_worker_times),'reported_decompression_time':max(decompression_worker_times),'reported_end_to_end_time':max(compression_worker_times)+max(decompression_worker_times),'all_ebs_execution_wall_time':execution_wall_time})
        write_json(output_dir/'summary.json',summary)
        print(json.dumps(json_safe(summary),ensure_ascii=False,allow_nan=False),flush=True)
    if stream_root.parent!=stream_parent:raise RuntimeError(f'拒绝清理意外目录：{stream_root}')
    shutil.rmtree(stream_root)

if __name__=='__main__':main()
