import argparse,concurrent.futures,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import check_cpus,default_data_dir,field_range_quality,field_slice_num,git_revision,optimal_assignment,project_root,quality_summary,read_metadata,sha256,shape_name,slurm_environment,statistics_version,worker_times,write_fields_csv,write_json

# 每个field组成完整tensor，用整个field的有效值range换算一个ABS EB；field沿点编号切成多个partition，各partition由单线程压缩器独立处理
compressor_names={'sz3':'sz3-nd-point-partitioned','qoz':'qoz2-q4-nd-point-partitioned'}
binary_names={'sz3':'compress_one_field_with_sz3_partitions','qoz':'compress_one_field_with_qoz_partitions'}
upstream_dirs={'sz3':'third_party/SZ3','qoz':'third_party/QoZ'}

def load_completed(path,codec,rel_eb,partition_num):
    if not path.is_file():return None
    result=json.loads(path.read_text())
    if 'valid_squared_error'in result and result.get('compressor')==compressor_names[codec]and result.get('rel_eb')==rel_eb and result.get('partition_num')==partition_num and result.get('valid'):return result
    return None

def unified_row(result,metadata,worker_id):
    quality=field_range_quality(result['squared_error'],result['valid_squared_error'],result['valid_value_num'],result['value_num'],result['field_range'])
    return {**result,**quality,'shape':shape_name(metadata,result['field']),'slice_num':field_slice_num(metadata,result['field']),'compression_preprocessing_time':0,'decompression_preprocessing_time':0,'worker_id':worker_id}

def run_worker(worker_id,fields,codec,rel_ebs,partition_num,data_dir,metadata,cpus,output_dirs,log_dir,binary,summarize_only):
    cpu_set=','.join(str(cpu)for cpu in cpus)
    environment={**os.environ,'OMP_NUM_THREADS':str(partition_num),'OMP_DYNAMIC':'false','OMP_PROC_BIND':'close','OMP_PLACES':'cores','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    results={rel_eb:[]for rel_eb in rel_ebs}
    failures=[]
    for field in fields:
        shape=metadata['fields'][field]['shape']
        pending=[]
        for rel_eb in rel_ebs:
            result_path=output_dirs[rel_eb]/f'{field}.json'
            result=load_completed(result_path,codec,rel_eb,partition_num)
            if result is None:pending.append((rel_eb,result_path))
            else:
                results[rel_eb].append(unified_row(result,metadata,worker_id))
                print(f'SKIP worker={worker_id},field={field},rel_eb={rel_eb:g}',flush=True)
        if not pending:continue
        if summarize_only:raise RuntimeError(f'{field}缺少已完成结果，不能只汇总')
        # 同一field的多个EB在一个进程中运行，只读取一次输入；每个EB重新切分partition
        if len(pending)==1:command=['taskset','--cpu-list',cpu_set,binary,data_dir,field,shape[0],shape[1],pending[0][0],partition_num,pending[0][1]]
        else:
            command=['taskset','--cpu-list',cpu_set,binary,data_dir,field,shape[0],shape[1],partition_num]
            for rel_eb,result_path in pending:command.extend((rel_eb,result_path))
        print(f'BEGIN worker={worker_id},codec={codec},field={field},shape={shape},partitions={partition_num},ebs={[rel_eb for rel_eb,_ in pending]},cpus={cpu_set}',flush=True)
        log_path=log_dir/f'{field}.log'
        with log_path.open('w')as log:returncode=subprocess.run([str(value)for value in command],cwd=project_root,stdout=log,stderr=subprocess.STDOUT,env=environment).returncode
        # 压缩器崩溃或误差超界时记录失败并继续处理该worker的其他field；崩溃前已完成的EB结果仍然有效
        for rel_eb,result_path in pending:
            result=load_completed(result_path,codec,rel_eb,partition_num)
            if result is None:
                failures.append({'field':field,'rel_eb':rel_eb,'returncode':returncode,'log':str(log_path)})
                print(f'FAIL worker={worker_id},codec={codec},field={field},rel_eb={rel_eb:g},returncode={returncode}',flush=True)
                continue
            results[rel_eb].append(unified_row(result,metadata,worker_id))
            print(f'END worker={worker_id},codec={codec},field={field},rel_eb={rel_eb:g},br={result["br"]:.6g},compression_time={result["compression_time"]:.6g},decompression_time={result["decompression_time"]:.6g},max_rss_kib={result["max_rss_kib"]}',flush=True)
    return worker_id,results,failures

def main():
    parser=argparse.ArgumentParser(description='Compress all 95 Ocean-large fields as tensors with field workers and point partitions')
    parser.add_argument('--codec',required=True,choices=tuple(compressor_names))
    parser.add_argument('--data-dir',type=Path,default=default_data_dir)
    parser.add_argument('--rel-eb',type=float,action='append',required=True)
    parser.add_argument('--field-workers',type=int,required=True)
    parser.add_argument('--partitions-per-field',type=int,required=True)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--field',action='append')
    parser.add_argument('--summarize-only',action='store_true')
    arguments=parser.parse_args()
    rel_ebs=arguments.rel_eb
    if any(not math.isfinite(rel_eb)or rel_eb<=0 for rel_eb in rel_ebs)or len(set(rel_ebs))!=len(rel_ebs):parser.error('rel-eb必须是不重复的有限正数')
    for value,name in((arguments.field_workers,'field-workers'),(arguments.partitions_per_field,'partitions-per-field')):
        if value<=0 or value&(value-1):parser.error(f'{name}必须是2的幂')
    data_dir=arguments.data_dir.resolve()
    metadata_path,metadata,fields=read_metadata(data_dir,arguments.field)
    assignments,loads,optimum,lower_bound=optimal_assignment({field:field_slice_num(metadata,field)for field in fields},arguments.field_workers)
    cpu_num=arguments.field_workers*arguments.partitions_per_field
    available_cpus=list(range(cpu_num))if arguments.summarize_only else check_cpus(cpu_num,not arguments.field)
    binary=(project_root/'build/field_relative'/binary_names[arguments.codec]).resolve()
    output_root=arguments.output_root or project_root/f'results/ocean_large_per_field_rd_{cpu_num}'
    configuration_dir=(output_root/arguments.codec/f'{arguments.field_workers}_fields_{arguments.partitions_per_field}_partitions').resolve()
    log_dir=configuration_dir/'logs'
    log_dir.mkdir(parents=True,exist_ok=True)
    output_dirs={rel_eb:configuration_dir/f'rel_{rel_eb:g}'for rel_eb in rel_ebs}
    snapshot=configuration_dir/binary.name
    if not arguments.summarize_only:shutil.copy2(binary,snapshot)
    binary_sha256=sha256(binary if arguments.summarize_only else snapshot)
    for rel_eb,output_dir in output_dirs.items():
        output_dir.mkdir(parents=True,exist_ok=True)
        write_json(output_dir/'manifest.json',{'codec':compressor_names[arguments.codec],'protocol':'field','field_workers':arguments.field_workers,'point_partitions_per_field':arguments.partitions_per_field,'cpu_num':cpu_num,'field_num':len(fields),'assignment_rule':'exact minimum of the maximum worker slice load, then of the maximum worker field count','assignments':assignments,'worker_slice_loads':loads,'worker_slice_load_lower_bound':lower_bound,'rel_eb':rel_eb,'rel_ebs_in_process':rel_ebs,'error_bound_rule':'REL*(field valid max-field valid min), excluding -99999','metadata_sha256':sha256(metadata_path),'binary_sha256':binary_sha256,'source_sha256':{str(path.relative_to(project_root)):sha256(path)for path in(Path(__file__).resolve(),project_root/'benchmarks/common.py',Path(__file__).with_name('compress_one_field_with_partitions.cpp'))},'git_revision':git_revision(),'upstream_revision':git_revision(project_root/upstream_dirs[arguments.codec]),**slurm_environment()})
    start=time.perf_counter()
    worker_results={}
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.field_workers)as executor:
        futures=[executor.submit(run_worker,worker_id,worker_fields,arguments.codec,rel_ebs,arguments.partitions_per_field,data_dir,metadata,available_cpus[worker_id*arguments.partitions_per_field:(worker_id+1)*arguments.partitions_per_field],output_dirs,log_dir,snapshot,arguments.summarize_only)for worker_id,worker_fields in enumerate(assignments)]
        for future in futures:
            worker_id,results,worker_failures=future.result()
            worker_results[worker_id]=results
            failures.extend(worker_failures)
    execution_wall_time=time.perf_counter()-start
    for rel_eb,output_dir in output_dirs.items():
        rows=[row for worker_id in range(arguments.field_workers)for row in worker_results[worker_id][rel_eb]]
        if not rows:raise RuntimeError(f'rel_eb={rel_eb:g}没有任何完成的field')
        write_fields_csv(output_dir/'fields.csv',rows)
        times=worker_times([sum(row['compression_time']for row in worker_results[worker_id][rel_eb])for worker_id in range(arguments.field_workers)],[sum(row['decompression_time']for row in worker_results[worker_id][rel_eb])for worker_id in range(arguments.field_workers)])
        eb_failures=[failure for failure in failures if failure['rel_eb']==rel_eb]
        summary={'codec':compressor_names[arguments.codec],'protocol':'field','statistics_version':statistics_version,'rel_eb':rel_eb,'field_workers':arguments.field_workers,'point_partitions_per_field':arguments.partitions_per_field,'cpu_num':cpu_num,'field_num':len(fields),'slice_num':sum(row['slice_num']for row in rows),'compressed_bytes':sum(row['compressed_bytes']for row in rows),**quality_summary(rows),**times,'all_ebs_execution_wall_time':execution_wall_time,'max_concurrent_rss_upper_bound_kib':sum(max((row['max_rss_kib']for row in worker_results[worker_id][rel_eb]),default=0)for worker_id in range(arguments.field_workers)),'max_error_over_eb':max(row['max_error_over_eb']for row in rows),'complete':len(rows)==len(fields),'failures':eb_failures,'valid':len(rows)==len(fields)and all(row['valid']for row in rows)}
        write_json(output_dir/'summary.json',summary)
        print(json.dumps({key:value for key,value in summary.items()if not isinstance(value,(list,dict))},ensure_ascii=False),flush=True)
    if failures:print(json.dumps({'failures':failures},ensure_ascii=False),flush=True)
    return 1 if failures else 0

if __name__=='__main__':
    sys.exit(main())
