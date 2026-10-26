import argparse,concurrent.futures,csv,json,math,os,shutil,subprocess,time
from pathlib import Path
from run_one_field import default_data_dir,json_safe,read_metadata,sha256,write_json

project_root=Path(__file__).resolve().parents[2]
compressor_names={'sz3':'sz3-nd-point-partitioned','qoz':'qoz2-q4-nd-point-partitioned'}
binary_names={'sz3':'compress_one_field_with_sz3_partitions','qoz':'compress_one_field_with_qoz_partitions'}

def make_assignments(fields,worker_num,metadata,point_num):
    assignments=[[]for _ in range(worker_num)]
    loads=[0]*worker_num
    field_loads={field:math.prod(metadata['fields'][field]['shape'])*point_num for field in fields}
    for field in sorted(fields,key=lambda name:(-math.prod(metadata['fields'][name]['shape']),name)):
        worker_id=min(range(worker_num),key=lambda index:(loads[index],index))
        assignments[worker_id].append(field)
        loads[worker_id]+=field_loads[field]
    while True:
        best_score=tuple(sorted(loads,reverse=True))
        best_operation=None
        for source in range(worker_num):
            for field in assignments[source]:
                field_load=field_loads[field]
                for destination in range(worker_num):
                    if source==destination:continue
                    candidate=loads.copy()
                    candidate[source]-=field_load
                    candidate[destination]+=field_load
                    score=tuple(sorted(candidate,reverse=True))
                    if score<best_score:
                        best_score=score
                        best_operation=('move',source,destination,field,None)
        for left in range(worker_num):
            for right in range(left+1,worker_num):
                for left_field in assignments[left]:
                    for right_field in assignments[right]:
                        left_load=field_loads[left_field]
                        right_load=field_loads[right_field]
                        if left_load==right_load:continue
                        candidate=loads.copy()
                        candidate[left]+=right_load-left_load
                        candidate[right]+=left_load-right_load
                        score=tuple(sorted(candidate,reverse=True))
                        if score<best_score:
                            best_score=score
                            best_operation=('swap',left,right,left_field,right_field)
        if best_operation is None:break
        operation,left,right,left_field,right_field=best_operation
        if operation=='move':
            assignments[left].remove(left_field)
            assignments[right].append(left_field)
            loads[left]-=field_loads[left_field]
            loads[right]+=field_loads[left_field]
        else:
            assignments[left].remove(left_field)
            assignments[right].remove(right_field)
            assignments[left].append(right_field)
            assignments[right].append(left_field)
            loads[left]+=field_loads[right_field]-field_loads[left_field]
            loads[right]+=field_loads[left_field]-field_loads[right_field]
    average_lower_bound=(sum(field_loads.values())+worker_num-1)//worker_num
    threshold_lower_bound=max(((sum(load>=threshold for load in field_loads.values())+worker_num-1)//worker_num)*threshold for threshold in set(field_loads.values()))
    max_load_lower_bound=max(average_lower_bound,threshold_lower_bound)
    if max(loads)!=max_load_lower_bound:raise RuntimeError(f'无法证明按field大小分配达到全局最优：当前最大负载={max(loads)}，可证明的下界={max_load_lower_bound}')
    return assignments,loads,max_load_lower_bound

def load_completed(path,codec,rel_eb,partition_num):
    if not path.is_file():return None
    result=json.loads(path.read_text())
    if result.get('compressor')==compressor_names[codec]and result.get('rel_eb')==rel_eb and result.get('partition_num')==partition_num and result.get('valid'):return result
    return None

def run_worker(worker_id,fields,codec,rel_ebs,partition_num,data_dir,metadata,cpus,output_dirs,log_dir,binary):
    cpu_set=','.join(str(cpu)for cpu in cpus)
    environment={**os.environ,'OMP_NUM_THREADS':str(partition_num),'OMP_DYNAMIC':'false','OMP_PROC_BIND':'close','OMP_PLACES':'cores','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    results={rel_eb:[]for rel_eb in rel_ebs}
    for field in fields:
        shape=metadata['fields'][field]['shape']
        pending=[]
        for rel_eb in rel_ebs:
            result_path=output_dirs[rel_eb]/f'{field}.json'
            result=load_completed(result_path,codec,rel_eb,partition_num)
            if result is None:pending.append((rel_eb,result_path))
            else:
                results[rel_eb].append(result)
                print(f'SKIP worker={worker_id},field={field},rel_eb={rel_eb:g}',flush=True)
        if not pending:continue
        log_path=log_dir/f'{field}.log'
        if len(pending)==1:
            rel_eb,result_path=pending[0]
            command=['taskset','--cpu-list',cpu_set,binary,data_dir,field,shape[0],shape[1],rel_eb,partition_num,result_path]
        else:
            command=['taskset','--cpu-list',cpu_set,binary,data_dir,field,shape[0],shape[1],partition_num]
            for rel_eb,result_path in pending:command.extend((rel_eb,result_path))
        print(f'BEGIN worker={worker_id},codec={codec},field={field},shape={shape},partitions={partition_num},ebs={[rel_eb for rel_eb,_ in pending]},cpus={cpu_set}',flush=True)
        start=time.perf_counter()
        with log_path.open('w')as log:returncode=subprocess.run([str(value)for value in command],cwd=project_root,stdout=log,stderr=subprocess.STDOUT,env=environment).returncode
        process_wall_time=time.perf_counter()-start
        if returncode:raise RuntimeError(f'{field}失败，returncode={returncode}，见{log_path}')
        for rel_eb,result_path in pending:
            if not result_path.is_file():raise RuntimeError(f'{field}缺少rel_eb={rel_eb:g}的结果，见{log_path}')
            result=json.loads(result_path.read_text())
            if not result.get('valid'):raise RuntimeError(f'{field}在rel_eb={rel_eb:g}时误差检查失败，见{result_path}')
            result['worker_id']=worker_id
            result['field_process_wall_time']=process_wall_time
            result['worker_execution_wall_time']=result.get('eb_processing_time_before_output',process_wall_time)+(result.get('input_and_range_time',0)if result.get('eb_index_in_process',0)==0 else 0)
            write_json(result_path,result)
            results[rel_eb].append(result)
            print(f'END worker={worker_id},codec={codec},field={field},rel_eb={rel_eb:g},br={result["br"]:.6g},compression_time={result["compression_time"]:.6g},decompression_time={result["decompression_time"]:.6g},max_rss_kib={result["max_rss_kib"]}',flush=True)
    return worker_id,results

def main():
    parser=argparse.ArgumentParser(description='Compress all 95 Ocean-large fields with field and point-partition parallelism')
    parser.add_argument('--codec',required=True,choices=tuple(compressor_names))
    parser.add_argument('--data-dir',type=Path,default=default_data_dir)
    parser.add_argument('--rel-eb',type=float,action='append',required=True)
    parser.add_argument('--field-workers',type=int,required=True)
    parser.add_argument('--partitions-per-field',type=int,required=True)
    parser.add_argument('--output-root',type=Path,default=project_root/'results/ocean_large_field_relative_five_ebs')
    arguments=parser.parse_args()
    rel_ebs=arguments.rel_eb
    if any(not math.isfinite(rel_eb)or rel_eb<=0 for rel_eb in rel_ebs):parser.error('rel-eb必须是有限正数')
    if len(set(rel_ebs))!=len(rel_ebs):parser.error('rel-eb不能重复')
    if arguments.field_workers<=0 or arguments.field_workers&(arguments.field_workers-1):parser.error('field-workers必须是2的幂')
    if arguments.partitions_per_field<=0 or arguments.partitions_per_field&(arguments.partitions_per_field-1):parser.error('partitions-per-field必须是2的幂')
    arguments.data_dir=arguments.data_dir.resolve()
    metadata_path,metadata=read_metadata(arguments.data_dir)
    fields=list(metadata['fields'])
    if len(fields)!=95:raise RuntimeError(f'预期95个fields，实际{len(fields)}个')
    point_num=(arguments.data_dir/f'{fields[0]}.dat.0').stat().st_size//4
    assignments,loads,max_load_lower_bound=make_assignments(fields,arguments.field_workers,metadata,point_num)
    binary=(project_root/'build/field_relative'/binary_names[arguments.codec]).resolve()
    if not binary.is_file():raise FileNotFoundError(binary)
    available_cpus=sorted(os.sched_getaffinity(0))
    required_cpus=arguments.field_workers*arguments.partitions_per_field
    if len(available_cpus)<required_cpus:raise RuntimeError(f'需要{required_cpus}个CPU，当前affinity只有{len(available_cpus)}个')
    configuration=f'{arguments.field_workers}_fields_{arguments.partitions_per_field}_partitions'
    configuration_dir=(arguments.output_root/arguments.codec/configuration).resolve()
    configuration_dir.mkdir(parents=True,exist_ok=True)
    log_dir=configuration_dir/'logs'
    log_dir.mkdir(exist_ok=True)
    output_dirs={rel_eb:configuration_dir/f'rel_{rel_eb:g}'for rel_eb in rel_ebs}
    binary_sha256=sha256(binary)
    git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip()
    upstream_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root/'third_party'/('SZ3'if arguments.codec=='sz3'else'QoZ'),text=True).strip()
    snapshots=[]
    for rel_eb,output_dir in output_dirs.items():
        output_dir.mkdir(parents=True,exist_ok=True)
        manifest={'codec':compressor_names[arguments.codec],'configuration':configuration,'field_workers':arguments.field_workers,'point_partitions_per_field':arguments.partitions_per_field,'active_cpus':required_cpus,'field_num':len(fields),'fields':fields,'assignments':assignments,'assignment_value_loads':loads,'assignment_max_value_load_lower_bound':max_load_lower_bound,'rel_eb':rel_eb,'rel_ebs_in_process':rel_ebs,'input_reused_across_rel_ebs':len(rel_ebs)>1,'metadata_sha256':sha256(metadata_path),'binary_sha256':binary_sha256,'git_revision':git_revision,'upstream_revision':upstream_revision}
        manifest_path=output_dir/'manifest.json'
        if manifest_path.is_file()and json.loads(manifest_path.read_text())!=manifest:raise RuntimeError(f'已有manifest不同：{manifest_path}')
        write_json(manifest_path,manifest)
        snapshot=output_dir/binary.name
        if not snapshot.is_file():shutil.copy2(binary,snapshot)
        if sha256(snapshot)!=binary_sha256:raise RuntimeError(f'保存的可执行文件SHA不一致：{snapshot}')
        snapshots.append(snapshot)
    snapshot=snapshots[0]
    start=time.perf_counter()
    worker_results={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.field_workers)as executor:
        futures=[]
        for worker_id,worker_fields in enumerate(assignments):
            cpus=available_cpus[worker_id*arguments.partitions_per_field:(worker_id+1)*arguments.partitions_per_field]
            futures.append(executor.submit(run_worker,worker_id,worker_fields,arguments.codec,rel_ebs,arguments.partitions_per_field,arguments.data_dir,metadata,cpus,output_dirs,log_dir,snapshot))
        for future in concurrent.futures.as_completed(futures):
            worker_id,results=future.result()
            worker_results[worker_id]=results
    execution_wall_time=time.perf_counter()-start
    for rel_eb in rel_ebs:
        output_dir=output_dirs[rel_eb]
        results=[]
        for field in fields:
            result_path=output_dir/f'{field}.json'
            if not result_path.is_file():raise RuntimeError(f'缺少field结果：rel_eb={rel_eb:g},field={field}')
            result=json.loads(result_path.read_text())
            if not result.get('valid'):raise RuntimeError(f'field结果无效：rel_eb={rel_eb:g},field={field}')
            results.append(result)
        with (output_dir/'fields.csv').open('w',newline='')as destination:
            fieldnames=sorted({key for result in results for key in result})
            writer=csv.DictWriter(destination,fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(json_safe(results))
        per_worker=[worker_results[worker_id][rel_eb]for worker_id in range(arguments.field_workers)]
        compression_worker_times=[sum(result['compression_time']for result in worker)for worker in per_worker]
        decompression_worker_times=[sum(result['decompression_time']for result in worker)for worker in per_worker]
        codec_compression_worker_times=[sum(result['codec_compression_time']for result in worker)for worker in per_worker]
        partition_worker_times=[sum(result['compression_partition_time']for result in worker)for worker in per_worker]
        codec_decompression_worker_times=[sum(result['codec_decompression_time']for result in worker)for worker in per_worker]
        merge_worker_times=[sum(result['decompression_merge_time']for result in worker)for worker in per_worker]
        input_worker_times=[sum(result.get('input_and_range_time',0)for result in worker)for worker in per_worker]
        verification_worker_times=[sum(result.get('verification_time',0)for result in worker)for worker in per_worker]
        helper_worker_times=[sum(result.get('compression_buffer_release_time',0)+result.get('decompression_buffer_allocation_time',0)+result.get('verification_time',0)+result.get('final_buffer_release_time',0)for result in worker)for worker in per_worker]
        eb_processing_worker_times=[sum(result.get('eb_processing_time_before_output',result.get('program_time_before_output',0))for result in worker)for worker in per_worker]
        execution_worker_times=[sum(result['worker_execution_wall_time']for result in worker)for worker in per_worker]
        compression_critical_worker=max(range(arguments.field_workers),key=lambda worker_id:compression_worker_times[worker_id])
        decompression_critical_worker=max(range(arguments.field_workers),key=lambda worker_id:decompression_worker_times[worker_id])
        summary={'codec':compressor_names[arguments.codec],'configuration':configuration,'field_workers':arguments.field_workers,'point_partitions_per_field':arguments.partitions_per_field,'active_cpus':required_cpus,'field_num':len(fields),'rel_eb':rel_eb,'rel_ebs_in_process':rel_ebs,'input_reused_across_rel_ebs':len(rel_ebs)>1,'value_num':sum(result['value_num']for result in results),'compressed_bytes':sum(result['compressed_bytes']for result in results),'br':8*sum(result['compressed_bytes']for result in results)/sum(result['value_num']for result in results),'reported_compression_time':compression_worker_times[compression_critical_worker],'reported_decompression_time':decompression_worker_times[decompression_critical_worker],'reported_codec_compression_time':codec_compression_worker_times[compression_critical_worker],'reported_partition_time':partition_worker_times[compression_critical_worker],'reported_codec_decompression_time':codec_decompression_worker_times[decompression_critical_worker],'reported_merge_time':merge_worker_times[decompression_critical_worker],'compression_critical_worker':compression_critical_worker,'decompression_critical_worker':decompression_critical_worker,'compression_worker_times':compression_worker_times,'decompression_worker_times':decompression_worker_times,'shared_input_worker_times':input_worker_times,'verification_worker_times':verification_worker_times,'helper_worker_times_excluding_shared_input':helper_worker_times,'eb_processing_worker_times':eb_processing_worker_times,'execution_worker_times':execution_worker_times,'all_ebs_execution_wall_time':execution_wall_time,'max_concurrent_rss_upper_bound_kib':sum(max(result['max_rss_kib']for result in worker)for worker in per_worker),'max_error_over_eb':max(result['max_error_over_eb']for result in results),'valid':all(result['valid']for result in results)}
        summary['reported_end_to_end_time']=summary['reported_compression_time']+summary['reported_decompression_time']
        write_json(output_dir/'summary.json',summary)
        print(json.dumps(json_safe(summary),ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':main()
