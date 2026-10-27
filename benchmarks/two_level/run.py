import argparse,json,math,os,re,shlex,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import check_cpus,default_data_dir,field_range_quality,field_slice_num,git_revision,ocean_large_slice_num,optimal_assignment,project_root,quality_columns,quality_summary,read_metadata,rel_eb_tags,sha256,shape_name,slurm_environment,statistics_version,worker_times,write_fields_csv,write_json

benchmark_dir=Path(__file__).resolve().parent
point_set_modes={'slice':'REL','field':'FIELD_REL'}
result_pattern=re.compile(r'^UNSZ FIELD RESULT side=(\w+) field=(\S+) slices=(\d+) values=(\d+) valid_values=(\d+) compressed_bytes=(\d+) squared_error=(\S+) valid_squared_error=(\S+) slice_normalized_valid_squared_error=(\S+) slice_normalized_squared_error=(\S+) field_min=(\S+) field_max=(\S+) preprocessing_time=(\S+) codec_time=(\S+) max_error_over_eb=(\S+)$',re.MULTILINE)

def point_set_config_path(protocol,rel_eb):
    # slice协议按每个slice的range换算REL，field协议按整个field的range换算FIELD_REL
    path=project_root/f'point_set_configs/Ocean_large_monthly_rbf_{"field_rel"if protocol=="field"else"rel"}_{rel_eb_tags[rel_eb]}.json'
    expected={'version':3,'labels':[{'method':'predict','eb_mode':point_set_modes[protocol],'eb_value':rel_eb,'condition':{'operator':'!=','value':-99999}},{'method':'constant'}]}
    if json.loads(path.read_text())!=expected:raise RuntimeError(f'Ocean-large {protocol}协议REL {rel_eb:g}点集配置与正式规则不一致')
    return path

def build_command(binary,data_dir,stream_dir,metadata_path,point_set_path,fields,worker_ids,field_worker_num,thread_num,decompress):
    command=[str(binary),'-point-set-config',str(point_set_path),'-data-dir',str(data_dir),'-output-dir',str(stream_dir),'-field-list','{'+','.join(fields)+'}','-tensor-metadata',str(metadata_path),'-coordinate-dim','3','-partition-num',str(thread_num),'-thread-num',str(thread_num)]
    if field_worker_num>1:command+=['-field-worker-num',str(field_worker_num),'-field-worker-map','{'+','.join(str(worker_id)for worker_id in worker_ids)+'}']
    if decompress:command.append('-skip-decompressed-output')
    return command

def run_logged(command,log_path):
    with log_path.open('w',encoding='utf-8')as log:subprocess.run(command,cwd=project_root,stdout=log,stderr=subprocess.STDOUT,check=True)

def read_field_results(text,fields,side):
    results={}
    for match in result_pattern.finditer(text):
        if match.group(1)!=side:continue
        field=match.group(2)
        if field in results:raise RuntimeError(f'{side}重复输出field结果：{field}')
        results[field]={'slice_num':int(match.group(3)),'value_num':int(match.group(4)),'valid_value_num':int(match.group(5)),'compressed_bytes':int(match.group(6)),'squared_error':float(match.group(7)),'valid_squared_error':float(match.group(8)),'slice_normalized_valid_squared_error':float(match.group(9)),'slice_normalized_squared_error':float(match.group(10)),'field_min':float(match.group(11)),'field_max':float(match.group(12)),'preprocessing_time':float(match.group(13)),'codec_time':float(match.group(14)),'max_error_over_eb':float(match.group(15))}
    if set(results)!=set(fields):raise RuntimeError(f'{side}没有完整输出全部{len(fields)}个field的结果：{len(results)}')
    if max(result['max_error_over_eb']for result in results.values())>1.01:raise RuntimeError(f'{side}误差验证失败')
    return results

def shared_plan_time(text):
    # 两级并行时依赖计划在worker启动前统一生成一次；单worker时由worker自己生成并已计入field外的预处理
    matches=re.findall(r'^two-level shared dependency-plan preprocessing time= ([^ ]+) s$',text,re.MULTILINE)
    if len(matches)>1:raise RuntimeError('共享依赖计划时间重复输出')
    if matches:return float(matches[0])
    matches=re.findall(r'^temporal dependency plan preprocessing time= ([^ ]+) s$',text,re.MULTILINE)
    if len(matches)!=1:raise RuntimeError('缺少依赖计划预处理时间')
    return float(matches[0])

def main():
    parser=argparse.ArgumentParser(description='Run UNSZ on Ocean-large with field workers and point partitions')
    parser.add_argument('--data-dir',type=Path,default=default_data_dir)
    parser.add_argument('--protocol',choices=tuple(point_set_modes),default='slice')
    parser.add_argument('--rel-eb',type=float,choices=tuple(rel_eb_tags),default=1e-3)
    parser.add_argument('--field-worker-num',type=int,default=16)
    parser.add_argument('--cpu-num',type=int,default=128)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--launcher',type=Path)
    parser.add_argument('--field',action='append')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--keep-output',action='store_true')
    arguments=parser.parse_args()
    field_worker_num=arguments.field_worker_num
    if field_worker_num<=0 or field_worker_num&(field_worker_num-1)or arguments.cpu_num%field_worker_num:parser.error('field-worker-num必须是2的幂并整除cpu-num')
    thread_num=arguments.cpu_num//field_worker_num
    rel_eb=arguments.rel_eb
    data_dir=arguments.data_dir.resolve()
    metadata_path,metadata,fields=read_metadata(data_dir,arguments.field)
    formal_complete_dataset=not arguments.field
    loads={field:field_slice_num(metadata,field)for field in fields}
    if formal_complete_dataset and sum(loads.values())!=ocean_large_slice_num:raise RuntimeError('完整Ocean-large slice数量不匹配')
    assignments,worker_loads,optimum,lower_bound=optimal_assignment(loads,field_worker_num)
    fields=[field for worker_fields in assignments for field in worker_fields]
    worker_ids=[worker_id for worker_id,worker_fields in enumerate(assignments)for _ in worker_fields]
    point_set_path=point_set_config_path(arguments.protocol,rel_eb)
    point_num=(data_dir/'coords.dat').stat().st_size//12
    binaries=[project_root/'install/bin/s1_compress',project_root/'install/bin/s1_decompress']
    if any(not path.is_file()for path in binaries):raise RuntimeError('缺少已安装的UNSZ二进制文件')
    run_id=os.environ.get('SLURM_JOB_ID',f'local_{os.getpid()}')
    stream_root=data_dir.parent/'two_level_outputs'
    stream_dir=stream_root/f'job_{run_id}'
    default_output_root=Path(f'results/ocean_large_unsz_{arguments.cpu_num}cpu/{arguments.protocol}_protocol/{field_worker_num}_field_workers_{thread_num}_partitions/rel_{rel_eb:g}')
    output_dir=(project_root/(arguments.output_root or default_output_root)).resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    compression_command=build_command(binaries[0],data_dir,stream_dir,metadata_path,point_set_path,fields,worker_ids,field_worker_num,thread_num,False)
    decompression_command=build_command(binaries[1],data_dir,stream_dir,metadata_path,point_set_path,fields,worker_ids,field_worker_num,thread_num,True)
    sources=[Path(__file__).resolve(),project_root/'benchmarks/common.py',project_root/'c_codes/compress_cli.cpp',project_root/'c_codes/s1_decompress.cpp']
    if arguments.launcher:sources.append(arguments.launcher.resolve())
    manifest={'codec':'unsz','protocol':arguments.protocol,'rel_eb':rel_eb,'field_worker_num':field_worker_num,'thread_num':thread_num,'partition_num':thread_num,'cpu_num':arguments.cpu_num,'field_num':len(fields),'slice_num':sum(loads.values()),'point_num':point_num,'assignment_rule':'exact minimum of the maximum worker slice load, then of the maximum worker field count','worker_slice_loads':worker_loads,'worker_slice_load_lower_bound':lower_bound,'worker_fields':assignments,'metadata_sha256':sha256(metadata_path),'point_set_config':str(point_set_path.relative_to(project_root)),'point_set_config_sha256':sha256(point_set_path),'binary_sha256':{path.name:sha256(path)for path in binaries},'git_revision':git_revision(),'source_sha256':{str(path.relative_to(project_root)):sha256(path)for path in sources},'commands':{'compression':shlex.join(compression_command),'decompression':shlex.join(decompression_command)},**slurm_environment()}
    write_json(output_dir/'manifest.json',manifest)
    if arguments.prepare_only:
        print(json.dumps({'output_dir':str(output_dir),'worker_slice_loads':worker_loads,'optimum':optimum,'max_fields_per_worker':max(len(worker_fields)for worker_fields in assignments)},ensure_ascii=False),flush=True)
        return
    check_cpus(arguments.cpu_num,formal_complete_dataset)
    if stream_dir.exists():raise RuntimeError(f'临时输出目录已经存在：{stream_dir}')
    stream_dir.mkdir(parents=True)
    compression_log=output_dir/'compression.log'
    decompression_log=output_dir/'decompression.log'
    print(f'BEGIN UNSZ {arguments.protocol} protocol REL={rel_eb:g},workers={field_worker_num},partitions={thread_num},output={stream_dir}',flush=True)
    run_logged(compression_command,compression_log)
    compressed_bytes=sum(path.stat().st_size for path in stream_dir.glob('*.unsz'))
    run_logged(decompression_command,decompression_log)
    compression_text=compression_log.read_text()
    decompression_text=decompression_log.read_text()
    compression_results=read_field_results(compression_text,fields,'compression')
    decompression_results=read_field_results(decompression_text,fields,'decompression')
    if sum(result['compressed_bytes']for result in compression_results.values())!=compressed_bytes:raise RuntimeError('UNSZ报告的码流字节数与磁盘文件不一致')
    rows=[]
    for worker_id,worker_fields in enumerate(assignments):
        for field in worker_fields:
            compression,decompression=compression_results[field],decompression_results[field]
            if compression['compressed_bytes']!=decompression['compressed_bytes']or not math.isclose(compression['squared_error'],decompression['squared_error'],rel_tol=1e-9,abs_tol=1e-12):raise RuntimeError(f'UNSZ压缩端和解压端结果不一致：{field}')
            field_range=compression['field_max']-compression['field_min']if math.isfinite(compression['field_max'])else 0.0
            if arguments.protocol=='slice':quality=quality_columns(decompression['slice_normalized_valid_squared_error'],decompression['valid_value_num'],decompression['slice_normalized_squared_error'],decompression['value_num'])
            else:quality=field_range_quality(decompression['squared_error'],decompression['valid_squared_error'],decompression['valid_value_num'],decompression['value_num'],field_range)
            rows.append({'field':field,'shape':shape_name(metadata,field),'slice_num':compression['slice_num'],'value_num':compression['value_num'],'compressed_bytes':compression['compressed_bytes'],'br':8*compression['compressed_bytes']/compression['value_num'],**quality,'field_range':field_range,'squared_error':decompression['squared_error'],'compression_preprocessing_time':compression['preprocessing_time'],'compression_time':compression['codec_time'],'decompression_preprocessing_time':decompression['preprocessing_time'],'decompression_time':decompression['codec_time'],'max_error_over_eb':max(compression['max_error_over_eb'],decompression['max_error_over_eb']),'valid':True,'worker_id':worker_id})
    write_fields_csv(output_dir/'fields.csv',rows)
    times=worker_times([sum(row['compression_preprocessing_time']+row['compression_time']for row in rows if row['worker_id']==worker_id)for worker_id in range(field_worker_num)],[sum(row['decompression_preprocessing_time']+row['decompression_time']for row in rows if row['worker_id']==worker_id)for worker_id in range(field_worker_num)])
    compression_plan_time,decompression_plan_time=shared_plan_time(compression_text),shared_plan_time(decompression_text)
    times['reported_compression_time']+=compression_plan_time
    times['reported_decompression_time']+=decompression_plan_time
    times['total_codec_time']=times['reported_compression_time']+times['reported_decompression_time']
    wall_times={}
    for side,text in(('compression',compression_text),('decompression',decompression_text)):
        matches=re.findall(rf'^two-level {side} wall time= ([^ ]+) s$',text,re.MULTILINE)
        wall_times[f'{side}_wall_time']=float(matches[0])if matches else None
    summary={'codec':'unsz','protocol':arguments.protocol,'statistics_version':statistics_version,'rel_eb':rel_eb,'cpu_num':arguments.cpu_num,'field_worker_num':field_worker_num,'thread_num':thread_num,'partition_num':thread_num,'field_num':len(fields),'slice_num':sum(loads.values()),'compressed_bytes':compressed_bytes,**quality_summary(rows),'shared_dependency_plan_time':{'compression':compression_plan_time,'decompression':decompression_plan_time},**times,**wall_times,'max_error_over_eb':max(row['max_error_over_eb']for row in rows),'shared_dag_compression_reuse_count':compression_text.count('Reusing shared spatial prediction DAG and weights'),'shared_dag_decompression_reuse_count':decompression_text.count('Reusing shared spatial prediction DAG and weights'),'valid':True}
    write_json(output_dir/'summary.json',summary)
    if not arguments.keep_output:
        if stream_dir.parent!=stream_root:raise RuntimeError(f'拒绝清理意外目录：{stream_dir}')
        shutil.rmtree(stream_dir)
    print(json.dumps({key:value for key,value in summary.items()if not isinstance(value,(list,dict))},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
