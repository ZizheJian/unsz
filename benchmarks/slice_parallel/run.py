import argparse,csv,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import check_cpus,default_data_dir,field_slice_num,git_revision,project_root,quality_columns,quality_summary,read_metadata,sha256,shape_name,slurm_environment,statistics_version,worker_times,write_fields_csv,write_json

# 每个slice作为一维数组独立压缩；slice在worker间轮流分配，每个slice内部再使用压缩器自身的OpenMP
codecs={'sz3':{'name':'sz3-1d','binary':'build/tensor_1d/benchmark_sz3','upstream':'third_party/SZ3','result_dir':'sz3_1d'},'pfpl':{'name':'pfpl','binary':'build/tensor_1d/benchmark_pfpl','upstream':'third_party/PFPL','result_dir':'pfpl'}}

def list_slices(data_dir,metadata,fields,max_slices):
    slices=[]
    for field in fields:
        slice_num=field_slice_num(metadata,field)if max_slices is None else min(field_slice_num(metadata,field),max_slices)
        for slice_id in range(slice_num):slices.append({'field':field,'slice_id':slice_id,'path':data_dir/f'{field}.dat.{slice_id}'})
    return slices

def run_workers(assignments,threads_per_slice,binary,rel_eb,worker_dir,available_cpus):
    environment={**os.environ,'OMP_NUM_THREADS':str(threads_per_slice),'OMP_THREAD_LIMIT':str(threads_per_slice),'OMP_DYNAMIC':'false','OMP_PROC_BIND':'close','OMP_PLACES':'cores','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    processes=[]
    start=time.perf_counter()
    for worker_id,assigned in enumerate(assignments):
        input_list=worker_dir/f'worker_{worker_id:03d}.inputs.txt'
        input_list.write_text(''.join(str(item['path'])+'\n'for item in assigned))
        cpu_set=','.join(str(cpu)for cpu in available_cpus[worker_id*threads_per_slice:(worker_id+1)*threads_per_slice])
        log=(worker_dir/f'worker_{worker_id:03d}.log').open('w')
        command=['taskset','--cpu-list',cpu_set,str(binary),str(input_list),str(rel_eb),str(threads_per_slice),str(worker_dir/f'worker_{worker_id:03d}.slices.csv')]
        processes.append((worker_id,subprocess.Popen(command,cwd=project_root,stdout=log,stderr=subprocess.STDOUT,env=environment),log))
    failures=[]
    for worker_id,process,log in processes:
        returncode=process.wait()
        log.close()
        if returncode:failures.append((worker_id,returncode))
    if failures:raise RuntimeError(f'worker失败：{failures}，见{worker_dir}')
    return time.perf_counter()-start

def summarize(codec,assignments,worker_dir,metadata,fields,threads_per_slice,rel_eb):
    field_slices={field:[]for field in fields}
    compression_times,decompression_times=[],[]
    for worker_id,assigned in enumerate(assignments):
        with(worker_dir/f'worker_{worker_id:03d}.slices.csv').open(newline='')as source:rows=list(csv.DictReader(source))
        if len(rows)!=len(assigned):raise RuntimeError(f'worker {worker_id}结果数量错误：{len(rows)}!={len(assigned)}')
        for local_slice_id,(item,row)in enumerate(zip(assigned,rows)):
            if int(row['slice_id'])!=local_slice_id:raise RuntimeError(f'worker {worker_id}的slice顺序错误')
            row['worker_id']=worker_id
            field_slices[item['field']].append(row)
        compression_times.append(sum(float(row['compression_time'])for row in rows))
        decompression_times.append(sum(float(row['decompression_time'])for row in rows))
    field_rows=[]
    for field in fields:
        rows=field_slices[field]
        # per-slice协议：每个slice的误差除以该slice普通值range的平方
        normalized_squared_error=normalized_squared_error_all_points=0.0
        for row in rows:
            slice_range=float(row['valid_max'])-float(row['valid_min'])if int(row['valid_point_num'])else 0.0
            if slice_range>0:
                normalized_squared_error+=float(row['valid_squared_error'])/slice_range**2
                normalized_squared_error_all_points+=float(row['squared_error'])/slice_range**2
            elif float(row['valid_squared_error'])!=0:raise RuntimeError(f'普通值为常量的slice未被精确恢复：{field}')
        value_num=sum(int(row['point_num'])for row in rows)
        compressed_bytes=sum(int(row['compressed_bytes'])for row in rows)
        field_row={'field':field,'shape':shape_name(metadata,field),'slice_num':len(rows),'value_num':value_num,'compressed_bytes':compressed_bytes,'br':8*compressed_bytes/value_num,**quality_columns(normalized_squared_error,sum(int(row['valid_point_num'])for row in rows),normalized_squared_error_all_points,value_num),'compression_preprocessing_time':0,'compression_time':sum(float(row['compression_time'])for row in rows),'decompression_preprocessing_time':0,'decompression_time':sum(float(row['decompression_time'])for row in rows),'max_error_over_eb':max(float(row['max_error_over_eb'])for row in rows),'valid':all(row['valid']=='1'for row in rows),'worker_id':None,'zero_eb_slice_num':sum(int(row['zero_eb'])for row in rows)}
        field_rows.append(field_row)
    summary={'codec':codecs[codec]['name'],'protocol':'slice','statistics_version':statistics_version,'rel_eb':rel_eb,'slice_workers':len(assignments),'threads_per_slice':threads_per_slice,'cpu_num':len(assignments)*threads_per_slice,'field_num':len(fields),'slice_num':sum(row['slice_num']for row in field_rows),'compressed_bytes':sum(row['compressed_bytes']for row in field_rows),**quality_summary(field_rows),**worker_times(compression_times,decompression_times),'worker_slice_counts':[len(assigned)for assigned in assignments],'max_error_over_eb':max(row['max_error_over_eb']for row in field_rows),'valid':all(row['valid']for row in field_rows)}
    return field_rows,summary

def main():
    parser=argparse.ArgumentParser(description='Run SZ3-1D or PFPL on Ocean-large slices with slice workers and threads inside each slice')
    parser.add_argument('--codec',required=True,choices=tuple(codecs))
    parser.add_argument('--data-dir',type=Path,default=default_data_dir)
    parser.add_argument('--rel-eb',type=float,action='append',required=True)
    parser.add_argument('--slice-workers',type=int,required=True)
    parser.add_argument('--threads-per-slice',type=int,required=True)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--dir-suffix',default='')
    parser.add_argument('--field',action='append')
    parser.add_argument('--max-slices',type=int)
    parser.add_argument('--summarize-only',action='store_true')
    arguments=parser.parse_args()
    for value,name in((arguments.slice_workers,'slice-workers'),(arguments.threads_per_slice,'threads-per-slice')):
        if value<=0 or value&(value-1):parser.error(f'{name}必须是2的幂')
    if any(not math.isfinite(rel_eb)or rel_eb<=0 for rel_eb in arguments.rel_eb)or len(set(arguments.rel_eb))!=len(arguments.rel_eb):parser.error('rel-eb必须是不重复的有限正数')
    data_dir=arguments.data_dir.resolve()
    metadata_path,metadata,fields=read_metadata(data_dir,arguments.field)
    formal_complete_dataset=not arguments.field and arguments.max_slices is None
    slices=list_slices(data_dir,metadata,fields,arguments.max_slices)
    cpu_num=arguments.slice_workers*arguments.threads_per_slice
    available_cpus=None if arguments.summarize_only else check_cpus(cpu_num,formal_complete_dataset)
    binary=(project_root/codecs[arguments.codec]['binary']).resolve()
    configuration=f'{arguments.slice_workers}_slice_workers_{arguments.threads_per_slice}_threads_per_slice'
    output_root=arguments.output_root or project_root/f'results/ocean_large_per_slice_rd_{cpu_num}{arguments.dir_suffix}'/codecs[arguments.codec]['result_dir']
    assignments=[slices[worker_id::arguments.slice_workers]for worker_id in range(arguments.slice_workers)]
    for rel_eb in arguments.rel_eb:
        output_dir=(output_root/configuration/f'rel_{rel_eb:g}').resolve()
        worker_dir=output_dir/'workers'
        if not arguments.summarize_only and(output_dir/'summary.json').is_file():
            previous=json.loads((output_dir/'summary.json').read_text())
            if previous.get('statistics_version')==statistics_version and previous.get('codec')==codecs[arguments.codec]['name']and previous.get('rel_eb')==rel_eb and previous.get('field_num')==len(fields)and previous.get('slice_num')==len(slices)and previous.get('valid'):
                print(f'SKIP completed REL={rel_eb:g},configuration={configuration}',flush=True)
                continue
        if arguments.summarize_only:
            binary_sha256=json.loads((output_dir/'manifest.json').read_text())['binary_sha256']
        else:
            worker_dir.mkdir(parents=True,exist_ok=True)
            snapshot=output_dir/binary.name
            shutil.copy2(binary,snapshot)
            binary_sha256=sha256(snapshot)
            write_json(output_dir/'manifest.json',{'codec':codecs[arguments.codec]['name'],'protocol':'slice','configuration':configuration,'slice_workers':arguments.slice_workers,'threads_per_slice':arguments.threads_per_slice,'cpu_num':cpu_num,'assignment_rule':'global slice order round-robin; all slices have the same size','field_num':len(fields),'slice_num':len(slices),'rel_eb':rel_eb,'error_bound_rule':'REL*(slice valid max-slice valid min), excluding -99999','binary_sha256':binary_sha256,'metadata_sha256':sha256(metadata_path),'source_sha256':{str(path.relative_to(project_root)):sha256(path)for path in(Path(__file__).resolve(),project_root/'benchmarks/common.py',project_root/'benchmarks/tensor_1d/benchmark.cpp')},'git_revision':git_revision(),'upstream_revision':git_revision(project_root/codecs[arguments.codec]['upstream']),**slurm_environment()})
            print(f'BEGIN codec={arguments.codec},REL={rel_eb:g},configuration={configuration}',flush=True)
            execution_wall_time=run_workers(assignments,arguments.threads_per_slice,snapshot,rel_eb,worker_dir,available_cpus)
        field_rows,summary=summarize(arguments.codec,assignments,worker_dir,metadata,fields,arguments.threads_per_slice,rel_eb)
        if not arguments.summarize_only:summary['execution_wall_time']=execution_wall_time
        elif(output_dir/'summary.json').is_file():summary['execution_wall_time']=json.loads((output_dir/'summary.json').read_text()).get('execution_wall_time')
        write_fields_csv(output_dir/'fields.csv',field_rows)
        write_json(output_dir/'summary.json',summary)
        print(json.dumps({key:value for key,value in summary.items()if not isinstance(value,(list,dict))},ensure_ascii=False),flush=True)
        if not summary['valid']:raise RuntimeError('误差检查失败')

if __name__=='__main__':main()
