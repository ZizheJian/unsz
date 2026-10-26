import argparse,csv,hashlib,json,math,os,re,shlex,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
default_data_dir=Path('/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/tensor_data')

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def json_safe(value):
    if isinstance(value,float)and not math.isfinite(value):return None
    if isinstance(value,dict):return {key:json_safe(item)for key,item in value.items()}
    if isinstance(value,list):return [json_safe(item)for item in value]
    return value

def write_json(path,value):
    Path(path).write_text(json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def read_metadata(data_dir):
    metadata_path=data_dir/'tensor_metadata.json'
    metadata=json.loads(metadata_path.read_text())
    fields=metadata.get('fields',{})
    shape_counts={}
    for field,config in fields.items():
        shape=config.get('shape')
        if not isinstance(shape,list)or len(shape)!=2 or any(not isinstance(value,int)or value<=0 for value in shape):raise RuntimeError(f'无效的tensor shape：{field}')
        shape_counts[tuple(shape)]=shape_counts.get(tuple(shape),0)+1
    if len(fields)!=95 or shape_counts!={(12,1):58,(12,80):33,(12,81):4}:raise RuntimeError(f'完整Ocean-large metadata不匹配：fields={len(fields)},shapes={shape_counts}')
    return metadata_path,metadata

def field_paths(data_dir,field,shape):
    return [data_dir/f'{field}.dat.{slice_id}'for slice_id in range(math.prod(shape))]

def field_statistics(data_dir,field,shape):
    paths=field_paths(data_dir,field,shape)
    if not all(path.is_file()for path in paths):raise FileNotFoundError(f'输入切片不完整：{field}')
    byte_num=paths[0].stat().st_size
    if not byte_num or byte_num%4:raise RuntimeError(f'输入不是float32数组：{paths[0]}')
    point_num=byte_num//4
    minimum=math.inf
    maximum=-math.inf
    valid_value_num=0
    start=time.perf_counter()
    for path in paths:
        if path.stat().st_size!=byte_num:raise RuntimeError(f'切片大小不一致：{path}')
        values=np.memmap(path,dtype=np.float32,mode='r',shape=(point_num,))
        if not np.isfinite(values).all():raise RuntimeError(f'输入包含非有限值：{path}')
        ordinary=values[values!=-99999.0]
        if ordinary.size:
            minimum=min(minimum,float(ordinary.min()))
            maximum=max(maximum,float(ordinary.max()))
            valid_value_num+=int(ordinary.size)
    if not valid_value_num:raise RuntimeError(f'field没有普通值：{field}')
    return {'point_num':point_num,'value_num':point_num*len(paths),'valid_value_num':valid_value_num,'field_min':minimum,'field_max':maximum,'field_range':maximum-minimum,'scan_time':time.perf_counter()-start}

def run_process(command,log_path,environment=None,cpu_set=None):
    command=[str(value)for value in command]
    if cpu_set:command=['taskset','--cpu-list',cpu_set,*command]
    with Path(log_path).open('w')as log:return subprocess.run(command,cwd=project_root,stdout=log,stderr=subprocess.STDOUT,env=environment).returncode

def parse_unsz_field(arguments,field,shape,output_dir):
    statistics=field_statistics(arguments.data_dir,field,shape)
    abs_eb=arguments.rel_eb*statistics['field_range']
    config_dir=output_dir/'point_set_configs'
    log_dir=output_dir/'logs'
    stream_dir=getattr(arguments,'stream_root',output_dir/'streams')/field
    config_dir.mkdir(parents=True,exist_ok=True)
    log_dir.mkdir(parents=True,exist_ok=True)
    if stream_dir.exists():shutil.rmtree(stream_dir)
    stream_dir.mkdir(parents=True)
    labels=[{'method':'constant'}]if abs_eb==0 else [{'method':'predict','eb_mode':'ABS','eb_value':abs_eb,'condition':{'operator':'!=','value':-99999}},{'method':'constant'}]
    point_set_path=config_dir/f'{field}.json'
    write_json(point_set_path,{'version':3,'labels':labels})
    threads=getattr(arguments,'threads',1)
    common=['-point-set-config',point_set_path,'-data-dir',arguments.data_dir,'-field',field,'-output-dir',stream_dir,'-tensor-metadata',arguments.data_dir/'tensor_metadata.json','-coordinate-dim',3,'-coordinate-transform','auto','-partition-num',threads,'-thread-num',threads]
    compression_command=[arguments.compression_binary,*common]
    decompression_command=[arguments.decompression_binary,*common,'-skip-decompressed-output']
    environment={**os.environ,'OMP_NUM_THREADS':str(threads),'OMP_DYNAMIC':'false','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    compression_log=log_dir/f'{field}.compression.log'
    decompression_log=log_dir/f'{field}.decompression.log'
    if run_process(compression_command,compression_log,environment,getattr(arguments,'cpu_set',None)):raise RuntimeError(f'UNSZ压缩失败：{field}，见{compression_log}')
    compression_text=compression_log.read_text()
    byte_match=re.search(r'^total label stream bytes= (\d+), total value stream bytes= (\d+), total shared metadata bytes= (\d+)$',compression_text,re.MULTILINE)
    if byte_match is None:raise RuntimeError(f'UNSZ输出缺少精确码流字节数：{field}')
    compressed_bytes=sum(int(value)for value in byte_match.groups())
    disk_compressed_bytes=sum(path.stat().st_size for path in stream_dir.glob('*.unsz'))
    if compressed_bytes!=disk_compressed_bytes:raise RuntimeError(f'UNSZ内存和磁盘码流字节数不一致：{field}，{compressed_bytes}!={disk_compressed_bytes}')
    if run_process(decompression_command,decompression_log,environment,getattr(arguments,'cpu_set',None)):raise RuntimeError(f'UNSZ解压失败：{field}，见{decompression_log}')
    decompression_text=decompression_log.read_text()
    from py_codes.parse_unsz_output import parse_unsz_output
    parsed=parse_unsz_output(compression_text,decompression_text)
    error_match=re.search(r'^total squared error= ([^,]+), total value count= (\d+)$',compression_text,re.MULTILINE)
    decompression_error_match=re.search(r'^total squared error= ([^,]+), total value count= (\d+)$',decompression_text,re.MULTILINE)
    if error_match is None or decompression_error_match is None:raise RuntimeError(f'UNSZ缺少平方误差统计：{field}')
    compression_squared_error=float(error_match.group(1))
    squared_error=float(decompression_error_match.group(1))
    if int(error_match.group(2))!=statistics['value_num']or int(decompression_error_match.group(2))!=statistics['value_num']:raise RuntimeError(f'UNSZ误差统计的值数量不一致：{field}')
    if not math.isclose(compression_squared_error,squared_error,rel_tol=1e-10,abs_tol=1e-12):raise RuntimeError(f'UNSZ压缩和解压平方误差不一致：{field}')
    psnr=math.inf if squared_error==0 else 10*math.log10(statistics['field_range']**2/(squared_error/statistics['value_num']))
    result={'field':field,'compressor':'unsz','layout':'time-point'if shape[1]==1 else'time-height-point','time_num':shape[0],'height_num':shape[1],**statistics,'input_bytes':statistics['value_num']*4,'rel_eb':arguments.rel_eb,'codec_error_bound_mode':'ABS','codec_abs_eb':abs_eb,'compressed_bytes':compressed_bytes,'br':8*compressed_bytes/statistics['value_num'],'compression_squared_error':compression_squared_error,'squared_error':squared_error,'psnr':psnr,'thread_num':threads,'compression_preprocessing_time':parsed['compression_preprocessing_time'],'decompression_preprocessing_time':parsed['decompression_preprocessing_time'],'compression_time':parsed['compression_time'],'decompression_time':parsed['decompression_time'],'max_error_over_eb':parsed['max_error_over_eb'],'valid':parsed['max_error_over_eb']<=1.01,'selected_tensor_flow':parsed['selected_value_flow'],'coordinate_transform_enabled':parsed['coordinate_transform_enabled'],'command':'compression: '+shlex.join(str(value)for value in compression_command)+'; decompression: '+shlex.join(str(value)for value in decompression_command)}
    shutil.rmtree(stream_dir)
    return result

def run_tensor_field(arguments,field,shape,output_dir,binary):
    log_dir=output_dir/'logs'
    log_dir.mkdir(parents=True,exist_ok=True)
    result_path=output_dir/f'{field}.json'
    if result_path.exists():result_path.unlink()
    command=[binary,arguments.data_dir,field,shape[0],shape[1],arguments.rel_eb,result_path]
    if arguments.point_limit:command.append(arguments.point_limit)
    threads=getattr(arguments,'threads',1)
    returncode=run_process(command,log_dir/f'{field}.log',{**os.environ,'OMP_NUM_THREADS':str(threads),'OMP_DYNAMIC':'false','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},getattr(arguments,'cpu_set',None))
    if not result_path.is_file():raise RuntimeError(f'{arguments.codec}没有生成结果：{field}，returncode={returncode}')
    result=json.loads(result_path.read_text())
    result['compression_preprocessing_time']=0
    result['decompression_preprocessing_time']=0
    result['thread_num']=threads
    result['returncode']=returncode
    if returncode or not result['valid']:raise RuntimeError(f'{arguments.codec}误差检查失败：{field}，returncode={returncode}')
    return result

def aggregate(results,codec,rel_eb):
    input_bytes=sum(item['input_bytes']for item in results)
    value_num=sum(item['value_num']for item in results)
    compressed_bytes=sum(item['compressed_bytes']for item in results)
    normalized_squared_error=sum(item['squared_error']/(item['field_range']**2)for item in results if item['field_range']>0)
    normalized_value_num=sum(item['value_num']for item in results if item['field_range']>0)
    psnr=math.inf if normalized_squared_error==0 else-10*math.log10(normalized_squared_error/normalized_value_num)
    compression_time=sum(item['compression_time']for item in results)
    decompression_time=sum(item['decompression_time']for item in results)
    return {'codec':codec,'rel_eb':rel_eb,'field_num':len(results),'constant_field_num':sum(item['field_range']==0 for item in results),'value_num':value_num,'input_bytes':input_bytes,'compressed_bytes':compressed_bytes,'br':8*compressed_bytes/value_num,'normalized_psnr':psnr,'compression_preprocessing_time':sum(item['compression_preprocessing_time']for item in results),'decompression_preprocessing_time':sum(item['decompression_preprocessing_time']for item in results),'compression_time':compression_time,'decompression_time':decompression_time,'compression_MB_s':input_bytes/1e6/compression_time,'decompression_MB_s':input_bytes/1e6/decompression_time,'max_error_over_eb':max(item['max_error_over_eb']for item in results),'valid':all(item['valid']for item in results)}

def main():
    parser=argparse.ArgumentParser(description='Compress one or more Ocean-large fields with a field-relative error bound')
    parser.add_argument('--codec',required=True,choices=('unsz','sz3','qoz'))
    parser.add_argument('--data-dir',type=Path,default=default_data_dir)
    parser.add_argument('--rel-eb',type=float,default=1e-3)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--field',action='append')
    parser.add_argument('--output-root',type=Path,default=project_root/'results/ocean_large_field_relative')
    parser.add_argument('--point-limit',type=int)
    arguments=parser.parse_args()
    arguments.data_dir=arguments.data_dir.resolve()
    if not math.isfinite(arguments.rel_eb)or arguments.rel_eb<=0:parser.error('REL EB必须是有限正数')
    if arguments.threads<=0:parser.error('threads必须是正数')
    if arguments.codec in('sz3','qoz')and arguments.threads!=1:parser.error('未分区的SZ3/QoZ程序固定使用1个线程；多核实验请使用run_all_95_fields.py')
    if arguments.point_limit is not None and(arguments.point_limit<=0 or arguments.codec=='unsz'):parser.error('point-limit必须是正数且仅用于tensor compressor小测试')
    metadata_path,metadata=read_metadata(arguments.data_dir)
    fields=arguments.field or list(metadata['fields'])
    if any(field not in metadata['fields']for field in fields):parser.error('指定field不在tensor metadata中')
    suffix=f'rel_{arguments.rel_eb:g}_t{arguments.threads}'+(f'_n{arguments.point_limit}'if arguments.point_limit else'')
    output_dir=(arguments.output_root/arguments.codec/suffix).resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    binaries={'s1_compress':project_root/'install/bin/s1_compress','s1_decompress':project_root/'install/bin/s1_decompress'}if arguments.codec=='unsz'else{f'compress_one_field_with_{arguments.codec}':project_root/'build/field_relative'/f'compress_one_field_with_{arguments.codec}'}
    missing_binaries=[str(path)for path in binaries.values()if not path.is_file()]
    if missing_binaries:raise FileNotFoundError('缺少可执行文件：'+','.join(missing_binaries))
    sources=[Path(__file__).resolve()]
    if arguments.codec in('sz3','qoz'):sources.append(Path(__file__).with_name('compress_one_field.cpp'))
    upstream_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root/'third_party'/('QoZ'if arguments.codec=='qoz'else'SZ3'),text=True).strip()if arguments.codec!='unsz'else subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip()
    manifest={'codec':arguments.codec,'rel_eb':arguments.rel_eb,'codec_error_bound_mode':'ABS','abs_eb_rule':'rel_eb*(field_max-field_min),excluding -99999','layout':'[time,point] when height=1; [time,height,point] otherwise; point is fastest','program_threads':arguments.threads,'point_limit':arguments.point_limit,'fields':fields,'metadata_sha256':sha256(metadata_path),'binary_sha256':{name:sha256(path)for name,path in binaries.items()},'upstream_revision':upstream_revision,'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip(),'source_sha256':{str(path.relative_to(project_root)):sha256(path)for path in sources}}
    manifest_path=output_dir/'manifest.json'
    if manifest_path.exists()and json.loads(manifest_path.read_text())!=manifest:raise RuntimeError(f'已有结果manifest不同：{manifest_path}')
    write_json(manifest_path,manifest)
    snapshots={}
    for name,binary in binaries.items():
        snapshot=output_dir/name
        if not snapshot.exists():shutil.copy2(binary,snapshot)
        if sha256(snapshot)!=manifest['binary_sha256'][name]:raise RuntimeError('保存的二进制SHA不一致：'+name)
        snapshots[name]=snapshot
    environment={'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_nodelist':os.environ.get('SLURM_NODELIST'),'slurm_cpus':int(os.environ['SLURM_CPUS_PER_TASK'])if os.environ.get('SLURM_CPUS_PER_TASK')else None,'hostname':os.uname().nodename}
    write_json(output_dir/f'environment_{os.environ.get("SLURM_JOB_ID",str(os.getpid()))}.json',environment)
    if arguments.codec=='unsz':
        arguments.compression_binary=snapshots['s1_compress']
        arguments.decompression_binary=snapshots['s1_decompress']
    else:arguments.tensor_binary=snapshots[f'compress_one_field_with_{arguments.codec}']
    results=[]
    for field in fields:
        shape=metadata['fields'][field]['shape']
        result_path=output_dir/f'{field}.json'
        if result_path.exists():
            result=json.loads(result_path.read_text())
            if result.get('rel_eb')==arguments.rel_eb and result.get('time_num')==shape[0]and result.get('height_num')==shape[1]and result.get('valid'):
                results.append(result)
                print(f'跳过已完成field：{field}',flush=True)
                continue
        print(f'BEGIN codec={arguments.codec},field={field},shape={shape},REL={arguments.rel_eb:g}',flush=True)
        result=parse_unsz_field(arguments,field,shape,output_dir)if arguments.codec=='unsz'else run_tensor_field(arguments,field,shape,output_dir,arguments.tensor_binary)
        write_json(result_path,result)
        results.append(result)
        with (output_dir/'fields.csv').open('w',newline='')as destination:
            fieldnames=sorted({key for item in results for key in item if key!='command'})
            writer=csv.DictWriter(destination,fieldnames=fieldnames,extrasaction='ignore')
            writer.writeheader()
            writer.writerows(json_safe(results))
        print(f'END codec={arguments.codec},field={field},br={result["br"]:.6g},psnr={result["psnr"]},max_error_over_eb={result["max_error_over_eb"]:.9g}',flush=True)
    summary=aggregate(results,arguments.codec,arguments.rel_eb)
    write_json(output_dir/'summary.json',summary)
    print(json.dumps(json_safe(summary),ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':main()
