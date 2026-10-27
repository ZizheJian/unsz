import csv,functools,hashlib,itertools,json,math,os,subprocess
from pathlib import Path

project_root=Path(__file__).resolve().parents[1]
default_data_dir=Path('/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/tensor_data')
ocean_large_shape_counts={(12,1):58,(12,80):33,(12,81):4}
ocean_large_slice_num=36264
rel_eb_tags={1e-1:'1e-1',1e-2:'1e-2',1e-3:'1e-3',1e-4:'1e-4',1e-5:'1e-5'}
fields_csv_columns=['field','shape','slice_num','value_num','valid_value_num','compressed_bytes','br','normalized_squared_error','nmse','psnr','normalized_squared_error_all_points','nmse_all_points','psnr_all_points','compression_preprocessing_time','compression_time','decompression_preprocessing_time','decompression_time','max_error_over_eb','valid','worker_id']
statistics_version=2

def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb')as source:
        for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def git_revision(path=project_root):
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip()

def json_safe(value):
    if isinstance(value,float)and not math.isfinite(value):return None
    if isinstance(value,dict):return {key:json_safe(item)for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(item)for item in value]
    return value

def write_json(path,value):
    Path(path).write_text(json.dumps(json_safe(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def read_metadata(data_dir,selected_fields=None):
    # 返回metadata路径、metadata和参与实验的field；不指定field时要求完整Ocean-large
    metadata_path=Path(data_dir)/'tensor_metadata.json'
    metadata=json.loads(metadata_path.read_text())
    all_fields=list(metadata.get('fields',{}))
    shape_counts={}
    for field in all_fields:
        shape=metadata['fields'][field].get('shape')
        if not isinstance(shape,list)or len(shape)!=2 or any(not isinstance(value,int)or value<=0 for value in shape):raise RuntimeError(f'无效的tensor shape：{field}')
        shape_counts[tuple(shape)]=shape_counts.get(tuple(shape),0)+1
    if len(all_fields)!=95 or shape_counts!=ocean_large_shape_counts:raise RuntimeError(f'完整Ocean-large metadata不匹配：fields={len(all_fields)},shapes={shape_counts}')
    fields=list(selected_fields)if selected_fields else all_fields
    if any(field not in metadata['fields']for field in fields):raise RuntimeError('指定field不在tensor metadata中')
    return metadata_path,metadata,fields

def field_slice_num(metadata,field):
    return math.prod(metadata['fields'][field]['shape'])

def shape_name(metadata,field):
    return 'x'.join(str(value)for value in metadata['fields'][field]['shape'])

def check_cpus(required_cpus,formal_complete_dataset):
    # 正式实验要求Slurm申请的CPU数恰好等于使用的CPU数，并返回可绑定的CPU编号
    if formal_complete_dataset and 'SLURM_JOB_ID'not in os.environ:raise RuntimeError('完整Ocean-large实验必须在Slurm作业中运行')
    if 'SLURM_JOB_ID'in os.environ and int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=required_cpus:raise RuntimeError(f'Slurm CPU数量必须等于{required_cpus}')
    available_cpus=sorted(os.sched_getaffinity(0))
    if len(available_cpus)<required_cpus:raise RuntimeError(f'需要{required_cpus}个CPU，当前affinity只有{len(available_cpus)}个')
    return available_cpus

def slurm_environment():
    return {'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'slurm_nodelist':os.environ.get('SLURM_NODELIST'),'slurm_cpus':int(os.environ['SLURM_CPUS_PER_TASK'])if os.environ.get('SLURM_CPUS_PER_TASK')else None,'hostname':os.uname().nodename}

def max_load_lower_bound(loads,worker_num):
    total=sum(loads)
    bound=max(-(-total//worker_num),max(loads))
    for threshold in set(loads):
        bound=max(bound,-(-sum(load>=threshold for load in loads)//worker_num)*threshold)
    return bound

def optimal_assignment(field_loads,worker_num):
    # 精确求解：把完整field分给worker_num个worker，首先使最大slice负载最小，其次使单个worker的最大field数最小
    # 第二个目标避免把大量小field堆到少数worker上，因为每个field还有与slice数无关的固定开销
    # 负载只有少数几种取值，因此对每组上限用“各种负载剩余数量”作状态做动态规划，求最少需要几个worker
    # 最小负载的field总是尽量填满当前worker，这不会增加所需worker数，可大幅减少分支
    if worker_num<=0:raise RuntimeError('worker数量必须为正数')
    sizes=sorted(set(field_loads.values()),reverse=True)
    counts=tuple(sum(load==size for load in field_loads.values())for size in sizes)
    smallest=sizes[-1]
    def plan_for(target,field_cap):
        @functools.lru_cache(maxsize=None)
        def solve(remaining):
            if not any(remaining):return 0,None
            best=(math.inf,None)
            for large in itertools.product(*(range(min(count,target//size,field_cap)+1)for size,count in zip(sizes[:-1],remaining[:-1]))):
                used=sum(count*size for count,size in zip(large,sizes))
                if used>target or sum(large)>field_cap:continue
                config=(*large,min(remaining[-1],(target-used)//smallest,field_cap-sum(large)))
                if not any(config):continue
                bin_num=solve(tuple(left-taken for left,taken in zip(remaining,config)))[0]+1
                if bin_num<best[0]:best=(bin_num,config)
            return best
        if solve(counts)[0]>worker_num:return None
        configs=[]
        remaining=counts
        while any(remaining):
            config=solve(remaining)[1]
            configs.append(config)
            remaining=tuple(left-taken for left,taken in zip(remaining,config))
        return configs
    def smallest_feasible(candidates,feasible):
        low,high=0,len(candidates)-1
        while low<high:
            middle=(low+high)//2
            if feasible(candidates[middle]):high=middle
            else:low=middle+1
        return candidates[low]
    field_num=len(field_loads)
    lower_bound=max_load_lower_bound(list(field_loads.values()),worker_num)
    candidates=sorted({sum(count*size for count,size in zip(combination,sizes))for combination in itertools.product(*(range(count+1)for count in counts))})
    optimum=smallest_feasible([target for target in candidates if target>=lower_bound],lambda target:plan_for(target,field_num)is not None)
    field_cap=smallest_feasible(list(range(-(-field_num//worker_num),field_num+1)),lambda cap:plan_for(optimum,cap)is not None)
    configs=sorted(plan_for(optimum,field_cap),key=lambda config:(-sum(count*size for count,size in zip(config,sizes)),[-count for count in config]))
    names={size:sorted(field for field,load in field_loads.items()if load==size)for size in sizes}
    assignments=[]
    for config in configs:
        worker_fields=[]
        for size,count in zip(sizes,config):
            worker_fields.extend(names[size][:count])
            names[size]=names[size][count:]
        assignments.append(worker_fields)
    assignments.extend([]for _ in range(worker_num-len(assignments)))
    loads=[sum(field_loads[field]for field in worker_fields)for worker_fields in assignments]
    if sorted(field for worker_fields in assignments for field in worker_fields)!=sorted(field_loads)or max(loads)!=optimum:raise RuntimeError('field分配结果不完整')
    return assignments,loads,optimum,lower_bound

def worker_times(compression_worker_times,decompression_worker_times):
    # 压缩端和解压端分别取worker累计时间的最大值，两端相加为total codec time
    return {'compression_worker_times':list(compression_worker_times),'decompression_worker_times':list(decompression_worker_times),'reported_compression_time':max(compression_worker_times),'reported_decompression_time':max(decompression_worker_times),'total_codec_time':max(compression_worker_times)+max(decompression_worker_times)}

def psnr_from_nmse(nmse):
    return math.inf if nmse==0 else-10*math.log10(nmse)

def quality_columns(normalized_squared_error,valid_value_num,normalized_squared_error_all_points,value_num):
    # per-slice协议中每个slice的误差除以该slice普通值range的平方，per-field协议中除以整个field普通值range的平方
    # 主口径只统计普通值，即排除-99999等异常值；同时保留包含全部点的口径备查
    nmse=normalized_squared_error/valid_value_num if valid_value_num else 0.0
    nmse_all_points=normalized_squared_error_all_points/value_num
    return {'valid_value_num':valid_value_num,'normalized_squared_error':normalized_squared_error,'nmse':nmse,'psnr':psnr_from_nmse(nmse),'normalized_squared_error_all_points':normalized_squared_error_all_points,'nmse_all_points':nmse_all_points,'psnr_all_points':psnr_from_nmse(nmse_all_points)}

def field_range_quality(squared_error,valid_squared_error,valid_value_num,value_num,field_range):
    # per-field协议：整个field只有一个range
    if field_range>0:return quality_columns(valid_squared_error/field_range**2,valid_value_num,squared_error/field_range**2,value_num)
    if valid_squared_error!=0:raise RuntimeError('普通值为常量的field未被精确恢复')
    return quality_columns(0.0,valid_value_num,0.0,value_num)

def aggregate_quality(rows):
    # BR按全部值数量加权；NMSE为所有归一化平方误差之和除以对应的值数量
    value_num=sum(row['value_num']for row in rows)
    valid_value_num=sum(row['valid_value_num']for row in rows)
    nmse=math.fsum(row['normalized_squared_error']for row in rows)/valid_value_num
    nmse_all_points=math.fsum(row['normalized_squared_error_all_points']for row in rows)/value_num
    return {'field_num':len(rows),'value_num':value_num,'valid_value_num':valid_value_num,'br':math.fsum(row['br']*row['value_num']for row in rows)/value_num,'nmse':nmse,'normalized_psnr':psnr_from_nmse(nmse),'nmse_all_points':nmse_all_points,'normalized_psnr_all_points':psnr_from_nmse(nmse_all_points)}

def quality_summary(rows):
    summary=aggregate_quality(rows)
    summary['shape_groups']={shape:aggregate_quality([row for row in rows if row['shape']==shape])for shape in sorted({row['shape']for row in rows},key=lambda name:[int(value)for value in name.split('x')])}
    return summary

def write_fields_csv(path,rows):
    extra_columns=sorted({key for row in rows for key in row}-set(fields_csv_columns))
    with Path(path).open('w',newline='')as destination:
        writer=csv.DictWriter(destination,fieldnames=fields_csv_columns+extra_columns,lineterminator='\n')
        writer.writeheader()
        writer.writerows(json_safe(rows))
