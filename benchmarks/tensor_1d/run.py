import argparse,csv,hashlib,json,math,os,shutil,subprocess
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser()
parser.add_argument('--codec',required=True,choices=('sz3','zfp','pfpl'))
parser.add_argument('--data-dir',default='/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean-large/tensor_data')
parser.add_argument('--rel-eb',type=float,default=1e-3)
parser.add_argument('--threads',type=int,default=1)
parser.add_argument('--field',action='append')
parser.add_argument('--output-root',default='results/ocean_large_tensor_1d')
parser.add_argument('--max-slices',type=int)
arguments=parser.parse_args()
if not math.isfinite(arguments.rel_eb) or arguments.rel_eb<=0 or arguments.threads<1:
    parser.error('误差界和线程数必须为正数')
data_dir=Path(arguments.data_dir).resolve()
metadata_path=data_dir/'tensor_metadata.json'
metadata=json.loads(metadata_path.read_text())
fields=arguments.field or list(metadata['fields'])
if any(field not in metadata['fields'] for field in fields):
    parser.error('指定field不在tensor metadata中')
output_dir=(project_root/arguments.output_root/arguments.codec/f'rel_{arguments.rel_eb:g}_t{arguments.threads}').resolve()
output_dir.mkdir(parents=True,exist_ok=True)
binary=project_root/'build/tensor_1d'/f'benchmark_{arguments.codec}'
binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()
manifest_path=output_dir/'manifest.json'
manifest={'codec':arguments.codec,'binary_sha256':binary_sha256,'metadata_sha256':hashlib.sha256(metadata_path.read_bytes()).hexdigest(),'rel_eb':arguments.rel_eb,'threads':arguments.threads,'layout':'one original-order unstructured slice per 1D array','range_excludes':-99999,'input_keeps_sentinel':True,'git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root,text=True).strip(),'upstream_revisions':{name:subprocess.check_output(['git','rev-parse','HEAD'],cwd=project_root/'third_party'/name,text=True).strip() for name in ('SZ3','zfp','PFPL')}}
manifest['codec_error_bound_mode']='ABS'
manifest['abs_eb_rule']='rel_eb*(valid_max-valid_min),computed separately for each slice'
if manifest_path.exists():
    previous=json.loads(manifest_path.read_text())
    if any(previous[key]!=manifest[key] for key in ('binary_sha256','metadata_sha256','rel_eb','threads','layout')):
        raise RuntimeError('已有结果的版本或参数不同，请使用新结果目录')
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
snapshot=output_dir/binary.name
if not snapshot.exists():
    shutil.copy2(binary,snapshot)
if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=binary_sha256:
    raise RuntimeError('保存的二进制SHA不一致')
results=[]
failures=[]
for field in fields:
    slice_num=math.prod(metadata['fields'][field]['shape'])
    if arguments.max_slices is not None:
        slice_num=min(slice_num,arguments.max_slices)
    summary_path=output_dir/f'{field}.json'
    if summary_path.exists():
        summary=json.loads(summary_path.read_text())
        if summary.get('slice_num')==slice_num:
            results.append(summary)
            if not summary['valid']:
                failures.append(field)
            print(f'跳过已完成field：{field}',flush=True)
            continue
    paths=[data_dir/f'{field}.dat.{slice_id}' for slice_id in range(slice_num)]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f'输入切片不完整：{field}')
    input_list=output_dir/f'{field}.inputs.txt'
    input_list.write_text(''.join(str(path)+'\n' for path in paths))
    slices_path=output_dir/f'{field}.slices.csv'
    print(f'BEGIN {arguments.codec} {field},slices={slice_num},threads={arguments.threads}',flush=True)
    with (output_dir/f'{field}.log').open('w') as log:
        result=subprocess.run([str(snapshot),str(input_list),str(arguments.rel_eb),str(arguments.threads),str(slices_path)],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':str(arguments.threads),'OMP_DYNAMIC':'false'})
    with slices_path.open(newline='') as source:
        rows=list(csv.DictReader(source))
    count=sum(int(row['point_num']) for row in rows)
    compressed_bytes=sum(int(row['compressed_bytes']) for row in rows)
    squared_error=sum(float(row['squared_error']) for row in rows)
    valid_rows=[row for row in rows if int(row['valid_point_num'])]
    data_range=max((float(row['valid_max']) for row in valid_rows),default=0)-min((float(row['valid_min']) for row in valid_rows),default=0)
    valid=result.returncode==0 and len(rows)==slice_num and all(row['valid']=='1' for row in rows)
    psnr=math.inf if squared_error==0 else 10*math.log10(data_range*data_range/(squared_error/count)) if data_range>0 and count else -math.inf
    summary={'field':field,'compressor':arguments.codec,'rel_eb':arguments.rel_eb,'thread_num':arguments.threads,'slice_num':slice_num,'completed_slice_num':len(rows),'input_bytes':count*4,'compressed_bytes':compressed_bytes,'br':8*compressed_bytes/count if count else math.nan,'psnr':psnr,'compression_time':sum(float(row['compression_time']) for row in rows),'decompression_time':sum(float(row['decompression_time']) for row in rows),'compression_preprocessing_time':0,'decompression_preprocessing_time':0,'max_error_over_eb':max((float(row['max_error_over_eb']) for row in rows),default=math.inf),'zero_eb_slice_num':sum(int(row['zero_eb']) for row in rows),'valid':valid,'returncode':result.returncode}
    summary['compression_MB_s']=summary['input_bytes']/1e6/summary['compression_time'] if summary['compression_time'] else 0
    summary['decompression_MB_s']=summary['input_bytes']/1e6/summary['decompression_time'] if summary['decompression_time'] else 0
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    results.append(summary)
    if not valid:
        failures.append(field)
    with (output_dir/'summary.csv').open('w',newline='') as destination:
        writer=csv.DictWriter(destination,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (output_dir/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if failures:
    raise RuntimeError('压缩或逐点误差验证未通过：'+','.join(failures))
