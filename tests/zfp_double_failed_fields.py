import argparse,csv,json,subprocess
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--binary',default='/tmp/unsz_zfp_error_diagnostic')
parser.add_argument('--source',default='results/ocean_large_tensor_1d/zfp/rel_0.001_t1')
parser.add_argument('--output',default='results/ocean_large_tensor_1d/zfp_double_worst_slices.json')
args=parser.parse_args()
source=Path(args.source)
with (source/'summary.csv').open(newline='') as stream:
    fields=[row['field'] for row in csv.DictReader(stream) if row['valid']!='True']
results=[]
for field in fields:
    with (source/f'{field}.slices.csv').open(newline='') as stream:
        worst=max(csv.DictReader(stream),key=lambda row:float(row['max_error_over_eb']))
    slice_id=int(worst['slice_id'])
    paths=(source/f'{field}.inputs.txt').read_text().splitlines()
    result=subprocess.run([args.binary,paths[slice_id],'--full-double',worst['abs_eb']],capture_output=True,text=True)
    if result.returncode not in (0,2):
        raise RuntimeError(result.stderr or result.stdout)
    row={'field':field,'slice_id':slice_id,'path':paths[slice_id],'original_float_ratio':float(worst['max_error_over_eb']),**json.loads(result.stdout),'valid':result.returncode==0}
    results.append(row)
    print(json.dumps(row),flush=True)
    Path(args.output).write_text(json.dumps({'scope':'one full worst-float-error slice per completed failed field,not all slices','source':str(source),'results':results},indent=2)+'\n')
print(f'Passed {sum(row["valid"] for row in results)}/{len(results)} failed-field worst slices',flush=True)
