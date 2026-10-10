import argparse,hashlib,json,subprocess
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--binary',default='/tmp/unsz_zfp_error_diagnostic')
parser.add_argument('--source',default='results/ocean_large_tensor_1d/zfp_double_worst_slices.json')
parser.add_argument('--output',default='results/ocean_large_tensor_1d/zfp_double_investigation.json')
args=parser.parse_args()
binary_sha256=hashlib.sha256(Path(args.binary).read_bytes()).hexdigest()
failed=[row for row in json.loads(Path(args.source).read_text())['results'] if not row['valid']]
results=[]
for row in failed:
    process=subprocess.run([args.binary,row['path'],'--investigate-double',str(row['abs_eb'])],capture_output=True,text=True)
    if process.returncode not in (0,2):
        raise RuntimeError(process.stderr or process.stdout)
    result={'field':row['field'],'slice_id':row['slice_id'],**json.loads(process.stdout)}
    results.append(result)
    print(json.dumps(result),flush=True)
    Path(args.output).write_text(json.dumps({'binary':args.binary,'binary_sha256':binary_sha256,'scope':'13 previously failed full slices;block trials are isolated worst blocks;full-half trial checks all points','trial_key':{'0':'original tolerance','1':'tolerance/2','2':'tolerance/4','3':'tolerance/8','4':'reversible','5':'sentinel replaced with zero,diagnostic only','6':'tolerance 1e-300,forces full 64-bit precision'},'results':results},indent=2)+'\n')
