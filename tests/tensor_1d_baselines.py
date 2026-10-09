import csv,pathlib,subprocess,tempfile
import numpy as np

project_root=pathlib.Path(__file__).resolve().parents[1]
random=np.random.default_rng(142857)
with tempfile.TemporaryDirectory(prefix='unsz-1d-baselines-') as temporary:
    root=pathlib.Path(temporary)
    values=random.normal(size=8195).astype(np.float32)
    values[::17]=-99999
    constant=np.full(8195,4,dtype=np.float32)
    constant[::17]=-99999
    cases=[values,constant,np.zeros(8195,dtype=np.float32),np.full(8195,-99999,dtype=np.float32),np.array([3],dtype=np.float32)]
    inputs=[]
    for index,case in enumerate(cases):
        path=root/f'case_{index}.dat'
        case.tofile(path)
        inputs.append(str(path))
    input_list=root/'inputs.txt'
    input_list.write_text('\n'.join(inputs)+'\n')
    for codec in ('sz3','zfp','pfpl'):
        for threads in (1,2):
            output=root/f'{codec}_{threads}.csv'
            result=subprocess.run([str(project_root/'build/tensor_1d'/f'benchmark_{codec}'),str(input_list),'0.001',str(threads),str(output)],capture_output=True,text=True)
            assert result.returncode==0,(codec,threads,result.stdout,result.stderr)
            with output.open() as source:
                rows=list(csv.DictReader(source))
            assert len(rows)==len(cases)
            assert all(row['valid']=='1' for row in rows)
            assert all(float(row['max_error'])==0 for row in rows[1:])
            print(f'{codec} t{threads}: random/sentinel/constant/zero/all-invalid/single-point passed',flush=True)
