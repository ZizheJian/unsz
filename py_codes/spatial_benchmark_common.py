import csv,math,re,subprocess
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[1]
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"

def run_command(command):
    completed=subprocess.run([str(value) for value in command],cwd=project_root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if completed.returncode!=0:
        print(completed.stdout,flush=True)
        raise RuntimeError("命令执行失败："+" ".join(str(value) for value in command))
    return completed.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def calculate_average_br_change(reference,test):
    reference_psnr=np.asarray([row["psnr"] for row in reference])
    reference_br=np.asarray([row["br"] for row in reference])
    test_psnr=np.asarray([row["psnr"] for row in test])
    test_br=np.asarray([row["br"] for row in test])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    degree=min(3,len(reference)-1,len(test)-1)
    reference_integral=np.polyint(np.polyfit(reference_psnr,reference_br,degree))
    test_integral=np.polyint(np.polyfit(test_psnr,test_br,degree))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (test_area/reference_area-1)*100

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def write_results(path,row_vec):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=row_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(row_vec)

def cleanup(data_dir,field,time_step):
    for suffix in (".unsz",".unsz.bin"):
        path=data_dir/f"{field}.dat.{time_step}{suffix}"
        if path.exists():
            path.unlink()
