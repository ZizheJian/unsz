import csv,math,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate

result_root=project_root/"results/coordinate_transform"

for path in sorted(result_root.glob("*.csv")):
    dataset=path.stem
    with path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    curves={}
    for mode in ("off","on"):
        task=f"coordinate_transform_{dataset}_{mode}"
        current=[row for row in rows if row["task"]==task]
        current.sort(key=lambda row:float(row["psnr"]))
        curves[mode]=current
    if not all(len(curves[mode])==5 for mode in curves):
        print(f"{dataset}:结果不完整，off={len(curves['off'])},on={len(curves['on'])}")
        continue
    reference_br=np.asarray([float(row["br"]) for row in curves["off"]])
    reference_psnr=np.asarray([float(row["psnr"]) for row in curves["off"]])
    test_br=np.asarray([float(row["br"]) for row in curves["on"]])
    test_psnr=np.asarray([float(row["psnr"]) for row in curves["on"]])
    bd_rate=calculate_bd_rate(reference_br,reference_psnr,test_br,test_psnr)
    compression_ratio=math.exp(sum(math.log(float(on["compression_time"])/float(off["compression_time"])) for off,on in zip(curves["off"],curves["on"]))/5)
    decompression_ratio=math.exp(sum(math.log(float(on["decompression_time"])/float(off["decompression_time"])) for off,on in zip(curves["off"],curves["on"]))/5)
    print(f"{dataset}:BD-rate(on vs off)={bd_rate:+.6f}%,compression_time_ratio={compression_ratio:.6f},decompression_time_ratio={decompression_ratio:.6f}")
