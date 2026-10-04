import csv,sys
from pathlib import Path

import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.plot_dataset_rd import calculate_bd_rate

result_path=project_root/"results/katrina_flow_staged_tuning/results.csv"
flow_vec=("katrina_staged_constant","katrina_staged_linear","katrina_staged_anchor8")

def main():
    with result_path.open(newline="") as file:
        rows=list(csv.DictReader(file))
    curve_map={flow:[] for flow in flow_vec}
    for row in rows:
        if row["task"] in curve_map:
            curve_map[row["task"]].append(row)
    for flow,curve in curve_map.items():
        curve.sort(key=lambda row:float(row["psnr"]))
        if len(curve)!=5:
            raise RuntimeError(f"{flow}应有5个RD点，实际为{len(curve)}")
    reference=curve_map[flow_vec[0]]
    reference_br=np.asarray([float(row["br"]) for row in reference])
    reference_psnr=np.asarray([float(row["psnr"]) for row in reference])
    print("flow,bd_rate_vs_constant_percent")
    for flow in flow_vec:
        curve=curve_map[flow]
        br=np.asarray([float(row["br"]) for row in curve])
        psnr=np.asarray([float(row["psnr"]) for row in curve])
        print(f"{flow},{calculate_bd_rate(reference_br,reference_psnr,br,psnr):+.9f}")

if __name__=="__main__":
    main()
