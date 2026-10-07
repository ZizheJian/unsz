import csv,math,re,sys
from pathlib import Path


project=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project/"py_codes"))
from parse_unsz_output import parse_unsz_output

result_dir=Path(__file__).resolve().parent
fields=(result_dir/"fields.txt").read_text().splitlines()
existing={row["field"]:row for row in csv.DictReader((result_dir/"summary.txt").open())}

def load(field,mode,rel_eb):
    if rel_eb==0.001:
        row=existing[field]
        return float(row[f"{mode}_psnr"]),float(row[f"{mode}_br"])
    eb_name=str(rel_eb).replace(".","p")
    compression=(result_dir/"logs"/f"{field}_{mode}_{eb_name}_compress.log").read_text()
    decompression=(result_dir/"logs"/f"{field}_{mode}_{eb_name}_decompress.log").read_text()
    result=parse_unsz_output(compression,decompression)
    return result["psnr"],result["br"]

def interpolate_log_br(points,psnr):
    points=sorted(points)
    for left,right in zip(points,points[1:]):
        if left[0]<=psnr<=right[0]:
            fraction=(psnr-left[0])/(right[0]-left[0])
            return math.exp(math.log(left[1])+fraction*(math.log(right[1])-math.log(left[1])))
    raise RuntimeError(f"PSNR {psnr}不在曲线范围内：{points}")

writer=csv.writer(sys.stdout)
writer.writerow(("field","common_psnr","on_br","off_br","off_over_on_br"))
for field in fields:
    curves={mode:[load(field,mode,rel_eb) for rel_eb in (0.002,0.001,0.0005)] for mode in ("on","off")}
    lower=max(min(point[0] for point in curve) for curve in curves.values())
    upper=min(max(point[0] for point in curve) for curve in curves.values())
    common_psnr=(lower+upper)/2
    on_br=interpolate_log_br(curves["on"],common_psnr)
    off_br=interpolate_log_br(curves["off"],common_psnr)
    writer.writerow((field,common_psnr,on_br,off_br,off_br/on_br))
