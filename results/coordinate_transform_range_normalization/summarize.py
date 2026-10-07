import csv,math
from pathlib import Path

import numpy as np

result_dir=Path(__file__).resolve().parent


def calculate_bd_rate(reference_br,reference_psnr,test_br,test_psnr):
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    if not upper>lower:
        raise RuntimeError("RD曲线没有共同PSNR区间")
    knots=np.unique(np.concatenate(([lower,upper],reference_psnr[(reference_psnr>lower)&(reference_psnr<upper)],test_psnr[(test_psnr>lower)&(test_psnr<upper)])))
    reference_log_br=np.interp(knots,reference_psnr,np.log(reference_br))
    test_log_br=np.interp(knots,test_psnr,np.log(test_br))
    mean_log_br_ratio=np.trapezoid(test_log_br-reference_log_br,knots)/(upper-lower)
    return math.expm1(mean_log_br_ratio)*100


def read_modes(path):
    with path.open(newline="",encoding="utf-8") as file:
        rows=list(csv.DictReader(file))
    result={}
    for mode in ("off","on"):
        selected=[row for row in rows if row["task"].endswith("_"+mode)]
        if len(selected)==5:
            result[mode]=selected
    return result


def value_at_rel_eb(rows,rel_eb):
    exponent=f"1e-{round(-math.log10(rel_eb))}"
    for row in rows:
        if row["rel_eb"]!="not_applicable" and float(row["rel_eb"])==rel_eb:
            return row
        if exponent in row["point_set_config"]:
            return row
    raise RuntimeError(f"结果中缺少REL EB={rel_eb:g}")


def main():
    print("field,original_flatness,transformed_flatness,flatness_ratio,bd_rate_on_vs_off_percent,br_ratio_on_over_off_at_1e-3,psnr_delta_on_minus_off_at_1e-3,current_auto_decision")
    for path in sorted(result_dir.glob("*.csv")):
        modes=read_modes(path)
        if len(modes)!=2:
            continue
        off=modes["off"]
        on=modes["on"]
        off_br=np.asarray([float(row["br"]) for row in off])
        off_psnr=np.asarray([float(row["psnr"]) for row in off])
        on_br=np.asarray([float(row["br"]) for row in on])
        on_psnr=np.asarray([float(row["psnr"]) for row in on])
        bd_rate=calculate_bd_rate(off_br,off_psnr,on_br,on_psnr)
        off_mid=value_at_rel_eb(off,1e-3)
        on_mid=value_at_rel_eb(on,1e-3)
        original=float(on_mid["coordinate_transform_original_flatness"])
        transformed=float(on_mid["coordinate_transform_transformed_flatness"])
        ratio=float(on_mid["coordinate_transform_flatness_ratio"])
        current_auto=original<=0.05 or (original<0.75 and ratio>=1.075)
        print(f"{path.stem},{original:.9f},{transformed:.9f},{ratio:.9f},{bd_rate:+.9f},{float(on_mid['br'])/float(off_mid['br']):.9f},{float(on_mid['psnr'])-float(off_mid['psnr']):+.9f},{int(current_auto)}")


if __name__=="__main__":
    main()
