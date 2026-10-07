import csv
from pathlib import Path

root=Path(__file__).resolve().parent
rows=[]
for path in sorted(root.glob("*.csv")):
    with path.open(newline="",encoding="utf-8") as file:
        field_rows=list(csv.DictReader(file))
    if not field_rows or "task" not in field_rows[0]:
        continue
    by_variant={row["task"].rsplit("_",1)[-1]:row for row in field_rows}
    if "off" not in by_variant:
        continue
    off_br=float(by_variant["off"]["br"])
    for variant in ("power1","power2","power3","power4"):
        if variant not in by_variant:
            continue
        row=by_variant[variant]
        br=float(row["br"])
        rows.append((path.stem,variant,br,(br/off_br-1)*100,float(row["psnr"]),row["coordinate_transform_diagonal"],float(row["coordinate_transform_time"])))

print("field,variant,br,br_change_vs_off_percent,psnr,diagonal,coordinate_transform_time")
for row in rows:
    print(f"{row[0]},{row[1]},{row[2]:.9g},{row[3]:.6f},{row[4]:.6f},{row[5]},{row[6]:.6f}")
