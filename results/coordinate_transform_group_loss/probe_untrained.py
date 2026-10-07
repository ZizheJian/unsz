import csv,re,shlex,subprocess
from pathlib import Path

root=Path(__file__).resolve().parent
output_path=root/"untrained_probe.csv"

def main():
    output_rows=[]
    for path in sorted(root.glob("*.csv")):
        with path.open(newline="",encoding="utf-8") as file:
            rows=list(csv.DictReader(file))
        if not rows or "task" not in rows[0]:
            continue
        matching=[row for row in rows if row["task"].endswith("_power1")]
        if len(matching)!=1:
            continue
        command_text=matching[0]["command"].split("; decompression:",1)[0].removeprefix("compression: ")
        result=subprocess.run(shlex.split(command_text),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        ratio_match=re.search(r"^coordinate transform untrained isotropic group MSE max/min=([^,]+),",result.stdout,re.MULTILINE)
        if ratio_match is None:
            raise RuntimeError(f"{path.name}没有输出训练前group MSE比值：\n{result.stdout}")
        output_rows.append({"field":path.stem,"untrained_group_mse_ratio":float(ratio_match.group(1))})
        print(f"{path.stem}: {ratio_match.group(1)}",flush=True)
    with output_path.open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=("field","untrained_group_mse_ratio"))
        writer.writeheader()
        writer.writerows(output_rows)

if __name__=="__main__":
    main()
