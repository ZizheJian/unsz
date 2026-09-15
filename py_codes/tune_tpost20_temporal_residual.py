import csv,math,re,subprocess,tempfile,time
from pathlib import Path

project_root=Path(__file__).resolve().parents[1]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Tpost20")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
result_path=project_root/"results/tpost20_scalar_temporal_residual_predictor_screen.csv"
rel_eb=1e-3
time_step_num=20
configuration_vec=(
    {"name":"idw4_p2","predictor":"inverse_proportion","ref_num":4,"power":2},
    {"name":"idw5_p2.5","predictor":"inverse_proportion","ref_num":5,"power":2.5},
    {"name":"idw10_p3.6","predictor":"inverse_proportion","ref_num":10,"power":3.6},
    {"name":"rbf8_s0.01","predictor":"rbf_fitting","ref_num":8,"power":3.6,"shape":0.01},
    {"name":"rbf8_s0.03","predictor":"rbf_fitting","ref_num":8,"power":3.6,"shape":0.03},
    {"name":"rbf10_s0.01","predictor":"rbf_fitting","ref_num":10,"power":3.6,"shape":0.01},
    {"name":"rbf10_s0.03","predictor":"rbf_fitting","ref_num":10,"power":3.6,"shape":0.03},
)

def linear_interpolation_time_order():
    order=[0,time_step_num-1]
    intervals=[(0,time_step_num-1)]
    for left,right in intervals:
        if right-left<=1:
            continue
        middle=(left+right)//2
        order.append(middle)
        intervals.append((left,middle))
        intervals.append((middle,right))
    return order

def run(command):
    start=time.perf_counter()
    completed=subprocess.run([str(value) for value in command],cwd=project_root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    elapsed=time.perf_counter()-start
    if completed.returncode!=0:
        print(completed.stdout,flush=True)
        raise RuntimeError("命令失败："+" ".join(str(value) for value in command))
    return completed.stdout,elapsed

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def cleanup(test_dir):
    for time_step in range(time_step_num):
        for suffix in (".unsz",".unsz.bin"):
            path=test_dir/f"scalar.dat.{time_step}{suffix}"
            if path.exists():
                path.unlink()

def main():
    expression="{"+",".join(str(value) for value in linear_interpolation_time_order())+"}"
    rows=[]
    with tempfile.TemporaryDirectory(prefix="unsz_tpost20_residual_tune_") as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"scalar.dat.{time_step}").symlink_to(data_dir/f"scalar.dat.{time_step}")
        for config in configuration_vec:
            structure_path=temporary_dir/f"r{config['ref_num']}.dag"
            common=["-data-dir",test_dir,"-variable","scalar","-time-step",expression,"-dim",3,"-M","REL",rel_eb,"-dag-cache",structure_path,"-predict-round-num",8,"-max-ref-num",config["ref_num"],"-actual-ref-num",config["ref_num"],"-inverse-distance-power",config["power"],"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",config["ref_num"],"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental","-cross-time-predictor-experimental","linear_interpolation_spatial_residual"]
            rbf=[] if config["predictor"]=="inverse_proportion" else ["-rbf-kernel","matern32","-rbf-shape",config["shape"],"-rbf-nugget",1e-6,"-rbf-polynomial-degree",0]
            compression_output,compression_wall=run([compressor,"-predictor",config["predictor"],*common,*rbf])
            decompression_output,decompression_wall=run([decompressor,*common,*rbf])
            psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
            decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
            if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
            row={"configuration":config["name"],"predictor":"idw" if config["predictor"]=="inverse_proportion" else "rbf","ref_num":config["ref_num"],"idw_power":config["power"] if config["predictor"]=="inverse_proportion" else "","rbf_kernel":"" if config["predictor"]=="inverse_proportion" else "matern32","rbf_shape":config.get("shape",""),"rbf_nugget":"" if config["predictor"]=="inverse_proportion" else 1e-6,"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"BR"),"psnr":psnr,"compression_wall_seconds":compression_wall,"decompression_wall_seconds":decompression_wall}
            rows.append(row)
            result_path.parent.mkdir(parents=True,exist_ok=True)
            with result_path.open("w",newline="") as file:
                writer=csv.DictWriter(file,fieldnames=rows[0].keys(),lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
            print(f"{config['name']}:BR={row['br']:.6f},PSNR={psnr:.6f}",flush=True)
            cleanup(test_dir)
    print("BR排名：")
    for row in sorted(rows,key=lambda value:value["br"]):
        print(f"  {row['configuration']}:BR={row['br']:.6f},PSNR={row['psnr']:.6f}")
    print(f"结果={result_path}")

if __name__=="__main__":
    main()
