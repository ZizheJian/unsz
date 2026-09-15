import csv,json,math,os,re,subprocess,tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
source_csv_path=data_dir/"attr0.csv"
result_path=project_root/"py_codes/trash/results/katrina_cross_time_label_rd.csv"
figure_path=project_root/"figures/katrina_cross_time_label_rd"
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
time_step_num=162
point_num=417642
total_mb=point_num*time_step_num*4/1e6

def run_command(command):
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(str(value) for value in command)}")
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def cleanup(test_dir):
    for time_step in range(time_step_num):
        for suffix in (".unsz",".unsz.bin"):
            path=test_dir/f"attr0.dat.{time_step}{suffix}"
            if path.exists():
                path.unlink()

def write_point_set_config(path,rel_eb):
    config={"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]}
    with open(path,"w",encoding="utf-8") as file:
        json.dump(config,file,ensure_ascii=False,indent=2)

def benchmark(test_dir,rel_eb,config_path,prediction_structure_path):
    common_arguments=["-data-dir",test_dir,"-variable","attr0","-time-step","[0,161]","-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",matrix_path,"-reverse-mean-distance-only-order-experimental","-dag-cache",prediction_structure_path,"-cross-time-label-predictor-experimental","previous"]
    compression_output=run_command([compressor,*common_arguments,"-label-stats"])
    br=read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR")
    psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR")
    preprocessing_time=read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间")
    compression_time=read_float(r"^total compression time= ([^ ]+) s$",compression_output,"总压缩时间")
    label_zstd_bytes=sum(int(value) for value in re.findall(r"label_zstd_bytes=([0-9]+)",compression_output))
    if label_zstd_bytes==0:
        raise RuntimeError("没有读取到标签Zstd码流长度")
    decompression_output=run_command([decompressor,*common_arguments])
    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"总解压时间")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
    cleanup(test_dir)
    return {"rel_eb":rel_eb,"br":br,"psnr":psnr,"label_zstd_bytes":label_zstd_bytes,"label_br":label_zstd_bytes*8/(point_num*time_step_num),"preprocessing_time":preprocessing_time,"compression_time":compression_time,"decompression_time":decompression_time,"compression_mb_s":total_mb/compression_time,"decompression_mb_s":total_mb/decompression_time}

def read_spatial_curves():
    with open(source_csv_path,newline="") as file:
        rows=[row for row in csv.DictReader(file) if int(row["TIME_STEP_NUM"])==time_step_num and float(row["REL_EB"]) in rel_eb_vec]
    if len(rows)!=len(rel_eb_vec):
        raise RuntimeError(f"现有Katrina CSV中应有{len(rel_eb_vec)}个点，实际为{len(rows)}")
    rows.sort(key=lambda row:float(row["REL_EB"]),reverse=True)
    return {name:{"br":np.asarray([float(row[f"{name}_BR"]) for row in rows]),"psnr":np.asarray([float(row[f"{name}_PSNR"]) for row in rows])} for name in ("UNSZ","UMC")}

def calculate_bd_rate(reference,test):
    lower=max(reference["psnr"].min(),test["psnr"].min())
    upper=min(reference["psnr"].max(),test["psnr"].max())
    reference_integral=np.polyint(np.polyfit(reference["psnr"],np.log(reference["br"]),3))
    test_integral=np.polyint(np.polyfit(test["psnr"],np.log(test["br"]),3))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (np.exp((test_area-reference_area)/(upper-lower))-1)*100

def write_results(results,curves):
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    temporal={"br":np.asarray([result["br"] for result in results]),"psnr":np.asarray([result["psnr"] for result in results])}
    figure,axis=plt.subplots(figsize=(6.4,4.6),layout="constrained")
    axis.plot(curves["UMC"]["br"],curves["UMC"]["psnr"],color="#D55E00",marker="o",linewidth=2,label="UMC NBP-IDW-INP")
    axis.plot(curves["UNSZ"]["br"],curves["UNSZ"]["psnr"],color="#0072B2",marker="s",linewidth=2,label="UNSZ spatial labels")
    axis.plot(temporal["br"],temporal["psnr"],color="#009E73",marker="^",linewidth=2,label="UNSZ previous-time labels")
    axis.set_xlim(left=0)
    axis.set_xlabel("Bit rate (bit/value)")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Katrina attr0: cross-time label prediction")
    axis.grid(True,linestyle=":",alpha=0.55)
    axis.legend(frameon=False)
    figure_path.parent.mkdir(parents=True,exist_ok=True)
    figure.savefig(figure_path.with_suffix(".png"),dpi=220,bbox_inches="tight",facecolor="white")
    figure.savefig(figure_path.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(figure)
    return temporal

def main():
    curves=read_spatial_curves()
    results=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_cross_time_label_",dir=temporary_root) as temporary_dir:
        test_dir=Path(temporary_dir)/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        prediction_structure_path=Path(temporary_dir)/"prediction_structure.bin"
        for rel_eb in rel_eb_vec:
            config_path=Path(temporary_dir)/f"point_sets_{rel_eb}.json"
            write_point_set_config(config_path,rel_eb)
            print(f"测试REL={rel_eb}",flush=True)
            result=benchmark(test_dir,rel_eb,config_path,prediction_structure_path)
            results.append(result)
            print(f"  BR={result['br']:.9f},PSNR={result['psnr']:.6f},label BR={result['label_br']:.9f},compression={result['compression_mb_s']:.3f} MB/s,decompression={result['decompression_mb_s']:.3f} MB/s",flush=True)
    temporal=write_results(results,curves)
    print(f"相对空间UNSZ BD-rate={calculate_bd_rate(curves['UNSZ'],temporal):.6f}%")
    print(f"相对UMC BD-rate={calculate_bd_rate(curves['UMC'],temporal):.6f}%")
    print(f"结果={result_path}")
    print(f"图像={figure_path.with_suffix('.png')}")

if __name__=="__main__":
    main()
