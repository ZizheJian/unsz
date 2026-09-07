import csv,json,math,os,re,subprocess,tempfile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
spatial_result_path=project_root/"py_codes/trash/results/katrina_current_spatial_rd.csv"
result_path=project_root/"py_codes/trash/results/katrina_current_cross_time_label_rd.csv"
figure_path=project_root/"figures/katrina_cross_time_label_rd"
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
time_step_num=162
point_num=417642
total_mb=point_num*time_step_num*4/1e6

def run_command(command):
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError("命令失败："+" ".join(str(value) for value in command))
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def calculate_bd_rate(reference_br,reference_psnr,test_br,test_psnr):
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    reference_integral=np.polyint(np.polyfit(reference_psnr,np.log(reference_br),3))
    test_integral=np.polyint(np.polyfit(test_psnr,np.log(test_br),3))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (np.exp((test_area-reference_area)/(upper-lower))-1)*100

def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values)/len(values))

def main():
    with open(spatial_result_path,newline="") as file:
        spatial_rows=list(csv.DictReader(file))
    spatial_rows.sort(key=lambda row:float(row["rel_eb"]),reverse=True)
    if [float(row["rel_eb"]) for row in spatial_rows]!=list(rel_eb_vec):
        raise RuntimeError("当前Katrina空间五点结果不完整")
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_current_cross_label_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        prediction_structure_path=temporary_dir/"prediction_structure.bin"
        for rel_eb,spatial_row in zip(rel_eb_vec,spatial_rows):
            config_path=temporary_dir/f"point_sets_{rel_eb}.json"
            with open(config_path,"w",encoding="utf-8") as file:
                json.dump({"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]},file,ensure_ascii=False)
            arguments=["-data-dir",test_dir,"-variable","attr0","-time-step","[0,161]","-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",matrix_path,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",4,"-actual-ref-num",4,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",4,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental","-cross-time-label-predictor-experimental","previous","-fused-two-point-set-prediction-experimental"]
            print(f"测试REL={rel_eb}",flush=True)
            compression_output=run_command([compressor,*arguments])
            decompression_output=run_command([decompressor,*arguments])
            br=read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"UNSZ BR")
            psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"UNSZ PSNR")
            decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"UNSZ解压PSNR")
            if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
            compression_time=read_float(r"^total compression time= ([^ ]+) s$",compression_output,"UNSZ压缩时间")
            decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"UNSZ解压时间")
            row={"rel_eb":rel_eb,"unsz_br":br,"unsz_psnr":psnr,"unsz_compression_mb_s":total_mb/compression_time,"unsz_decompression_mb_s":total_mb/decompression_time,"spatial_br":float(spatial_row["unsz_br"]),"spatial_psnr":float(spatial_row["unsz_psnr"]),"spatial_compression_mb_s":float(spatial_row["unsz_compression_mb_s"]),"spatial_decompression_mb_s":float(spatial_row["unsz_decompression_mb_s"]),"umc_br":float(spatial_row["umc_br"]),"umc_psnr":float(spatial_row["umc_psnr"]),"umc_compression_mb_s":float(spatial_row["umc_compression_mb_s"]),"umc_decompression_mb_s":float(spatial_row["umc_decompression_mb_s"])}
            result_vec.append(row)
            print(f"  BR={br:.6f},PSNR={psnr:.6f},compression={row['unsz_compression_mb_s']:.3f} MB/s,decompression={row['unsz_decompression_mb_s']:.3f} MB/s",flush=True)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    unsz_br=np.asarray([row["unsz_br"] for row in result_vec])
    unsz_psnr=np.asarray([row["unsz_psnr"] for row in result_vec])
    spatial_br=np.asarray([row["spatial_br"] for row in result_vec])
    spatial_psnr=np.asarray([row["spatial_psnr"] for row in result_vec])
    umc_br=np.asarray([row["umc_br"] for row in result_vec])
    umc_psnr=np.asarray([row["umc_psnr"] for row in result_vec])
    figure,axis=plt.subplots(figsize=(6.4,4.6),layout="constrained")
    axis.plot(umc_br,umc_psnr,color="#D55E00",marker="o",linewidth=2,label="UMC NBP-IDW-INP")
    axis.plot(spatial_br,spatial_psnr,color="#0072B2",marker="s",linewidth=2,label="UNSZ spatial labels")
    axis.plot(unsz_br,unsz_psnr,color="#009E73",marker="^",linewidth=2,label="UNSZ previous-time labels")
    axis.set_xlim(left=0)
    axis.set_xlabel("Bit rate (bit/value)")
    axis.set_ylabel("PSNR (dB)")
    axis.set_title("Katrina attr0: cross-time label prediction")
    axis.grid(True,linestyle=":",alpha=0.55)
    axis.legend(frameon=False)
    figure.savefig(figure_path.with_suffix(".png"),dpi=220,bbox_inches="tight",facecolor="white")
    figure.savefig(figure_path.with_suffix(".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(figure)
    print(f"相对当前空间UNSZ BD-rate={calculate_bd_rate(spatial_br,spatial_psnr,unsz_br,unsz_psnr):.6f}%")
    print(f"相对UMC BD-rate={calculate_bd_rate(umc_br,umc_psnr,unsz_br,unsz_psnr):.6f}%")
    print(f"UMC/UNSZ压缩速度={geometric_mean([row['umc_compression_mb_s']/row['unsz_compression_mb_s'] for row in result_vec]):.6f}x")
    print(f"UMC/UNSZ解压速度={geometric_mean([row['umc_decompression_mb_s']/row['unsz_decompression_mb_s'] for row in result_vec]):.6f}x")
    print(f"空间融合UNSZ/跨标签UNSZ压缩速度={geometric_mean([row['spatial_compression_mb_s']/row['unsz_compression_mb_s'] for row in result_vec]):.6f}x")
    print(f"空间融合UNSZ/跨标签UNSZ解压速度={geometric_mean([row['spatial_decompression_mb_s']/row['unsz_decompression_mb_s'] for row in result_vec]):.6f}x")
    print(f"结果={result_path}")

if __name__=="__main__":
    main()
