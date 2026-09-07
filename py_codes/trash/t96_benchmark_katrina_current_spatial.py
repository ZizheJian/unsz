import csv,json,math,os,re,subprocess,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
source_csv_path=data_dir/"attr0.csv"
result_path=project_root/"py_codes/trash/results/katrina_current_spatial_rd.csv"
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
    with open(source_csv_path,newline="") as file:
        source_rows=[row for row in csv.DictReader(file) if int(row["TIME_STEP_NUM"])==time_step_num and float(row["REL_EB"]) in rel_eb_vec]
    source_rows.sort(key=lambda row:float(row["REL_EB"]),reverse=True)
    if len(source_rows)!=len(rel_eb_vec):
        raise RuntimeError("Katrina现有CSV中的UMC五点曲线不完整")
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_current_spatial_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        prediction_structure_path=temporary_dir/"prediction_structure.bin"
        for rel_eb,source_row in zip(rel_eb_vec,source_rows):
            config_path=temporary_dir/f"point_sets_{rel_eb}.json"
            with open(config_path,"w",encoding="utf-8") as file:
                json.dump({"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]},file,ensure_ascii=False)
            arguments=["-data-dir",test_dir,"-variable","attr0","-time-step","[0,161]","-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",matrix_path,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",4,"-actual-ref-num",4,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",4,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental","-fused-two-point-set-prediction-experimental","-normalized-weight-experimental"]
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
            row={"rel_eb":rel_eb,"unsz_br":br,"unsz_psnr":psnr,"unsz_compression_mb_s":total_mb/compression_time,"unsz_decompression_mb_s":total_mb/decompression_time,"umc_br":float(source_row["UMC_BR"]),"umc_psnr":float(source_row["UMC_PSNR"]),"umc_compression_mb_s":float(source_row["UMC_COMPRESSION_MB_S"]),"umc_decompression_mb_s":float(source_row["UMC_DECOMPRESSION_MB_S"])}
            result_vec.append(row)
            print(f"  UNSZ BR={br:.6f},PSNR={psnr:.6f},compression={row['unsz_compression_mb_s']:.3f} MB/s,decompression={row['unsz_decompression_mb_s']:.3f} MB/s",flush=True)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    unsz_br=np.asarray([row["unsz_br"] for row in result_vec])
    unsz_psnr=np.asarray([row["unsz_psnr"] for row in result_vec])
    umc_br=np.asarray([row["umc_br"] for row in result_vec])
    umc_psnr=np.asarray([row["umc_psnr"] for row in result_vec])
    bd_rate=calculate_bd_rate(umc_br,umc_psnr,unsz_br,unsz_psnr)
    compression_ratio=geometric_mean([row["umc_compression_mb_s"]/row["unsz_compression_mb_s"] for row in result_vec])
    decompression_ratio=geometric_mean([row["umc_decompression_mb_s"]/row["unsz_decompression_mb_s"] for row in result_vec])
    print(f"UNSZ相对UMC BD-rate={bd_rate:.6f}%")
    print(f"UMC/UNSZ压缩速度={compression_ratio:.6f}x")
    print(f"UMC/UNSZ解压速度={decompression_ratio:.6f}x")
    print(f"结果={result_path}")

if __name__=="__main__":
    main()
