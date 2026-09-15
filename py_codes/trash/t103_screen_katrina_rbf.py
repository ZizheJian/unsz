import csv,json,math,os,re,subprocess,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
idw_result_path=project_root/"py_codes/trash/results/katrina_current_idw_parameter_screen.csv"
result_path=project_root/"py_codes/trash/results/katrina_rbf_screen.csv"
time_step_num=20
rel_eb_vec=(1e-2,1e-3,1e-4)
shape_vec=(0.01,0.03,0.1,0.3,1,3,10)
extra_arguments=[]
ref_num=10
cache_threshold=8
verify_decompression=False

def run_command(command):
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError("命令执行失败："+" ".join(str(value) for value in command))
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def write_point_set_config(path,rel_eb):
    config={"version":1,"rule_type":"ordered","labels":[{"method":"rbf_fitting","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]}
    with open(path,"w",encoding="utf-8") as file:
        json.dump(config,file,ensure_ascii=False)

def calculate_bd_rate(reference,test):
    reference_psnr=np.asarray([result["psnr"] for result in reference])
    reference_br=np.asarray([result["br"] for result in reference])
    test_psnr=np.asarray([result["psnr"] for result in test])
    test_br=np.asarray([result["br"] for result in test])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    reference_integral=np.polyint(np.polyfit(reference_psnr,np.log(reference_br),2))
    test_integral=np.polyint(np.polyfit(test_psnr,np.log(test_br),2))
    return (np.exp(((np.polyval(test_integral,upper)-np.polyval(test_integral,lower))-(np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)))/(upper-lower))-1)*100

def main():
    with open(idw_result_path,newline="") as file:
        idw_reference=[row for row in csv.DictReader(file) if row["name"]=="ref4_power3.6"]
    idw_reference=[{"br":float(row["br"]),"psnr":float(row["psnr"])} for row in idw_reference]
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_rbf_screen_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        prediction_structure_path=temporary_dir/"prediction_structure.bin"
        for shape in shape_vec:
            current=[]
            for rel_eb in rel_eb_vec:
                config_path=temporary_dir/f"point_sets_{rel_eb}.json"
                write_point_set_config(config_path,rel_eb)
                arguments=["-data-dir",test_dir,"-variable","attr0","-time-step",f"[0,{time_step_num-1}]","-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",matrix_path,"-dag-cache",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",ref_num,"-actual-ref-num",ref_num,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",cache_threshold,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental","-cross-time-label-predictor-experimental","previous","-rbf-kernel","matern32","-rbf-shape",shape,"-rbf-nugget",1e-6,*extra_arguments]
                output=run_command([compressor,*arguments])
                psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",output,"总PSNR")
                decompression_time=float("nan")
                if verify_decompression:
                    decompression_output=run_command([decompressor,*arguments])
                    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
                    decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")
                result={"ref_num":ref_num,"shape":shape,"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",output,"总BR"),"psnr":psnr,"preprocessing_time":read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",output,"预处理时间"),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",output,"压缩时间"),"decompression_time":decompression_time}
                result_vec.append(result)
                current.append(result)
                for time_step in range(time_step_num):
                    output_path=test_dir/f"attr0.dat.{time_step}.unsz"
                    if output_path.exists():
                        output_path.unlink()
            decompression_text=f"，decompression={math.exp(sum(math.log(result['decompression_time']) for result in current)/len(current)):.6f} s" if verify_decompression else ""
            print(f"ref={ref_num},shape={shape:g}：BD-rate={calculate_bd_rate(idw_reference,current):+.6f}%，compression={math.exp(sum(math.log(result['compression_time']) for result in current)/len(current)):.6f} s{decompression_text}",flush=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_vec)

if __name__=="__main__":
    main()
