import csv,json,math,os,re,subprocess,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
result_path=project_root/"py_codes/trash/results/katrina_temporal_label_predictors.csv"
time_step_num=int(os.environ.get("UNSZ_TEMPORAL_LABEL_TIME_STEP_NUM","162"))
point_num=417642
rel_eb=1e-3

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

def hierarchical_time_step_order(time_step_num):
    if time_step_num<=1:
        return [0]
    order=[0,time_step_num-1]
    interval_vec=[(0,time_step_num-1)]
    while interval_vec:
        next_interval_vec=[]
        for left,right in interval_vec:
            if right-left<=1:
                continue
            middle=(left+right)//2
            order.append(middle)
            next_interval_vec.extend(((left,middle),(middle,right)))
        interval_vec=next_interval_vec
    if len(order)!=time_step_num or len(set(order))!=time_step_num:
        raise RuntimeError("层次时间顺序没有恰好覆盖全部time step")
    return order

def aggregate_label_statistics(output):
    match_vec=re.findall(r"^label_prediction_error_ratio=([^,]+), label_position_delta_bytes=(\d+), label_zstd_bytes=(\d+), zstd_level=\d+$",output,re.MULTILINE)
    if len(match_vec)!=time_step_num:
        label_lines="\n".join(line for line in output.splitlines() if "label_" in line)
        raise RuntimeError(f"标签统计数量不正确：{len(match_vec)}!={time_step_num}\n{label_lines}")
    target_num=point_num*time_step_num
    estimated_error_num=sum(point_num*float(error_ratio) for error_ratio,_,_ in match_vec)
    position_delta_bytes=sum(int(byte_num) for _,byte_num,_ in match_vec)
    zstd_bytes=sum(int(byte_num) for _,_,byte_num in match_vec)
    return target_num,estimated_error_num/target_num,position_delta_bytes,zstd_bytes

def main():
    if time_step_num<2 or time_step_num>162:
        raise ValueError("time_step_num必须在[2,162]内")
    result_vec=[]
    mode_vec=[("spatial",None,list(range(time_step_num))),("previous","previous",list(range(time_step_num))),("lorenzo3","lorenzo3",list(range(time_step_num))),("lorenzo4","lorenzo4",list(range(time_step_num))),("endpoint_agreement","endpoint_agreement",hierarchical_time_step_order(time_step_num))]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_temporal_label_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        config_path=temporary_dir/"point_sets.json"
        with open(config_path,"w",encoding="utf-8") as file:
            json.dump({"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]},file,ensure_ascii=False)
        prediction_structure_path=temporary_dir/"prediction_structure.bin"
        for mode_name,predictor,time_step_order in mode_vec:
            time_step_expression="{"+",".join(str(time_step) for time_step in time_step_order)+"}"
            arguments=["-data-dir",test_dir,"-variable","attr0","-time-step",time_step_expression,"-dim",3,"-point-set-config",config_path,"-adjust-coord-matrix-file",matrix_path,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",4,"-actual-ref-num",4,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",4,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental"]
            if predictor is not None:
                arguments.extend(("-cross-time-label-predictor-experimental",predictor))
            print(f"测试{mode_name}，time_step_num={time_step_num}",flush=True)
            compression_output=run_command([compressor,*arguments])
            decompression_output=run_command([decompressor,*arguments])
            br=read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR")
            psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
            decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
            if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
            target_num,error_ratio,position_delta_bytes,label_zstd_bytes=aggregate_label_statistics(compression_output)
            compression_time=read_float(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间")
            decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间")
            total_mb=point_num*time_step_num*4/1e6
            row={"mode":mode_name,"time_step_num":time_step_num,"rel_eb":rel_eb,"br":br,"psnr":psnr,"label_target_num":target_num,"label_prediction_error_ratio":error_ratio,"label_position_delta_bytes":position_delta_bytes,"label_zstd_bytes":label_zstd_bytes,"label_br":label_zstd_bytes*8/(point_num*time_step_num),"compression_mb_s":total_mb/compression_time,"decompression_mb_s":total_mb/decompression_time}
            result_vec.append(row)
            print(f"  BR={br:.6f},label BR={row['label_br']:.6f},label error={error_ratio:.6f},label zstd={label_zstd_bytes} bytes",flush=True)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys(),lineterminator="\n")
        writer.writeheader()
        writer.writerows(result_vec)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
