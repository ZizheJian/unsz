import argparse,csv,json,re,subprocess,tempfile
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

project_root=Path(__file__).resolve().parents[1]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Tpost20")
compressor=project_root/"install/bin/s1_compress"
result_path=project_root/"results/tpost20_scalar_joint_screen.csv"
time_step_num=20
spatial_dim=3

def run_command(command):
    print("运行："," ".join(str(value) for value in command),flush=True)
    result=subprocess.run([str(value) for value in command],cwd=project_root,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError("命令失败："+" ".join(str(value) for value in command))
    return result.stdout

def read_float(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def write_config(path,rel_eb):
    with path.open("w",encoding="utf-8") as file:
        json.dump({"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":rel_eb}]},file)

def prepare_joint_inputs(work_dir,time_scale):
    work_dir.mkdir(exist_ok=True)
    (work_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
    for time_step in range(time_step_num):
        (work_dir/f"scalar.dat.{time_step}").symlink_to(data_dir/f"scalar.dat.{time_step}")
    time_coordinate_path=work_dir/"time_coordinates.dat"
    (np.arange(time_step_num,dtype=np.float64)*time_scale).tofile(time_coordinate_path)
    return time_coordinate_path

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--rel-eb",type=float,default=1e-3)
    parser.add_argument("--time-scale-multipliers",type=float,nargs="+",default=(0.25,0.5,1,2,4))
    parser.add_argument("--partition-num",type=int,default=32)
    parser.add_argument("--thread-num",type=int,default=16)
    args=parser.parse_args()
    if not compressor.is_file():
        raise FileNotFoundError(compressor)
    coord=np.fromfile(data_dir/"coords.dat",dtype=np.float32).reshape(-1,spatial_dim)
    point_num=coord.shape[0]
    nearest_distance=cKDTree(coord.astype(np.float64)).query(coord,k=2,workers=1)[0][:,1]
    median_distance=float(np.median(nearest_distance))
    rows=[]
    with tempfile.TemporaryDirectory(prefix="unsz_tpost20_joint_") as temporary_dir_string:
        work_dir=Path(temporary_dir_string)
        config_path=work_dir/"point_sets.json"
        write_config(config_path,args.rel_eb)
        for multiplier in args.time_scale_multipliers:
            time_scale=median_distance*multiplier
            joint_dir=work_dir/f"scale_{multiplier:g}"
            time_coordinate_path=prepare_joint_inputs(joint_dir,time_scale)
            prediction_structure_path=work_dir/f"prediction_structure_scale_{multiplier:g}.dag"
            command=[compressor,"-data-dir",joint_dir,"-variable","scalar","-time-step","[0,20)","-time-step-coordinate-file",time_coordinate_path,"-dim",spatial_dim,"-point-set-config",config_path,"-dag-cache",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",4,"-actual-ref-num",4,"-inverse-distance-power",2,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",3,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental","-cross-time-spatial-point-num-experimental",point_num,"-partition-num",args.partition_num,"-thread-num",args.thread_num]
            output=run_command(command)
            row={"rel_eb":args.rel_eb,"time_scale_multiplier":multiplier,"time_scale":time_scale,"partition_num":args.partition_num,"thread_num":args.thread_num,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",output,"BR"),"psnr":read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",output,"PSNR"),"compression_time_seconds":read_float(r"^total compression time= ([^ ]+) s$",output,"核心压缩时间"),"prediction_structure_bytes":prediction_structure_path.stat().st_size}
            rows.append(row)
            print(f"scale={multiplier:g},BR={row['br']:.6f},PSNR={row['psnr']:.6f},核心压缩时间={row['compression_time_seconds']:.3f}s",flush=True)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with result_path.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"结果={result_path}")

if __name__=="__main__":
    main()
