import csv,math,re,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project_root))
from py_codes.py_unsz_core.adjust_coord import fit_sample_transform_matrix,normalize_constant_matrix
from py_codes.py_unsz_core.init import init
from py_codes.py_unsz_core.param import Param
source_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Ocean")
compressor=project_root/"build/s1_compress"
decompressor=project_root/"build/s1_decompress"
field="zTop"
rel_eb_vec=(1e-3,9e-4)
result_path=project_root/"py_codes/trash/results/ocean_ztop_partition_adjust_screen.csv"

def split_partition(coord,partition_num):
    partition_vec=[np.arange(coord.shape[0],dtype=np.int64)]
    while len(partition_vec)<partition_num:
        next_partition=[]
        for point_ids in partition_vec:
            current_coord=coord[point_ids]
            split_dim=int(np.argmax(np.ptp(current_coord,axis=0)))
            order=np.lexsort((point_ids,current_coord[:,split_dim]))
            middle=point_ids.size//2
            next_partition.extend((point_ids[order[:middle]],point_ids[order[middle:]]))
        partition_vec=next_partition
    return partition_vec

def train_partition_matrices(param,data,partition_vec):
    rng=np.random.default_rng(0)
    matrix_vec=[]
    time_dict={name:0 for name in ("第一次采样、应用已知变换","有效点检查和第二次采样","建树","最近邻查询","整理点对","合并点对数组","优化变换矩阵")}
    start_time=time.perf_counter()
    for partition_id,point_ids in enumerate(partition_vec):
        transform_matrix=np.eye(param.dim,dtype=np.float64)
        first_sample_num=max(param.actual_ref_num+1,int(np.ceil(point_ids.size*0.25)))
        for iteration in range(4):
            first_sample_ids=np.sort(rng.choice(point_ids,size=first_sample_num,replace=False))
            fitted_matrix,_,_,_=fit_sample_transform_matrix(param,data[None,:],[0],first_sample_ids,transform_matrix,"second_power",0.1,100,"gauss_newton",rng,time_dict,0.5,3.6,1e-16)
            transform_matrix=fitted_matrix if iteration==0 else normalize_constant_matrix(0.5*transform_matrix+0.5*fitted_matrix)
        matrix_vec.append(transform_matrix)
        print(f"partition {partition_id}:point_num={point_ids.size},matrix=\n{transform_matrix}",flush=True)
    print(f"partition矩阵训练时间={time.perf_counter()-start_time:.6f}s",flush=True)
    return matrix_vec

def transform_partitions(coord,partition_vec,matrix_vec):
    transformed_coord=np.empty_like(coord,dtype=np.float64)
    for point_ids,matrix in zip(partition_vec,matrix_vec):
        center=np.mean(coord[point_ids].astype(np.float64),axis=0)
        transformed_coord[point_ids]=center+(coord[point_ids]-center)@matrix
    return transformed_coord.astype(np.float32)

def run_command(command):
    result=subprocess.run([str(value) for value in command],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(str(value) for value in command)}")
    return result.stdout

def read_float(pattern,output,name):
    match=re.search(pattern,output,re.MULTILINE)
    if match is None:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def benchmark(test_dir,config,rel_eb,prediction_structure_path):
    arguments=["-data-dir",test_dir,"-variable",field,"-time-step","{0}","-M","REL",rel_eb,"-dim",3,"-prediction-structure",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",10,"-actual-ref-num",10,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",8,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental","-global-order-before-round-split-experimental"]
    compression_output=run_command([compressor,*arguments])
    decompression_output=run_command([decompressor,*arguments])
    psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"压缩PSNR")
    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr}!={decompressed_psnr}")
    for suffix in (".unsz",".unsz.bin"):
        output_path=Path(str(test_dir/f"{field}.dat.0")+suffix)
        if output_path.is_file() or output_path.is_symlink():
            output_path.unlink()
    return {"config":config,"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"BR"),"psnr":psnr}

def interpolate_log_br(row_vec,target_psnr):
    row_vec=sorted(row_vec,key=lambda row:row["psnr"])
    lower,upper=row_vec
    ratio=(target_psnr-lower["psnr"])/(upper["psnr"]-lower["psnr"])
    return math.exp(math.log(lower["br"])*(1-ratio)+math.log(upper["br"])*ratio)

def main():
    param=Param()
    param.coord_path=str(source_dir/"coords.dat")
    param.original_data_path_vec=[str(source_dir/f"{field}.dat.0")]
    param.dim=3
    param.rel_eb=1e-3
    init(param)
    coord=param.coord_array
    data=np.fromfile(param.original_data_path_vec[0],dtype=param.data_dtype)
    coord_config={"original_coord":coord}
    for partition_num in (2,4):
        partition_vec=split_partition(coord,partition_num)
        matrix_vec=train_partition_matrices(param,data,partition_vec)
        coord_config[f"partition_{partition_num}"]=transform_partitions(coord,partition_vec,matrix_vec)
    row_vec=[]
    with tempfile.TemporaryDirectory(prefix="unsz_partition_adjust_") as temporary_dir:
        temporary_dir=Path(temporary_dir)
        for config,current_coord in coord_config.items():
            config_dir=temporary_dir/config
            config_dir.mkdir()
            current_coord.tofile(config_dir/"coords.dat")
            (config_dir/f"{field}.dat.0").symlink_to(source_dir/f"{field}.dat.0")
            prediction_structure_path=config_dir/"prediction.bin"
            for rel_eb in rel_eb_vec:
                row=benchmark(config_dir,config,rel_eb,prediction_structure_path)
                row_vec.append(row)
                print(f"{config} REL={rel_eb:g}:BR={row['br']:.9f},PSNR={row['psnr']:.6f}",flush=True)
    target_psnr=next(row["psnr"] for row in row_vec if row["config"]=="original_coord" and row["rel_eb"]==rel_eb_vec[0])
    summary=[]
    original_br=interpolate_log_br([row for row in row_vec if row["config"]=="original_coord"],target_psnr)
    for config in coord_config:
        matched_br=interpolate_log_br([row for row in row_vec if row["config"]==config],target_psnr)
        summary.append({"config":config,"target_psnr":target_psnr,"matched_br":matched_br,"br_change_percent":(matched_br/original_br-1)*100})
        print(f"{config}:同PSNR BR变化={(matched_br/original_br-1)*100:.6f}%",flush=True)
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with result_path.open("w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

if __name__=="__main__":
    main()
