import csv,json,math,os,re,subprocess,tempfile
from pathlib import Path
import numpy as np

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Katrina")
compressor=project_root/"install/bin/s1_compress"
decompressor=project_root/"install/bin/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Katrina_attr0_spatial.txt"
result_path=project_root/"py_codes/trash/results/katrina_reference_num_rd.csv"
summary_path=project_root/"py_codes/trash/results/katrina_reference_num_rd_summary.csv"
time_step_num=162
point_num=417642
total_mb=point_num*time_step_num*4/1e6
rel_eb_vec=(1e-1,1e-2,1e-3,1e-4,1e-5)
config_vec=(
    {"name":"ref10_cache8","ref_num":10,"cache_threshold":8},
    {"name":"ref4_cache4","ref_num":4,"cache_threshold":4},
    {"name":"ref4_cache3","ref_num":4,"cache_threshold":3},
)

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

def sum_stage_time(text,name):
    return sum(float(value) for value in re.findall(rf"compression stage time: .*?{name}=([0-9.eE+-]+) s",text))

def write_point_set_config(path,rel_eb):
    config={"version":1,"rule_type":"ordered","labels":[{"method":"inverse_proportion","eb_mode":"REL","eb_value":rel_eb,"condition":{"operator":">=","value":-1000}},{"method":"constant"}]}
    with open(path,"w",encoding="utf-8") as file:
        json.dump(config,file,ensure_ascii=False,indent=2)

def cleanup(test_dir):
    for time_step in range(time_step_num):
        for suffix in (".unsz",".unsz.bin"):
            path=test_dir/f"attr0.dat.{time_step}{suffix}"
            if path.exists():
                path.unlink()

def benchmark(test_dir,point_set_config_path,config,prediction_structure_path,rel_eb):
    ref_num=config["ref_num"]
    cache_threshold=config["cache_threshold"]
    common_arguments=["-data-dir",test_dir,"-variable","attr0","-time-step","[0,161]","-dim",3,"-point-set-config",point_set_config_path,"-adjust-coord-matrix-file",matrix_path,"-dag-cache",prediction_structure_path,"-predict-round-num",8,"-max-ref-num",ref_num,"-actual-ref-num",ref_num,"-inverse-distance-power",3.6,"-kd-leaf-size",32,"-d-only-cache-refill-threshold-experimental",cache_threshold,"-eb-schedule-experimental","round_point_ratio",0.8,"-reverse-mean-distance-only-order-experimental"]
    compression_output=run_command([compressor,*common_arguments,"-compression-stage-time"])
    decompression_output=run_command([decompressor,*common_arguments])
    br=read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR")
    psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR")
    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
    preprocessing_time=read_float(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间")
    compression_time=read_float(r"^total compression time= ([^ ]+) s$",compression_output,"总压缩时间")
    decompression_time=read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"总解压时间")
    cache_match=re.search(r"^D-only score cache: (\d+) hits, (\d+) misses$",compression_output,re.MULTILINE)
    reduced_match=re.search(r"^D-only reduced-reference cache hits: (\d+)$",compression_output,re.MULTILINE)
    result={"name":config["name"],"ref_num":ref_num,"cache_threshold":cache_threshold,"rel_eb":rel_eb,"br":br,"psnr":psnr,"preprocessing_time":preprocessing_time,"compression_time":compression_time,"decompression_time":decompression_time,"compression_mb_s":total_mb/compression_time,"decompression_mb_s":total_mb/decompression_time,"point_set_time":sum_stage_time(compression_output,"point_set"),"label_prediction_time":sum_stage_time(compression_output,"label_prediction"),"value_prediction_time":sum_stage_time(compression_output,"value_prediction"),"lossless_encoding_time":sum_stage_time(compression_output,"lossless_encoding"),"prediction_structure_bytes":prediction_structure_path.stat().st_size,"cache_hit_num":int(cache_match.group(1)) if cache_match else -1,"cache_miss_num":int(cache_match.group(2)) if cache_match else -1,"reduced_ref_cache_hit_num":int(reduced_match.group(1)) if reduced_match else -1}
    cleanup(test_dir)
    return result

def calculate_bd_rate(reference,test):
    reference_psnr=np.asarray([result["psnr"] for result in reference])
    reference_br=np.asarray([result["br"] for result in reference])
    test_psnr=np.asarray([result["psnr"] for result in test])
    test_br=np.asarray([result["br"] for result in test])
    lower=max(reference_psnr.min(),test_psnr.min())
    upper=min(reference_psnr.max(),test_psnr.max())
    reference_integral=np.polyint(np.polyfit(reference_psnr,np.log(reference_br),3))
    test_integral=np.polyint(np.polyfit(test_psnr,np.log(test_br),3))
    reference_area=np.polyval(reference_integral,upper)-np.polyval(reference_integral,lower)
    test_area=np.polyval(test_integral,upper)-np.polyval(test_integral,lower)
    return (np.exp((test_area-reference_area)/(upper-lower))-1)*100

def geometric_mean(value_vec):
    return math.exp(sum(math.log(value) for value in value_vec)/len(value_vec))

def main():
    result_vec=[]
    result_map={}
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_katrina_reference_num_rd_",dir=temporary_root) as temporary_dir_string:
        temporary_dir=Path(temporary_dir_string)
        test_dir=temporary_dir/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        for time_step in range(time_step_num):
            (test_dir/f"attr0.dat.{time_step}").symlink_to(data_dir/f"attr0.dat.{time_step}")
        for config in config_vec:
            config_result_vec=[]
            prediction_structure_path=temporary_dir/f"{config['name']}.prediction_structure.bin"
            for rel_eb in rel_eb_vec:
                point_set_config_path=temporary_dir/f"point_sets_{config['name']}_{rel_eb}.json"
                write_point_set_config(point_set_config_path,rel_eb)
                print(f"测试{config['name']}，REL={rel_eb}",flush=True)
                result=benchmark(test_dir,point_set_config_path,config,prediction_structure_path,rel_eb)
                result_vec.append(result)
                config_result_vec.append(result)
                print(f"  BR={result['br']:.9f},PSNR={result['psnr']:.6f},preprocessing={result['preprocessing_time']:.6f} s,compression={result['compression_mb_s']:.3f} MB/s,decompression={result['decompression_mb_s']:.3f} MB/s",flush=True)
            result_map[config["name"]]=config_result_vec
    result_path.parent.mkdir(parents=True,exist_ok=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    baseline=result_map[config_vec[0]["name"]]
    summary_vec=[]
    for config in config_vec:
        current=result_map[config["name"]]
        summary={"name":config["name"],"ref_num":config["ref_num"],"cache_threshold":config["cache_threshold"],"bd_rate_vs_ref10_cache8":calculate_bd_rate(baseline,current),"first_preprocessing_time":current[0]["preprocessing_time"],"compression_mb_s_geomean":geometric_mean([result["compression_mb_s"] for result in current]),"decompression_mb_s_geomean":geometric_mean([result["decompression_mb_s"] for result in current]),"prediction_structure_bytes":current[0]["prediction_structure_bytes"],"cache_hit_num":current[0]["cache_hit_num"],"cache_miss_num":current[0]["cache_miss_num"],"reduced_ref_cache_hit_num":current[0]["reduced_ref_cache_hit_num"]}
        summary_vec.append(summary)
        print(f"{summary['name']}：BD-rate={summary['bd_rate_vs_ref10_cache8']:+.6f}%，首次预处理={summary['first_preprocessing_time']:.6f} s，压缩={summary['compression_mb_s_geomean']:.3f} MB/s，解压={summary['decompression_mb_s_geomean']:.3f} MB/s",flush=True)
    with open(summary_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=summary_vec[0].keys())
        writer.writeheader()
        writer.writerows(summary_vec)
    print(f"逐点结果已写入{result_path}",flush=True)
    print(f"汇总结果已写入{summary_path}",flush=True)

if __name__=="__main__":
    main()
