import csv,math,os,re,subprocess,tempfile
from pathlib import Path

project_root=Path(__file__).resolve().parents[2]
data_dir=Path("/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data/Tsunami")
compressor=project_root/"build/s1_compress"
decompressor=project_root/"build/s1_decompress"
matrix_path=project_root/"adjust_coord_matrices/Tsunami_momentum_x_spatial.txt"
result_path=project_root/"py_codes/trash/results/tsunami_momentum_x_predictor_screen.csv"
rel_eb_vec=(1e-3,1e-5)
power_vec=(1,2,3,3.6,4,5,6,8)
ref_num_vec=(4,6,8,10)
config_vec=[{"name":f"idw_p{power:g}_r{ref_num}","predictor":"inverse_proportion","power":power,"ref_num":ref_num} for ref_num in ref_num_vec for power in power_vec]
config_vec+=[{"name":f"fitting_r{ref_num}","predictor":"fitting","power":3.6,"ref_num":ref_num} for ref_num in ref_num_vec]
config_vec+=[{"name":f"hybrid_loo_r{ref_num}","predictor":"hybrid_loo","power":3.6,"ref_num":ref_num} for ref_num in ref_num_vec]

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
    for suffix in (".unsz",".unsz.bin"):
        path=test_dir/f"momentum_x.dat.0{suffix}"
        if path.exists():
            path.unlink()

def benchmark(test_dir,prediction_structure_path,config,rel_eb):
    common_arguments=["-data-dir",test_dir,"-variable","momentum_x","-time-step","{0}","-dim",2,"-M","REL",rel_eb,"-adjust-coord-matrix-file",matrix_path,"-reverse-mean-distance-only-order-experimental",*config.get("extra_arguments",()),"-dag-cache",prediction_structure_path,"-max-ref-num",config.get("max_ref_num",10),"-actual-ref-num",config["ref_num"],"-inverse-distance-power",config["power"]]
    compression_output=run_command([compressor,*common_arguments,"-predictor",config["predictor"],"-prediction-residual-stats"])
    decompression_output=run_command([decompressor,*common_arguments])
    psnr=read_float(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR")
    decompressed_psnr=read_float(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    if not math.isclose(psnr,decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{psnr} != {decompressed_psnr}")
    stats_match=re.search(r"^prediction_residual_stats: num=(\d+), mse=([^,]+), mae=([^,]+), qb_entropy_br=([^,]+), qb_zero_ratio=([^,]+), qb_same_as_previous_ratio=([^ ]+)$",compression_output,re.MULTILINE)
    bytes_match=re.search(r"^value_huffman_bytes=(\d+), value_zstd_bytes=(\d+), file_bytes=(\d+)$",compression_output,re.MULTILINE)
    if stats_match is None or bytes_match is None:
        raise RuntimeError("输出中缺少预测残差或码流长度统计")
    cleanup(test_dir)
    return {"config":config["name"],"predictor":config["predictor"],"power":config["power"],"ref_num":config["ref_num"],"rel_eb":rel_eb,"br":read_float(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),"psnr":psnr,"prediction_mse":float(stats_match.group(2)),"prediction_mae":float(stats_match.group(3)),"qb_entropy_br":float(stats_match.group(4)),"qb_zero_ratio":float(stats_match.group(5)),"qb_same_as_previous_ratio":float(stats_match.group(6)),"value_huffman_bytes":int(bytes_match.group(1)),"value_zstd_bytes":int(bytes_match.group(2)),"file_bytes":int(bytes_match.group(3)),"compression_time":read_float(r"^total compression time= ([^ ]+) s$",compression_output,"总压缩时间"),"decompression_time":read_float(r"^total decompression time= ([^ ]+) s$",decompression_output,"总解压时间")}

def print_pareto(results,rel_eb):
    current=[result for result in results if result["rel_eb"]==rel_eb]
    pareto=[]
    for result in current:
        dominated=any(other["br"]<=result["br"] and other["psnr"]>=result["psnr"] and (other["br"]<result["br"] or other["psnr"]>result["psnr"]) for other in current)
        if not dominated:
            pareto.append(result)
    pareto.sort(key=lambda result:result["br"])
    print(f"REL={rel_eb}的Pareto配置：")
    for result in pareto:
        print(f"  {result['config']}:BR={result['br']:.6f},PSNR={result['psnr']:.6f},prediction MSE={result['prediction_mse']:.9g},qb entropy={result['qb_entropy_br']:.6f},Zstd={result['value_zstd_bytes']} B")

def main():
    result_path.parent.mkdir(parents=True,exist_ok=True)
    results=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    with tempfile.TemporaryDirectory(prefix="unsz_tsunami_predictor_",dir=temporary_root) as temporary_dir:
        test_dir=Path(temporary_dir)/"data"
        test_dir.mkdir()
        (test_dir/"coords.dat").symlink_to(data_dir/"coords.dat")
        (test_dir/"momentum_x.dat.0").symlink_to(data_dir/"momentum_x.dat.0")
        prediction_structure_path=Path(temporary_dir)/"prediction_structure.bin"
        for config_id,config in enumerate(config_vec):
            for rel_eb in rel_eb_vec:
                result=benchmark(test_dir,prediction_structure_path,config,rel_eb)
                results.append(result)
                print(f"{config_id+1}/{len(config_vec)} {config['name']} REL={rel_eb}:BR={result['br']:.6f},PSNR={result['psnr']:.6f},MSE={result['prediction_mse']:.9g},entropy={result['qb_entropy_br']:.6f},Zstd={result['value_zstd_bytes']} B",flush=True)
    with open(result_path,"w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    for rel_eb in rel_eb_vec:
        print_pareto(results,rel_eb)
    print(f"结果={result_path}")

if __name__=="__main__":
    main()
