import csv,math,os,re,subprocess,tempfile

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
compressor=os.path.join(project_root,"build/s1_compress")
decompressor=os.path.join(project_root,"build/s1_decompress")
result_path=os.path.join(project_root,"independent_block_loo_benchmark.csv")
dataset_vec=(
    {"name":"Ocean-temperature","folder":"Ocean","variable":"temperature","adjust_coord":False},
    {"name":"Ocean-layerThickness","folder":"Ocean","variable":"layerThickness","adjust_coord":False},
    {"name":"Katrina-attr0","folder":"Katrina","variable":"attr0","adjust_coord":True},
)
block_num_vec=(1,2,4)
rel_eb_vec=(0.02,0.01,0.005)

def run_command(command):
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(command)}")
    return result.stdout

def read_value(pattern,text,name):
    match=re.search(pattern,text,re.MULTILINE)
    if not match:
        raise RuntimeError(f"输出中缺少{name}")
    return float(match.group(1))

def calculate_bd_rate(base_result_vec,test_result_vec):
    import numpy as np
    base_psnr=np.asarray([result["psnr"] for result in base_result_vec])
    test_psnr=np.asarray([result["psnr"] for result in test_result_vec])
    base_log_br=np.log(np.asarray([result["br"] for result in base_result_vec]))
    test_log_br=np.log(np.asarray([result["br"] for result in test_result_vec]))
    min_psnr=max(base_psnr.min(),test_psnr.min())
    max_psnr=min(base_psnr.max(),test_psnr.max())
    if not min_psnr<max_psnr:
        return math.nan,min_psnr,max_psnr
    base_polynomial=np.polyfit(base_psnr,base_log_br,2)
    test_polynomial=np.polyfit(test_psnr,test_log_br,2)
    base_integral=np.polyval(np.polyint(base_polynomial),max_psnr)-np.polyval(np.polyint(base_polynomial),min_psnr)
    test_integral=np.polyval(np.polyint(test_polynomial),max_psnr)-np.polyval(np.polyint(test_polynomial),min_psnr)
    return (math.exp((test_integral-base_integral)/(max_psnr-min_psnr))-1)*100,min_psnr,max_psnr

def benchmark(dataset,block_num,rel_eb,test_dir):
    point_num=os.path.getsize(os.path.join(data_root,dataset["folder"],"coords.dat"))//(3*4)
    block_size=(point_num+block_num-1)//block_num if block_num>1 else 0
    config_name="global" if block_num==1 else f"block_num_{block_num}"
    variable=f"{dataset['variable']}_{config_name}_{str(rel_eb).replace('.','_')}"
    source_dir=os.path.join(data_root,dataset["folder"])
    os.symlink(os.path.join(source_dir,f"{dataset['variable']}.dat.0"),os.path.join(test_dir,f"{variable}.dat.0"))
    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step","{0}","-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental","-dag-cache",os.path.join(test_dir,f"prediction_structure_{config_name}.bin"),"-predictor","hybrid_loo"]
    if block_num>1:
        arguments.extend(["-partition-num",str(block_num)])
    if dataset["adjust_coord"]:
        arguments.append("-adjust-coord")
    compression_output=run_command([compressor,*arguments])
    actual_block_num=1 if block_num==1 else int(read_value(r"^Independent spatial blocks: ([0-9]+),",compression_output,"独立块数量"))
    if actual_block_num!=block_num:
        raise RuntimeError(f"期望生成{block_num}个块，实际生成{actual_block_num}个块")
    decompression_arguments=[]
    skip_next=False
    for argument in arguments:
        if skip_next:
            skip_next=False
            continue
        if argument=="-predictor":
            skip_next=True
            continue
        decompression_arguments.append(argument)
    decompression_output=run_command([decompressor,*decompression_arguments])
    result={
        "dataset":dataset["name"],
        "block_num":block_num,
        "block_size":block_size,
        "rel_eb":rel_eb,
        "br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),
        "psnr":read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR"),
        "preprocessing_time":read_value(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),
        "compression_time":read_value(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),
        "decompression_time":read_value(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间"),
        "fitting_ratio":read_value(r"^hybrid_loo_fitting_ratio=([^,]+),",compression_output,"混合拟合比例"),
    }
    decompressed_psnr=read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    if not math.isclose(result["psnr"],decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{result['psnr']} != {decompressed_psnr}")
    return result

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    for dataset in dataset_vec:
        with tempfile.TemporaryDirectory(prefix=f"unsz_block_loo_{dataset['folder']}_",dir=temporary_root) as test_dir:
            source_dir=os.path.join(data_root,dataset["folder"])
            os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
            result_map={}
            for block_num in block_num_vec:
                block_result_vec=[]
                for rel_eb in rel_eb_vec:
                    result=benchmark(dataset,block_num,rel_eb,test_dir)
                    result_vec.append(result)
                    block_result_vec.append(result)
                    print(f"{dataset['name']} block_num={block_num},block_size={result['block_size']},rel_eb={rel_eb}: br={result['br']:.6f},psnr={result['psnr']:.6f},fitting_ratio={result['fitting_ratio']:.6f},preprocessing={result['preprocessing_time']:.6f} s,compression={result['compression_time']:.6f} s,decompression={result['decompression_time']:.6f} s",flush=True)
                result_map[block_num]=block_result_vec
            for block_num in block_num_vec[1:]:
                bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map[1],result_map[block_num])
                print(f"{dataset['name']} BD-rate(block_num={block_num} vs global)={bd_rate:.6f}%,overlap_psnr=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
