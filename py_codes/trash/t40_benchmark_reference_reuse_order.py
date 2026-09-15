import csv,math,os,re,subprocess,tempfile
import numpy as np

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
compressor=os.path.join(project_root,"build/s1_compress")
decompressor=os.path.join(project_root,"build/s1_decompress")
result_path=os.path.join(project_root,"reference_reuse_order_benchmark.csv")
rel_eb_vec=(0.02,0.01,0.005)
dataset_vec=(
    {"name":"Ocean-temperature","folder":"Ocean","variable":"temperature","adjust_coord":False},
    {"name":"Katrina-attr0","folder":"Katrina","variable":"attr0","adjust_coord":True},
)
config_vec=(
    {"name":"d_only","flag":"-reverse-mean-distance-only-order-experimental"},
    {"name":"reference_reuse_first","flag":"-reverse-reference-reuse-order-experimental"},
)

def run_command(command):
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(command)}")
    return result.stdout

def read_value(pattern,text,name,required=True):
    match=re.search(pattern,text,re.MULTILINE)
    if not match:
        if required:
            raise RuntimeError(f"输出中缺少{name}")
        return math.nan
    return float(match.group(1))

def calculate_bd_rate(base_result_vec,test_result_vec):
    base_psnr=np.asarray([result["psnr"] for result in base_result_vec])
    test_psnr=np.asarray([result["psnr"] for result in test_result_vec])
    base_log_br=np.log(np.asarray([result["br"] for result in base_result_vec]))
    test_log_br=np.log(np.asarray([result["br"] for result in test_result_vec]))
    min_psnr=max(base_psnr.min(),test_psnr.min())
    max_psnr=min(base_psnr.max(),test_psnr.max())
    base_polynomial=np.polyfit(base_psnr,base_log_br,2)
    test_polynomial=np.polyfit(test_psnr,test_log_br,2)
    base_integral=np.polyval(np.polyint(base_polynomial),max_psnr)-np.polyval(np.polyint(base_polynomial),min_psnr)
    test_integral=np.polyval(np.polyint(test_polynomial),max_psnr)-np.polyval(np.polyint(test_polynomial),min_psnr)
    return (math.exp((test_integral-base_integral)/(max_psnr-min_psnr))-1)*100,min_psnr,max_psnr

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    for dataset in dataset_vec:
        source_dir=os.path.join(data_root,dataset["folder"])
        with tempfile.TemporaryDirectory(prefix=f"unsz_reuse_{dataset['folder']}_",dir=temporary_root) as test_dir:
            os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
            result_map={}
            for config in config_vec:
                prediction_structure_path=os.path.join(test_dir,f"{config['name']}.dag")
                config_result_vec=[]
                for rel_eb in rel_eb_vec:
                    variable=f"{config['name']}_{str(rel_eb).replace('.','_')}"
                    os.symlink(os.path.join(source_dir,f"{dataset['variable']}.dat.0"),os.path.join(test_dir,f"{variable}.dat.0"))
                    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step","{0}","-M","REL",str(rel_eb),config["flag"],"-dag-cache",prediction_structure_path]
                    if dataset["adjust_coord"]:
                        arguments.append("-adjust-coord")
                    compression_output=run_command([compressor,*arguments])
                    decompression_output=run_command([decompressor,*arguments])
                    result={
                        "dataset":dataset["name"],
                        "config":config["name"],
                        "rel_eb":rel_eb,
                        "br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),
                        "psnr":read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR"),
                        "huffman_bytes":read_value(r"^value_huffman_bytes=([0-9]+),",compression_output,"Huffman长度"),
                        "zstd_bytes":read_value(r"^value_huffman_bytes=[0-9]+, value_zstd_bytes=([0-9]+),",compression_output,"Zstd长度"),
                        "preprocessing_time":read_value(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),
                        "reference_overlap":read_value(r"^Selected adjacent reference overlap: ([^ ]+) /",compression_output,"参考重叠",False),
                    }
                    decompressed_psnr=read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压PSNR")
                    if not math.isclose(result["psnr"],decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
                        raise RuntimeError("压缩和解压PSNR不一致")
                    result_vec.append(result)
                    config_result_vec.append(result)
                    print(f"{dataset['name']} {config['name']} rel_eb={rel_eb}: br={result['br']:.6f},psnr={result['psnr']:.6f},huffman={result['huffman_bytes']:.0f},zstd={result['zstd_bytes']:.0f},preprocessing={result['preprocessing_time']:.6f} s",flush=True)
                result_map[config["name"]]=config_result_vec
            bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map["d_only"],result_map["reference_reuse_first"])
            print(f"{dataset['name']} BD-rate(reference_reuse_first vs d_only)={bd_rate:.6f}%,overlap_psnr=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
