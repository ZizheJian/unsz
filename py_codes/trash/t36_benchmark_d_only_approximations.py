import csv,math,os,re,subprocess,tempfile
import numpy as np

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
compressor=os.path.join(project_root,"build/s1_compress")
decompressor=os.path.join(project_root,"build/s1_decompress")
result_path=os.path.join(project_root,"d_only_order_benchmark.csv")
rel_eb_vec=(0.02,0.01,0.005)
dataset_vec=(
    {"name":"Ocean-temperature","folder":"Ocean","variable":"temperature","adjust_coord":False},
    {"name":"Katrina-attr0","folder":"Katrina","variable":"attr0","adjust_coord":True},
)
config_vec=(
    {"name":"exact_10","flags":[]},
    {"name":"cache_9","flags":["-d-only-cache-refill-threshold-experimental","9"]},
    {"name":"candidate_5","flags":["-order-candidate-num-experimental","5"]},
    {"name":"candidate_4","flags":["-order-candidate-num-experimental","4"]},
)

def run_command(command):
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode!=0:
        print(result.stdout,flush=True)
        raise RuntimeError(f"命令失败：{' '.join(command)}")
    return result.stdout

def read_value(pattern,text,name,value_type=float):
    match=re.search(pattern,text,re.MULTILINE)
    if not match:
        raise RuntimeError(f"输出中缺少{name}")
    return value_type(match.group(1))

def calculate_bd_rate(base_result_vec,test_result_vec):
    base_psnr=np.asarray([result["psnr"] for result in base_result_vec])
    test_psnr=np.asarray([result["psnr"] for result in test_result_vec])
    base_log_br=np.log(np.asarray([result["br"] for result in base_result_vec]))
    test_log_br=np.log(np.asarray([result["br"] for result in test_result_vec]))
    min_psnr=max(base_psnr.min(),test_psnr.min())
    max_psnr=min(base_psnr.max(),test_psnr.max())
    if not min_psnr<max_psnr:
        raise RuntimeError("两条RD曲线没有重叠的PSNR区间")
    polynomial_degree=min(3,len(base_result_vec)-1,len(test_result_vec)-1)
    base_polynomial=np.polyfit(base_psnr,base_log_br,polynomial_degree)
    test_polynomial=np.polyfit(test_psnr,test_log_br,polynomial_degree)
    base_integral=np.polyval(np.polyint(base_polynomial),max_psnr)-np.polyval(np.polyint(base_polynomial),min_psnr)
    test_integral=np.polyval(np.polyint(test_polynomial),max_psnr)-np.polyval(np.polyint(test_polynomial),min_psnr)
    return (math.exp((test_integral-base_integral)/(max_psnr-min_psnr))-1)*100,min_psnr,max_psnr

def benchmark(dataset,config,rel_eb,test_dir):
    variable=f"{config['name']}_{str(rel_eb).replace('.','_')}"
    source_dir=os.path.join(data_root,dataset["folder"])
    os.symlink(os.path.join(source_dir,f"{dataset['variable']}.dat.0"),os.path.join(test_dir,f"{variable}.dat.0"))
    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step","[0,0]","-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental",*config["flags"]]
    if dataset["adjust_coord"]:
        arguments.append("-adjust-coord")
    compression_output=run_command([compressor,*arguments])
    decompression_output=run_command([decompressor,*arguments])
    result={
        "dataset":dataset["name"],
        "config":config["name"],
        "rel_eb":rel_eb,
        "preprocessing":read_value(r"^preprocessing time excluding adjust coord= ([^ ]+) s$",compression_output,"预处理时间"),
        "sorting":read_value(r"^Target sorting time: ([^ ]+) s$",compression_output,"排序时间"),
        "reference_generation":read_value(r"^Prediction reference generation time: ([^ ]+) s$",compression_output,"参考结构生成时间"),
        "cache_hits":read_value(r"^D-only score cache: ([0-9]+) hits,",compression_output,"缓存命中数",int),
        "reduced_ref_hits":read_value(r"^D-only reduced-reference cache hits: ([0-9]+)$",compression_output,"少参考点缓存命中数",int),
        "cache_misses":read_value(r"^D-only score cache: [0-9]+ hits, ([0-9]+) misses$",compression_output,"缓存未命中数",int),
        "br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),
        "psnr":read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR"),
        "decompressed_psnr":read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR"),
    }
    if not math.isclose(result["psnr"],result["decompressed_psnr"],rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{result['psnr']} != {result['decompressed_psnr']}")
    return result

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    for dataset in dataset_vec:
        source_dir=os.path.join(data_root,dataset["folder"])
        with tempfile.TemporaryDirectory(prefix=f"unsz_d_only_{dataset['folder']}_",dir=temporary_root) as test_dir:
            os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
            result_map={}
            for config in config_vec:
                config_result_vec=[]
                for rel_eb in rel_eb_vec:
                    result=benchmark(dataset,config,rel_eb,test_dir)
                    result_vec.append(result)
                    config_result_vec.append(result)
                    print(f"{dataset['name']} {config['name']} rel_eb={rel_eb}: preprocessing={result['preprocessing']:.6f} s,sorting={result['sorting']:.6f} s,hits={result['cache_hits']},reduced={result['reduced_ref_hits']},misses={result['cache_misses']},br={result['br']:.6f},psnr={result['psnr']:.6f}",flush=True)
                result_map[config["name"]]=config_result_vec
            for config in config_vec[1:]:
                bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map["exact_10"],result_map[config["name"]])
                print(f"{dataset['name']} BD-rate({config['name']} vs exact_10)={bd_rate:.6f}%,overlap_psnr=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
