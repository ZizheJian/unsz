import csv,math,os,re,subprocess,tempfile
import numpy as np

project_root=os.path.dirname(os.path.abspath(__file__))
data_root="/anvil/projects/x-cis240192/x-zjian1-in-cis240192/unstructured-data"
compressor=os.path.join(project_root,"build/s1_compress")
decompressor=os.path.join(project_root,"build/s1_decompress")
result_path=os.path.join(project_root,"predictor_formula_benchmark.csv")
rel_eb_vec=(0.02,0.01,0.005)
dataset_vec=(
    {"name":"Ocean-temperature","folder":"Ocean","variable":"temperature","adjust_coord":False},
    {"name":"Ocean-layerThickness","folder":"Ocean","variable":"layerThickness","adjust_coord":False},
    {"name":"Katrina-attr0","folder":"Katrina","variable":"attr0","adjust_coord":True},
)
config_vec=tuple({"name":f"idw_p{power:g}_r{ref_num}","predictor":"inverse_proportion","power":power,"ref_num":ref_num} for power in (2,3.61,5) for ref_num in (4,5,10))+(
    {"name":"idw_p7.22_r10","predictor":"inverse_proportion","power":7.22,"ref_num":10},
    {"name":"fitting_r10","predictor":"fitting","power":3.61,"ref_num":10},
    {"name":"hybrid_loo_r10","predictor":"hybrid_loo","power":3.61,"ref_num":10},
)
base_name="idw_p3.61_r10"

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
    base_psnr=np.asarray([result["psnr"] for result in base_result_vec])
    test_psnr=np.asarray([result["psnr"] for result in test_result_vec])
    base_log_br=np.log(np.asarray([result["br"] for result in base_result_vec]))
    test_log_br=np.log(np.asarray([result["br"] for result in test_result_vec]))
    min_psnr=max(base_psnr.min(),test_psnr.min())
    max_psnr=min(base_psnr.max(),test_psnr.max())
    if not min_psnr<max_psnr:
        return math.nan,min_psnr,max_psnr
    degree=min(2,len(base_result_vec)-1,len(test_result_vec)-1)
    base_polynomial=np.polyfit(base_psnr,base_log_br,degree)
    test_polynomial=np.polyfit(test_psnr,test_log_br,degree)
    base_integral=np.polyval(np.polyint(base_polynomial),max_psnr)-np.polyval(np.polyint(base_polynomial),min_psnr)
    test_integral=np.polyval(np.polyint(test_polynomial),max_psnr)-np.polyval(np.polyint(test_polynomial),min_psnr)
    return (math.exp((test_integral-base_integral)/(max_psnr-min_psnr))-1)*100,min_psnr,max_psnr

def benchmark(dataset,config,rel_eb,test_dir,prediction_structure_path):
    variable=f"{config['name']}_{str(rel_eb).replace('.','_')}"
    source_dir=os.path.join(data_root,dataset["folder"])
    os.symlink(os.path.join(source_dir,f"{dataset['variable']}.dat.0"),os.path.join(test_dir,f"{variable}.dat.0"))
    arguments=["-data-dir",test_dir,"-variable",variable,"-time-step","{0}","-M","REL",str(rel_eb),"-reverse-mean-distance-only-order-experimental","-dag-cache",prediction_structure_path,"-predictor",config["predictor"],"-actual-ref-num",str(config["ref_num"]),"-inverse-distance-power",str(config["power"])]
    if dataset["adjust_coord"]:
        arguments.append("-adjust-coord")
    compression_output=run_command([compressor,*arguments])
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
        "config":config["name"],
        "rel_eb":rel_eb,
        "br":read_value(r"^total cr= [^,]+, total br= ([^,]+),",compression_output,"总BR"),
        "psnr":read_value(r"^total cr= [^,]+, total br= [^,]+, total psnr= ([^ ]+)$",compression_output,"总PSNR"),
        "compression_time":read_value(r"^total compression time= ([^ ]+) s$",compression_output,"压缩时间"),
        "decompression_time":read_value(r"^total decompression time= ([^ ]+) s$",decompression_output,"解压时间"),
        "fitting_ratio":read_value(r"^hybrid_loo_fitting_ratio=([^,]+),",compression_output,"混合拟合比例") if config["predictor"]=="hybrid_loo" else math.nan,
    }
    decompressed_psnr=read_value(r"^total decompressed psnr= ([^ ]+)$",decompression_output,"解压总PSNR")
    if not math.isclose(result["psnr"],decompressed_psnr,rel_tol=1e-6,abs_tol=1e-6):
        raise RuntimeError(f"压缩和解压PSNR不一致：{result['psnr']} != {decompressed_psnr}")
    return result

def main():
    result_vec=[]
    temporary_root=os.environ.get("SLURM_TMPDIR","/tmp")
    for dataset in dataset_vec:
        source_dir=os.path.join(data_root,dataset["folder"])
        with tempfile.TemporaryDirectory(prefix=f"unsz_predictor_{dataset['folder']}_",dir=temporary_root) as test_dir:
            os.symlink(os.path.join(source_dir,"coords.dat"),os.path.join(test_dir,"coords.dat"))
            prediction_structure_path=os.path.join(test_dir,"prediction_structure.bin")
            result_map={}
            for config in config_vec:
                config_result_vec=[]
                for rel_eb in rel_eb_vec:
                    result=benchmark(dataset,config,rel_eb,test_dir,prediction_structure_path)
                    result_vec.append(result)
                    config_result_vec.append(result)
                    print(f"{dataset['name']} {config['name']} rel_eb={rel_eb}: br={result['br']:.6f},psnr={result['psnr']:.6f},compression={result['compression_time']:.6f} s,decompression={result['decompression_time']:.6f} s",flush=True)
                result_map[config["name"]]=config_result_vec
            for config in config_vec:
                if config["name"]==base_name:
                    continue
                bd_rate,min_psnr,max_psnr=calculate_bd_rate(result_map[base_name],result_map[config["name"]])
                print(f"{dataset['name']} BD-rate({config['name']} vs {base_name})={bd_rate:.6f}%,overlap_psnr=[{min_psnr:.6f},{max_psnr:.6f}]",flush=True)
    with open(result_path,"w",newline="") as result_file:
        writer=csv.DictWriter(result_file,fieldnames=result_vec[0].keys())
        writer.writeheader()
        writer.writerows(result_vec)
    print(f"结果已写入{result_path}",flush=True)

if __name__=="__main__":
    main()
